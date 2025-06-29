'''
  This code is designed to handle 'h5' format files and convert it to 'nc' format along with extraction of required variable, resampling due to change in projection, parallel and GPU based execution 
  Flowchart of code
  1. Function for converson of projection from metarcor to global coordinate
  2. Function for extraction of required variables from INSAT file 
  3. Function for saving extracted and resampled data to 'nc' file under same name
  4. Function for executing process in parallel to reduce run   # <------- files directory
'''
import h5py  # For reading HDF5 files
import numpy as np  # For array and numerical operations
import xarray as xr  # For working with multi-dimensional labeled data
import pyproj  # For map projections and coordinate conversions
import torch  # For GPU-based computations
import torch.nn.functional as F  # For grid sampling interpolation on GPU
from pathlib import Path  # For easy and portable file path handling
from concurrent.futures import ProcessPoolExecutor  # For parallel CPU processing
import multiprocessing  # To get number of available CPU cores

# Interpolates data from source lat/lon grid to target lat/lon grid using PyTorch on GPU
def interpolate_torch(data, src_lat, src_lon, target_lat, target_lon):
    
    # Normalize coordinate grid from source to [-1, 1] range for grid_sample
    def normalize(coords, min_val, max_val):
        return 2 * (coords - min_val) / (max_val - min_val) - 1

    # Convert input data array to a 4D PyTorch tensor and move it to GPU
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    # Normalize target grid coordinates (lat and lon)
    norm_lon = normalize(target_lon, src_lon.min(), src_lon.max())
    norm_lat = normalize(target_lat, src_lat.min(), src_lat.max())

    # Create the sampling grid for grid_sample, stacking normalized lon and lat
    grid = torch.stack([
        torch.tensor(norm_lon, dtype=torch.float32),
        torch.tensor(norm_lat, dtype=torch.float32)
    ], dim=-1).unsqueeze(0).cuda()

    # Use PyTorch's grid_sample for bilinear interpolation
    interpolated = F.grid_sample(data_tensor, grid, mode='bilinear', align_corners=True)
    
    # Return interpolated data back as a NumPy array on CPU
    return interpolated.squeeze().cpu().numpy()

# Reads data from one INSAT HDF5 file and prepares lat/lon grids for interpolation
def prepare_file_data(insat_path):
    with h5py.File(insat_path, "r") as insat_file:
        # Load raw satellite image channels and related temperature lookup tables
        x = insat_file["X"][:]  # Projection X-coordinates
        y = insat_file["Y"][:]  # Projection Y-coordinates
        tir1 = insat_file["IMG_TIR1"][0]  # Thermal Infrared Band 1 image data
        tir2 = insat_file["IMG_TIR2"][0]  # Thermal Infrared Band 2 image data
        wv = insat_file["IMG_WV"][0]      # Water vapor channel image data
        tir1_temp = insat_file["IMG_TIR1_TEMP"][:]  # TIR1 temp lookup table
        tir2_temp = insat_file["IMG_TIR2_TEMP"][:]  # TIR2 temp lookup table
        wv_temp = insat_file["IMG_WV_TEMP"][:]      # WV temp lookup table
        greycount = insat_file["GreyCount"][:]      # Grey count information

    # Convert X/Y grid to lon/lat using pyproj (from projection space to geo space)
    # constant values for conversion are available as metadata in each file
    xx, yy = np.meshgrid(x, y)
    proj = pyproj.Proj(proj='merc', lat_ts=17.75, lon_0=77.25, x_0=0, y_0=0,
                       a=6378137.0, b=6356752.3142, datum='WGS84')
    lon, lat = proj(xx, yy, inverse=True)

    # Generate a regular lat/lon target grid with 0.1 degree spacing
    lat_grid = np.arange(lat.min(), lat.max(), 0.10)
    lon_grid = np.arange(lon.min(), lon.max(), 0.10)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

    # Return all needed data for this file
    return {
        "filename": insat_path.stem,  # File name without extension
        "tir1": tir1,
        "tir2": tir2,
        "wv": wv,
        "tir1_temp": tir1_temp,
        "tir2_temp": tir2_temp,
        "wv_temp": wv_temp,
        "greycount": greycount,
        "lat": lat,  # Source latitude grid
        "lon": lon,  # Source longitude grid
        "lat_grid": lat_grid,  # Target grid (1D lat)
        "lon_grid": lon_grid,  # Target grid (1D lon)
        "lat2d": lat2d,        # Target grid (2D lat)
        "lon2d": lon2d         # Target grid (2D lon)
    }

# Performs interpolation and calculations on GPU, then saves output as NetCDF
def process_on_gpu(data, output_dir):
    
    # Interpolate raw TIR1, TIR2, and WV image data onto the new lat/lon grid
    tir1_interp = interpolate_torch(data["tir1"], data["lat"], data["lon"], data["lat2d"], data["lon2d"])
    tir2_interp = interpolate_torch(data["tir2"], data["lat"], data["lon"], data["lat2d"], data["lon2d"])
    wv_interp   = interpolate_torch(data["wv"],   data["lat"], data["lon"], data["lat2d"], data["lon2d"])

    # Map pixel values to temperatures using the lookup tables
    tir1_int = torch.clamp(torch.tensor(tir1_interp, dtype=torch.long), 0, len(data["tir1_temp"]) - 1)
    tir2_int = torch.clamp(torch.tensor(tir2_interp, dtype=torch.long), 0, len(data["tir2_temp"]) - 1)
    wv_int   = torch.clamp(torch.tensor(wv_interp, dtype=torch.long), 0, len(data["wv_temp"]) - 1)

    tir1_temp_grid = torch.tensor(data["tir1_temp"])[tir1_int]
    tir2_temp_grid = torch.tensor(data["tir2_temp"])[tir2_int]
    wv_temp_grid   = torch.tensor(data["wv_temp"])[wv_int]

    # Calculate derived fields (differences and ratios between temperature grids)
    difference_tir = tir1_temp_grid - tir2_temp_grid
    difference_tir1_wv = tir1_temp_grid - wv_temp_grid
    difference_tir2_wv = tir2_temp_grid - wv_temp_grid

    # Division between TIR1 and TIR2 brightness temperatures
    division_tir = torch.divide(tir1_temp_grid, tir2_temp_grid)
    # Replace NaN values (from division by zero) with 0
    division_tir[torch.isnan(division_tir)] = 0

    # Build the final xarray.Dataset to store all output fields
    ds = xr.Dataset(
        {
            "IMG_TIR1": (("lat", "lon"), tir1_interp),
            "IMG_TIR2": (("lat", "lon"), tir2_interp),
            "IMG_WV": (("lat", "lon"), wv_interp),
            "IMG_TIR1_TB": (("lat", "lon"), tir1_temp_grid.numpy()),
            "IMG_TIR2_TB": (("lat", "lon"), tir2_temp_grid.numpy()),
            "IMG_WV_TB": (("lat", "lon"), wv_temp_grid.numpy()),
            "TIR_TB_DIFFERENCE": (("lat", "lon"), difference_tir.numpy()),
            "TIR_TB_DIVISION": (("lat", "lon"), division_tir.numpy()),
            "TIR1_WV_TB_DIFFERENCE": (("lat", "lon"), difference_tir1_wv.numpy()),
            "TIR2_WV_TB_DIFFERENCE": (("lat", "lon"), difference_tir2_wv.numpy()),
            "GreyCount": (("GreyCount",), data["greycount"]),
            "IMG_TIR1_TEMP_COUNT": (("TIR1_TEMP_count",), data["tir1_temp"]),
            "IMG_TIR2_TEMP_COUNT": (("TIR2_TEMP_count",), data["tir2_temp"]),
            "IMG_WV_TEMP_COUNT": (("WV_TEMP_count",), data["wv_temp"])
        },
        coords={
            "lat": data["lat_grid"],  # Latitude coordinates for the grid
            "lon": data["lon_grid"]   # Longitude coordinates for the grid
        }
    )

    # Save the dataset as NetCDF file
    out_file = output_dir / f"{data['filename']}.nc"
    ds.to_netcdf(out_file)
    print(f" Saved: {out_file.name}")

# This function handles the entire workflow:
# 1. Reads all input HDF5 files in parallel (CPU)
# 2. Processes each file one by one on GPU
def process_all_files_parallel(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all .h5 files in the input directory
    files = list(input_dir.glob("*.h5"))
    print(f" Found {len(files)} files")

    # Step 1: Parallel reading/preparation of file data using multiple CPU cores
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        file_data_list = list(executor.map(prepare_file_data, files))

    # Step 2: GPU-based interpolation and saving (sequential because GPUs usually handle 1 process at a time)
    for data in file_data_list:
        process_on_gpu(data, output_dir)
# Run the full pipeline (Read all files and process)
process_all_files_parallel("/kaggle/input/insat-level-1c", "/kaggle/working")
