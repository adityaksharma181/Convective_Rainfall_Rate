'''
    This code is designed to take .h5 format files from directory (to be out in code ) and convert each file to .nc format with only required variables that is precipitation, latitude and longitude.
'''
import h5py                    # For reading HDF5 files
import xarray as xr            # For creating and manipulating labeled datasets 
import numpy as np             # For numerical operations
import os                      # For working with file paths
from glob import glob          # For finding files matching a pattern (e.g., *.HDF5)

# Set input directory
input_dir = "/kaggle/input/imerg-late-run/" # <----- HDF5 files are stored

# Set output directory
output_dir = "/kaggle/working/" # <----- save NetCDF files

# Define spatial boundary using INSAT as reference ( INSAT is restricted to these global co-ordinate lat, lon )
lat_min, lat_max = -10.0, 45.5         # Minimum and maximum latitude
lon_min, lon_max = 44.5, 110.0         # Minimum and maximum longitude

# Find all HDF5 files in input directory
hdf_files = glob(os.path.join(input_dir, "*.HDF5"))  # List of all files ending with .HDF5

# Loop over each HDF5 file
for file_path in hdf_files:
    # Get file name without path
    filename = os.path.basename(file_path)

    # Create output filename by replacing .HDF5 extension with .nc
    output_filename = filename.replace(".HDF5", ".nc")

    # Full output path
    output_path = os.path.join(output_dir, output_filename)

    # Open and read data from HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Read long, lat and precipitation data from HDF5
        lon = f['Grid']['lon'][:]                         # 1D array of longitudes
        lat = f['Grid']['lat'][:]                         # 1D array of latitudes
        precip = f['Grid']['precipitation'][0, :, :]      # 2D array of precipitation values

        # Handle missing data byconvert fill value -9999.9 to NaN
        precip = np.where(precip == -9999.9, np.nan, precip)

    # Transpose the precipitation array to match xarray convention (lat, lon)
    # Original shape: (lon, lat), changing to (lat, lon)
    precip = precip.T

    # Create an xarray Dataset for easy NetCDF saving
    output_imerg = xr.Dataset(
        {
            "precipitation": (["lat", "lon"], precip)     # Main variable with dimensions (lat, lon)
        },
        coords={
            "lat": lat,                                   # Latitude coordinate
            "lon": lon                                    # Longitude coordinate
        }
    )

    # Subset data based on the desired lat/lon boundary box
    output_imerg_subset = output_imerg.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Add metadata / attributes for better understanding in the output NetCDF
    output_imerg_subset["precipitation"].attrs["units"] = "mm/hr"                  # Unit of precipitation
    output_imerg_subset["precipitation"].attrs["long_name"] = "IMERG Precipitation Rate"  # Description
    output_imerg_subset["lat"].attrs["units"] = "degrees_north"                    # Latitude unit
    output_imerg_subset["lon"].attrs["units"] = "degrees_east"                     # Longitude unit

    # Save the subset dataset to NetCDF format
    output_imerg_subset.to_netcdf(output_path)

    # Print confirmation for each file processed
    print(f"Saved: {output_path}")
