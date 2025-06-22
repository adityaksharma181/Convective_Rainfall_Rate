# INSAT files uses Metarcor projection which needs to be converted into Global Co-ordinate system along with resampling of required variables
# USE CUSTOM PATH BEFORE ECECUTING THE FILE
import h5py
import numpy as np
import xarray as xr
import pyproj
from scipy.interpolate import griddata

# INSAT-3DR level 1C file path
insat_path = "Folder/3RIMG_28MAY2025_1315_L1C_ASIA_MER_V01R00.h5" #     <---------- File Path

# Extraction of important variables
with h5py.File(insat_path, "r") as insat_file:

    x = insat_file["X"][:]
    y = insat_file["Y"][:]
    tir1 = insat_file["IMG_TIR1"][0]
    tir2 = insat_file["IMG_TIR2"][0]
    wv = insat_file["IMG_WV"][0]
    greycount = insat_file["GreyCount"][:]
    tir1_temp = insat_file["IMG_TIR1_TEMP"][:]
    tir2_temp = insat_file["IMG_TIR2_TEMP"][:]
    wv_temp = insat_file["IMG_WV_TEMP"][:]

# Projection conversion: values are available in form of metadata and can be extracted using software "Panoply" 
    xx, yy = np.meshgrid(x, y)
    proj = pyproj.Proj(proj='merc', lat_ts=17.75, lon_0=77.25, x_0=0, y_0=0,
                       a=6378137.0, b=6356752.3142, datum='WGS84')
    lon, lat = proj(xx, yy, inverse=True)

# Flatten and interpolate of Geo 2D variables 
lat_flat = lat.flatten()
lon_flat = lon.flatten()
tir1_flat = tir1.flatten()
tir2_flat = tir2.flatten()
wv_flat = wv.flatten()

# Regular lat-lon grid for resampling 
lat_grid = np.arange(lat.min(), lat.max(), 0.10)
lon_grid = np.arange(lon.min(), lon.max(), 0.10)
lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)

# Interpolate raw data into regular grid using linear interpolation
tir1_interp = griddata((lat_flat, lon_flat), tir1_flat, (lat2d, lon2d), method='linear')
tir2_interp = griddata((lat_flat, lon_flat), tir2_flat, (lat2d, lon2d), method='linear')
wv_interp = griddata((lat_flat, lon_flat), wv_flat, (lat2d, lon2d), method='linear')

# Convert counts to brightness temperature
tir1_int = np.clip(tir1_interp.astype(int), 0, len(tir1_temp) - 1)
tir2_int = np.clip(tir2_interp.astype(int), 0, len(tir2_temp) - 1)
wv_int = np.clip(wv_interp.astype(int), 0, len(wv_temp) - 1)

# Lookup tables to convert digital counts to brightness temperature
tir1_temp_grid = tir1_temp[tir1_int]
tir2_temp_grid = tir2_temp[tir2_int]
wv_temp_grid = wv_temp[wv_int]

# Derived fields from temperature gird
difference_tir = tir1_temp_grid - tir2_temp_grid
difference_tir1_wv = tir1_temp_grid - wv_temp_grid
difference_tir2_wv = tir1_temp_grid - wv_temp_grid
division_tir = np.divide(tir1_temp_grid, tir2_temp_grid, out=np.full_like(tir1_temp_grid, np.nan), where=tir2_temp_grid != 0)

# Xarray dataset to store all required variables
insat_output = xr.Dataset(
    {
        "IMG_TIR1": (("lat", "lon"), tir1_interp),
        "IMG_TIR2": (("lat", "lon"), tir2_interp),
        "IMG_WV": (("lat", "lon"), wv_interp),
        "IMG_TIR1_TB": (("lat", "lon"), tir1_temp_grid),
        "IMG_TIR2_TB": (("lat", "lon"), tir2_temp_grid),
        "IMG_WV_TB": (("lat", "lon"), wv_temp_grid),
        "TIR_TB_DIFFERENCE": (("lat", "lon"), difference_tir),
        "TIR_TB_DIVISION": (("lat", "lon"), division_tir),
        "TIR1_WV_TB_DIFFERENCE": (("lat", "lon"), difference_tir1_wv),
        "TIR2_WV_TB_DIFFERENCE": (("lat", "lon"), difference_tir2_wv),
        "GreyCount": (("GreyCount",), greycount),
        "IMG_TIR1_TEMP_COUNT": (("TIR1_TEMP_count",), tir1_temp),
        "IMG_TIR2_TEMP_COUNT": (("TIR2_TEMP_count",), tir2_temp),
        "IMG_WV_TEMP_COUNT": (("WV_TEMP_count",), wv_temp)
    },
    coords={
        "lat": lat_grid,
        "lon": lon_grid
    }
)
# save and completion text
print(insat_output)
insat_output.to_netcdf("insat_resampled.nc")  #  <------------ File Path
print("✅ INSAT processing complete.")
