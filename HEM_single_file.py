#result
    #NOAA TPW dataset created.
    #IMERG precipitation dataset created.
    #INSAT insat_data dataset created.
    #Minimum Brightness Temperature (T_min): 179.86 K
    #Mean Precipitable Water (PW): 37.62 mm
    #Computed Coefficients: A = 94216830.8462, B = 0.079393


#working#final
import os
import h5py
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta, timezone
from pyproj import CRS, Transformer
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xesmf as xe
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator

#file directory
noaa_files = 'C:/Users/adity/Downloads/Blended-Hydro_TPW_MAP_d20250528/BHP-TPW_v04r0_blend_s202505281159300_e202505282300000_c202505282357373.nc'
imerg_files = 'C:/Users/adity/Desktop/Data_Set/GPM_IMERG_late_run/3B-HHR-L.MS.MRG.3IMERG.20250528-S000000-E002959.0000.V07B.HDF5'
insat_files = 'C:/Users/adity/Desktop/Data_Set/INSAT_level1C/3RIMG_28MAY2025_0015_L1C_ASIA_MER_V01R00.h5'


## Define bounding box from insat metadata
lat_min, lat_max = -10.0, 45.5
lon_min, lon_max = 44.5, 110.0



# extract data from NOAA_TPW
noaa_data = nc.Dataset(noaa_files)

# Read lat/lon
lat_noaa = noaa_data.variables['lat'][:]
lon_noaa = noaa_data.variables['lon'][:]

# boundation on noaa file
lat_idx = np.where((lat_noaa >= lat_min) & (lat_noaa <= lat_max))[0]
lon_idx = np.where((lon_noaa >= lon_min) & (lon_noaa <= lon_max))[0]

# TPW dataset
TPW = noaa_data.variables['TPW'][
    lat_idx.min():lat_idx.max() + 1,
    lon_idx.min():lon_idx.max() + 1
]
noaa_data.close()
print("NOAA TPW dataset created.")



#extract data from IMERG
with h5py.File(imerg_files, 'r') as imerg:
    imerg_lat = imerg['/Grid/lat'][:]
    imerg_lon = imerg['/Grid/lon'][:]
    imerg_precip = imerg['/Grid/precipitation'][0]

# 2D meshgrid
lon_grid, lat_grid = np.meshgrid(imerg_lon, imerg_lat, indexing='ij')
mask = (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)
precip_masked = np.where(mask, imerg_precip, np.nan)

#precipitation dataset
precipitation = xr.Dataset(
    {"precipitation": (["lon", "lat"], precip_masked)
        },
    coords={
        "lon": imerg_lon,
        "lat": imerg_lat
    }
)
print("IMERG precipitation dataset created.")



# extract data from INSAT
with h5py.File(insat_files, 'r') as insat:
    insat_gc = insat['GreyCount'][:]
    insat_temp = insat['IMG_TIR1_TEMP'][:]
    IMG_TIR1 = insat['IMG_TIR1'][0, :, :]  # First time slice
    X = insat['X'][:]
    Y = insat['Y'][:]

#PRojection information
    proj_info = insat['Projection_Information']
    central_lon = proj_info.attrs['longitude_of_projection_origin'].item()
    standard_parallel = proj_info.attrs['standard_parallel'].item()
    semi_major = proj_info.attrs['semi_major_axis'].item()
    semi_minor = proj_info.attrs['semi_minor_axis'].item()

#projection transformation
proj_str = (
    f"+proj=merc +lon_0={central_lon} +lat_ts={standard_parallel} "
    f"+a={semi_major} +b={semi_minor} +units=m +no_defs"
)
crs_proj = CRS.from_proj4(proj_str)
crs_wgs84 = CRS.from_epsg(4326)
transformer = Transformer.from_crs(crs_proj, crs_wgs84, always_xy=True)

# Transforming X,Y to lat,lon
lat, lon = np.meshgrid(X, Y)
lat_insat, lon_insat = transformer.transform(lat, lon)

# Interpolate TIR1 to temperature
func = RegularGridInterpolator((insat_gc,), insat_temp, bounds_error=False, fill_value=np.nan)
IMG_TIR1_TEMP_grid = func(IMG_TIR1.ravel()).reshape(IMG_TIR1.shape)

# insar_dat dataset
insat_data = xr.Dataset(
    {
        'IMG_TIR1_TEMP_grid': (['Y', 'X'], IMG_TIR1_TEMP_grid)
    },
    coords={
        'lat': (['Y', 'X'], lat_insat),
        'lon': (['Y', 'X'], lon_insat),
    }
)
print("INSAT insat_data dataset created.")

#Dyanmic coefficients calculation for core fraction only
def coefficients(T_min, PW_mm):
    """
    Computing A and B coefficients for RR = A * exp(-B * T)
    """
    #converting to inches from ATBD
    PW_in = PW_mm / 25.4
    # Rain rate at 240K
    RR_240 = 0.5  
    # Rain rate at T_min
    RR_max = 40 * PW_in  

    ln_RR_240 = np.log(RR_240)
    ln_RR_max = np.log(RR_max)

    B = (ln_RR_240 - ln_RR_max) / (T_min - 240)
    A = RR_240 / np.exp(-B * 240)

    return A, B

# Get T_min from INSAT
T_min = np.nanmin(IMG_TIR1_TEMP_grid)
print(f"Minimum Brightness Temperature (T_min): {T_min:.2f} K")

# Get average TPW (in mm) from NOAA
PW_mean = np.nanmean(TPW)
print(f"Mean Precipitable Water (PW): {PW_mean:.2f} mm")

# Compute coefficients
A, B = coefficients(T_min, PW_mean)
print(f"Computed Coefficients: A = {A:.4f}, B = {B:.6f}")
