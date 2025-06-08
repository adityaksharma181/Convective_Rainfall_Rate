#for single file each from insat, noaa, imerg
import os
import h5py
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from pyproj import CRS
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import xesmf as xe
from scipy.optimize import curve_fit

"""
required insat data
    #insat_seconds_since_imerg# as seconds from time epoch of imerg 6-1-1980 UTM
    #output_insat_data= xr.dataset#an xarray Dataset conataining all data(including metadata) of lat, lon, thermal_TEMP, IMG_MIR_TEMP
an xarray Dataset conataining all data(including metadata) of lat, lon, thermal_TEMP, IMG_MIR_TEMP

required noaa data
    #noaa_seconds_since_imerg# as seconds from time epoch of imerg 6-1-1980 UTM
    #output_noaa_data= xr.open_dataset# as xarray dataset conataining all data(including metadata) from noaa of lat, lon, TPW 

required imerg data
    #output_imerg_data=xr.dataset# as xarray Dataset conataining all data(including metadata) of precipitation, lon, lat, epoch time
"""

#importing files
insat_files = 'C:/Users/adity/Desktop/Data_Set/INSAT_level1C/3RIMG_28MAY2025_0015_L1C_ASIA_MER_V01R00.h5'
imerg_files = 'C:/Users/adity/Desktop/Data_Set/GPM_IMERG_late_run/3B-HHR-L.MS.MRG.3IMERG.20250528-S003000-E005959.0030.V07B.HDF5'
noaa_files = 'C:/Users/adity/Downloads/Blended-Hydro_TPW_MAP_d20250528/BHP-TPW_v04r0_blend_s202505281159300_e202505282300000_c202505282357373.nc'


#INSAT FILE OPERATIONS-------------------------------------------------------------------------------------------------------------------
#give nc file consists of lat,lon, Greycount, Thermal_MIR, IMG_MIR_TEMP
#separately converted the time epoch to imerg and gives value in seconds variable=insat_seconds_since_imerg
with h5py.File(insat_files, 'r') as insat:
    # Read the MIR (middle infrared) band data, mask fill values with NaN
    insat_mir = insat['IMG_MIR'][0, :, :]
    fill_value = insat['IMG_MIR'].attrs['_FillValue']
    insat_mir = np.where(insat_mir == fill_value, np.nan, insat_mir)

    # Read projection parameters and compute affine transform
    ulx, uly = insat['Projection_Information'].attrs['upper_left_xy(meters)']
    x_res = (insat['X'][-1] - insat['X'][0]) / (len(insat['X']) - 1)
    y_res = (insat['Y'][0] - insat['Y'][-1]) / (len(insat['Y']) - 1)
    transform = Affine.translation(ulx, uly) * Affine.scale(x_res, -y_res)

    # Dimensions of the input image
    width = insat_mir.shape[1]
    height = insat_mir.shape[0]

    # Read Greycount, time and MIR temperature datasets
    greycount = insat['GreyCount'][:]  # shape may be 1D or 2D
    img_mir_temp = insat['IMG_MIR_TEMP'][:]  # shape may be 1D or 2D
    insat_time = insat['time'][:]

    # Attempt to extract acquisition time
    if 'time' in insat.attrs:
        acquisition_time = insat.attrs['time']
    elif 'time' in insat:
        acquisition_time = insat['time'][:]
    else:
        acquisition_time = 'Unknown'

# Debug: print dataset shapes
print("insat_mir shape:", insat_mir.shape)
print("greycount shape:", greycount.shape)
print("IMG_MIR_TEMP shape:", img_mir_temp.shape)

# Define source (INSAT) and target (WGS84) coordinate systems
src_crs = CRS.from_proj4(
    '+proj=merc +lon_0=77.25 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6356752.3142 +units=m +no_defs'
)
tgt_crs = CRS.from_epsg(4326)

# Calculate bounds of the source image
left, bottom, right, top = array_bounds(height, width, transform)

# Calculate transformation for reprojection
transform_dst, width_dst, height_dst = calculate_default_transform(
    src_crs, tgt_crs, width, height, left, bottom, right, top
)

# Prepare an empty array for the reprojected data
dst_array = np.empty((height_dst, width_dst), dtype=insat_mir.dtype)

# Reproject the MIR image to lat/lon (WGS84)
reproject(
    source=insat_mir,
    destination=dst_array,
    src_transform=transform,
    src_crs=src_crs,
    dst_transform=transform_dst,
    dst_crs=tgt_crs,
    resampling=Resampling.nearest,
)

# Generate 1D coordinate arrays for latitude and longitude
cols = np.arange(width_dst)
rows = np.arange(height_dst)
lons = transform_dst * (cols + 0.5, np.zeros_like(cols) + 0.5)
lats = transform_dst * (np.zeros_like(rows) + 0.5, rows + 0.5)

lons_1d = np.array(lons[0])  # longitude values
lats_1d = np.array(lats[1])  # latitude values

# Setup coordinate arrays for greycount and MIR temperature
if len(greycount.shape) == 1:
    x_coord = np.arange(greycount.shape[0])
    y_coord = None
elif len(greycount.shape) == 2:
    y_coord = np.arange(greycount.shape[0])
    x_coord = np.arange(greycount.shape[1])
else:
    raise ValueError(f"Unexpected greycount shape: {greycount.shape}")

if len(img_mir_temp.shape) == 1:
    dim0 = np.arange(img_mir_temp.shape[0])
    dim1 = None
elif len(img_mir_temp.shape) == 2:
    dim0 = np.arange(img_mir_temp.shape[0])
    dim1 = np.arange(img_mir_temp.shape[1])
else:
    raise ValueError(f"Unexpected IMG_MIR_TEMP shape: {img_mir_temp.shape}")

# Build coordinates dictionary for xarray dataset
coords = {
    'lat': lats_1d,
    'lon': lons_1d,
}

if y_coord is not None:
    coords['y'] = y_coord
coords['x'] = x_coord

if dim1 is not None:
    coords['dim0'] = dim0
    coords['dim1'] = dim1
    ds_vars = {
        'thermal_infrared_1': (('lat', 'lon'), dst_array),
        'Greycount': (('y', 'x'), greycount),
        'IMG_MIR_TEMP': (('dim0', 'dim1'), img_mir_temp),
    }
else:
    coords['dim0'] = dim0
    ds_vars = {
        'thermal_MIR': (('lat', 'lon'), dst_array),
        'Greycount': (('x',), greycount),
        'IMG_MIR_TEMP': (('dim0',), img_mir_temp),
    }

# Create an xarray Dataset
output_insat_data = xr.Dataset(ds_vars, coords=coords, attrs={
    'acquisition_time': acquisition_time,
    'source_file': insat_files
})

# Add variable metadata
output_insat_data['thermal_MIR'].attrs['units'] = 'unknown'
output_insat_data['thermal_MIR'].attrs['long_name'] = 'INSAT Thermal Infrared Band 1'

output_insat_data['Greycount'].attrs['description'] = 'Greycount data from original HDF5'
output_insat_data['IMG_MIR_TEMP'].attrs['description'] = 'IMG_MIR_TEMP data from original HDF5'




#conversion to time epoch of imerg------
insat_ref = datetime(2000, 1, 1, 0, 0, 0)
imerg_ref = datetime(1980, 1, 6, 0, 0, 0)
with h5py.File(imerg_files, 'r') as f:
    gtime_data = f['Grid/time'][:] 
    #print(f"IMERG time raw data (seconds since {imerg_ref.date()}):\n", gtime_data)

    #print(f"INSAT time raw data (minutes since {insat_ref.date()}):\n", insat_time)
    #Convert INSAT time to seconds since IMERG reference
    insat_datetimes = [insat_ref + timedelta(minutes=m) for m in insat_time]
    # Convert INSAT datetimes to seconds since IMERG reference
    insat_seconds_since_imerg = np.array([(dt - imerg_ref).total_seconds() for dt in insat_datetimes])
    #print(f"INSAT time converted to seconds since {imerg_ref.date()} (IMERG reference):\n", insat_seconds_since_imerg)





#working #imerg data into GLOBAL_CE_16km system #output==lat,lon,precip,time------------------------------------------------------------------


# --------------------
# Step 1: Load HDF5 File
# --------------------
# Open dataset using h5netcdf (xarray backend)
imerg_data = xr.open_dataset(imerg_files, group="Grid", engine="h5netcdf")

# Extract required variables
precip = imerg_data['precipitation']
imerg_time = imerg_data['time']

# Replace fill values with NaN
precip = precip.where(precip != -9999.9)

# --------------------
# Step 2: Build Target Grid (~16 km ≈ 0.15°)
# --------------------
lon_start = 44.5
lon_end = 110.0
lat_start = -10.0
lat_end = 45.5
resolution = 0.15

lon = np.arange(lon_start, lon_end + resolution, resolution)
lat = np.arange(lat_start, lat_end + resolution, resolution)

imerg_data_out = xr.Dataset(
    {
        "lon": (["lon"], lon),
        "lat": (["lat"], lat)
    }
)

# --------------------
# Step 3: Regrid Using xESMF
# --------------------
regridder = xe.Regridder(precip, imerg_data_out, method="bilinear", filename="weights_GLOBAL_CE_16km.nc", reuse_weights=False)
precip_regridded = regridder(precip)

# --------------------
# Step 4: Save Output
# --------------------
output_imerg_data = xr.Dataset(
    {
        "precipitation": precip_regridded
    },
    coords={
        "lat": imerg_data_out['lat'],
        "lon": imerg_data_out['lon']
    }
)






#noaa----------------------------------------------------------------------------------------------------------------------------------------
#working#changing to time epoch for NOAA wrt imerg ==== gives seconds from epoch


with h5py.File(noaa_files, 'r') as noaa:
    noaa_lon = noaa['lon'][:]
    noaa_lat = noaa['lat'][:]
    noaa_tpw = noaa['TPW'][:]


gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

# Get just the filename part
base_filename = os.path.basename(noaa_files)

# Extract the 's' timestamp (starts with 'sYYYYMMDDHHMMSSS')
parts = base_filename.split('_')
start_str = next(part for part in parts if part.startswith('s'))[1:]

# Parse date and time
date_part = start_str[:8]        # '20250528'
time_part = start_str[8:]        # '1159300' -> HHMMSS + tenth of second

# Convert time components
hour = int(time_part[:2])
minute = int(time_part[2:4])
second = int(time_part[4:6])
tenth_second = int(time_part[6]) / 10.0

# Construct datetime object
timestamp = datetime(
    year=int(date_part[:4]),
    month=int(date_part[4:6]),
    day=int(date_part[6:8]),
    hour=hour,
    minute=minute,
    second=second,
    microsecond=int(tenth_second * 1e6),
    tzinfo=timezone.utc
)

# Compute seconds since GPS epoch
noaa_seconds_since_imerg = int((timestamp - gps_epoch).total_seconds())

# xarray dataset of noaa
output_noaa_data = xr.open_dataset(noaa_files, engine="h5netcdf")


"""
required insat data
    #insat_seconds_since_imerg# as seconds from time epoch of imerg 6-1-1980 UTM
    #output_insat_data= xr.dataset#an xarray Dataset conataining all data(including metadata) of lat, lon, thermal_TEMP, IMG_MIR_TEMP
an xarray Dataset conataining all data(including metadata) of lat, lon, thermal_TEMP, IMG_MIR_TEMP

required noaa data
    #noaa_seconds_since_imerg# as seconds from time epoch of imerg 6-1-1980 UTM
    #output_noaa_data= xr.open_dataset# as xarray dataset conataining all data(including metadata) from noaa of lat, lon, TPW 

required imerg data
    #output_imerg_data=xr.dataset# as xarray Dataset conataining all data(including metadata) of precipitation, lon, lat, epoch time


extraction of Brightness temperature from output_insat_data
extraction of TPW from output_noaa_data
extraction of precipitaion from output_imerg_data"""



#temporal condition
#for insat and imerg
time_diff1 = abs((np.datetime64(insat_seconds_since_imerg) - np.datetime64(imerg_time)) / np.timedelta64(1, 's'))
#for noaa and imerg
time_diff2 = abs((np.datetime64(noaa_seconds_since_imerg) - np.datetime64(imerg_time)) / np.timedelta64(1, 's'))
#condition to be within 30 min w
if time_diff1 <30 and time_diff2 <30:
    TPW = np.array([output_noaa_data['TPW']])
    TB = np.array([output_insat_data['IMG_MIR_TEMP']])
    R = np.array([output_imerg_data['precipitation']])

    # Define the brightness temperature threshold (Kelvin)
    Tc = 240 #ATBD mosdac
    #HEM model function to fit rainfall data
def hem_model(inputs, a0, a1, a2, b0, b1, b2):
    """
    HEM rainfall estimation model:
    R = a(TPW) * exp(-b(TPW) * (TB - Tc)) if TB < Tc, else 0

    Parameters:
    - inputs: TPW and TB arrays
    - a0, a1, a2: coefficients for quadratic function of a(TPW)
    - b0, b1, b2: coefficients for quadratic function of b(TPW)

    Returns:
    - Estimated rainfall array corresponding to inputs
    """
    tpw, tb = inputs

    # Calculate empirical coefficient 'a' as quadratic function of TPW
    a = a0 + a1 * tpw + a2 * tpw**2

    # Calculate empirical coefficient 'b' as quadratic function of TPW
    b = b0 + b1 * tpw + b2 * tpw**2

    # Apply model formula only where TB < Tc, else rainfall = 0
    rainfall_est = np.where(tb < Tc, a * np.exp(-b * (tb - Tc)), 0.0)

    return rainfall_est

# Pack TPW and TB as inputs to the model
inputs = (TPW, TB)

# Initial guess for the parameters to start curve fitting
initial_guess = [0.1, 0.01, 0.001, 0.01, 0.001, 0.0001]

# Use curve_fit to optimize parameters minimizing the difference
# between observed rainfall (R) and model estimates
params_opt, params_cov = curve_fit(hem_model, inputs, R, p0=initial_guess, maxfev=10000)

# Extract optimized coefficients
a0, a1, a2, b0, b1, b2 = params_opt

# Print the fitted empirical coefficients
print("Fitted coefficients:")
print(f"a0 = {a0}, a1 = {a1}, a2 = {a2}")
print(f"b0 = {b0}, b1 = {b1}, b2 = {b2}")







