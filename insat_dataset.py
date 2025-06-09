#insat with time and dataset
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from affine import Affine
import os
import h5py
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from pyproj import CRS



#from pyproj import CRS
#from affine import Affine
#from rasterio.transform import array_bounds
#from rasterio.warp import calculate_default_transform, reproject, Resampling
#import xesmf as xe
#from scipy.optimize import curve_fit


"""
required insat data
    #insat_seconds_since_imerg# as seconds from time epoch of imerg 6-1-1980 UTM
    #output_insat_data= xr.dataset#an xarray Dataset conataining all data(including metadata) of lat, lon, thermal_TEMP, IMG_MIR_TEMP
an xarray Dataset conatining all data(including metadata) of lat, lon, thermal_TEMP, IMG_MIR_TEMP

required noaa data
    #noaa_seconds_since_imerg# as seconds from time epoch of imerg 6-1-1980 UTM
    #output_noaa_data= xr.open_dataset# as xarray dataset conataining all data(including metadata) from noaa of lat, lon, TPW 

required imerg data
    #output_imerg_data=xr.dataset# as xarray Dataset conatining all data(including metadata) of precipitation, lon, lat, epoch time
"""

# Directories for the datasets
insat_dir = 'C:/Users/adity/Desktop/Data_Set/INSAT_level1C/'
imerg_dir = 'C:/Users/adity/Desktop/Data_Set/GPM_IMERG_late_run/'
noaa_dir = 'C:/Users/adity/Desktop/Data_Set/noaa tpw/Blended-Hydro_TPW_MAP_d20250528'

# List of files in each directory
insat_files = [os.path.join(insat_dir, f) for f in os.listdir(insat_dir) if f.endswith('.h5')]
imerg_files = [os.path.join(imerg_dir, f) for f in os.listdir(imerg_dir) if f.endswith('.HDF5')]
noaa_files = [os.path.join(noaa_dir, f) for f in os.listdir(noaa_dir) if f.endswith('.nc')]



# Sort the IMERG files in ascending order by filename
imerg_files.sort()
insat_files.sort()

# Reference datetimes
insat_ref = datetime(2000, 1, 1, 0, 0, 0)
imerg_ref = datetime(1980, 1, 6, 0, 0, 0)

# Process each INSAT file
for insat_file in insat_files:
    print(f"Processing INSAT file: {insat_file}")
    
    # Open INSAT file and read time data
    with h5py.File(insat_file, 'r') as f_insat:
        time_data = f_insat['time'][:]  # minutes since insat_ref
        
        img_data = f_insat['IMG_MIR'][0, :, :]  # Thermal Infrared1 band
        fill_value = f_insat['IMG_MIR'].attrs['_FillValue']
        img_data = np.where(img_data == fill_value, np.nan, img_data)

        greycount = f_insat['GreyCount'][:]
        img_mir_temp = f_insat['IMG_MIR_TEMP'][:]

        # Projection info
        ulx, uly = f_insat['Projection_Information'].attrs['upper_left_xy(meters)']
        x_res = (f_insat['X'][-1] - f_insat['X'][0]) / (len(f_insat['X']) - 1)
        y_res = (f_insat['Y'][0] - f_insat['Y'][-1]) / (len(f_insat['Y']) - 1)
        transform = Affine.translation(ulx, uly) * Affine.scale(x_res, -y_res)
        width = img_data.shape[1]
        height = img_data.shape[0]
    
    #print(f"INSAT time raw data (minutes since {insat_ref.date()}):\n", time_data)
    
    # Convert INSAT time (minutes since insat_ref) to datetime objects
    insat_datetimes = [insat_ref + timedelta(minutes=m) for m in time_data]
    
    # Convert INSAT datetimes to seconds since imerg_ref as integers
    insat_seconds_aligned = np.array([
        int((dt - imerg_ref).total_seconds()) 
        for dt in insat_datetimes
    ])
    #print(f"INSAT time converted to seconds since {imerg_ref.date()} (IMERG reference) as integers:\n", insat_seconds_aligned)
    
    # For each IMERG file, open and print the time data
    for imerg_file in imerg_files:
        #print(f"Processing IMERG file: {imerg_file}")
        
        with h5py.File(imerg_file, 'r') as f_imerg:
            gtime_data = f_imerg['Grid/time'][:]
            print(f"IMERG time raw data (seconds since {imerg_ref.date()}):\n", gtime_data)
    
    print("----\n")
    
    #Projection 
    # Step 2: Define CRS and transform
    src_crs = CRS.from_proj4('+proj=merc +lon_0=77.25 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6356752.3142 +units=m +no_defs')
    tgt_crs = CRS.from_epsg(4326)
    left, bottom, right, top = array_bounds(height, width, transform)

    # Step 3: Calculate destination transform and size
    transform_dst, width_dst, height_dst = calculate_default_transform(
        src_crs, tgt_crs, width, height, left, bottom, right, top
    )

    # Prepare output array for reprojected img_data
    img_data_reproj = np.full((height_dst, width_dst), np.nan, dtype=img_data.dtype)

    # Step 4: Reproject img_data to lat/lon
    reproject(
        source=img_data,
        destination=img_data_reproj,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=transform_dst,
        dst_crs=tgt_crs,
        resampling=Resampling.nearest
    )

    # Calculate coordinates for lat/lon grid based on transform_dst
    lon = np.linspace(transform_dst.c, transform_dst.c + transform_dst.a * width_dst, width_dst)
    lat = np.linspace(transform_dst.f, transform_dst.f + transform_dst.e * height_dst, height_dst)
    # Reverse if necessary
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        img_data_reproj = img_data_reproj[::-1, :]

    # Step 5: Create xarray Dataset with metadata
    ds = xr.Dataset(
        {
            "IMG_MIR": (("lat", "lon"), img_data_reproj),
            "IMG_MIR_TEMP": (("GreyCount",), img_mir_temp),
            "GreyCount": (("GreyCount",), greycount),
        },
        coords={
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "title": "Processed INSAT IMG_TIR1 data reprojected to EPSG:4326",
            "institution": "BES,SAC/ISRO,Ahmedabad,INDIA.",
            "source": "INSAT-3DR IMG Level1C",
            "projection": "EPSG:4326"
        }
    )

    # Add variable attributes
    ds["IMG_MIR"].attrs.update({
        "long_name": "Thermal MIR Count",
        "units": "counts",
        "_FillValue": np.nan,
        "coordinates": "lat lon"
    })
    ds["IMG_MIR_TEMP"].attrs.update({
        "long_name": "Middle Infrared Brightness Temperature",
        "units": "K"
    })

    # Now `ds` contains all your data + metadata and can be used in-memory
    print(ds)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
