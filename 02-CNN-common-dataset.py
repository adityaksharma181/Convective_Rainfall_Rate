'''
  Use of Custom weights for Heavy rainfalls.
  This code is designed to create common normalised dataset from data extraction from recently created nc format files of INSAT and IMERG, with temporal and spatial alignment with proper resampling wherever required
  Flowchart of code
  1. input directories
  2. Function for time extraction using file names
  3. Function for temporal alignment
  4. Function for data extraction from valid files along with masking of bad data points
  5. normalise vall extracted variables using minmax ( linear scaling )
  6. Create common dataset
  7. Function for custom weights
'''
# Import necessary libraries for file operations, regex, and datetime handling
import os
import re
from glob import glob
from datetime import datetime

# Numerical and scientific computing libraries
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# Import TensorFlow and Keras for building the CNN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer, Conv2D, MaxPooling2D,
    BatchNormalization, Flatten, Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define input data directories
INSAT_DIR = "/kaggle/input/nc-insat"  # Directory where INSAT satellite data (NetCDF files) are stored
IMERG_DIR = "/kaggle/input/nc-imerg"  # Directory where IMERG rainfall data (NetCDF files) are stored

# Define spatial boundary (latitude and longitude) for clipping the data
lat_min_clip, lat_max_clip = 8.07, 37.1
lon_min_clip, lon_max_clip = 68.12, 97.42

# Parse date and time from INSAT filenames
def parse_insat_time(filename):
    match = re.search(r'3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})', filename)  # Extract day, month string, year, and time
    if not match:
        return None  # Return None if the pattern doesn't match
    day, mon_str, year, time_str = match.groups()
    month_map = {
        'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
    }  # Map month strings to month numbers
    hour, minute = int(time_str[:2]), int(time_str[2:])  # Extract hour and minute
    return datetime(int(year), month_map[mon_str], int(day), hour, minute)  # Return datetime object

# Parse date and time from IMERG filenames
def parse_imerg_time(filename):
    match = re.search(r'3IMERG\.(\d{8})-S(\d{6})', filename)  # Extract date and start time
    if not match:
        return None
    date_str, time_str = match.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")  # Convert to datetime object

# Find closest IMERG file for a given INSAT timestamp (within 30 minutes tolerance)
def find_closest_imerg(insat_time, imerg_list, tol_secs=1800):
    best, mindiff = None, tol_secs + 1  # Initialize variables
    for path, imt in imerg_list:
        if imt is None:
            continue  # Skip files with invalid timestamps
        diff = abs((insat_time - imt).total_seconds())  # Calculate time difference in seconds
        if diff < mindiff:  # Keep track of the closest match
            mindiff, best = diff, path
    return best  # Return path to closest IMERG file

# Process one INSAT-IMERG pair and return feature-target grids
def process_pair(insat_path, insat_time, imerg_list):
    imerg_path = find_closest_imerg(insat_time, imerg_list)  # Find closest IMERG file
    if not imerg_path or not os.path.exists(imerg_path):
        return None  # Skip if no matching IMERG file exists

    # Load both datasets using xarray
    insat_ds = xr.open_dataset(insat_path)
    imerg_ds = xr.open_dataset(imerg_path)

    # Clip both datasets to defined spatial bounds (lat-lon box)
    insat_ds = insat_ds.sel(
        lat=slice(lat_min_clip, lat_max_clip),
        lon=slice(lon_min_clip, lon_max_clip)
    )
    imerg_ds = imerg_ds.sel(
        lat=slice(lat_min_clip, lat_max_clip),
        lon=slice(lon_min_clip, lon_max_clip)
    )

    # Rename coordinates if INSAT dataset uses different names (latitude, longitude → lat, lon)
    if 'latitude' in insat_ds.coords and 'longitude' in insat_ds.coords:
        insat_ds = insat_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Extract required INSAT image bands (brightness temperatures)
    try:
        tir1 = insat_ds['IMG_TIR1_TB']
        tir2 = insat_ds['IMG_TIR2_TB']
        wv = insat_ds['IMG_WV_TB']
    except KeyError:
        return None  # Skip files missing required bands

    # Generate derived features (differences between channels)
    tir1_wv = tir1 - wv
    tir2_wv = tir2 - wv

    # Combine all features into one xarray Dataset
    features = xr.Dataset({
        'WV': wv,
        'TIR1': tir1,
        'TIR2': tir2,
        'TIR1_WV': tir1_wv,
        'TIR2_WV': tir2_wv
    })

    # Interpolate INSAT features to IMERG grid for spatial alignment
    features_interp = features.interp(
        lat=imerg_ds['lat'],
        lon=imerg_ds['lon'],
        method='nearest'
    )

    # Convert xarray dataset into NumPy arrays
    X_grid = np.stack(
        [features_interp[var].values for var in features_interp.data_vars],  # Stack each feature
        axis=-1  # Last axis holds feature channels
    )
    y_grid = imerg_ds['precipitation'].values  # Get IMERG precipitation target

    # Create a mask to filter out invalid values (NaNs, Infs)
    mask = np.all(np.isfinite(X_grid), axis=-1) & np.isfinite(y_grid)
    if not mask.any():
        return None  # Skip if all values are invalid

    # Replace invalid values with zeros
    X_grid[~mask] = 0.0
    y_grid[~mask] = 0.0

    return X_grid, y_grid  # Return feature and target grids

# Load INSAT and IMERG file lists with parsed times
# Load INSAT file paths and corresponding times
insat_files = [
    (f, parse_insat_time(f)) for f in sorted(glob(os.path.join(INSAT_DIR, '*.nc')))
]
insat_files = [(f, t) for f, t in insat_files if t is not None]  # Filter out files with invalid timestamps

# Load IMERG file paths and corresponding times
imerg_files = [
    (f, parse_imerg_time(f)) for f in sorted(glob(os.path.join(IMERG_DIR, '*.nc')))
]
imerg_files = [(f, t) for f, t in imerg_files if t is not None]  # Filter out files with invalid timestamps

# Extract feature-target pairs from dataset
X_list, y_list = [], []  # Initialize empty lists for features and targets
for insat_path, insat_time in insat_files:
    result = process_pair(insat_path, insat_time, imerg_files)  # Process each INSAT file
    if result:
        X_list.append(result[0])  # Append feature grid
        y_list.append(result[1])  # Append target grid

# Convert lists into NumPy arrays
X = np.array(X_list)
y = np.array(y_list)
print(f"Loaded {X.shape[0]} samples: X={X.shape}, y={y.shape}")  # Print dataset summary

# Split into train, validation, and test sets
# 70% of data goes to train, 15% to val, 15% to test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42
)

# Reshape target data for model training
# Flatten target grids for regression (CNN will output flattened predictions)
y_train_flat = y_train.reshape((y_train.shape[0], -1))
y_val_flat   = y_val.reshape((y_val.shape[0], -1))
y_test_flat  = y_test.reshape((y_test.shape[0], -1))

# Define custom Weighted MSE Loss Function
# This gives more penalty to errors on heavy rainfall areas
def weighted_mse(y_true, y_pred):
    weight = (
        1.0
        + 5.0 * tf.cast(y_true > 20.0, tf.float32)  # Apply weight multiplier for rainfall > 20 mm/hr
        + 10.0 * tf.cast(y_true > 40.0, tf.float32)  # Larger multiplier for rainfall > 40 mm/hr
    )
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))  # Weighted Mean Squared Error

