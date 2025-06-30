# Common dataset
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

import os  # Operating system interface for file path handling
import re  # Regular expressions for filename parsing
from glob import glob  # For file pattern matching
from datetime import datetime  # For handling datetime objects
import numpy as np  # Numerical computing
import xarray as xr  # Working with multi-dimensional scientific data
import matplotlib.pyplot as plt  # Plotting (though not used in this script)
import warnings  # Handling warnings
import tensorflow as tf  # TensorFlow for machine learning
from tensorflow.keras.models import Sequential  # Sequential model API
from tensorflow.keras.layers import (  # Layers for the neural network
    InputLayer, Conv2D, MaxPooling2D,
    BatchNormalization, Flatten, Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam  # Optimizer
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.metrics import mean_squared_error  # Evaluation metric

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# Define input data directories
INSAT_DIR = "/kaggle/working/"  # Directory for INSAT satellite data
IMERG_DIR = "/kaggle/working/"  # Directory for IMERG rainfall data

# Define spatial boundary (latitude and longitude) for clipping the data
lat_min_clip, lat_max_clip = 8.07, 37.1  # Latitude range
lon_min_clip, lon_max_clip = 68.12, 97.42  # Longitude range

# Parse date and time from INSAT filenames
def parse_insat_time(filename):
    match = re.search(r'3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})', filename)  # Regex match
    if not match:
        return None  # Return None if pattern not found
    day, mon_str, year, time_str = match.groups()  # Extract parts
    month_map = {  # Month abbreviation to number mapping
        'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
    }
    hour, minute = int(time_str[:2]), int(time_str[2:])  # Extract hour and minute
    return datetime(int(year), month_map[mon_str], int(day), hour, minute)  # Return datetime object

# Parse date and time from IMERG filenames
def parse_imerg_time(filename):
    match = re.search(r'3IMERG\.(\d{8})-S(\d{6})', filename)  # Regex match
    if not match:
        return None  # Return None if pattern not found
    date_str, time_str = match.groups()  # Extract parts
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")  # Return datetime object

# Find closest IMERG file for a given INSAT timestamp (within 30 minutes tolerance)
def find_closest_imerg(insat_time, imerg_list, tol_secs=1800):
    best, mindiff = None, tol_secs + 1  # Initialize best match
    for path, imt in imerg_list:  # Loop through IMERG files
        if imt is None:
            continue  # Skip if IMERG time is None
        diff = abs((insat_time - imt).total_seconds())  # Time difference in seconds
        if diff < mindiff:  # Check if closer
            mindiff, best = diff, path  # Update best match
    return best  # Return closest IMERG file path

# Process one INSAT-IMERG pair and return feature-target grids
def process_pair(insat_path, insat_time, imerg_list):
    imerg_path = find_closest_imerg(insat_time, imerg_list)  # Get closest IMERG file
    if not imerg_path or not os.path.exists(imerg_path):
        return None  # Skip if IMERG file not found

    insat_ds = xr.open_dataset(insat_path)  # Open INSAT dataset
    imerg_ds = xr.open_dataset(imerg_path)  # Open IMERG dataset

    insat_ds = insat_ds.sel(  # Clip INSAT spatially
        lat=slice(lat_min_clip, lat_max_clip),
        lon=slice(lon_min_clip, lon_max_clip)
    )
    imerg_ds = imerg_ds.sel(  # Clip IMERG spatially
        lat=slice(lat_min_clip, lat_max_clip),
        lon=slice(lon_min_clip, lon_max_clip)
    )

    if 'latitude' in insat_ds.coords and 'longitude' in insat_ds.coords:
        insat_ds = insat_ds.rename({'latitude': 'lat', 'longitude': 'lon'})  # Rename coords

    try:
        tir1 = insat_ds['IMG_TIR1_TB']  # Extract TIR1 band
        tir2 = insat_ds['IMG_TIR2_TB']  # Extract TIR2 band
        wv = insat_ds['IMG_WV_TB']  # Extract WV band
    except KeyError:
        return None  # Skip if missing variables

    tir1_wv = tir1 - wv  # Feature: TIR1 minus WV
    tir2_wv = tir2 - wv  # Feature: TIR2 minus WV

    features = xr.Dataset({  # Create feature dataset
        'WV': wv,
        'TIR1': tir1,
        'TIR2': tir2,
        'TIR1_WV': tir1_wv,
        'TIR2_WV': tir2_wv
    })

    features_interp = features.interp(  # Interpolate features to IMERG grid
        lat=imerg_ds['lat'],
        lon=imerg_ds['lon'],
        method='nearest'
    )

    X_grid = np.stack(  # Stack features into numpy array
        [features_interp[var].values for var in features_interp.data_vars],
        axis=-1
    )
    y_grid = imerg_ds['precipitation'].values  # Target rainfall grid

    mask = np.all(np.isfinite(X_grid), axis=-1) & np.isfinite(y_grid)  # Valid data mask
    if not mask.any():
        return None  # Skip if no valid data

    X_grid[~mask] = 0.0  # Set invalid features to 0
    y_grid[~mask] = 0.0  # Set invalid targets to 0

    return X_grid, y_grid  # Return feature and target grid

# Load INSAT and IMERG file lists with parsed times
insat_files = [
    (f, parse_insat_time(f)) for f in sorted(glob(os.path.join(INSAT_DIR, '*.nc')))
]  # Get INSAT file paths and times
insat_files = [(f, t) for f, t in insat_files if t is not None]  # Filter out files with no time

imerg_files = [
    (f, parse_imerg_time(f)) for f in sorted(glob(os.path.join(IMERG_DIR, '*.nc')))
]  # Get IMERG file paths and times
imerg_files = [(f, t) for f, t in imerg_files if t is not None]  # Filter out files with no time

# Extract feature-target pairs from dataset
X_list, y_list = [], []  # Initialize lists
for insat_path, insat_time in insat_files:
    result = process_pair(insat_path, insat_time, imerg_files)  # Process each pair
    if result:
        X_list.append(result[0])  # Append features
        y_list.append(result[1])  # Append targets

X = np.array(X_list)  # Convert feature list to numpy array
y = np.array(y_list)  # Convert target list to numpy array

# Approach 2: Mask 0 mm/hr rainfall points by setting them to NaN
y = np.where(y == 0.0, np.nan, y)  # Mask 0 rainfall values

# NEW: Create flattened data and grid indices for NaN-masking
X_all = X.reshape((-1, X.shape[-1]))  # Flatten X
y_all = y.flatten()  # Flatten y

grid_indices_all = np.array(np.where(np.isfinite(y_all))).flatten()  # Get valid indices
X_all = X_all[grid_indices_all]  # Filter X for valid points
y_all = y_all[grid_indices_all]  # Filter y for valid points

# Rainfall distribution stats (ignoring NaNs)
y_flat = y.flatten()  # Flatten y again
y_flat_valid = y_flat[np.isfinite(y_flat)]  # Valid y values (non-NaN)

print(f"Rainfall distribution:")  # Print rainfall stats
print(f"  0 mm/hr (masked as NaN): {np.sum(np.isnan(y_flat))} samples ({100*np.sum(np.isnan(y_flat))/len(y_flat):.1f}%)")
print(f"  0-5 mm/hr: {np.sum((y_flat_valid > 0) & (y_flat_valid <= 5))} samples ({100*np.sum((y_flat_valid > 0) & (y_flat_valid <= 5))/len(y_flat):.1f}%)")
print(f"  5-15 mm/hr: {np.sum((y_flat_valid > 5) & (y_flat_valid <= 15))} samples ({100*np.sum((y_flat_valid > 5) & (y_flat_valid <= 15))/len(y_flat):.1f}%)")
print(f"  15-30 mm/hr: {np.sum((y_flat_valid > 15) & (y_flat_valid <= 30))} samples ({100*np.sum((y_flat_valid > 15) & (y_flat_valid <= 30))/len(y_flat):.1f}%)")
print(f"  >30 mm/hr: {np.sum(y_flat_valid > 30)} samples ({100*np.sum(y_flat_valid > 30)/len(y_flat):.1f}%)")
print(f"Rainfall mean: {np.nanmean(y_flat):.4f}, max: {np.nanmax(y_flat):.4f}")

# NEW: Print dataset shape after removing 0 mm/hr points
print(f"After removing 0 mm/hr samples:")
print(f"  X shape: {X_all.shape}, y shape: {y_all.shape}, grid_indices shape: {grid_indices_all.shape}")
print(f"Loaded {X.shape[0]} samples: X={X.shape}, y={y.shape}")

# Split into train, validation, and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42  # Split train+val and test
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42  # Split train and val (0.1765 * 0.85 ≈ 0.15 total val)
)

# Reshape target data for model training
y_train_flat = y_train.reshape((y_train.shape[0], -1))  # Flatten train targets
y_val_flat   = y_val.reshape((y_val.shape[0], -1))  # Flatten val targets
y_test_flat  = y_test.reshape((y_test.shape[0], -1))  # Flatten test targets

# Replace any remaining NaNs in target data with 0.0
y_train_flat = np.nan_to_num(y_train_flat, nan=0.0)  # Clean train targets
y_val_flat = np.nan_to_num(y_val_flat, nan=0.0)  # Clean val targets
y_test_flat = np.nan_to_num(y_test_flat, nan=0.0)  # Clean test targets

# ALSO clean the original splits for evaluation
y_train = np.nan_to_num(y_train, nan=0.0)  # Clean train
y_val   = np.nan_to_num(y_val,   nan=0.0)  # Clean val
y_test  = np.nan_to_num(y_test,  nan=0.0)  # Clean test

# Define custom Weighted MSE Loss Function
def weighted_mse(y_true, y_pred):
    weight = (  # Calculate weights
        1.0
        + 5.0 * tf.cast(y_true > 20.0, tf.float32)  # Higher weight for y > 20 mm/hr
        + 10.0 * tf.cast(y_true > 40.0, tf.float32)  # Even higher for y > 40 mm/hr
    )
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))  # Weighted MSE
