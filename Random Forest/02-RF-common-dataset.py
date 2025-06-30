'''
  This code is designed to create common normalised dataset from data extraction from recently created nc format files of INSAT and IMERG, with temporal and spatial alignment with proper resampling wherever required
  Flowchart of code
  1. input directories
  2. Function for time extraction using file names
  3. Function for temporal alignment
  4. Function for data extraction from valid files along with masking of bad data points
  5. masking zero rain
  6. normalise vall extracted variables using minmax ( linear scaling )
  7. Create common dataset
'''

import os  # For file path operations
import re  # For regex-based filename parsing
from glob import glob  # For file pattern matching
from datetime import datetime  # For working with timestamps
import numpy as np  # For numerical array operations
import xarray as xr  # For handling NetCDF data
from sklearn.model_selection import train_test_split  # For splitting datasets
from sklearn.preprocessing import MinMaxScaler  # For feature scaling
from concurrent.futures import ProcessPoolExecutor, as_completed  # For parallel processing

# Directories
INSAT_DIR = "/kaggle/working/"  # Directory containing INSAT files
IMERG_DIR = "/kaggle/working/"  # Directory containing IMERG files

# Clipping boundary (India region)
lat_min_clip, lat_max_clip = 8.07, 37.1  # Latitude boundaries
lon_min_clip, lon_max_clip = 68.12, 97.42  # Longitude boundaries

# Function to parse INSAT file timestamp from filename
def parse_insat_time(filename):
    match = re.search(r'3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})', filename)  # Regex match for timestamp
    if not match:
        return None  # Return None if pattern not found
    day, mon_str, year, time_str = match.groups()  # Extract date parts
    month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
                 'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}  # Month mapping
    hour = int(time_str[:2])  # Extract hour
    minute = int(time_str[2:])  # Extract minute
    return datetime(int(year), month_map[mon_str], int(day), hour, minute)  # Return datetime object

# Function to parse IMERG file timestamp from filename
def parse_imerg_time(filename):
    match = re.search(r'3IMERG\.(\d{8})-S(\d{6})', filename)  # Regex match for timestamp
    if not match:
        return None  # Return None if pattern not found
    date_str, time_str = match.groups()  # Extract date and time parts
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")  # Return datetime object

# Function to find closest IMERG file to a given INSAT time (within tolerance)
def find_closest_imerg(insat_time, imerg_files, tol_secs=1800):
    min_diff = float('inf')  # Initialize minimum time difference
    closest_file = None  # Initialize closest file
    for imerg_file, imerg_time in imerg_files:
        diff = abs((insat_time - imerg_time).total_seconds())  # Time difference in seconds
        if diff < min_diff and diff <= tol_secs:  # Check within tolerance
            min_diff = diff  # Update min difference
            closest_file = imerg_file  # Update closest file
    return closest_file  # Return closest file path

# Function to process one INSAT-IMERG file pair
def process_pair(insat_path, insat_time, imerg_files):
    try:
        imerg_path = find_closest_imerg(insat_time, imerg_files)  # Find closest IMERG file
        if not imerg_path or not os.path.exists(imerg_path):  # Check file exists
            return None  # Skip if not found

        insat_ds = xr.open_dataset(insat_path)  # Open INSAT NetCDF file
        imerg_ds = xr.open_dataset(imerg_path)  # Open IMERG NetCDF file

        insat_ds = insat_ds.sel(lat=slice(lat_min_clip, lat_max_clip), lon=slice(lon_min_clip, lon_max_clip))  # Clip INSAT to region
        imerg_ds = imerg_ds.sel(lat=slice(lat_min_clip, lat_max_clip), lon=slice(lon_min_clip, lon_max_clip))  # Clip IMERG to region

        if 'latitude' in insat_ds.coords and 'longitude' in insat_ds.coords:
            insat_ds = insat_ds.rename({'latitude': 'lat', 'longitude': 'lon'})  # Rename INSAT coords to match IMERG

        required_coords = {'lat', 'lon'}  # Required coordinate names
        if not required_coords.issubset(insat_ds.coords) or not required_coords.issubset(imerg_ds.coords):
            print(f"Skipping {insat_path}: Missing lat/lon.")  # Skip if coords missing
            return None

        try:
            tir1 = insat_ds['IMG_TIR1_TB']  # Extract TIR1 band
            tir2 = insat_ds['IMG_TIR2_TB']  # Extract TIR2 band
            wv = insat_ds['IMG_WV_TB']  # Extract WV band
        except KeyError as e:
            print(f"Skipping {insat_path}: Missing INSAT variables: {e}")  # Skip if variables missing
            return None

        tir1_wv = tir1 - wv  # Feature: TIR1 - WV
        tir2_wv = tir2 - wv  # Feature: TIR2 - WV

        features = xr.Dataset({
            'WV': wv,
            'TIR1': tir1,
            'TIR2': tir2,
            'TIR1_WV': tir1_wv,
            'TIR2_WV': tir2_wv
        })  # Create feature dataset

        features_interp = features.interp(
            lat=imerg_ds['lat'],
            lon=imerg_ds['lon'],
            method='nearest'
        )  # Interpolate features onto IMERG grid

        X = np.stack([
            features_interp['WV'].values.flatten(),
            features_interp['TIR1'].values.flatten(),
            features_interp['TIR2'].values.flatten(),
            features_interp['TIR1_WV'].values.flatten(),
            features_interp['TIR2_WV'].values.flatten()
        ], axis=1)  # Stack features into 2D array

        y = imerg_ds['precipitation'].values.flatten()  # Get target variable (rainfall)

        valid_mask = np.all(~np.isnan(X), axis=1) & ~np.isnan(y)  # Mask for non-NaN values
        X_valid = X[valid_mask]  # Filter features
        y_valid = y[valid_mask]  # Filter targets

        if y_valid.size == 0 or np.mean(y_valid == 0) > 0.95:
            return None  # Skip if almost all rainfall is zero or empty

        lat_len, lon_len = imerg_ds['lat'].size, imerg_ds['lon'].size  # Grid size
        grid_indices = np.array(np.unravel_index(np.where(valid_mask)[0], (lat_len, lon_len))).T  # Get grid indices for valid points

        return X_valid, y_valid, grid_indices  # Return processed arrays

    except Exception as e:
        print(f"Error processing {os.path.basename(insat_path)}: {e}")  # Log errors
        return None  # Skip on error

# Load INSAT filenames and their parsed times
insat_files = [(f, parse_insat_time(f)) for f in sorted(glob(os.path.join(INSAT_DIR, '*.nc')))]

# Load IMERG filenames and their parsed times
imerg_files = [(f, parse_imerg_time(f)) for f in sorted(glob(os.path.join(IMERG_DIR, '*.nc')))]
imerg_files = [f for f in imerg_files if f[1] is not None]  # Keep only valid timestamps

X_list, y_list, grid_list = [], [], []  # Initialize lists for dataset storage

# Process all file pairs in parallel using multiprocessing
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(process_pair, insat_path, insat_time, imerg_files)
        for insat_path, insat_time in insat_files if insat_time  # Submit jobs for each INSAT file with valid time
    ]
    for future in as_completed(futures):
        res = future.result()  # Get result from each process
        if res:
            X_list.append(res[0])  # Append features
            y_list.append(res[1])  # Append targets
            grid_list.append(res[2])  # Append grid indices

if len(X_list) == 0 or len(y_list) == 0:
    print("No valid data after spatial clipping.")  # Check for empty dataset
    exit()  # Exit if no data

X_all = np.vstack(X_list)  # Combine all feature arrays
y_all = np.hstack(y_list)  # Combine all target arrays
grid_indices_all = np.vstack(grid_list)  # Combine all grid index arrays

# Print overall dataset statistics
print(f"Final dataset shape: X = {X_all.shape}, y = {y_all.shape}, grid_indices = {grid_indices_all.shape}")

print(f"\n--- Overall Rainfall Distribution in Full Dataset ---")
print(f"Total samples: {X_all.shape[0]}")
print(f"  0 mm/hr: {np.sum(y_all == 0)} samples ({100*np.sum(y_all == 0)/len(y_all):.1f}%)")
print(f"  0-5 mm/hr: {np.sum((y_all > 0) & (y_all <= 5))} samples ({100*np.sum((y_all > 0) & (y_all <= 5))/len(y_all):.1f}%)")
print(f"  5-15 mm/hr: {np.sum((y_all > 5) & (y_all <= 15))} samples ({100*np.sum((y_all > 5) & (y_all <= 15))/len(y_all):.1f}%)")
print(f"  15-30 mm/hr: {np.sum((y_all > 15) & (y_all <= 30))} samples ({100*np.sum((y_all > 15) & (y_all <= 30))/len(y_all):.1f}%)")
print(f"  >30 mm/hr: {np.sum(y_all > 30)} samples ({100*np.sum(y_all > 30)/len(y_all):.1f}%)")
print(f"Rainfall mean: {y_all.mean():.4f}, max: {y_all.max():.4f}")

# Remove zero-rainfall samples (keep only positive rainfall)
positive_mask = y_all > 0
X_all = X_all[positive_mask]  # Keep only positive samples
y_all = y_all[positive_mask]  # Keep targets accordingly
grid_indices_all = grid_indices_all[positive_mask]  # Keep grid indices accordingly

print(f"After removing 0 mm/hr samples:")
print(f"  X shape: {X_all.shape}, y shape: {y_all.shape}, grid_indices shape: {grid_indices_all.shape}")
print(f"------------------------------------------------------\n")

# Normalize features to 0-1 range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)  # Fit scaler on all features

# Split into Train, Temp (val+test)
X_train, X_temp, y_train, y_temp, grid_train, grid_temp = train_test_split(
    X_scaled, y_all, grid_indices_all, test_size=0.3, random_state=42  # 70% train, 30% temp
)

# Split Temp into Val and Test (50-50)
X_val, X_test, y_val, y_test, grid_val, grid_test = train_test_split(
    X_temp, y_temp, grid_temp, test_size=0.5, random_state=42  # Equal val/test split
)

# Final dataset shapes after splitting
print(f"Split Sizes After Removing 0 mm/hr:")
print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
