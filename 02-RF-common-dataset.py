'''
  This code is designed to create common normalised dataset from data extraction from recently created nc format files of INSAT and IMERG, with temporal and spatial alignment with proper resampling wherever required
  Flowchart of code
  1. input directories
  2. Function for time extraction using file names
  3. Function for temporal alignment
  4. Function for data extraction from valid files along with masking of bad data points
  5. normalise vall extracted variables using minmax ( linear scaling )
  6. Create common dataset
'''
import os  # For handling file paths
import re  # For regular expressions used to parse filenames
from glob import glob  # For finding files matching a pattern
from datetime import datetime  # For handling dates and times
import numpy as np  # For numerical operations
import xarray as xr  # For handling NetCDF datasets
from sklearn.model_selection import train_test_split  # For splitting data into train/val/test
from sklearn.preprocessing import MinMaxScaler  # For scaling/normalizing data
from concurrent.futures import ProcessPoolExecutor, as_completed  # For parallel processing

# Define directories where INSAT and IMERG NetCDF files are stored
INSAT_DIR = "/kaggle/working/"  # <------ Change dir accordingly
IMERG_DIR = "/kaggle/working/"   # <------ Change dir accordingly

# Define clipping boundary for India region (in latitudes and longitudes)
lat_min_clip, lat_max_clip = 8.07, 37.1
lon_min_clip, lon_max_clip = 68.12, 97.42

# Function to extract timestamp from INSAT filename
def parse_insat_time(filename):
    # Match pattern from filename like: 3RIMG_08JUN2021_0300
    match = re.search(r'3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})', filename)
    if not match:
        return None  # If no match found, return None
    day, mon_str, year, time_str = match.groups()
    # Map month abbreviation to number
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    hour = int(time_str[:2])  # Extract hour
    minute = int(time_str[2:])  # Extract minute
    return datetime(int(year), month_map[mon_str], int(day), hour, minute)  # Return datetime object

# Function to extract timestamp from IMERG filename
def parse_imerg_time(filename):
    # Match pattern from filename like: 3IMERG.20210608-S030000
    match = re.search(r'3IMERG\.(\d{8})-S(\d{6})', filename)
    if not match:
        return None
    date_str, time_str = match.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")  # Convert to datetime object

# Function to find IMERG file closest in time to given INSAT time (within tolerance)
def find_closest_imerg(insat_time, imerg_files, tol_secs=1800):
    min_diff = float('inf')  # Initialize with infinity
    closest_file = None
    for imerg_file, imerg_time in imerg_files:
        diff = abs((insat_time - imerg_time).total_seconds())  # Time difference in seconds
        if diff < min_diff and diff <= tol_secs:  # Keep if within tolerance and smaller diff
            min_diff = diff
            closest_file = imerg_file
    return closest_file  # Return closest matching IMERG file path

# Function to process each INSAT-IMERG file pair and extract features and target
def process_pair(insat_path, insat_time, imerg_files):
    try:
        # Find closest IMERG file for this INSAT timestamp
        imerg_path = find_closest_imerg(insat_time, imerg_files)
        if not imerg_path or not os.path.exists(imerg_path):
            return None  # Skip if no valid IMERG file found

        # Open INSAT and IMERG NetCDF files as xarray Datasets
        insat_ds = xr.open_dataset(insat_path)
        imerg_ds = xr.open_dataset(imerg_path)

        # Clip INSAT data spatially to India region
        insat_ds = insat_ds.sel(
            lat=slice(lat_min_clip, lat_max_clip),
            lon=slice(lon_min_clip, lon_max_clip)
        )
        # Clip IMERG data spatially to India region
        imerg_ds = imerg_ds.sel(
            lat=slice(lat_min_clip, lat_max_clip),
            lon=slice(lon_min_clip, lon_max_clip)
        )

        # Rename coordinates if needed to ensure both datasets use 'lat' and 'lon'
        if 'latitude' in insat_ds.coords and 'longitude' in insat_ds.coords:
            insat_ds = insat_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Check that both datasets have lat/lon coordinates
        required_coords = {'lat', 'lon'}
        if not required_coords.issubset(insat_ds.coords) or not required_coords.issubset(imerg_ds.coords):
            print(f"Skipping {insat_path}: Missing lat/lon.")
            return None

        try:
            # Extract INSAT variables (WV, TIR1, TIR2 bands)
            tir1 = insat_ds['IMG_TIR1_TB']
            tir2 = insat_ds['IMG_TIR2_TB']
            wv = insat_ds['IMG_WV_TB']
        except KeyError as e:
            # If required variables missing, skip this file
            print(f"Skipping {insat_path}: Missing INSAT variables: {e}")
            return None

        # Calculate feature differences
        tir1_wv = tir1 - wv
        tir2_wv = tir2 - wv

        # Create xarray Dataset of selected features
        features = xr.Dataset({
            'WV': wv,
            'TIR1': tir1,
            'TIR2': tir2,
            'TIR1_WV': tir1_wv,
            'TIR2_WV': tir2_wv
        })

        # Interpolate INSAT features onto IMERG grid for spatial alignment
        features_interp = features.interp(
            lat=imerg_ds['lat'],
            lon=imerg_ds['lon'],
            method='nearest'
        )

        # Flatten feature variables into 2D array for ML input
        X = np.stack([
            features_interp['WV'].values.flatten(),
            features_interp['TIR1'].values.flatten(),
            features_interp['TIR2'].values.flatten(),
            features_interp['TIR1_WV'].values.flatten(),
            features_interp['TIR2_WV'].values.flatten()
        ], axis=1)

        # Target variable: IMERG precipitation
        y = imerg_ds['precipitation'].values.flatten()

        # Create a mask to remove any rows where features or target are NaN
        valid_mask = np.all(~np.isnan(X), axis=1) & ~np.isnan(y)

        # Skip if most precipitation values are zero (non-rainy scene)
        if y.size == 0 or np.mean(y == 0) > 0.95:
            return None

        # Filter out invalid samples
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Track grid position (lat index, lon index) for each valid sample
        lat_len, lon_len = imerg_ds['lat'].size, imerg_ds['lon'].size
        grid_indices = np.array(np.unravel_index(np.where(valid_mask)[0], (lat_len, lon_len))).T  # shape: (n_points, 2)

        # Return features, target, and grid positions for this file pair
        return X_valid, y_valid, grid_indices

    except Exception as e:
        # Handle any unexpected error
        print(f"❌ Error processing {os.path.basename(insat_path)}: {e}")
        return None

# Get sorted list of INSAT files with their timestamps
insat_files = [(f, parse_insat_time(f)) for f in sorted(glob(os.path.join(INSAT_DIR, '*.nc')))]

# Get sorted list of IMERG files with their timestamps
imerg_files = [(f, parse_imerg_time(f)) for f in sorted(glob(os.path.join(IMERG_DIR, '*.nc')))]

# Filter out IMERG files where timestamp parsing failed
imerg_files = [f for f in imerg_files if f[1] is not None]

# Lists to collect data from all pairs
X_list, y_list, grid_list = [], [], []

# Use parallel processing to speed up INSAT-IMERG file pairing and feature extraction
with ProcessPoolExecutor(max_workers=8) as executor:
    # Submit each INSAT file for processing in parallel
    futures = [
        executor.submit(process_pair, insat_path, insat_time, imerg_files)
        for insat_path, insat_time in insat_files if insat_time  # Skip files with invalid timestamp
    ]
    # As each parallel job completes, collect the results
    for future in as_completed(futures):
        res = future.result()
        if res:
            X_list.append(res[0])  # Features
            y_list.append(res[1])  # Target (precipitation)
            grid_list.append(res[2])  # Grid positions

# If no valid data found after processing, stop the program
if len(X_list) == 0 or len(y_list) == 0:
    print("❌ No valid data after spatial clipping.")
    exit()

# Combine all feature arrays vertically (stack all samples)
X_all = np.vstack(X_list)
# Combine all target arrays into single 1D array
y_all = np.hstack(y_list)
# Combine all grid index arrays
grid_indices_all = np.vstack(grid_list)

# Print final dataset shape (features, target, grid positions)
print(f"✅ Final dataset shape: X = {X_all.shape}, y = {y_all.shape}, grid_indices = {grid_indices_all.shape}")

# Normalize feature values using Min-Max scaling (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)

# Split into 70% train and 30% temp (which will later split into val and test)
X_train, X_temp, y_train, y_temp, grid_train, grid_temp = train_test_split(
    X_scaled, y_all, grid_indices_all, test_size=0.3, random_state=42
)

# From the 30% temp, split into 50% val and 50% test => making final ratio 15% val and 15% test
X_val, X_test, y_val, y_test, grid_val, grid_test = train_test_split(
    X_temp, y_temp, grid_temp, test_size=0.5, random_state=42
)

# Print final split sizes
print(f"✅ Split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
