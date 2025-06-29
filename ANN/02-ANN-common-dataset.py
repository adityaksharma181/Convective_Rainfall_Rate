'''
  This code is designed to create common normalised dataset from data extraction from recently created nc format files of INSAT and IMERG, with temporal and spatial alignment with proper resampling wherever required
  Flowchart of code
  1. input directories
  2. Function for time extraction using file names
  3. Function for temporal alignment
  4. Function for data extraction from valid files along with masking of bad data points
  5. Use of sample weights for heavy rainfalls
  5. normalise vall extracted variables using minmax ( linear scaling )
  6. Create common dataset
'''
import os                               # For file and directory operations
import re                               # For parsing filenames using regular expressions
from glob import glob                   # For file pattern matching
from datetime import datetime           # For handling datetime operations
import numpy as np                      # For numerical array operations
import xarray as xr                     # For working with NetCDF files easily
from concurrent.futures import ProcessPoolExecutor, as_completed  # For parallel processing to speed up data extraction
import tensorflow as tf                 # For building and training the neural network
from sklearn.preprocessing import RobustScaler  # For scaling input features while being robust to outliers
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets

# Define directories where INSAT and IMERG NetCDF data files are stored
INSAT_DIR = "/kaggle/working/"
IMERG_DIR = "/kaggle/working/"

# Latitude and longitude range to clip the data to the Indian region
lat_min_clip, lat_max_clip = 8.07, 37.1
lon_min_clip, lon_max_clip = 68.12, 97.42

# Function to parse INSAT file timestamp from filename
def parse_insat_time(filename):
    match = re.search(r'3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})', filename)  # Extract date and time info
    if not match:
        return None
    day, mon_str, year, time_str = match.groups()  # Extract components
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    hour = int(time_str[:2])
    minute = int(time_str[2:])
    return datetime(int(year), month_map[mon_str], int(day), hour, minute)  # Return datetime object

# Function to parse IMERG file timestamp from filename
def parse_imerg_time(filename):
    match = re.search(r'3IMERG\.(\d{8})-S(\d{6})', filename)  # Extract date and time info
    if not match:
        return None
    date_str, time_str = match.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")  # Return datetime object

# Find closest IMERG file to given INSAT time within a tolerance (30 minutes)
def find_closest_imerg(insat_time, imerg_files, tol_secs=1800):
    min_diff = float('inf')
    closest_file = None
    for imerg_file, imerg_time in imerg_files:
        if imerg_time is None:
            continue
        diff = abs((insat_time - imerg_time).total_seconds())  # Time difference in seconds
        if diff < min_diff and diff <= tol_secs:
            min_diff = diff
            closest_file = imerg_file
    return closest_file  # Return path of closest IMERG file

# Function to create new features from INSAT dataset
def create_additional_features(insat_ds):
    """Create additional derived features for better precipitation estimation"""
    tir1 = insat_ds['IMG_TIR1_TB']
    tir2 = insat_ds['IMG_TIR2_TB']
    wv = insat_ds['IMG_WV_TB']
    
    # Basic features
    tir1_wv = tir1 - wv
    tir2_wv = tir2 - wv
    
    # Additional advanced features
    tir_avg = (tir1 + tir2) / 2
    tir_ratio = tir1 / (tir2 + 1e-6)  # Avoid division by zero
    temp_gradient = np.abs(tir1 - tir2)
    cold_cloud_indicator = (tir1 < 220) & (tir2 < 220)  # Binary indicator for cold clouds
    wv_tir_interaction = wv * tir_avg
    
    # Simple texture feature: spatial gradient of TIR1
    tir1_grad_x = np.gradient(tir1.values, axis=1)
    tir1_grad_y = np.gradient(tir1.values, axis=0)
    spatial_texture = np.sqrt(tir1_grad_x**2 + tir1_grad_y**2)  # Magnitude of gradient
    
    # Return all features as xarray Dataset
    features = xr.Dataset({
        'WV': wv,
        'TIR1': tir1,
        'TIR2': tir2,
        'TIR1_WV': tir1_wv,
        'TIR2_WV': tir2_wv,
        'TIR_AVG': tir_avg,
        'TIR_RATIO': tir_ratio,
        'TEMP_GRADIENT': temp_gradient,
        'COLD_CLOUD': cold_cloud_indicator.astype(float),
        'WV_TIR_INTERACTION': wv_tir_interaction,
        'SPATIAL_TEXTURE': (['lat', 'lon'], spatial_texture)
    })
    
    return features

# Process a single INSAT-IMERG pair
def process_pair(insat_path, insat_time, imerg_files):
    try:
        imerg_path = find_closest_imerg(insat_time, imerg_files)
        if not imerg_path or not os.path.exists(imerg_path):
            return None

        insat_ds = xr.open_dataset(insat_path)  # Open INSAT NetCDF
        imerg_ds = xr.open_dataset(imerg_path)  # Open IMERG NetCDF

        # Clip both datasets to desired geographic region
        insat_ds = insat_ds.sel(lat=slice(lat_min_clip, lat_max_clip),
                                lon=slice(lon_min_clip, lon_max_clip))
        imerg_ds = imerg_ds.sel(lat=slice(lat_min_clip, lat_max_clip),
                                lon=slice(lon_min_clip, lon_max_clip))

        # Rename coordinates for consistency
        if 'latitude' in insat_ds.coords and 'longitude' in insat_ds.coords:
            insat_ds = insat_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

        required_coords = {'lat', 'lon'}
        if not required_coords.issubset(insat_ds.coords) or not required_coords.issubset(imerg_ds.coords):
            return None

        try:
            features = create_additional_features(insat_ds)  # Generate feature set
        except KeyError as e:
            print(f"Missing required variables in {os.path.basename(insat_path)}: {e}")
            return None

        # Interpolate INSAT features onto IMERG grid
        features_interp = features.interp(
            lat=imerg_ds['lat'],
            lon=imerg_ds['lon'],
            method='nearest'
        )

        # Select feature names to use
        feature_names = ['WV', 'TIR1', 'TIR2', 'TIR1_WV', 'TIR2_WV', 
                        'TIR_AVG', 'TIR_RATIO', 'TEMP_GRADIENT', 
                        'COLD_CLOUD', 'WV_TIR_INTERACTION', 'SPATIAL_TEXTURE']
        
        # Flatten feature grids into vectors for ML
        X = np.stack([
            features_interp[feat].values.flatten() for feat in feature_names
        ], axis=1)

        y = imerg_ds['precipitation'].values.flatten()  # Target: precipitation

        y = np.clip(y, 0, 50)  # Cap precipitation to 50 mm/hr

        valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)  # Filter out NaNs or invalid data

        # Filter out poor-quality samples (dominated by zeros)
        if y.size == 0 or np.mean(y == 0) > 0.95:
            return None

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        return X_valid, y_valid, features_interp, imerg_ds  # Return processed feature and label set

    except Exception as e:
        print(f"Error processing {os.path.basename(insat_path)}: {e}")
        return None

# Custom loss function for better handling of high rainfall
def custom_loss(y_true, y_pred):
    """Custom loss that gives more weight to higher precipitation values"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))  # Standard MSE
    high_precip_mask = tf.cast(y_true > 10, tf.float32)  # Focus on high rainfall (>10 mm/hr)
    high_precip_penalty = tf.reduce_mean(
        high_precip_mask * tf.square(tf.maximum(0.0, y_true - y_pred))
    )
    return mse + 2.0 * high_precip_penalty  # Final loss combining both terms

# Load INSAT file list with their parsed times
insat_files = [(f, parse_insat_time(f)) for f in sorted(glob(os.path.join(INSAT_DIR, '*.nc')))]
insat_files = [(f, t) for (f, t) in insat_files if t is not None]  # Keep only valid ones

# Load IMERG file list with parsed times
imerg_files = [(f, parse_imerg_time(f)) for f in sorted(glob(os.path.join(IMERG_DIR, '*.nc')))]
imerg_files = [(f, t) for (f, t) in imerg_files if t is not None]

print(f"Found {len(insat_files)} INSAT files and {len(imerg_files)} IMERG files")

# Prepare lists to store extracted data
X_list, y_list = [], []
ds_interp_list, imrg_list = [], []

# Process data in parallel to speed up I/O and feature extraction
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(process_pair, insat_path, insat_time, imerg_files)
        for insat_path, insat_time in insat_files
    ]
    for future in as_completed(futures):
        res = future.result()
        if res:
            X_list.append(res[0])
            y_list.append(res[1])
            ds_interp_list.append(res[2])
            imrg_list.append(res[3])

# If no data processed, exit
if len(X_list) == 0 or len(y_list) == 0:
    print("No valid data after processing.")
    exit()

# Combine all samples from all files into one big training set
X = np.vstack(X_list)
y = np.hstack(y_list)
print(f"Final dataset shape: X = {X.shape}, y = {y.shape}")

# Store last processed INSAT and IMERG grid (for later full-grid predictions)
ds_interp = ds_interp_list[-1]
imrg = imrg_list[-1]
feature_names = ['WV', 'TIR1', 'TIR2', 'TIR1_WV', 'TIR2_WV', 
                'TIR_AVG', 'TIR_RATIO', 'TEMP_GRADIENT', 
                'COLD_CLOUD', 'WV_TIR_INTERACTION', 'SPATIAL_TEXTURE']

# Filter out any outliers in target rainfall data
mask = (y >= 0) & (y <= 50)  # Only keep targets within 0-50 mm/hr
X = X[mask]
y = y[mask]

# Create sample weights to balance class imbalance (very low rainfall dominates data)
def create_sample_weights(y_values):
    """Create sample weights that better represent high precipitation events"""
    weights = np.ones_like(y_values)
    weights[y_values == 0] = 0.1  # Very low weight for 0 rainfall
    weights[(y_values > 0) & (y_values <= 1)] = 1.0
    weights[(y_values > 1) & (y_values <= 5)] = 2.0
    weights[(y_values > 5) & (y_values <= 10)] = 4.0
    weights[(y_values > 10) & (y_values <= 20)] = 8.0
    weights[(y_values > 20) & (y_values <= 35)] = 16.0
    weights[y_values > 35] = 25.0  # Very high weight for extreme rainfall
    
    weights = weights / np.mean(weights)  # Normalize to mean 1
    return weights

sample_weights = create_sample_weights(y)  # Calculate sample weights

print(f"After filtering: {X.shape[0]} samples remain")
print(f"Rainfall distribution:")
print(f"  0 mm/hr: {np.sum(y == 0)} samples ({100*np.sum(y == 0)/len(y):.1f}%)")
print(f"  0-5 mm/hr: {np.sum((y > 0) & (y <= 5))} samples ({100*np.sum((y > 0) & (y <= 5))/len(y):.1f}%)")
print(f"  5-15 mm/hr: {np.sum((y > 5) & (y <= 15))} samples ({100*np.sum((y > 5) & (y <= 15))/len(y):.1f}%)")
print(f"  15-30 mm/hr: {np.sum((y > 15) & (y <= 30))} samples ({100*np.sum((y > 15) & (y <= 30))/len(y):.1f}%)")
print(f"  >30 mm/hr: {np.sum(y > 30)} samples ({100*np.sum(y > 30)/len(y):.1f}%)")
print(f"Rainfall mean: {y.mean():.4f}, max: {y.max():.4f}")

# Scale input features to handle different ranges (using median and IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Create bins for stratified sampling to preserve rainfall distribution in train-test split
y_bins = np.digitize(y, bins=[0.1, 1, 5, 15, 30])

# First, split into 70% train and 30% temp (for test + validation)
X_train, X_temp, y_train, y_temp, sw_train, sw_temp, bins_train, bins_temp = train_test_split(
    X_scaled, y, sample_weights, y_bins,
    test_size=0.3, random_state=42, stratify=y_bins
)

# Then split the 30% temp set into 15% test and 15% validation (50-50 split of temp)
X_test, X_val, y_test, y_val, sw_test, sw_val = train_test_split(
    X_temp, y_temp, sw_temp,
    test_size=0.5, random_state=42, stratify=bins_temp
)

input_shape = X_train.shape[1]
print(f"Input shape: {input_shape}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
