import os  # For file path checking
import numpy as np  # For numerical array operations
import xarray as xr  # For handling NetCDF data
import matplotlib.pyplot as plt  # For plotting
import cartopy.crs as ccrs  # For map projections
import cartopy.feature as cfeature  # For map features like coastlines
import tensorflow as tf  # For loading and using the trained ANN model
import joblib  # For loading the scaler object
from tensorflow.keras.models import load_model  # For loading the Keras model

# File paths and geographic clipping limits
INSAT_FILE_PATH = "/kaggle/input/nc-insat/3RIMG_28MAY2025_0715_L1C_ASIA_MER_V01R00.nc"  # Path to input INSAT file
MODEL_PATH = "insat_imerg_ann_model_improved.h5"  # Path to trained ANN model
SCALER_PATH = "scaler_improved.save"  # Path to saved scaler
lat_min_clip, lat_max_clip = 8.07, 37.1  # Latitude range for clipping
lon_min_clip, lon_max_clip = 68.12, 97.42  # Longitude range for clipping

# Function to create input features from INSAT data
def create_features(ds):
    tir1, tir2, wv = ds['IMG_TIR1_TB'], ds['IMG_TIR2_TB'], ds['IMG_WV_TB']  # Extract input channels
    return xr.Dataset({
        'WV': wv,  # Water vapor channel
        'TIR1': tir1,  # TIR1 brightness temperature
        'TIR2': tir2,  # TIR2 brightness temperature
        'TIR1_WV': tir1 - wv,  # Derived feature: TIR1 - WV
        'TIR2_WV': tir2 - wv   # Derived feature: TIR2 - WV
    })

# Custom loss function used during ANN model training (not used during inference)
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))  # Mean squared error
    high_mask = tf.cast(y_true > 10, tf.float32)  # Mask for high rainfall pixels
    penalty = tf.reduce_mean(high_mask * tf.square(tf.maximum(0.0, y_true - y_pred)))  # Extra penalty for underestimation
    return mse + 2.0 * penalty  # Total loss

# Function to predict rainfall from INSAT data
def predict_rainfall(insat_path):
    # Check if input files exist
    if not os.path.exists(insat_path) or not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Missing input files.")  # Raise error if any file is missing
    
    model = load_model(MODEL_PATH, custom_objects={'custom_loss': custom_loss})  # Load the ANN model
    scaler = joblib.load(SCALER_PATH)  # Load the scaler for feature normalization
    
    # Open the INSAT NetCDF file and clip spatially
    ds = xr.open_dataset(insat_path).sel(lat=slice(lat_min_clip, lat_max_clip),
                                         lon=slice(lon_min_clip, lon_max_clip))
    
    # Rename coordinates if needed
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    features = create_features(ds)  # Create feature dataset
    feature_names = ['WV', 'TIR1', 'TIR2', 'TIR1_WV', 'TIR2_WV']  # List of feature names
    
    # Flatten feature arrays and stack into input matrix
    X = np.stack([features[f].values.flatten() for f in feature_names], axis=1)
    mask = np.all(np.isfinite(X), axis=1)  # Mask for valid (non-NaN) input pixels
    
    if np.sum(mask) == 0:
        raise ValueError("No valid pixels.")  # Error if no valid input pixels
    
    X_scaled = scaler.transform(X[mask])  # Scale the valid input features
    y_pred = model.predict(X_scaled, verbose=0).flatten()  # Predict rainfall using the ANN model
    y_pred = np.maximum(y_pred, 0)  # Clip negative predictions to zero
    
    y_full = np.full(X.shape[0], np.nan)  # Create full-size output array with NaNs
    y_full[mask] = y_pred  # Fill predictions into valid pixel positions
    
    # Reshape output back to 2D grid and return
    return y_full.reshape(features['WV'].shape), features['lat'].values, features['lon'].values

# Function to plot predicted rainfall grid
def plot_prediction(grid, lats, lons, insat_path):
    grid = np.where(grid <= 0, np.nan, grid)  # Mask out non-positive rainfall values
    valid = grid[~np.isnan(grid)]  # Get valid (positive) rainfall pixels
    
    if len(valid) == 0:
        print("No positive rainfall to plot.")  # Skip plotting if no valid data
        return
    
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]  # Map extent
    
    # Create plot with Cartopy projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent)  # Set map extent
    
    # Display the rainfall grid
    im = ax.imshow(grid, origin='lower', extent=extent, cmap='viridis', 
                   vmin=0, vmax=max(20, valid.max()), transform=ccrs.PlateCarree(), alpha=0.8)
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='white')  # Add coastlines
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.1)  # Light land shading
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)  # Light ocean shading
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)  # Add gridlines
    gl.top_labels = gl.right_labels = False  # Remove top and right labels
    
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, label='Rainfall (mm/hr)')  # Colorbar with label
    plt.title(f'Predicted Rainfall\n{os.path.basename(insat_path)}')  # Plot title
    
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display the plot
    plt.savefig(f"rainfall_{os.path.splitext(os.path.basename(insat_path))[0]}.png", dpi=300, bbox_inches='tight')  # Save plot as PNG

# Main execution block
if __name__ == "__main__":
    try:
        grid, lats, lons = predict_rainfall(INSAT_FILE_PATH)  # Run rainfall prediction
        plot_prediction(grid, lats, lons, INSAT_FILE_PATH)  # Plot predicted rainfall
    except Exception as e:
        print(f"Error: {e}")  # Print error if something fails
