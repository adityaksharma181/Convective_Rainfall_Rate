# Model prediction
'''
  
'''
# Import required libraries
import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the custom weighted MSE loss function (needed when loading the trained model)
def weighted_mse(y_true, y_pred):
    weight = (
        1.0
        + 5.0 * tf.cast(y_true > 20.0, tf.float32)  # Higher weight if true rainfall > 20 mm/hr
        + 10.0 * tf.cast(y_true > 40.0, tf.float32)  # Even higher weight if > 40 mm/hr
    )
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))  # Weighted mean squared error

# Configuration: paths to model and input INSAT file
MODEL_PATH = "/kaggle/working/insat_imerg_cnn_weighted_model.h5"  # Trained model path
INSAT_FILE_PATH = "/kaggle/input/nc-insat/3RIMG_28MAY2025_0645_L1C_ASIA_MER_V01R00.nc"  # INSAT NetCDF input file

# Define the spatial clipping boundary (same area used during training)
lat_min_clip, lat_max_clip = 8.07, 37.1
lon_min_clip, lon_max_clip = 68.12, 97.42

# Import scipy functions for resizing and interpolation
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
SCIPY_AVAILABLE = True

# Function to parse date and time from INSAT filename
def parse_insat_time(filename):
    match = re.search(r'3RIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})', filename)  # Regex to match INSAT filename pattern
    if not match:
        return None  # Return None if pattern not found
    day, mon_str, year, time_str = match.groups()  # Extract day, month, year, and time
    month_map = {
        'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
    }  # Map month abbreviation to number
    hour, minute = int(time_str[:2]), int(time_str[2:])  # Extract hour and minute
    return datetime(int(year), month_map[mon_str], int(day), hour, minute)  # Return datetime object

# Function to process a single INSAT file and prepare it for prediction
def process_single_insat(insat_path):
    # Load INSAT dataset using xarray
    insat_ds = xr.open_dataset(insat_path)
    
    # Clip dataset to the defined lat/lon bounds
    insat_ds = insat_ds.sel(
        lat=slice(lat_min_clip, lat_max_clip),
        lon=slice(lon_min_clip, lon_max_clip)
    )

    # Extract required image bands (brightness temperatures)
    tir1 = insat_ds['IMG_TIR1_TB']
    tir2 = insat_ds['IMG_TIR2_TB']
    wv = insat_ds['IMG_WV_TB']

    # Calculate derived features (differences between channels)
    tir1_wv = tir1 - wv
    tir2_wv = tir2 - wv
    
    # Combine all features into one xarray dataset
    features = xr.Dataset({
        'WV': wv,
        'TIR1': tir1,
        'TIR2': tir2,
        'TIR1_WV': tir1_wv,
        'TIR2_WV': tir2_wv
    })
    
    # Convert feature dataset to numpy array
    X_grid = np.stack(
        [features[var].values for var in features.data_vars],
        axis=-1
    )
    
    # Mask non-finite values (replace with 0.0)
    mask = np.all(np.isfinite(X_grid), axis=-1)
    X_grid[~mask] = 0.0
    
    # Add batch dimension (required by TensorFlow model)
    X_input = np.expand_dims(X_grid, axis=0)
    
    return X_input, features.lat.values, features.lon.values, insat_ds  # Return inputs and metadata

# Function to predict and plot rainfall from INSAT data
def plot_rainfall_prediction(insat_file_path, model_path, save_plot=True, output_dir="./"):

    # Load trained model with custom loss function
    model = load_model(model_path, custom_objects={'weighted_mse': weighted_mse})
    
    # Process the INSAT file to get input and coordinates
    X_input, lats, lons, insat_ds = process_single_insat(insat_file_path)
    
    # Get model's expected output size (flattened)
    model_output_size = model.output_shape[1]
    current_grid_size = X_input.shape[1] * X_input.shape[2]  # Calculate input grid size
    
    # If grid size mismatch between model and input
    if current_grid_size != model_output_size:
        print(f"Warning: Grid size mismatch!")
        
        # Estimate expected grid height and width from output size
        import math
        expected_h = int(math.sqrt(model_output_size))
        expected_w = model_output_size // expected_h
        
        # Adjust height and width until product matches output size
        while expected_h * expected_w != model_output_size:
            expected_h -= 1
            expected_w = model_output_size // expected_h
        
        # Calculate zoom factors for resizing input grid
        zoom_h = expected_h / X_input.shape[1]
        zoom_w = expected_w / X_input.shape[2]
                
        # Resize each input channel to match expected grid size
        X_resized = np.zeros((1, expected_h, expected_w, X_input.shape[3]))
        for c in range(X_input.shape[3]):
            X_resized[0, :, :, c] = zoom(X_input[0, :, :, c], (zoom_h, zoom_w), order=1)
        
        X_input = X_resized  # Replace with resized input
        
        # Interpolate latitude and longitude arrays to new grid size
        lat_interp = interp1d(np.arange(len(lats)), lats, kind='linear')
        lon_interp = interp1d(np.arange(len(lons)), lons, kind='linear')
        
        new_lat_indices = np.linspace(0, len(lats)-1, expected_h)
        new_lon_indices = np.linspace(0, len(lons)-1, expected_w)
        
        lats = lat_interp(new_lat_indices)  # New latitudes
        lons = lon_interp(new_lon_indices)  # New longitudes
        
        print(f"new coordinates: lats({len(lats)}), lons({len(lons)})")
        
        grid_h, grid_w = expected_h, expected_w  # Update grid size variables
    
    grid_h, grid_w = X_input.shape[1], X_input.shape[2]  # Final grid size
    
    # Run model prediction
    y_pred_flat = model.predict(X_input, verbose=0)
    
    # Reshape flat prediction back to 2D spatial grid
    y_pred = y_pred_flat.reshape((grid_h, grid_w))
    
    # Check if prediction grid matches coordinate arrays
    print(f"Final dimensions - Prediction: {y_pred.shape}, Lats: {len(lats)}, Lons: {len(lons)}")
    
    if y_pred.shape[0] != len(lats) or y_pred.shape[1] != len(lons):
        # If mismatch, attempt to fix coordinate array length
        if y_pred.size == len(lats) * len(lons):
            print("Attempting to fix coordinate array lengths...")
            lats = np.linspace(lats[0], lats[-1], y_pred.shape[0])  # Generate evenly spaced lats
            lons = np.linspace(lons[0], lons[-1], y_pred.shape[1])  # Generate evenly spaced lons
            print(f"Fixed coordinates: lats({len(lats)}) x lons({len(lons)})")
        
        raise ValueError("Cannot resolve dimension mismatch!")  # Raise error if still mismatch
    
    # Extract timestamp from INSAT filename
    filename = os.path.basename(insat_file_path)
    timestamp = parse_insat_time(filename)
    time_str = timestamp.strftime("%Y-%m-%d %H:%M UTC") if timestamp else "Unknown Time"
    
    # Create matplotlib figure
    fig = plt.figure(figsize=(12, 8))
    
    # Define rainfall color levels and colors
    rainfall_levels = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 100]
    rainfall_colors = ['white', '#E0F7FA', '#B2EBF2', '#80DEEA', '#4DD0E1', 
                      '#26C6DA', '#00BCD4', '#00ACC1', '#0097A7', '#00838F', '#006064']
    rainfall_cmap = colors.ListedColormap(rainfall_colors)
    rainfall_norm = colors.BoundaryNorm(rainfall_levels, len(rainfall_colors))
    
    # Plot rainfall prediction on map with Cartopy
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip], ccrs.PlateCarree())
    
    # Add map features (coastline, land, ocean)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    
    # Plot predicted rainfall using contour fill
    im = ax.contourf(lons, lats, y_pred, levels=rainfall_levels, 
                     cmap=rainfall_cmap, norm=rainfall_norm, extend='max', 
                     transform=ccrs.PlateCarree())
    ax.set_title(f'Predicted Rainfall Rate (mm/hr)\n{time_str}', fontsize=14, fontweight='bold')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Add colorbar for rainfall intensity
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Rainfall Rate (mm/hr)', fontsize=12)
    
    plt.tight_layout()  # Adjust layout
    
    # Save the plot as PNG if requested
    if save_plot:
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M") if timestamp else "unknown_time"
        output_filename = f"rainfall_prediction_{timestamp_str}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()  # Display the plot
    
    return y_pred, lats, lons  # Return predicted rainfall and coordinates

# Main execution block
if __name__ == "__main__":
    # Change INSAT path here if needed
    INSAT_FILE_PATH = "/kaggle/input/nc-insat/3RIMG_28MAY2025_0645_L1C_ASIA_MER_V01R00.nc"
    
    # Run the rainfall prediction and plot
    predicted_rainfall, latitudes, longitudes = plot_rainfall_prediction(
        INSAT_FILE_PATH, 
        MODEL_PATH, 
        save_plot=True, 
        output_dir="./"
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("RAINFALL PREDICTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Prediction shape: {predicted_rainfall.shape}")
    print(f"Max rainfall predicted: {np.max(predicted_rainfall):.2f} mm/hr")
    print(f"Mean rainfall: {np.mean(predicted_rainfall):.2f} mm/hr")
