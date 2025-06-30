'''
  This code handles the prediction from model solely using one insat custom or modified file
  Flowchart of code:
  1. clipping for india
  2. function for rainfall prediction from model using all the features
  3. plot the spatial graph
'''
import os  # For file and path operations
import numpy as np  # For numerical array operations
import xarray as xr  # For handling multi-dimensional scientific data (NetCDF, etc.)
import matplotlib.pyplot as plt  # For plotting
import joblib  # For loading saved machine learning models and scalers
import cartopy.crs as ccrs  # For map projections
import cartopy.feature as cfeature  # For adding map features like coastlines, land, ocean

# Clipping region for India
lat_min_clip, lat_max_clip = 8.07, 37.1  # Latitude bounds
lon_min_clip, lon_max_clip = 68.12, 97.42  # Longitude bounds

# Load trained Random Forest model and scaler
rf = joblib.load("random_forest_rainfall_model.joblib")  # Load Random Forest model
scaler = joblib.load("scaler.joblib")  # Load scaler for feature normalization
print("Model and Scaler loaded.")  # Confirm load success

def predict_rainfall_from_insat(insat_path):
    try:
        insat_ds = xr.open_dataset(insat_path)  # Open INSAT NetCDF file as xarray dataset
        insat_ds = insat_ds.sel(  # Clip dataset to India's lat-lon extent
            lat=slice(lat_min_clip, lat_max_clip),
            lon=slice(lon_min_clip, lon_max_clip)
        )

        # Rename lat/lon if needed
        if 'latitude' in insat_ds.coords and 'longitude' in insat_ds.coords:
            insat_ds = insat_ds.rename({'latitude': 'lat', 'longitude': 'lon'})  # Rename coordinates if needed

        if not {'lat', 'lon'}.issubset(insat_ds.coords):  # Check if lat/lon are present
            print(f"❌ Missing lat/lon in {insat_path}")  # Print error if missing
            return  # Exit function

        lat_grid = insat_ds['lat'].values  # Get latitude values
        lon_grid = insat_ds['lon'].values  # Get longitude values

        # Extract INSAT features
        tir1 = insat_ds['IMG_TIR1_TB']  # Thermal Infrared channel 1 brightness temperature
        tir2 = insat_ds['IMG_TIR2_TB']  # Thermal Infrared channel 2 brightness temperature
        wv = insat_ds['IMG_WV_TB']  # Water vapor channel brightness temperature

        # Derived features
        tir1_wv = tir1 - wv  # Difference between TIR1 and WV
        tir2_wv = tir2 - wv  # Difference between TIR2 and WV

        # Create feature dataset
        features = xr.Dataset({  # Create new xarray dataset for features
            'WV': wv,
            'TIR1': tir1,
            'TIR2': tir2,
            'TIR1_WV': tir1_wv,
            'TIR2_WV': tir2_wv
        })

        # Prepare feature array
        X_new = np.stack([  # Stack feature variables into 2D numpy array
            features['WV'].values.flatten(),
            features['TIR1'].values.flatten(),
            features['TIR2'].values.flatten(),
            features['TIR1_WV'].values.flatten(),
            features['TIR2_WV'].values.flatten()
        ], axis=1)

        # Handle NaNs
        valid_mask = np.all(~np.isnan(X_new), axis=1)  # Create mask for rows without NaNs
        X_valid = X_new[valid_mask]  # Keep only valid rows

        if X_valid.shape[0] == 0:  # Check if there are valid pixels
            print("No valid pixels to predict after filtering NaNs.")  # Print error
            return  # Exit function

        # Scale and predict
        X_valid_scaled = scaler.transform(X_valid)  # Apply scaling to valid features
        y_pred = rf.predict(X_valid_scaled)  # Predict rainfall using Random Forest model

        # Fill predictions back into grid
        pred_grid_flat = np.full(X_new.shape[0], np.nan)  # Initialize flat array with NaNs
        pred_grid_flat[valid_mask] = y_pred  # Fill predicted values at valid positions
        pred_grid_2d = pred_grid_flat.reshape(len(lat_grid), len(lon_grid))  # Reshape flat array back to 2D grid

        # Plot with coastlines only
        plt.figure(figsize=(10, 8))  # Create figure with specified size
        ax = plt.axes(projection=ccrs.PlateCarree())  # Set map projection
        mesh = ax.pcolormesh(lon_grid, lat_grid, pred_grid_2d, cmap='Blues', shading='auto', transform=ccrs.PlateCarree())  # Create rainfall heatmap
        plt.colorbar(mesh, ax=ax, label='Predicted Rainfall (mm/hr)')  # Add colorbar

        # Add only coastlines and land/ocean background
        ax.coastlines(resolution='10m', color='black', linewidth=1)  # Add coastlines
        ax.add_feature(cfeature.LAND, facecolor='lightgray')  # Add land background
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')  # Add ocean background

        ax.set_extent([lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip], crs=ccrs.PlateCarree())  # Set map extent to India

        plt.title(f"Predicted Rainfall from INSAT file: {os.path.basename(insat_path)}")  # Add plot title
        plt.tight_layout()  # Adjust layout to avoid clipping
        plt.show()  # Display the plot

    except Exception as e:  # Catch any errors during processing
        print(f"❌ Error processing {insat_path}: {e}")  # Print error message
 
# Change path below to your external INSAT file
example_insat_path = "/kaggle/input/nc-insat/3RIMG_28MAY2025_0115_L1C_ASIA_MER_V01R00.nc"  # Example INSAT file path

predict_rainfall_from_insat(example_insat_path)  # Call function with example file
