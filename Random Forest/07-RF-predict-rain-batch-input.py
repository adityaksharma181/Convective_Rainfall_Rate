'''
  This code is designed to handle batch files of insat modeified nc located in a folder
  Flowchart of code:
  1. Clipping condition for india
  2. load model
  3. loop for extractinf features from insat custom files
  4. prediction using model 
  5. plot spatial map
'''
import os  # For file path handling
import numpy as np  # For numerical operations
import xarray as xr  # For handling NetCDF datasets
import matplotlib.pyplot as plt  # For plotting
import joblib  # For loading the saved model and scaler
import cartopy.crs as ccrs  # For map projections
import cartopy.feature as cfeature  # For adding map features like land and ocean

# Clipping region for India
lat_min_clip, lat_max_clip = 8.07, 37.1  # Minimum and maximum latitude for clipping
lon_min_clip, lon_max_clip = 68.12, 97.42  # Minimum and maximum longitude for clipping

# Load trained Random Forest model and scaler
rf = joblib.load("random_forest_rainfall_model.joblib")  # Load Random Forest model
scaler = joblib.load("scaler.joblib")  # Load scaler for feature normalization
print(" Model and Scaler loaded.")  # Confirm successful loading

def predict_for_folder(insat_folder):  # Function to predict rainfall for all files in a folder
    all_preds = []  # List to store prediction grids
    lat_grid = None  # Placeholder for latitude grid
    lon_grid = None  # Placeholder for longitude grid

    insat_files = [os.path.join(insat_folder, f) for f in os.listdir(insat_folder) if f.endswith('.nc')]  # Get list of .nc files
    print(f"Found {len(insat_files)} INSAT files in folder.")  # Print number of files found

    for insat_path in insat_files:  # Loop through each INSAT file
        try:
            insat_ds = xr.open_dataset(insat_path)  # Open NetCDF dataset
            insat_ds = insat_ds.sel(  # Clip dataset to India region
                lat=slice(lat_min_clip, lat_max_clip),
                lon=slice(lon_min_clip, lon_max_clip)
            )

            # Rename lat/lon if needed
            if 'latitude' in insat_ds.coords and 'longitude' in insat_ds.coords:  # Check for alternative coordinate names
                insat_ds = insat_ds.rename({'latitude': 'lat', 'longitude': 'lon'})  # Rename to 'lat' and 'lon'

            if not {'lat', 'lon'}.issubset(insat_ds.coords):  # Check if both lat and lon exist
                print(f" Skipping {insat_path}: Missing lat/lon")  # Skip file if lat/lon missing
                continue  # Move to next file

            if lat_grid is None or lon_grid is None:  # Initialize lat/lon grid once
                lat_grid = insat_ds['lat'].values  # Get latitude values
                lon_grid = insat_ds['lon'].values  # Get longitude values

            # Extract INSAT features
            tir1 = insat_ds['IMG_TIR1_TB']  # Extract TIR1 band
            tir2 = insat_ds['IMG_TIR2_TB']  # Extract TIR2 band
            wv = insat_ds['IMG_WV_TB']  # Extract WV band

            # Derived features
            tir1_wv = tir1 - wv  # TIR1 minus WV
            tir2_wv = tir2 - wv  # TIR2 minus WV

            features = xr.Dataset({  # Create a dataset with all features
                'WV': wv,
                'TIR1': tir1,
                'TIR2': tir2,
                'TIR1_WV': tir1_wv,
                'TIR2_WV': tir2_wv
            })

            X_new = np.stack([  # Stack features into a 2D array (pixels × features)
                features['WV'].values.flatten(),
                features['TIR1'].values.flatten(),
                features['TIR2'].values.flatten(),
                features['TIR1_WV'].values.flatten(),
                features['TIR2_WV'].values.flatten()
            ], axis=1)

            valid_mask = np.all(~np.isnan(X_new), axis=1)  # Mask to identify non-NaN pixels
            X_valid = X_new[valid_mask]  # Keep only valid (non-NaN) pixels

            if X_valid.shape[0] == 0:  # Skip if no valid pixels
                print(f"No valid pixels in {os.path.basename(insat_path)}")  # Print message
                continue  # Skip to next file

            X_valid_scaled = scaler.transform(X_valid)  # Scale features using saved scaler
            y_pred = rf.predict(X_valid_scaled)  # Predict rainfall using Random Forest model

            # Fill back into grid
            pred_grid_flat = np.full(X_new.shape[0], np.nan)  # Initialize grid with NaNs
            pred_grid_flat[valid_mask] = y_pred  # Fill predictions for valid pixels
            pred_grid_2d = pred_grid_flat.reshape(len(lat_grid), len(lon_grid))  # Reshape to 2D grid

            all_preds.append(pred_grid_2d)  # Add prediction grid to list

        except Exception as e:  # Handle errors during file processing
            print(f" Error processing {os.path.basename(insat_path)}: {e}")  # Print error message

    if len(all_preds) == 0:  # Check if no predictions were made
        print("No valid predictions from folder.")  # Print message
        return  # Exit function

    # Calculate mean across time (all files)
    mean_precip = np.nanmean(np.stack(all_preds), axis=0)  # Compute mean rainfall grid ignoring NaNs

    # Plot mean precipitation
    plt.figure(figsize=(10, 8))  # Create figure
    ax = plt.axes(projection=ccrs.PlateCarree())  # Create map axes
    mesh = ax.pcolormesh(lon_grid, lat_grid, mean_precip, cmap='Blues', shading='auto', transform=ccrs.PlateCarree())  # Plot rainfall grid
    plt.colorbar(mesh, ax=ax, label='Mean Predicted Rainfall (mm/hr)')  # Add colorbar

    # Coastlines and land/ocean (No country borders)
    ax.coastlines(resolution='10m', color='black', linewidth=1)  # Draw coastlines
    ax.add_feature(cfeature.LAND, facecolor='lightgray')  # Fill land areas
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')  # Fill ocean areas

    ax.set_extent([lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip], crs=ccrs.PlateCarree())  # Set map extent to India region

    plt.title(f"Mean Predicted Rainfall from {len(all_preds)} INSAT files")  # Set plot title
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot

    # Print mean over entire area (single number)
    mean_value = np.nanmean(mean_precip)  # Calculate overall mean rainfall
    print(f"✅ Overall Mean Predicted Precipitation (mm/hr): {mean_value:.4f}")  # Print mean rainfall

# Replace this with your folder path
example_insat_folder = "/kaggle/input/nc-insat"  # Path to folder containing INSAT files

predict_for_folder(example_insat_folder)  
