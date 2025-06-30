'''
  This code is designed to plot true vs predicted rainfall from the model
'''
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import cartopy.crs as ccrs

# Predict target values for the test set and flatten the result to 1D array
y_pred_test = model.predict(X_test).flatten()

# Check if there is at least one interpolated dataset for full grid visualization
if len(ds_interp_list) > 0:
    # Stack features from the interpolated dataset into a 2D array (samples x features)
    X_full = np.stack([ds_interp[feat].values.flatten() for feat in feature_names], axis=1)
    
    # Extract true precipitation values from IMERG dataset and flatten
    y_full = imrg['precipitation'].values.flatten()
    
    # Clip true precipitation to a maximum of 50 mm/hr
    y_full = np.clip(y_full, 0, 50)
    
    # Create mask for finite feature values and precipitation > 0
    mask_full = np.isfinite(X_full).all(axis=1) & np.isfinite(y_full) & (y_full > 0)
    
    # Proceed only if there are valid points after masking
    if np.sum(mask_full) > 0:
        # Scale valid feature inputs using the pre-fitted scaler
        X_full_scaled = scaler.transform(X_full[mask_full])
        
        # Predict rainfall for the valid points
        y_pred_flat = model.predict(X_full_scaled).flatten()
        
        # Initialize full prediction array with NaNs
        y_pred_full = np.full(y_full.shape, np.nan)
        
        # Fill predicted values only at valid locations
        y_pred_full[mask_full] = y_pred_flat
        
        # Initialize full true value array with NaNs
        y_full_masked = np.full(y_full.shape, np.nan)
        
        # Fill true precipitation values only at valid locations
        y_full_masked[mask_full] = y_full[mask_full]
        
        # Reshape predicted values back to original grid shape
        predicted_grid = y_pred_full.reshape(imrg['precipitation'].shape)
        
        # Reshape true values back to original grid shape
        true_grid = y_full_masked.reshape(imrg['precipitation'].shape)
        
        # Define spatial extent for plotting (lon_min, lon_max, lat_min, lat_max)
        extent = [float(imrg['lon'].min()), float(imrg['lon'].max()),
                  float(imrg['lat'].min()), float(imrg['lat'].max())]
        
        # Create figure with two side-by-side map subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot true precipitation on first subplot
        im1 = ax1.imshow(true_grid, origin='lower', cmap='viridis', 
                         extent=extent, vmin=0, vmax=50, transform=ccrs.PlateCarree())
        ax1.set_title("True IMERG Precipitation (>0 mm/hr)")  # Title for true plot
        ax1.set_xlabel("Longitude")  # X-axis label
        ax1.set_ylabel("Latitude")   # Y-axis label
        ax1.coastlines(resolution='10m')  # Add coastlines only (no borders)
        plt.colorbar(im1, ax=ax1, label='Precipitation (mm/hr)')  # Colorbar for true plot
        
        # Plot predicted precipitation on second subplot
        im2 = ax2.imshow(predicted_grid, origin='lower', cmap='viridis', 
                         extent=extent, vmin=0, vmax=50, transform=ccrs.PlateCarree())
        ax2.set_title("Predicted Precipitation (Improved Model)")  # Title for predicted plot
        ax2.set_xlabel("Longitude")  # X-axis label
        ax2.set_ylabel("Latitude")  # Y-axis label
        ax2.coastlines(resolution='10m')  # Add coastlines only
        plt.colorbar(im2, ax=ax2, label='Predicted Rainfall (mm/hr)')  # Colorbar for predicted plot

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()

