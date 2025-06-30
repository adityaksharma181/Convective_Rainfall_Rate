# Plot true vs predicted
'''
  This code is designed to plot true rain and predicted rainfall
  Flowchat of code
  1.
'''
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import cartopy.crs as ccrs  # Import Cartopy coordinate reference systems
import cartopy.feature as cfeature  # Import Cartopy map features

# Select sample index for visualization
sample_idx = 0  # Choose the first sample to visualize

# Replace NaNs in target (ground truth) data with 0.0
y_test = np.nan_to_num(y_test, nan=0.0)  # Fill NaNs in ground truth data with 0.0

# Get true precipitation grid for the sample
true_grid = y_test[sample_idx]  # Extract the true precipitation grid for selected sample

# Predict precipitation grid using the trained model
pred_grid = model.predict(X_test[sample_idx:sample_idx+1]).reshape(true_grid.shape)  # Get model prediction and reshape to match true grid

# Replace any NaNs in prediction with 0.0
pred_grid = np.nan_to_num(pred_grid, nan=0.0)  # Fill NaNs in prediction with 0.0

# Mask zero and negative precipitation values
masked_true_grid = np.ma.masked_where(true_grid <= 0, true_grid)  # Mask true grid values ≤ 0
masked_pred_grid = np.ma.masked_where(pred_grid <= 0, pred_grid)  # Mask predicted grid values ≤ 0

# Color scale limits (ignoring masked values)
vmin = min(masked_true_grid.min(), masked_pred_grid.min())  # Minimum value for color scale
vmax = max(masked_true_grid.max(), masked_pred_grid.max())  # Maximum value for color scale

# Custom colormap (white for masked areas)
cmap = plt.cm.viridis.copy()  # Copy the viridis colormap
cmap.set_bad(color='white')  # Set color for masked (bad) data to white

# Plotting with Cartopy projections
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})  # Create 2 side-by-side map plots

for ax, grid, title in zip(
    axes,
    [masked_true_grid, masked_pred_grid],  # Data grids to plot
    ['True Precipitation (Masked ≤ 0 mm/hr)', 'Predicted Precipitation (Masked ≤ 0 mm/hr)']  # Titles for plots
):
    # Plot precipitation grid
    im = ax.imshow(
        grid,  # Data to display
        origin='lower',  # Set image origin to lower left
        cmap=cmap,  # Use custom colormap
        extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip],  # Geographic extent of the plot
        vmin=vmin,  # Minimum color scale value
        vmax=vmax,  # Maximum color scale value
        transform=ccrs.PlateCarree()  # Map projection transform
    )

    # Add coastlines and optional land/ocean features
    ax.coastlines(resolution='10m', color='black', linewidth=1)  # Draw coastlines

    # Optional: Add borders or land features for better context
    ax.add_feature(cfeature.LAND, facecolor='lightgray')  # Add land with light gray color

    ax.set_title(title)  # Set title for each subplot

    # Add colorbar for each plot
    fig.colorbar(im, ax=ax, orientation='vertical', label='mm/hr')  # Add vertical colorbar with label

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()  # Display the plots
