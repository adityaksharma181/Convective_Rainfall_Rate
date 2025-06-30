'''
  This code is designed to create the plot of avg True rainfall and predcited rainfall from the trained model 
  Flowchart of code
  1. Loop insat files for avg true GPM IMERG rainfall
  2. Predicted rainfall from model
  3. Define custom color for range of rainfalls
  4. Plot avg True and predicted from model
'''
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr

# Initialize list to store precipitation data from all IMERG files
all_imerg_precip = []

# Loop through each IMERG file path
for imerg_path, _ in imerg_files:
    ds = xr.open_dataset(imerg_path)  # Open NetCDF file
    ds = ds.sel(lat=slice(lat_min_clip, lat_max_clip), lon=slice(lon_min_clip, lon_max_clip))  # Clip to lat-lon range
    all_imerg_precip.append(ds['precipitation'].values)  # Append precipitation values

# Stack precipitation arrays along time axis and calculate mean rainfall
all_imerg_precip = np.stack(all_imerg_precip, axis=0)
mean_imerg_precip = np.nanmean(all_imerg_precip, axis=0)

# Open last IMERG file for grid reference and clip to lat-lon range
imerg_new = xr.open_dataset(imerg_files[-1][0]).sel(
    lat=slice(lat_min_clip, lat_max_clip),
    lon=slice(lon_min_clip, lon_max_clip)
)

# Get grid size (lat and lon dimensions)
lat_len, lon_len = imerg_new['lat'].size, imerg_new['lon'].size

# Initialize empty arrays for true and predicted rainfall maps
true_map = np.full((lat_len, lon_len), np.nan)
pred_map = np.full((lat_len, lon_len), np.nan)

# Run Random Forest model to predict rainfall for test samples
y_pred_test = rf.predict(X_test)

# Fill true and predicted rainfall maps at test grid locations
for idx, (lat_idx, lon_idx) in enumerate(grid_test):
    if 0 <= lat_idx < lat_len and 0 <= lon_idx < lon_len:  # Check grid bounds
        true_map[lat_idx, lon_idx] = mean_imerg_precip[lat_idx, lon_idx]  # True rainfall
        pred_map[lat_idx, lon_idx] = y_pred_test[idx]  # Predicted rainfall

# Create mask where true rainfall is 0 mm/hr
zero_rain_mask = (true_map == 0)
true_map[zero_rain_mask] = np.nan  # Mask 0 rainfall points in true map
pred_map[zero_rain_mask] = np.nan  # Mask same points in predicted map

# Create lat-lon grids for spatial masking
lats = imerg_new['lat'].values
lons = imerg_new['lon'].values
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')  # Create 2D grid arrays

# Apply spatial mask to limit map to desired lat-lon range
spatial_mask = (
    (lat_grid >= lat_min_clip) & (lat_grid <= lat_max_clip) &
    (lon_grid >= lon_min_clip) & (lon_grid <= lon_max_clip)
)

# Apply spatial mask to both true and predicted rainfall maps
true_map[~spatial_mask] = np.nan
pred_map[~spatial_mask] = np.nan

# Clip rainfall values between 0 and 50 mm/hr for display
true_capped = np.clip(true_map, 0, 50)
pred_capped = np.clip(pred_map, 0, 50)

# Define color bins for rainfall ranges
bounds = [0, 1, 5, 10, 20, 30, 40, 50]

# Define colors for each rainfall bin
colors = [
    (1, 1, 0.6),    # Light yellow for 0–1 mm/hr
    (0.8, 0.8, 0.8),  # Light gray for 1–5 mm/hr
    (0.6, 0.6, 0.6),  # Medium gray for 5–10 mm/hr
    (0.4, 0.4, 0.4),  # Dark gray for 10–20 mm/hr
    (0.6, 0, 0),      # Dark red for 20–30 mm/hr
    (0.4, 0, 0),      # Deeper red for 30–40 mm/hr
    (0, 0, 0),        # Black for 40–50 mm/hr
]

# Create colormap and normalization for color bins
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

# Create side-by-side figure for true and predicted rainfall maps
fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                         subplot_kw={'projection': ccrs.PlateCarree()})

# Titles for each subplot
titles = ["Mean IMERG Rainfall (All Times)", "RF Predicted Rainfall (Test Samples)"]
data = [true_capped, pred_capped]  # Data arrays for plotting

# Plot each map (true and predicted)
for ax, d, title in zip(axes, data, titles):
    im = ax.imshow(
        d,
        origin='upper',  # Top of array is top of plot
        extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip],  # Map extent
        transform=ccrs.PlateCarree(),  # Projection
        cmap=cmap,  # Color map
        norm=norm  # Color normalization
    )
    ax.coastlines(resolution='10m', linewidth=1)  # Add coastlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color='gray', alpha=0.5, linestyle='--')  # Gridlines
    gl.top_labels = gl.right_labels = False  # Turn off top and right labels
    ax.set_title(title, fontsize=14, fontweight='bold')  # Set subplot title

# Add a shared horizontal colorbar below the plots
cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                    pad=0.05, fraction=0.08, ticks=bounds)
cbar.set_label('Rainfall (mm/hr)')  # Colorbar label
cbar.set_ticklabels([f"{b:g}" for b in bounds])  # Custom tick labels

# Add overall figure title
plt.suptitle("Rainfall Map: Mean IMERG (True) vs RF Prediction (Test Grid)", fontsize=16, fontweight='bold')
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()  # Display the plot


# Adjust spacing to prevent overlaps
plt.tight_layout()

# Display the plot
plt.show()
