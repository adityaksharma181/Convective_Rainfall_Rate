'''
  This code is designed to create the plot of avg True rainfall and predcited rainfall from the trained model 
  Flowchart of code
  1. Loop insat files for avg true GPM IMERG rainfall
  2. Predicted rainfall from model
  3. Define custom color for range of rainfalls
  4. Plot avg True and predicted from model
'''
import numpy as np                           # For numerical operations and array handling
import matplotlib.pyplot as plt              # For plotting and visualization
import cartopy.crs as ccrs                   # For map projections and geographic plotting
from matplotlib.colors import ListedColormap, BoundaryNorm  # For custom color mapping
import xarray as xr                          # For working with NetCDF and gridded datasets

# List to store precipitation data from all IMERG files
all_imerg_precip = []

# Loop through all IMERG files (paths stored in imerg_files)
for imerg_path, _ in imerg_files:
    ds = xr.open_dataset(imerg_path)  # Open each IMERG NetCDF file as an xarray dataset
    
    # Clip the data to the desired latitude and longitude range
    ds = ds.sel(lat=slice(lat_min_clip, lat_max_clip), lon=slice(lon_min_clip, lon_max_clip))
    
    # Extract precipitation values and store in the list
    all_imerg_precip.append(ds['precipitation'].values)

# Stack all time slices into a single 3D array: (time, lat, lon)
all_imerg_precip = np.stack(all_imerg_precip, axis=0)

# Calculate mean precipitation across time dimension (axis 0)
mean_imerg_precip = np.nanmean(all_imerg_precip, axis=0)

# Prepare Lat-Lon Grid for Mapping
# Open the last IMERG file to get lat-lon grid reference (since all files have same grid)
imerg_new = xr.open_dataset(imerg_files[-1][0]).sel(
    lat=slice(lat_min_clip, lat_max_clip),    # Clip latitude range
    lon=slice(lon_min_clip, lon_max_clip)     # Clip longitude range
)

# number of latitude and longitude grid points
lat_len, lon_len = imerg_new['lat'].size, imerg_new['lon'].size

# Create empty arrays to hold true and predicted rainfall maps
true_map = np.full((lat_len, lon_len), np.nan)  # Initialize with NaNs
pred_map = np.full((lat_len, lon_len), np.nan)  # Initialize with NaNs

# Fill True and Predicted Rainfall Maps
# Run the trained Random Forest model to predict rainfall for test samples
y_pred_test = rf.predict(X_test)  # Predict using the test feature set

# Loop over each test grid point
for idx, (lat_idx, lon_idx) in enumerate(grid_test):
    
    # Check if the grid index is within valid lat-lon dimensions
    if 0 <= lat_idx < lat_len and 0 <= lon_idx < lon_len:
        
        # Assign the true rainfall (from IMERG mean) to the true_map
        true_map[lat_idx, lon_idx] = mean_imerg_precip[lat_idx, lon_idx]
        
        # Assign the model's predicted rainfall to the pred_map
        pred_map[lat_idx, lon_idx] = y_pred_test[idx]

# Apply Spatial Masking to Remove Unwanted Areas
# Get latitude and longitude values from the IMERG grid
lats = imerg_new['lat'].values
lons = imerg_new['lon'].values

# Create 2D meshgrid of lat and lon for masking
lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

# Create a spatial mask: True for points within lat-lon clipping bounds, False outside
spatial_mask = (
    (lat_grid >= lat_min_clip) & (lat_grid <= lat_max_clip) &   # Lat within range
    (lon_grid >= lon_min_clip) & (lon_grid <= lon_max_clip)     # Lon within range
)

# Apply mask: Set points outside the desired area to NaN
true_map[~spatial_mask] = np.nan
pred_map[~spatial_mask] = np.nan

# Limit true rainfall values between 0 and 50 mm/hr
true_capped = np.clip(true_map, 0, 50)

# Limit predicted rainfall values between 0 and 50 mm/hr
pred_capped = np.clip(pred_map, 0, 50)

# Define Color Map and Bins for Plotting
# Define rainfall bins (thresholds) for color levels
bounds = [0, 1, 5, 10, 20, 30, 40, 50]

# Define corresponding colors for each bin (from white to black through grays and reds)
colors = [
    (1,1,1),         # 0–1 mm/hr : white
    (0.8,0.8,0.8),   # 1–5 mm/hr : light gray
    (0.6,0.6,0.6),   # 5–10 mm/hr: medium gray
    (0.4,0.4,0.4),   # 10–20 mm/hr: dark gray
    (0.6,0,0),       # 20–30 mm/hr: dark red
    (0.4,0,0),       # 30–40 mm/hr: deeper red
    (0,0,0),         # 40–50 mm/hr: black
]

# Create a custom color map using the defined colors
cmap = ListedColormap(colors)

# Define normalization: Map data values to color bins
norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

# Plot True vs Predicted Rainfall Maps Side-by-Side
# Create figure with two subplots (side by side), each with a Plate Carree map projection
fig, axes = plt.subplots(1, 2, figsize=(16,6),
                         subplot_kw={'projection': ccrs.PlateCarree()})

# Define titles for the two subplots
titles = ["Mean IMERG Rainfall (All Times)", "RF Predicted Rainfall (Test Samples)"]

# Data to plot in each panel: true first, then predicted
data = [true_capped, pred_capped]

# Loop through each subplot: one for true, one for predicted
for ax, d, title in zip(axes, data, titles):
    
    # Plot rainfall data as an image on the map
    im = ax.imshow(
        d,                                           # Rainfall data (2D array)
        origin='upper',                              # Keep top at the top
        extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip],  # Map extent
        transform=ccrs.PlateCarree(),                # Coordinate reference system
        cmap=cmap,                                   # Use custom color map
        norm=norm                                    # Apply color bin normalization
    )
    
    # Draw coastlines for better geographic context
    ax.coastlines(resolution='10m', linewidth=1)
    
    # Add gridlines with lat-lon labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color='gray', alpha=0.5, linestyle='--')
    
    # Remove top and right labels to avoid clutter
    gl.top_labels = gl.right_labels = False
    
    # Set subplot title
    ax.set_title(title, fontsize=14, fontweight='bold')

# Add Shared Colorbar
# Add horizontal colorbar below both plots
cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                    pad=0.05, fraction=0.08, ticks=bounds)

# Label for the colorbar
cbar.set_label('Rainfall (mm/hr)')

# Set tick labels (matching bin edges)
cbar.set_ticklabels([f"{b:g}" for b in bounds])

# Add Overall Figure Title and Show Plot
# Main title for the whole figure
plt.suptitle("Rainfall Map: Mean IMERG (True) vs RF Prediction (Test Grid)", fontsize=16, fontweight='bold')

# Adjust spacing to prevent overlaps
plt.tight_layout()

# Display the plot
plt.show()
