'''
  This code is designed to plot avg temp spatially from insat files
  1. Loop over each insat file
  2. computing mean BT 
  3. Plot avg BT 
'''
import numpy as np                           # For numerical array operations
import matplotlib.pyplot as plt              # For plotting and visualization
import cartopy.crs as ccrs                   # For handling map projections (Cartopy)
import xarray as xr                          # For working with multi-dimensional gridded data (NetCDF etc.)

# BT from insat files
bt_sum = None                                # Initialize BT (Brightness Temperature) sum array as None
count = 0                                    # Initialize file counter

# Load INSAT file paths
insat_folder = "/kaggle/input/nc-insat"        # Path to folder containing INSAT NetCDF files
insat_files = [(file, None) for file in sorted(glob(os.path.join(insat_folder, "*.nc")))]   # Load all .nc files into a list of tuples (file, None)


# Loop over each INSAT file
for file, _ in insat_files:                  # 'insat_files' is assumed to be a list of (file, something) tuples
    ds = xr.open_dataset(file).sel(          # Open each NetCDF dataset file and select clipped latitude/longitude range
        lat=slice(lat_min_clip, lat_max_clip),        # Clip latitude range (from lat_min_clip to lat_max_clip)
        lon=slice(lon_min_clip, lon_max_clip)         # Clip longitude range (from lon_min_clip to lon_max_clip)
    )
    
    # Renaming coordinates if needed
    if 'latitude' in ds.coords and 'longitude' in ds.coords:  # Check if file uses 'latitude' and 'longitude' as coordinate names
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})  # Rename them to 'lat' and 'lon' for consistency
    
    # Get TIR1 Brightness Temperature
    bt = ds['IMG_TIR1_TB'].values            # Extract TIR1 Brightness Temperature data as a numpy array
    
    # Initialize sum array if first file
    if bt_sum is None:                       # If this is the first file (bt_sum not initialized yet)
        bt_sum = np.zeros_like(bt, dtype=np.float64)    # Initialize bt_sum array with zeros, same shape as bt, as float
    
    # Accumulate sum, ignoring NaNs
    valid_mask = ~np.isnan(bt)               # Create a boolean mask where bt values are NOT NaN (valid data locations)
    bt_sum[valid_mask] += bt[valid_mask]     # Add valid BT values to the running sum
    
    # Increment count where data is valid
    if count == 0:                           # For first file, initialize valid_counts array
        valid_counts = np.zeros_like(bt, dtype=np.int32)  # Count array for how many valid (non-NaN) values per grid cell
    valid_counts[valid_mask] += 1            # Increment count where BT data was valid
    
    count += 1                               # Increment the total file counter

# Mean BT
avg_bt_map = np.where(valid_counts > 0, bt_sum / valid_counts, np.nan)   # Calculate mean BT where there was at least one valid observation, else set NaN

# plot avg true BT
bt_vmin = np.nanmin(avg_bt_map)              # Minimum BT value across the map (ignoring NaNs), for color scale
bt_vmax = np.nanmax(avg_bt_map)              # Maximum BT value across the map (ignoring NaNs), for color scale

# Create a figure and axes for plotting with Cartopy projection
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})   # Create figure and axis with PlateCarree projection (lat-lon map)

# Set map extent to match data clipping range
ax.set_extent([lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip], crs=ccrs.PlateCarree())  # Set map limits (lon_min, lon_max, lat_min, lat_max)

# Plot the average BT map
im = ax.imshow(
    avg_bt_map,                              # The 2D data array to plot (average BT)
    origin='upper',                          # Origin of image is at the top
    extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip],   # Set extent of the image on the map
    transform=ccrs.PlateCarree(),            # Define data coordinate system (regular lat-lon grid)
    cmap='inferno',                          # Color map style for the image (inferno colormap)
    vmin=bt_vmin,                            # Minimum color scale value
    vmax=bt_vmax                             # Maximum color scale value
)

ax.coastlines(resolution='10m', linewidth=1)  # Add coastlines with 10m resolution and 1px line width for reference

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--', alpha=0.5)  # Draw gridlines (lat/lon lines) with labels and styling
gl.top_labels = False                      # Do not show labels on top
gl.right_labels = False                    # Do not show labels on right side

# Add title to the plot
ax.set_title("Brightness Temperature (TIR1)", fontsize=14, fontweight='bold')   # Set plot title with font styling

# Add colorbar for reference
cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='BT (K)')   # Add colorbar, shrink it to fit better, label as BT in Kelvin

# Add super title with file count info
plt.suptitle(f"INSAT TIR1 Brightness Temperature Average (From {count} Files)", fontsize=16, fontweight='bold')  # Big title showing number of files averaged

plt.tight_layout()                          # Adjust layout to avoid overlapping of labels and titles
plt.show()                                   # Display the plot
