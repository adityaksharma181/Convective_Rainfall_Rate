import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# File Paths and extraction from each file
insat = xr.open_dataset("insat_resampled.nc")  #  <---------------- File Paths
imerg = xr.open_dataset("gpm-imerg-resampled.nc")  #  <---------------- File Paths

# Interpolating INSAT data to match IMERG’s lat/lon resolution
insat_interp = insat.interp(lat=imerg['lat'], lon=imerg['lon'], method='linear')

# Extraction 2D variables from the dataset and flatten them to 1D arrays
def get_flat(var):
    return insat_interp[var].values.flatten()

# Brightness temperature fields and derived features
X = np.stack([
    get_flat('IMG_TIR1_TB'),
    get_flat('IMG_TIR2_TB'),
    get_flat('IMG_WV_TB'),
    get_flat('TIR1_WV_TB_DIFFERENCE'),
    get_flat('TIR2_WV_TB_DIFFERENCE'),
    get_flat('TIR_TB_DIFFERENCE'),
    get_flat('TIR_TB_DIVISION')
], axis=1)

# Addition of Latitude and Longitude as features
lat_grid, lon_grid = np.meshgrid(imerg['lat'].values, imerg['lon'].values, indexing='ij')
X_lat = lat_grid.flatten()
X_lon = lon_grid.flatten()

X = np.column_stack((X, X_lat, X_lon))

# Flattening IMERG precipitation to align with X 
y = imerg['precipitation'].values.flatten()

# Removing invalid or NaN values and zero-rainfall pixels
mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & (y > 0)
X = X[mask]
y = y[mask]

# More weight to heavy rainfall cases to avoid model bias toward light
weights = y.copy()
weights = weights / weights.max()  
weights = 1 + 4 * weights          # <------------- boost heavy rainfall (weight 1–5)

# Split data into training and testing sets
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Train Random Forest with weights
rf = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train, sample_weight=w_train)

# Using RMSE and R² to evaluate model performance
y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.3f} mm/hr")
print(f"✅ R² Score: {r2:.3f}")

# Removing the lat, lon from importance graph
all_features = [
    'TIR1_TB', 'TIR2_TB', 'WV_TB',
    'TIR1-WV_DIFF', 'TIR2-WV_DIFF', 'TIR_DIFF', 'TIR_DIV',
    'Lat', 'Lon'
]
all_importances = rf.feature_importances_
features = all_features[:7]
importances = all_importances[:7]

plt.figure(figsize=(8, 4))
plt.bar(features, importances, color='steelblue')
plt.xticks(rotation=45)
plt.title("Random Forest Feature Importance (Excluding Lat/Lon)")
plt.ylabel("Importance")
plt.grid(True)
plt.tight_layout()
plt.show()

# Use entire dataset for model prediction
X_all = np.stack([
    insat_interp['IMG_TIR1_TB'].values.flatten(),
    insat_interp['IMG_TIR2_TB'].values.flatten(),
    insat_interp['IMG_WV_TB'].values.flatten(),
    insat_interp['TIR1_WV_TB_DIFFERENCE'].values.flatten(),
    insat_interp['TIR2_WV_TB_DIFFERENCE'].values.flatten(),
    insat_interp['TIR_TB_DIFFERENCE'].values.flatten(),
    insat_interp['TIR_TB_DIVISION'].values.flatten(),
    X_lat, X_lon
], axis=1)

# Remove invalid pixels before prediction
valid_mask = np.isfinite(X_all).all(axis=1)
y_pred_all = np.full(X_all.shape[0], np.nan)
y_pred_all[valid_mask] = rf.predict(X_all[valid_mask])

# Reshape predictions to 2D for Plotting
shape = imerg['precipitation'].shape
pred_map = y_pred_all.reshape(shape)
true_map = imerg['precipitation'].values

#  Monochrome Style
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vmax = max(np.nanmax(true_map), np.nanmax(pred_map))
vmin = 0

# Plot 1: True Precipitation 
im0 = axes[0].imshow(true_map, cmap='Blues', vmin=vmin, vmax=vmax)
axes[0].set_title("IMERG Precipitation (True)", fontsize=13)
fig.colorbar(im0, ax=axes[0], shrink=0.8).ax.tick_params(labelsize=10)

# Plot 2: Predicted Precipitation 
im1 = axes[1].imshow(pred_map, cmap='Greens', vmin=vmin, vmax=vmax)
axes[1].set_title("Predicted Precipitation (RF)", fontsize=13)
fig.colorbar(im1, ax=axes[1], shrink=0.8).ax.tick_params(labelsize=10)

for ax in axes:
    ax.axis('off')

plt.suptitle("Rainfall Estimation: Single-Color, High-Contrast Maps", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# Vibrant Color Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: True Precipitation 
im0 = axes[0].imshow(true_map, cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title("IMERG Precipitation (True)", fontsize=13, fontweight='bold')
fig.colorbar(im0, ax=axes[0], shrink=0.8).ax.tick_params(labelsize=10)

# Plot 2: Predicted Precipitation 
im1 = axes[1].imshow(pred_map, cmap='plasma', vmin=vmin, vmax=vmax)
axes[1].set_title("Predicted Precipitation (RF)", fontsize=13, fontweight='bold')
fig.colorbar(im1, ax=axes[1], shrink=0.8).ax.tick_params(labelsize=10)

for ax in axes:
    ax.axis('off')

plt.suptitle("Rainfall Estimation: Vibrant High-Contrast Maps", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# Geospatial Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})

lat_min, lat_max = -10.0, 45.5
lon_min, lon_max = 44.5, 110.0

titles = [
    "IMERG Precipitation (True)",
    "Predicted Precipitation (RF)"
]

cmaps = ['viridis', 'plasma']
data_maps = [true_map, pred_map]

for i, ax in enumerate(axes):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    im = ax.imshow(
        data_maps[i],
        origin='upper',
        extent=[imerg.lon.min(), imerg.lon.max(), imerg.lat.min(), imerg.lat.max()],
        transform=ccrs.PlateCarree(),
        cmap=cmaps[i],
        vmin=vmin,
        vmax=vmax
    )
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False

    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, shrink=0.8, orientation='vertical')
    cb.ax.tick_params(labelsize=10)

plt.suptitle("Rainfall Estimation with Geospatial Context", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
