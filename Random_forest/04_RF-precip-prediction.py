import numpy as np
import xarray as xr
import joblib
import h5py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error, r2_score

# Validation dataset files
insat_new = xr.open_dataset("/kaggle/working/TB_insat_new.nc")
imerg_new = xr.open_dataset("/kaggle/working/precip_new.nc")

# Align INSAT to IMERG
insat_interp = insat_new.interp(lat=imerg_new['lat'], lon=imerg_new['lon'], method='linear')

# Preaparing feature to compare
def get_flat(var):
    return insat_interp[var].values.flatten()

X_new = np.stack([
    get_flat('IMG_TIR1_TB'),
    get_flat('IMG_TIR2_TB'),
    get_flat('IMG_WV_TB'),
    get_flat('TIR1_WV_TB_DIFFERENCE'),
    get_flat('TIR2_WV_TB_DIFFERENCE'),
    get_flat('TIR_TB_DIFFERENCE'),
    get_flat('TIR_TB_DIVISION')
], axis=1)

# Add lat,lon
lat_grid, lon_grid = np.meshgrid(imerg_new['lat'].values, imerg_new['lon'].values, indexing='ij')
X_lat = lat_grid.flatten()
X_lon = lon_grid.flatten()

X_new = np.column_stack((X_new, X_lat, X_lon))

# True precipitation
y_true = imerg_new['precipitation'].values.flatten()

# Filter out valid and rainfall greater than 0 mm/hr
valid_mask = np.isfinite(X_new).all(axis=1) & np.isfinite(y_true) & (y_true > 0)
X_new = X_new[valid_mask]
y_true = y_true[valid_mask]

# Apply weight to heavy rainfall
weights = y_true / y_true.max()
weights = 1 + 4 * weights  # scale from 1 to 5

# Load trained Model
model = joblib.load("/kaggle/working/random_forest_model.pkl")

# Prediction
y_pred = model.predict(X_new)

# Evaluate
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"📊 RMSE: {rmse:.3f} mm/hr")


# Save Pridiction to h5 format
with h5py.File("/kaggle/working/validation_output.h5", "w") as f:
    f.create_dataset("y_true", data=y_true)
    f.create_dataset("y_pred", data=y_pred)
    f.attrs["RMSE"] = rmse
    f.attrs["R2_Score"] = r2

# Maps
true_map_full = imerg_new['precipitation'].values
lat_len, lon_len = true_map_full.shape

# Initialize maps with NaNs
true_map = np.full((lat_len, lon_len), np.nan)
pred_map = np.full((lat_len, lon_len), np.nan)

# Assign predicted/true values where valid
flat_index = np.where(valid_mask)[0]
true_map_flat = true_map.flatten()
pred_map_flat = pred_map.flatten()
true_map_flat[valid_mask] = y_true
pred_map_flat[valid_mask] = y_pred

true_map = true_map_flat.reshape(lat_len, lon_len)
pred_map = pred_map_flat.reshape(lat_len, lon_len)

# PLot 
vmin, vmax = 0, max(y_true.max(), y_pred.max())

# PLot 1
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(true_map, cmap='Blues', vmin=vmin, vmax=vmax)
axes[0].set_title("IMERG Precipitation (True)", fontsize=13)
fig.colorbar(im0, ax=axes[0], shrink=0.8).ax.tick_params(labelsize=10)

im1 = axes[1].imshow(pred_map, cmap='Greens', vmin=vmin, vmax=vmax)
axes[1].set_title("Predicted Precipitation (RF)", fontsize=13)
fig.colorbar(im1, ax=axes[1], shrink=0.8).ax.tick_params(labelsize=10)

for ax in axes:
    ax.axis('off')

plt.suptitle("Rainfall Estimation: Single-Color, High-Contrast Maps", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# Plot 2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(true_map, cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title("IMERG Precipitation (True)", fontsize=13, fontweight='bold')
fig.colorbar(im0, ax=axes[0], shrink=0.8).ax.tick_params(labelsize=10)

im1 = axes[1].imshow(pred_map, cmap='plasma', vmin=vmin, vmax=vmax)
axes[1].set_title("Predicted Precipitation (RF)", fontsize=13, fontweight='bold')
fig.colorbar(im1, ax=axes[1], shrink=0.8).ax.tick_params(labelsize=10)

for ax in axes:
    ax.axis('off')

plt.suptitle("Rainfall Estimation: Vibrant High-Contrast Maps", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# Plot 3
fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})
lat_min, lat_max = float(imerg_new.lat.min()), float(imerg_new.lat.max())
lon_min, lon_max = float(imerg_new.lon.min()), float(imerg_new.lon.max())

titles = ["IMERG Precipitation (True)", "Predicted Precipitation (RF)"]
cmaps = ['viridis', 'plasma']
data_maps = [true_map, pred_map]

for i, ax in enumerate(axes):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    im = ax.imshow(
        data_maps[i],
        origin='upper',
        extent=[lon_min, lon_max, lat_min, lat_max],
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
