import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Files path and parameters
insat = "/kaggle/working/TB_insat_output.nc"
imerg = "/kaggle/working/output_precipitation_data.nc"
MODEL_SAVE_PATH = "insat_imerg_ann_model_mse.h5"

# Model parameters
LEARNING_RATE = 0.0005
EPOCHS = 100
BATCH_SIZE = 8000

#Extraction of necessory varaibles
ds = xr.open_dataset(insat)
imrg = xr.open_dataset(imerg)

# Use of files that have similat time stamp
if 'time' in ds.dims:
    ds = ds.squeeze('time', drop=True)
if 'time' in imrg.dims:
    imrg = imrg.squeeze('time', drop=True)

#Align of insat and imerg using lat,lon
ds_interp = ds.interp(lat=imrg['lat'], lon=imrg['lon'], method="linear")

# Use of variable for model
features = [
    'IMG_TIR1_TB', 'IMG_TIR2_TB', 'IMG_WV_TB',
    'TIR1_WV_TB_DIFFERENCE', 'TIR2_WV_TB_DIFFERENCE',
    'TIR_TB_DIVISION', 'TIR_TB_DIFFERENCE'
]
X = np.stack([ds_interp[feat].values.flatten() for feat in features], axis=1)
y = imrg['precipitation'].values.flatten()

# Remove NaN values and condition that precipitation is greater than 0 mm/hr
mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & (y > 0)
X = X[mask]
y = y[mask]

# use of Weights as Model was ignoring Heavy rainfall in prediction
rain_bins = np.digitize(y, bins=[2, 5, 8])  
bin_weights = np.array([2, 4, 8.0, 25.0])
sample_weights = bin_weights[rain_bins]
sample_weights /= np.mean(sample_weights)

print(f"After filtering: {X.shape[0]} samples remain")
print(f"Rainfall mean: {y.mean():.4f}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dataset Spliting in training set and test set
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X_scaled, y, sample_weights, test_size=0.2, random_state=42
)

input_shape = X_train.shape[1]
print(f"Input shape: {input_shape}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Model 
model = Sequential([
    Dense(50, input_dim=input_shape, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
             tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

model.summary()

# Model training 
history = model.fit(
    X_train, y_train,
    sample_weight=sw_train,
    validation_data=(X_test, y_test, sw_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Save model to predict precipitation
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# PLOTS
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Val MSE')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()


# BIN wise evaluation
y_pred_test = model.predict(X_test).flatten()
test_bins = np.digitize(y_test, bins=[2, 5, 8])
bin_names = ["0-2", "2-5", "5-8", "8-inf"]

print("📊 Evaluation by Rainfall Intensity Bin (Test Set):")
for i, label in enumerate(bin_names):
    idx = np.where(test_bins == i)[0]
    if len(idx) == 0:
        continue
    rmse = np.sqrt(mean_squared_error(y_test[idx], y_pred_test[idx]))
    mae = mean_absolute_error(y_test[idx], y_pred_test[idx])
    print(f"Range {label} mm/hr: RMSE = {rmse:.3f}, MAE = {mae:.3f}, Samples = {len(idx)}")

# Grid prediction
X_full = np.stack([ds_interp[feat].values.flatten() for feat in features], axis=1)
y_full = imrg['precipitation'].values.flatten()
mask_full = np.isfinite(X_full).all(axis=1) & np.isfinite(y_full) & (y_full > 0)

X_full_scaled = scaler.transform(X_full[mask_full])
y_pred_flat = model.predict(X_full_scaled).flatten()

y_pred_full = np.full(y_full.shape, np.nan)
y_pred_full[mask_full] = y_pred_flat
predicted_grid = y_pred_full.reshape(imrg['precipitation'].shape)

# Results shown using plots
plt.figure(figsize=(8, 6))
plt.imshow(imrg['precipitation'].values, origin='lower', cmap='viridis',
           extent=[float(imrg['lon'].min()), float(imrg['lon'].max()),
                   float(imrg['lat'].min()), float(imrg['lat'].max())])
plt.colorbar(label='Precipitation (mm/hr)')
plt.title("True IMERG Precipitation")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(predicted_grid, origin='lower', cmap='viridis',
           extent=[float(imrg['lon'].min()), float(imrg['lon'].max()),
                   float(imrg['lat'].min()), float(imrg['lat'].max())])
plt.colorbar(label='Predicted Rainfall (mm/hr)')
plt.title("Predicted Precipitation (MSE, Bin-Weighted)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(False)
plt.show()


