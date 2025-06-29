'''
  This code is designed to plot true vs predicted rainfall from the model
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Predict the target values for the test set and flatten the result to 1D array
y_pred_test = model.predict(X_test).flatten()

# Calculate Mean Squared Error (MSE) for the test set
test_mse = mean_squared_error(y_test, y_pred_test)

# rint the MSE for the test set
print(f"Test MSE: {test_mse:.4f}")

# Print the range of predicted and true precipitation values for the test set
print(f"Test set prediction range: {y_pred_test.min():.2f} - {y_pred_test.max():.2f} mm/hr")
print(f"Test set true range: {y_test.min():.2f} - {y_test.max():.2f} mm/hr")

# Analyze model performance across different precipitation intensity ranges (using only MSE now)
ranges = [(0, 1), (1, 5), (5, 15), (15, 30), (30, 50)]  # Precipitation ranges in mm/hr

# print("\n Performance by precipitation range (MSE only):")
for low, high in ranges:
    # Create a boolean mask selecting samples that fall in the current precipitation range
    mask_range = (y_test >= low) & (y_test < high)
    
    # Only calculate MSE if there are more than 10 samples in this range
    if np.sum(mask_range) > 10:
        # calculate MSE for this precipitation range
        mse_range = mean_squared_error(y_test[mask_range], y_pred_test[mask_range])
        
        # Calculate mean of predicted and true values for this range
        mean_pred = np.mean(y_pred_test[mask_range])
        mean_true = np.mean(y_test[mask_range])
        
        # Print MSE and means for this precipitation range
        print(f"  {low}-{high} mm/hr: MSE={mse_range:.3f}, "
              f"Mean_pred={mean_pred:.3f}, Mean_true={mean_true:.3f}, N={np.sum(mask_range)}")

# Full grid prediction for visualization (only if ds_interp_list is not empty)
if len(ds_interp_list) > 0:
    # Prepare the full input feature grid by stacking all features as columns
    X_full = np.stack([ds_interp[feat].values.flatten() for feat in feature_names], axis=1)
    
    # Get the true precipitation values for the full grid
    y_full = imrg['precipitation'].values.flatten()
    
    # Clip true precipitation values to a maximum of 50 mm/hr for visualization
    y_full = np.clip(y_full, 0, 50)
    
    # Create a mask to filter out any non-finite (NaN or inf) values from features and target
    mask_full = np.isfinite(X_full).all(axis=1) & np.isfinite(y_full)
    
    # Only proceed if there are any valid samples
    if np.sum(mask_full) > 0:
        # Apply the same scaling transformation used during training
        X_full_scaled = scaler.transform(X_full[mask_full])
        
        # Make model predictions for the full grid and flatten the result
        y_pred_flat = model.predict(X_full_scaled).flatten()
        
        # Create an array to hold predicted precipitation for the full grid, filled with NaNs initially
        y_pred_full = np.full(y_full.shape, np.nan)
        
        # Fill the predicted values only at valid (finite) grid points
        y_pred_full[mask_full] = y_pred_flat
        
        # Reshape the flat predicted array back to the original grid shape
        predicted_grid = y_pred_full.reshape(imrg['precipitation'].shape)
        
        # Start plotting: Create a figure with 3 subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Define the spatial extent of the grid for correct axis scaling on the plot
        extent = [float(imrg['lon'].min()), float(imrg['lon'].max()),
                  float(imrg['lat'].min()), float(imrg['lat'].max())]
        
        # Plot the true precipitation values (clipped at 50 mm/hr)
        im1 = ax1.imshow(np.clip(imrg['precipitation'].values, 0, 50), 
                         origin='lower', cmap='viridis', extent=extent, vmin=0, vmax=50)
        ax1.set_title("True IMERG Precipitation")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        plt.colorbar(im1, ax=ax1, label='Precipitation (mm/hr)')
        
        # Plot the predicted precipitation grid from the model
        im2 = ax2.imshow(predicted_grid, origin='lower', cmap='viridis', 
                         extent=extent, vmin=0, vmax=50)
        ax2.set_title("Predicted Precipitation (Improved Model)")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        plt.colorbar(im2, ax=ax2, label='Predicted Rainfall (mm/hr)')

# Predict on training set and calculate MSE
y_pred_train = model.predict(X_train).flatten()
train_mse = mean_squared_error(y_train, y_pred_train)

# Predict on validation set and calculate MSE
y_pred_val = model.predict(X_val).flatten()
val_mse = mean_squared_error(y_val, y_pred_val)

# Print MSE values for training, validation, and test sets
print("\n📈 Overall MSE Metrics:")
print(f"✅ Training MSE: {train_mse:.4f}")
print(f"✅ Validation MSE: {val_mse:.4f}")
print(f"✅ Test MSE: {test_mse:.4f}")
