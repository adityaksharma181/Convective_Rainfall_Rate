# Model
'''
    This code is designed to train CNN model by taking BT variables, precipitation to predict rainfall
    Flowchart of Code
    1. Model save path
    2. hyperparameters 
    3. CNN model to take (lat, lon, channels) input
    4. Train model on training dataset
    5. Validate model
'''
import numpy as np                           # For numerical operations
import matplotlib.pyplot as plt              # For plotting graphs
from tensorflow.keras.models import Sequential      # For creating sequential CNN model
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout  # CNN layers
from tensorflow.keras.optimizers import Adam         # Optimizer
import tensorflow as tf                      # TensorFlow library
from sklearn.metrics import mean_squared_error       # For calculating MSE for evaluation

# Save model 
MODEL_SAVE_PATH = "insat_imerg_cnn_weighted_model.h5"  # Path to save trained model

# Define hyperparameters for the CNN training
LEARNING_RATE = 1e-4   # Learning rate for optimizer
EPOCHS = 200           # Number of epochs to train
BATCH_SIZE = 24        # Batch size for training

# Build CNN model
grid_h, grid_w, channels = X.shape[1], X.shape[2], X.shape[3]   # Get input image dimensions from X

# Create the CNN model
model = Sequential([                          # Initialize sequential model

    InputLayer(input_shape=(grid_h, grid_w, channels)),    # Input layer with image shape

    Conv2D(64, (3,3), activation='relu', padding='same'),  # 1st Conv layer with 64 filters
    BatchNormalization(),                      # Normalize activations
    MaxPooling2D((2,2)),                       # Downsample feature maps

    Conv2D(64, (3,3), activation='relu', padding='same'),  # 2nd Conv layer
    BatchNormalization(),                      # Normalize activations
    MaxPooling2D((2,2)),                       # Downsample again

    Conv2D(128, (3,3), activation='relu', padding='same'), # 3rd Conv layer
    BatchNormalization(),                      # Normalize activations
    MaxPooling2D((2,2)),                       # Further downsampling

    Flatten(),                                 # Flatten to 1D vector
    Dense(128, activation='relu'),             # Fully connected dense layer
    Dropout(0.3),                              # Dropout to reduce overfitting
    Dense(grid_h * grid_w, activation='linear') # Output layer (regression)
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),     # Compile with Adam optimizer
    loss=weighted_mse,                               # Custom weighted MSE loss
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]  # Track RMSE metric
)
model.summary()                                      # Print model summary

# Train model with validation data
history = model.fit(
    X_train, y_train_flat,                           # Training data
    validation_data=(X_val, y_val_flat),             # Validation data
    epochs=EPOCHS,                                    # Total epochs
    batch_size=BATCH_SIZE,                            # Batch size
    verbose=1                                         # Show progress
)

# Save trained model
model.save(MODEL_SAVE_PATH)                          # Save model to file
print(f"Model saved to {MODEL_SAVE_PATH}")           # Confirm save location

# Plot training history
plt.figure(figsize=(10,5))                           # Set plot size
plt.plot(history.history['loss'], label='Train Weighted MSE')         # Plot training loss
plt.plot(history.history['val_loss'], label='Val Weighted MSE')       # Plot validation loss
plt.xlabel('Epoch')                                  # X-axis label
plt.ylabel('Weighted MSE')                           # Y-axis label
plt.legend()                                         # Show legend
plt.grid(True)                                       # Show grid
plt.show()                                           # Display plot

# Evaluation helper function
def eval_split(name, X_split, y_split):              # Define eval function for dataset splits
    y_pred_flat = model.predict(X_split)             # Predict flattened output
    y_pred = y_pred_flat.reshape(y_split.shape)      # Reshape to original target shape
    mse = mean_squared_error(y_split.flatten(), y_pred.flatten())  # Calculate MSE
    print(f"{name} MSE: {mse:.4f}")                  # Print MSE result

# Evaluate on all splits
eval_split("Training", X_train, y_train)            # Evaluate on training data
eval_split("Validation", X_val, y_val)              # Evaluate on validation data
eval_split("Test", X_test, y_test)                  # Evaluate on test data

