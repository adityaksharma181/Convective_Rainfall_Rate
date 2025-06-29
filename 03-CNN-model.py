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
MODEL_SAVE_PATH = "insat_imerg_cnn_weighted_model.h5"  # Filepath to save the trained model

# Define hyperparameters for the CNN training
LEARNING_RATE = 1e-4   # Learning rate for the optimizer
EPOCHS = 200           # Total number of training epochs
BATCH_SIZE = 32        # Number of samples per training batch

# Build CNN model
grid_h, grid_w, channels = X.shape[1], X.shape[2], X.shape[3]   

# Get the input shape (height, width, channels) from X (input data)
model = Sequential([                          # Initialize a Sequential model (layers will be added one after another)
    InputLayer(input_shape=(grid_h, grid_w, channels)),    # Input layer that takes images with shape (grid_h, grid_w, channels)

    Conv2D(64, (3,3), activation='relu', padding='same'),  # First convolutional layer with 64 filters, 3x3 kernel, ReLU activation, same padding
    BatchNormalization(),                      # Normalize activations to stabilize and speed up training
    MaxPooling2D((2,2)),                       # Downsample the feature map using 2x2 max pooling

    Conv2D(64, (3,3), activation='relu', padding='same'),  # Second convolutional layer with 64 filters, 3x3 kernel
    BatchNormalization(),                      # Normalize again
    MaxPooling2D((2,2)),                       # Another max pooling layer to reduce spatial dimensions

    Conv2D(128, (3,3), activation='relu', padding='same'), # Third convolutional layer with 128 filters
    BatchNormalization(),                      # Normalize again
    MaxPooling2D((2,2)),                       # Another max pooling layer

    Flatten(),                                 # Flatten the 3D feature map to 1D vector for Dense layers
    Dense(128, activation='relu'),             # Fully connected Dense layer with 128 neurons and ReLU activation
    Dropout(0.3),                              # Dropout layer to prevent overfitting by randomly dropping 30% neurons
    Dense(grid_h * grid_w, activation='linear') # Output layer with size equal to (grid_h * grid_w), linear activation (since it's regression)
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),     # Compile model with Adam optimizer and a custom learning rate
    loss=weighted_mse,                               # Use custom weighted mean squared error as loss
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]  # Monitor RMSE during training
)
model.summary()                                      # Print a summary of the model (layers and parameters)

# Train model with validation set
history = model.fit(
    X_train, y_train_flat,                           # Training data (input and target)
    validation_data=(X_val, y_val_flat),             # Validation data (for monitoring overfitting)
    epochs=EPOCHS,                                    # Total number of training epochs
    batch_size=BATCH_SIZE,                            # Number of samples per training batch
    verbose=1                                         # Display progress bar during training
)

# Save trained model
model.save(MODEL_SAVE_PATH)                          # Save the entire model to disk at specified path
print(f"Model saved to {MODEL_SAVE_PATH}")           # Print save location

# Plot training history
plt.figure(figsize=(10,5))                           # Create a new figure with specified size
plt.plot(history.history['loss'], label='Train Weighted MSE')         # Plot training loss
plt.plot(history.history['val_loss'], label='Val Weighted MSE')       # Plot validation loss
plt.xlabel('Epoch')                                  # Label for x-axis
plt.ylabel('Weighted MSE')                           # Label for y-axis
plt.legend()                                         # Display legend for the plot
plt.grid(True)                                       # Display grid in plot
plt.show()                                           # Show the plot

# Evaluation helper
def eval_split(name, X_split, y_split):              # Define a function to evaluate model on a given dataset split
    y_pred_flat = model.predict(X_split)             # Use the model to predict target values (flattened)
    y_pred = y_pred_flat.reshape(y_split.shape)      # Reshape the prediction back to the original shape
    mse = mean_squared_error(y_split.flatten(), y_pred.flatten())  # Calculate Mean Squared Error between true and predicted
    print(f"{name} MSE: {mse:.4f}")  # Print MSE and RMSE for this split

# Evaluate on all splits
eval_split("Training", X_train, y_train)            # Evaluate performance on Training set
eval_split("Validation", X_val, y_val)              # Evaluate performance on Validation set
eval_split("Test", X_test, y_test)                  # Evaluate performance on Test set
