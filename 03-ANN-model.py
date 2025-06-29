# Import necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting the training history
import tensorflow as tf  # For building and training the neural network
from tensorflow.keras.models import Sequential  # For sequential model building
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout  # Layers for CNN
from tensorflow.keras.optimizers import Adam  # Optimizer
from tensorflow.keras.losses import MeanSquaredError  # Loss function (optional, but commonly imported)

# Assuming X, X_train, y_train_flat, X_val, y_val_flat, LEARNING_RATE, EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH, weighted_mse are already defined in your environment.

# Get input shape from data
grid_h, grid_w, channels = X.shape[1], X.shape[2], X.shape[3]  
# grid_h: Height of each input sample
# grid_w: Width of each input sample
# channels: Number of channels (like RGB = 3 channels, grayscale = 1 channel)

# Build CNN model
model = Sequential([  # Create a sequential model where layers are stacked one after another

    InputLayer(input_shape=(grid_h, grid_w, channels)),  
    # First layer: Input layer that matches the shape of the input data

    Conv2D(64, (3,3), activation='relu', padding='same'),  
    # 2D Convolutional layer with 64 filters, 3x3 kernel size, ReLU activation
    # Padding='same' means output size stays the same as input size after convolution

    BatchNormalization(),  
    # Normalize activations for faster training and stability

    MaxPooling2D((2,2)),  
    # Reduce the spatial size (height and width) by taking the maximum value over 2x2 patches

    Conv2D(64, (3,3), activation='relu', padding='same'),  
    # Another Conv2D layer with same settings

    BatchNormalization(),  
    # Normalize again

    MaxPooling2D((2,2)),  
    # Another MaxPooling to reduce size further

    Conv2D(128, (3,3), activation='relu', padding='same'),  
    # Third Conv2D layer with more filters (128)

    BatchNormalization(),  
    # Normalize again

    MaxPooling2D((2,2)),  
    # Final pooling layer to reduce dimensions again

    Flatten(),  
    # Flatten the 3D output from the last layer into a 1D vector so it can go into Dense layers

    Dense(128, activation='relu'),  
    # Fully connected (dense) layer with 128 neurons and ReLU activation

    Dropout(0.3),  
    # Dropout layer: Randomly drops 30% of the neurons during training to prevent overfitting

    Dense(grid_h * grid_w, activation='linear')  
    # Final output layer: Number of outputs equals total grid size
    # Activation is linear because we are predicting continuous values (regression task)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),  
    # Optimizer: Adam with user-defined learning rate

    loss=weighted_mse,  
    # Custom loss function called weighted_mse (assumed to be defined already)

    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]  
    # Evaluation metric: Root Mean Squared Error (RMSE)
)

# Print model architecture summary
model.summary()  

# Train model with validation set
history = model.fit(
    X_train, y_train_flat,  
    # Training inputs and flattened training targets

    validation_data=(X_val, y_val_flat),  
    # Validation data inputs and targets for monitoring during training

    epochs=EPOCHS,  
    # Total number of training epochs (user-defined constant)

    batch_size=BATCH_SIZE,  
    # Number of samples per gradient update (user-defined constant)

    verbose=1  
    # Verbosity mode: 1 means progress bar and training info will be shown during training
)

# Save trained model
model.save(MODEL_SAVE_PATH)  
# Save the entire model (architecture + weights + optimizer state) to the given path
print(f"Model saved to {MODEL_SAVE_PATH}")  
# Print confirmation of save location

# Plot training history
plt.figure(figsize=(10,5))  
# Create a new figure with specified size (width=10, height=5 inches)

plt.plot(history.history['loss'], label='Train Weighted MSE')  
# Plot training loss over epochs

plt.plot(history.history['val_loss'], label='Val Weighted MSE')  
# Plot validation loss over epochs

plt.xlabel('Epoch')  
# Label for x-axis

plt.ylabel('Weighted MSE')  
# Label for y-axis (showing loss)

plt.legend()  
# Show legend to differentiate between training and validation curves

plt.grid(True)  
# Display grid on plot

plt.show()  
# Display the plot
