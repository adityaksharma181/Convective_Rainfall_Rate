'''
  This code is designed to train ANN model using custom L1 and L2 regularisation with early stop fnuctionality
  Flowchart of code
  1. Hyperparameters
  2. Define Model
  3. Training and Validation of Model
'''
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting training history
import joblib  # For saving the scaler object
import tensorflow as tf  # TensorFlow main library
from tensorflow.keras.models import Sequential  # For creating a linear stack of layers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # Common layers for neural networks
from tensorflow.keras.optimizers import Adam  # Optimizer for training
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Callbacks to control training process
from tensorflow.keras.regularizers import l1_l2  # For L1 and L2 regularization to avoid overfitting

# Define the path where the trained model will be saved
MODEL_SAVE_PATH = "insat_imerg_ann_model_improved.h5"

# hyperparameters for better model performance
LEARNING_RATE = 0.0001                  # Learning rate for optimizer
EPOCHS = 200                            # Number of training epochs
BATCH_SIZE = 4000                        # Number of samples per training batch
DROPOUT_RATE = 0.3                      # Dropout rate to prevent overfitting
L1_REG = 1e-6                           # L1 regularization strength
L2_REG = 1e-5                           # L2 regularization strength

# Build the improved Artificial Neural Network (ANN) model
model = Sequential([  # Start a Sequential model (layers are stacked one after another)

    # First hidden layer with 512 neurons, ReLU activation, and L1/L2 regularization
    Dense(512, input_dim=input_shape, activation='relu', 
          kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)),
    BatchNormalization(),  # Normalize the outputs of the previous layer to stabilize training
    Dropout(DROPOUT_RATE),  # Randomly drop out some neurons during training to prevent overfitting
    
    # Second hidden layer with 256 neurons
    Dense(256, activation='relu', 
          kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)),
    BatchNormalization(),  # Batch normalization again for stability
    Dropout(DROPOUT_RATE),  # Apply dropout
    
    # Third hidden layer with 128 neurons
    Dense(128, activation='relu', 
          kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)),
    BatchNormalization(),  # Normalize outputs
    Dropout(DROPOUT_RATE/2),  # Apply smaller dropout
    
    # Fourth hidden layer with 64 neurons (no regularization here)
    Dense(64, activation='relu'),
    Dropout(DROPOUT_RATE/2),  # Dropout to prevent overfitting
    
    # Fifth hidden layer with 32 neurons
    Dense(32, activation='relu'),
    
    # Output layer for regression task (1 output, linear activation)
    Dense(1, activation='linear')  # Linear output for continuous regression targets
])

# Compile the model 
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),  # Use Adam optimizer with custom learning rate
    loss=custom_loss,  # Use your custom-defined loss function for training
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae")  # Calculate MAE during training
    ]
)

# Print the model summary (shows layer details and parameters)
model.summary()

# Define callbacks to control training and avoid overfitting
callbacks = [
    EarlyStopping(  # Stop training early if validation loss doesn't improve
        monitor='val_loss',  # Watch validation loss
        patience=20,  # Wait for 20 epochs before stopping
        restore_best_weights=True,  # Keep the best model weights
        verbose=1  # Print messages when early stopping triggers
    ),
    ReduceLROnPlateau(  # Reduce learning rate if model stops improving
        monitor='val_loss',  # Watch validation loss
        factor=0.5,  # Reduce learning rate by half
        patience=10,  # Wait 10 epochs before reducing LR
        min_lr=1e-6,  # Minimum allowed learning rate
        verbose=1  # Print messages when LR is reduced
    )
]

# Train the model on the training data
print("Starting model training...")
history = model.fit(
    X_train, y_train,  # Training features and labels
    sample_weight=sw_train,  # Optional weights for each training sample
    validation_data=(X_test, y_test, sw_test),  # Validation data and sample weights
    epochs=EPOCHS,  # Total number of training epochs
    batch_size=BATCH_SIZE,  # Number of samples per batch
    callbacks=callbacks,  # Use the defined callbacks
    verbose=1  # Print progress bar during training
)

# Save the trained model and the scaler for future use
#model.save(MODEL_SAVE_PATH)  # Save the trained model to the given path
#joblib.dump(scaler, "scaler_improved.save")  # Save the scaler used during data preprocessing
#print(f"Model and scaler saved successfully to: {MODEL_SAVE_PATH}")

# Plot training history (Loss and Metrics over epochs)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 grid for plots

# Plot training vs validation loss
ax1.plot(history.history['loss'], label='Train Loss')  # Plot training loss
ax1.plot(history.history['val_loss'], label='Val Loss')  # Plot validation loss
ax1.set_title('Model Loss')  # Set plot title
ax1.set_xlabel('Epoch')  # X-axis label
ax1.set_ylabel('Loss')  # Y-axis label
ax1.legend()  # Show legend
ax1.grid(True)  # Show grid for better readability
