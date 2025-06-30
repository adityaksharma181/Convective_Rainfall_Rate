'''
  This code is designed to train ANN model using custom L1 and L2 regularisation with early stop fnuctionality
  Flowchart of code
  1. Hyperparameters
  2. Define Model
  3. Training and Validation of Model
'''
# Import necessary libraries
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

# Hyperparameters for better model performance
LEARNING_RATE = 0.0001                  # Learning rate for optimizer
EPOCHS = 100                            # Number of training epochs
BATCH_SIZE = 4000                       # Number of samples per training batch
DROPOUT_RATE = 0.3                      # Dropout rate to prevent overfitting
L1_REG = 1e-6                           # L1 regularization strength
L2_REG = 1e-5                           # L2 regularization strength

# Build the improved Artificial Neural Network (ANN) model
model = Sequential([  # Start defining sequential model

    Dense(128, input_dim=input_shape, activation='relu',  # First dense layer with ReLU activation and L1/L2 regularization
          kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)),
    BatchNormalization(),  # Apply batch normalization
    Dropout(DROPOUT_RATE),  # Apply dropout to reduce overfitting
    
    Dense(64, activation='relu',  # Second dense layer
          kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG)),
    BatchNormalization(),  # Apply batch normalization
    Dropout(DROPOUT_RATE/2),  # Apply smaller dropout
    
    Dense(32, activation='relu'),  # Third dense layer without regularization
    Dropout(DROPOUT_RATE/2),  # Apply dropout again

    Dense(1, activation='linear')  # Output layer for regression (rainfall prediction)
])

# Compile the model 
model.compile(  # Compile the model with optimizer, loss and metrics
    optimizer=Adam(learning_rate=LEARNING_RATE),  # Adam optimizer with defined learning rate
    loss=custom_loss,  # Custom loss function
    metrics=[
        tf.keras.metrics.MeanSquaredError(name="mse")  # Mean Squared Error as evaluation metric
    ]
)

# Model summary
model.summary()  # Print model architecture summary

# Define callbacks
callbacks = [  # Create list of callbacks for training control
    EarlyStopping(  # Stop training early if val_loss doesn't improve
        monitor='val_loss',  # Monitor validation loss
        patience=20,  # Stop if no improvement for 20 epochs
        restore_best_weights=True,  # Restore weights from best epoch
        verbose=1  # Verbosity for logging
    ),
    ReduceLROnPlateau(  # Reduce learning rate when val_loss plateaus
        monitor='val_loss',  # Monitor validation loss
        factor=0.5,  # Reduce LR by factor of 0.5
        patience=10,  # Wait 10 epochs before reducing LR
        min_lr=1e-6,  # Minimum learning rate
        verbose=1  # Verbosity for logging
    )
]

# Train the model
print("Starting model training...")  # Print start message
history = model.fit(  # Train the model and store training history
    X_train, y_train,  # Training features and labels
    sample_weight=sw_train,  # Sample weights for training
    validation_data=(X_test, y_test, sw_test),  # Validation data and sample weights
    epochs=EPOCHS,  # Number of epochs
    batch_size=BATCH_SIZE,  # Batch size
    callbacks=callbacks,  # Callbacks
    verbose=1  # Verbose output
)

# ---- Calculate and Print MSE for Train and Test ----
train_mse = model.evaluate(X_train, y_train, sample_weight=sw_train, verbose=0)[1]  # Evaluate MSE on training data
test_mse = model.evaluate(X_test, y_test, sample_weight=sw_test, verbose=0)[1]  # Evaluate MSE on test data

print("\nFinal Mean Squared Error (MSE) Results:")  # Print header
print(f"Training MSE: {train_mse:.6f}")  # Print training MSE
print(f"Testing MSE: {test_mse:.6f}")  # Print testing MSE
# If you have a separate validation set, uncomment and modify below:
val_mse = model.evaluate(X_val, y_val, sample_weight=sw_val, verbose=0)[1]  # Evaluate MSE on validation data
print(f"Validation MSE: {val_mse:.6f}")  # Print validation MSE

# ---- Only MSE and Loss curves ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Create two side-by-side plots

# Plot Loss (Training vs Validation)
ax1.plot(history.history['loss'], label='Train Loss')  # Plot training loss
ax1.plot(history.history['val_loss'], label='Val Loss')  # Plot validation loss
ax1.set_title('Model Loss')  # Set plot title
ax1.set_xlabel('Epoch')  # X-axis label
ax1.set_ylabel('Loss')  # Y-axis label
ax1.legend()  # Show legend
ax1.grid(True)  # Show grid

# Plot MSE (Training vs Validation)
ax2.plot(history.history['mse'], label='Train MSE')  # Plot training MSE
ax2.plot(history.history['val_mse'], label='Val MSE')  # Plot validation MSE
ax2.set_title('Mean Squared Error (MSE)')  # Set plot title
ax2.set_xlabel('Epoch')  # X-axis label
ax2.set_ylabel('MSE')  # Y-axis label
ax2.legend()  # Show legend
ax2.grid(True)  # Show grid

plt.tight_layout()  # Adjust layout
plt.show()  # Display plots

# ---- Optional: Save model and scaler ----
model.save(MODEL_SAVE_PATH)  # Save trained model
joblib.dump(scaler, "scaler_improved.save")  # Save scaler used during training
print(f"Model and scaler saved successfully to: {MODEL_SAVE_PATH}")  # Confirmation message

ax1.set_title('Model Loss')  # Set plot title
ax1.set_xlabel('Epoch')  # X-axis label
ax1.set_ylabel('Loss')  # Y-axis label
ax1.legend()  # Show legend
ax1.grid(True)  # Show grid for better readability
