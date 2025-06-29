'''
  This code is designed to create Random Forest Regression model, which is trained on 70% of datset and remaining 30 % is divided equally into testing and validation
  Flowchart of code
  1. Create Random Forest
  2. Function  to train model
  3.Validation of model
  4. Plot importance feature
'''
from sklearn.ensemble import RandomForestRegressor  # Random Forest model for regression tasks
from sklearn.metrics import mean_squared_error, r2_score  # Metrics to evaluate model performance
import matplotlib.pyplot as plt  # For plotting graphs

# Create a Random Forest Regressor model with specified hyperparameters
rf = RandomForestRegressor(
    n_estimators=100,      # Number of trees in the forest
    max_depth=20,          # Maximum depth of each tree to control overfitting
    max_samples=0.5,       # Use 50% of the training samples for each tree (bagging)
    n_jobs=-1,             # Use all available CPU cores for faster training
    random_state=42        # Set random seed for reproducibility
)

print("\nTraining Random Forest...")  # Print a message before training starts
rf.fit(X_train, y_train)  # Fit (train) the model on the training data (features and target)

# Define a function to evaluate the model and plot results
def evaluate(X, y, title):
    y_pred = rf.predict(X)  # Predict target values (rainfall) using the trained model

    # Calculate Mean Squared Error (MSE) between true and predicted values
    mse = mean_squared_error(y, y_pred)

    # Calculate R-squared (R²) score - measures how well the model explains the variance in the data
    r2 = r2_score(y, y_pred)

    # Print the evaluation results (RMSE and R² score)
    print(f"{title} - MSE: {mse:.3f} | R²: {r2:.3f}")

# Evaluate model performance on Train, Validation, and Test datasets
train_mse = evaluate(X_train, y_train, "Train")  # Evaluate on Training data
val_mse = evaluate(X_val, y_val, "Validation")  # Evaluate on Validation data
test_mse = evaluate(X_test, y_test, "Test")  # Evaluate on Test data


# Plot Feature Importances from Random Forest
# Define feature names 
features = ['WV', 'TIR1', 'TIR2', 'TIR1-WV', 'TIR2-WV']

# Get the calculated feature importances from the trained Random Forest model
importances = rf.feature_importances_

# Create a bar plot for feature importances
plt.figure(figsize=(6, 4))  # Set figure size
plt.bar(features, importances)  # Draw bar chart (x=feature names, y=importances)
plt.title("Random Forest Feature Importance")  # Title of the plot
plt.ylabel("Importance")  # Y-axis label
plt.grid(True)  # Add grid lines for better readability
plt.tight_layout()  # Adjust layout
plt.show()  # Display the plot
