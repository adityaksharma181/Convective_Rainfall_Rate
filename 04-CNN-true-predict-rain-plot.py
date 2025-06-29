'''
  This code is designed to plot true rain and predicted rainfall
  Flowchat of code
  1.
'''
import numpy as np               # For numerical operations like array manipulation
import matplotlib.pyplot as plt  # For plotting and visualization

# Visualize a sample prediction from the model
sample_idx = 0  # Select the index of the sample to visualize ( the first sample in the test set)

# Get the true precipitation grid (ground truth) for the selected sample from the test labels (y_test)
true_grid = y_test[sample_idx]  

# Predict the precipitation grid for the selected sample using the trained model
# X_test[sample_idx:sample_idx+1] selects one sample from the test features
# .predict() generates the model's predicted output for that sample
# .reshape(true_grid.shape) reshapes the prediction to match the shape of the true grid for easy comparison
pred_grid = model.predict(X_test[sample_idx:sample_idx+1]).reshape(true_grid.shape)  


# Plot the True Precipitation Grid
plt.figure(figsize=(8,6))  

# Create a new figure with specified size (8 inches wide by 6 inches tall)
plt.imshow(
    true_grid,                 # The 2D true precipitation data to display
    origin='lower',            # Place the origin (0,0) of the grid at the bottom left corner
    cmap='viridis',            # Use the 'viridis' color map for coloring values
    # Define the extent (coordinates range) of the plot: x-axis (longitude) and y-axis (latitude)
    extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip]  
)

# Set the title of the plot 
plt.title('True Precipitation')  

# Add a colorbar to show the scale (precipitation intensity in millimeters per hour)
plt.colorbar(label='mm/hr')  

# Display the plot on the screen
plt.show()  

# Plot the Predicted Precipitation Grid
plt.figure(figsize=(8,6))  
# Create another new figure with the same size for the predicted data plot

plt.imshow(
    pred_grid,                 # The 2D predicted precipitation data from the model
    origin='lower',            # Again, put the origin at the bottom left for consistency
    cmap='viridis',            # Use the same colormap for consistency in color scaling
    # Use the same geographic extent as the true grid for direct visual comparison
    extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip]  
)

# Title showing model's predicted precipitation output
plt.title('Predicted Precipitation (Weighted CNN)')  
plt.colorbar(label='mm/hr')  
plt.show()  

