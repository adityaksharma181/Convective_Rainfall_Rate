'''
  This code is designed to plot true rain 
'''
import numpy as np               # For numerical operations like array manipulation
import matplotlib.pyplot as plt  # For plotting and visualization

# Visualize a sample prediction from the model

sample_idx = 0  # Select the index of the sample to visualize (here, the first sample in the test set)

true_grid = y_test[sample_idx]  
# Get the true precipitation grid (ground truth) for the selected sample from the test labels (y_test)

pred_grid = model.predict(X_test[sample_idx:sample_idx+1]).reshape(true_grid.shape)  
# Predict the precipitation grid for the selected sample using the trained model
# X_test[sample_idx:sample_idx+1] selects one sample from the test features
# .predict() generates the model's predicted output for that sample
# .reshape(true_grid.shape) reshapes the prediction to match the shape of the true grid for easy comparison

# ------------------------ Plot the True Precipitation Grid ------------------------

plt.figure(figsize=(8,6))  
# Create a new figure with specified size (8 inches wide by 6 inches tall)

plt.imshow(
    true_grid,                 # The 2D true precipitation data to display
    origin='lower',            # Place the origin (0,0) of the grid at the bottom left corner
    cmap='viridis',            # Use the 'viridis' color map for coloring values
    extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip]  
    # Define the extent (coordinates range) of the plot: x-axis (longitude) and y-axis (latitude)
)

plt.title('True Precipitation')  
# Set the title of the plot to describe that this shows the true precipitation values

plt.colorbar(label='mm/hr')  
# Add a colorbar to show the scale (precipitation intensity in millimeters per hour)

plt.show()  
# Display the plot on the screen

# ------------------------ Plot the Predicted Precipitation Grid ------------------------

plt.figure(figsize=(8,6))  
# Create another new figure with the same size for the predicted data plot

plt.imshow(
    pred_grid,                 # The 2D predicted precipitation data from the model
    origin='lower',            # Again, put the origin at the bottom left for consistency
    cmap='viridis',            # Use the same colormap for consistency in color scaling
    extent=[lon_min_clip, lon_max_clip, lat_min_clip, lat_max_clip]  
    # Use the same geographic extent as the true grid for direct visual comparison
)

plt.title('Predicted Precipitation (Weighted CNN)')  
# Title showing this is the model's predicted precipitation output

plt.colorbar(label='mm/hr')  
# Add a colorbar to indicate the precipitation scale (same units as before)

plt.show()  
# Display this second plot on the screen
