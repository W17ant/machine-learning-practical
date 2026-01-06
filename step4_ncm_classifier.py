# =============================================================================
# STEP 5: NEAREST-CLASS-MEAN (NCM) CLASSIFIER
# =============================================================================

# What is the Nearest-Class-Mean classifier?
# - A simple classification method based on distance to class centroids
# - For each test point, calculate distance to mu0 and distance to mu1
# - Assign the point to whichever class centroid is closer

# What is Euclidean distance?
# - The straight-line distance between two points
# - For 2D points: distance = sqrt((x1_a - x1_b)^2 + (x2_a - x2_b)^2)
# - Example: distance from (0,0) to (3,4) = sqrt(9 + 16) = sqrt(25) = 5

# Function: nearest_mean_predict
# Predicts class labels for test data using nearest class mean
#
# Parameters:
# - test_df: DataFrame containing test samples with x1, x2 columns
# - means: DataFrame containing mean vectors for each class
def nearest_mean_predict(test_df, means):
    # Get the centroid coordinates as a numpy array
    # Shape: (2, 2) - 2 classes, each with 2 coordinates (x1, x2)
    centers = means.values

    # Get the class labels (0 and 1) in the same order as centers
    labels = means.index.to_numpy()

    # Get test point coordinates as numpy array
    # Shape: (n_samples, 2) - each row is a point with x1, x2
    X = test_df[["x1", "x2"]].to_numpy()

    # Calculate squared Euclidean distance from each test point to each centroid
    # X[:, None, :] reshapes X to (n_samples, 1, 2) for broadcasting
    # centers[None, :, :] reshapes centers to (1, 2, 2) for broadcasting
    # Result shape: (n_samples, 2) - distance to each of the 2 centroids
    # Note: we use squared distance (no sqrt) because it preserves ordering
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    # For each test point, find which centroid is closest (index 0 or 1)
    pred_idx = np.argmin(dists, axis=1)

    # Convert centroid indices back to class labels
    return labels[pred_idx]

# Run the classifier on test data
y_pred = nearest_mean_predict(test_df, means)

# Display some predictions alongside actual labels
print("Sample predictions (first 10 test points):")
print(f"Predicted: {y_pred[:10]}")
print(f"Actual:    {test_df['label'].values[:10]}")
