# =============================================================================
# STEP 4: COMPUTE CLASS MEAN VECTORS
# =============================================================================

# What are class mean vectors?
# - The "mean" (average) position of all points belonging to each class
# - For class 0: average of all x1 values, average of all x2 values = (mu0_x1, mu0_x2)
# - For class 1: average of all x1 values, average of all x2 values = (mu1_x1, mu1_x2)
# - These are the "centroids" (center points) of each class

# Why use only training data?
# - Test data must remain unseen during model training
# - Using test data would be "cheating" - the model would have seen the answers
# - This ensures a fair evaluation of the model's performance

# Calculate the mean of x1 and x2 for each class (0 and 1) in training data
# .groupby("label") - separates data by class
# [["x1","x2"]] - selects only the feature columns
# .mean() - calculates the average of each column per group
means = train_df.groupby("label")[["x1", "x2"]].mean()

# Display the class means
print("Class mean vectors (centroids):")
print(means)

# Extract individual mean vectors for clarity
mu0 = means.loc[0].values  # Mean vector for class 0: [x1_mean, x2_mean]
mu1 = means.loc[1].values  # Mean vector for class 1: [x1_mean, x2_mean]

print(f"\nmu0 (class 0 centroid): x1={mu0[0]:.4f}, x2={mu0[1]:.4f}")
print(f"mu1 (class 1 centroid): x1={mu1[0]:.4f}, x2={mu1[1]:.4f}")
