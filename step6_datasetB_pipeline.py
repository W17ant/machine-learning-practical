# =============================================================================
# DATASET B - COMPLETE NCM CLASSIFICATION PIPELINE
# =============================================================================
# This repeats the same process as Dataset A, but for Dataset B
# Dataset B has "unequal covariance" - the spread of data differs between classes

import pandas as pd
import numpy as np

# =============================================================================
# STEP 1: LOAD DATASET B
# =============================================================================

# Load Dataset B (unequal covariance)
df_b = pd.read_csv("datasetB.csv")

# Verify the dataset
print("DATASET B - LOADING")
print(f"Number of rows: {df_b.shape[0]}")
print(f"Column names: {list(df_b.columns)}")
print(df_b.head())

# =============================================================================
# STEP 2: STRATIFIED 80/20 SPLIT
# =============================================================================

def stratified_train_test_split(df, test_frac=0.2, random_state=42):
    # Set random seed so results are reproducible
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []

    # Split each class proportionally
    for cls, group in df.groupby("label"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_test = int(len(idx) * test_frac)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.loc[sorted(test_idx)].reset_index(drop=True)
    return train_df, test_df

# Perform the split
train_df_b, test_df_b = stratified_train_test_split(df_b, test_frac=0.2)

print("\nDATASET B - SPLIT VERIFICATION")
print(f"Training set size: {len(train_df_b)} rows")
print(f"Test set size: {len(test_df_b)} rows")

# =============================================================================
# STEP 3: COMPUTE CLASS MEANS
# =============================================================================

# Calculate mean vectors from training data only
means_b = train_df_b.groupby("label")[["x1", "x2"]].mean()

print("\nDATASET B - CLASS MEANS")
print(means_b)

# =============================================================================
# STEP 4: NCM CLASSIFIER
# =============================================================================

def nearest_mean_predict(test_df, means):
    centers = means.values
    labels = means.index.to_numpy()
    X = test_df[["x1", "x2"]].to_numpy()

    # Calculate squared Euclidean distance to each centroid
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    # Predict class with nearest centroid
    pred_idx = np.argmin(dists, axis=1)
    return labels[pred_idx]

# Make predictions
y_pred_b = nearest_mean_predict(test_df_b, means_b)

# =============================================================================
# STEP 5: EVALUATE ACCURACY
# =============================================================================

y_true_b = test_df_b["label"].to_numpy()
accuracy_b = (y_true_b == y_pred_b).mean()

print("\n" + "=" * 50)
print("DATASET B - NCM CLASSIFIER RESULTS")
print("=" * 50)
print(f"Total test samples: {len(y_true_b)}")
print(f"Correct predictions: {(y_true_b == y_pred_b).sum()}")
print(f"Incorrect predictions: {(y_true_b != y_pred_b).sum()}")
print(f"Test Accuracy: {accuracy_b:.4f} ({accuracy_b * 100:.2f}%)")
print("=" * 50)

# Save predictions
results_df_b = test_df_b.copy()
results_df_b["predicted"] = y_pred_b
results_df_b["correct"] = (y_true_b == y_pred_b)
results_df_b.to_csv("datasetB_predictions.csv", index=False)
print("\nPredictions saved to: datasetB_predictions.csv")
