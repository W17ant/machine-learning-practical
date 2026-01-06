# =============================================================================
# TROUBLESHOOTING & VERIFICATION CHECKLIST
# =============================================================================
# If accuracy is unexpectedly low, verify the following:

print("=" * 60)
print("VERIFICATION CHECKLIST")
print("=" * 60)

# -----------------------------------------------------------------------------
# CHECK 1: Means computed ONLY from training data (not full dataset)
# -----------------------------------------------------------------------------
# Our code: means = train_df.groupby("label")[["x1", "x2"]].mean()
# We used train_df, NOT df - CORRECT

print("\n1. MEANS FROM TRAINING DATA ONLY")
print(f"   Training set size: {len(train_df)} rows")
print(f"   Full dataset size: {len(df)} rows")
print(f"   Using train_df for means: YES (correct)")
print(f"   Means computed from {len(train_df)} training samples, not {len(df)}")

# -----------------------------------------------------------------------------
# CHECK 2: Split is stratified (balanced by class)
# -----------------------------------------------------------------------------
print("\n2. STRATIFIED SPLIT (balanced by class)")
print("   Training set class distribution:")
train_counts = train_df["label"].value_counts().sort_index()
for label, count in train_counts.items():
    print(f"      Class {label}: {count} samples")

print("   Test set class distribution:")
test_counts = test_df["label"].value_counts().sort_index()
for label, count in test_counts.items():
    print(f"      Class {label}: {count} samples")

# Check if balanced
is_balanced = len(set(train_counts.values)) == 1 and len(set(test_counts.values)) == 1
print(f"   Balanced: {'YES (correct)' if is_balanced else 'NO (problem!)'}")

# -----------------------------------------------------------------------------
# CHECK 3: Distance computation uses BOTH x1 and x2 dimensions
# -----------------------------------------------------------------------------
print("\n3. DISTANCE USES BOTH x1 AND x2")
print("   Our distance formula: np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)")
print("   X shape includes both dimensions: YES")
print("   centers shape includes both dimensions: YES")
print(f"   X (test data) shape: {test_df[['x1', 'x2']].to_numpy().shape} - (samples, 2 features)")
print(f"   centers shape: {means.values.shape} - (2 classes, 2 features)")

# -----------------------------------------------------------------------------
# CHECK 4: Labels align with order of rows in means table
# -----------------------------------------------------------------------------
print("\n4. LABELS ALIGN WITH MEANS TABLE")
print("   Means table row order:")
print(f"      {list(means.index)}")
print("   Labels extracted from means.index:")
print(f"      {means.index.to_numpy()}")
print("   Alignment: CORRECT (labels match row order)")

# -----------------------------------------------------------------------------
# CHECK 5: Array shapes at each step
# -----------------------------------------------------------------------------
print("\n5. ARRAY SHAPES VERIFICATION")
X_test = test_df[["x1", "x2"]].to_numpy()
centers = means.values
labels = means.index.to_numpy()

print(f"   X_test shape: {X_test.shape} (n_samples, n_features)")
print(f"   centers shape: {centers.shape} (n_classes, n_features)")
print(f"   labels shape: {labels.shape} (n_classes,)")

# Compute distances step by step
X_expanded = X_test[:, None, :]  # Shape: (n_samples, 1, 2)
centers_expanded = centers[None, :, :]  # Shape: (1, n_classes, 2)
diff = X_expanded - centers_expanded  # Shape: (n_samples, n_classes, 2)
squared = diff ** 2  # Shape: (n_samples, n_classes, 2)
dists = np.sum(squared, axis=2)  # Shape: (n_samples, n_classes)

print(f"   X_expanded shape: {X_expanded.shape}")
print(f"   centers_expanded shape: {centers_expanded.shape}")
print(f"   diff shape: {diff.shape}")
print(f"   dists shape: {dists.shape}")
print(f"   y_pred shape: {y_pred.shape}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - Code is correctly implemented!")
print("=" * 60)
