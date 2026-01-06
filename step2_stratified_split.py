# =============================================================================
# STEP 3: STRATIFIED 80/20 TRAIN/TEST SPLIT
# =============================================================================

# Import numpy for random number generation
import numpy as np

# What is a stratified split?
# - "Stratified" means each class (0 and 1) is split proportionally
# - If class 0 has 500 samples, 80% (400) go to train, 20% (100) go to test
# - This ensures both train and test sets have balanced class representation

# Why stratified?
# - Prevents all samples of one class ending up in train or test by chance
# - Gives a fair evaluation of model performance on both classes

# Function: stratified_train_test_split
# Split data into train and test sets, keeping class proportions equal.
#
# Parameters:
# - df: the DataFrame to split
# - test_frac: fraction of data for testing (0.2 = 20%)
# - random_state: seed for reproducibility (same split every time)
def stratified_train_test_split(df, test_frac=0.2, random_state=42):
    # Set random seed so results are reproducible
    rng = np.random.default_rng(random_state)

    # Lists to store indices for train and test sets
    train_idx, test_idx = [], []

    # Loop through each class (label 0 and label 1) separately
    for cls, group in df.groupby("label"):
        # Get all row indices for this class
        idx = group.index.to_numpy()

        # Shuffle the indices randomly
        rng.shuffle(idx)

        # Calculate how many samples go to test set (20%)
        n_test = int(len(idx) * test_frac)

        # First n_test indices go to test, the rest go to train
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    # Create the train and test DataFrames using the collected indices
    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    test_df = df.loc[sorted(test_idx)].reset_index(drop=True)

    return train_df, test_df

# Perform the split: 80% train, 20% test
train_df, test_df = stratified_train_test_split(df, test_frac=0.2)

# =============================================================================
# VERIFY THE SPLIT
# =============================================================================

# Check the sizes of each set
print(f"Training set size: {len(train_df)} rows (80%)")
print(f"Test set size: {len(test_df)} rows (20%)")

# Verify stratification - check class distribution in each set
print(f"\nTraining set class distribution:")
print(train_df["label"].value_counts().sort_index())

print(f"\nTest set class distribution:")
print(test_df["label"].value_counts().sort_index())
