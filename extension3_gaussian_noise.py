# =============================================================================
# EXTENSION 3: GAUSSIAN NOISE FEATURES - ROBUSTNESS STUDY
# =============================================================================
# Add random noise features to the data and measure how classifiers perform.
# This tests robustness to irrelevant/uninformative features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("EXTENSION 3: ROBUSTNESS TO GAUSSIAN NOISE FEATURES")
print("=" * 70)

# -----------------------------------------------------------------------------
# ADD NOISE FEATURES TO DATA
# -----------------------------------------------------------------------------

def add_noise_features(df, n_noise_features, seed=42):
    """
    Add random Gaussian noise features to a dataset.
    These features have NO predictive power - they're pure noise.

    Parameters:
    - df: original DataFrame with x1, x2, label
    - n_noise_features: number of noise features to add
    - seed: random seed

    Returns:
    - df_noisy: DataFrame with original features + noise features
    """
    rng = np.random.default_rng(seed)
    df_noisy = df.copy()

    # Add noise features with random Gaussian values
    for i in range(n_noise_features):
        # Noise has mean=0, std=1 (no relationship to class)
        df_noisy[f'noise_{i+1}'] = rng.normal(0, 1, len(df))

    return df_noisy

# -----------------------------------------------------------------------------
# TEST NCM AND QDA WITH INCREASING NOISE FEATURES
# -----------------------------------------------------------------------------

noise_counts = [0, 1, 2, 5, 10, 20, 50, 100]
results_a = {'ncm': [], 'qda': []}
results_b = {'ncm': [], 'qda': []}

print("\nTesting with increasing noise features...")
print("-" * 60)

for n_noise in noise_counts:
    # --- Dataset A ---
    train_noisy_a = add_noise_features(train_df, n_noise)
    test_noisy_a = add_noise_features(test_df, n_noise)

    feature_cols = ['x1', 'x2'] + [f'noise_{i+1}' for i in range(n_noise)]

    X_train_a = train_noisy_a[feature_cols].to_numpy()
    y_train_a = train_noisy_a["label"].to_numpy()
    X_test_a = test_noisy_a[feature_cols].to_numpy()
    y_test_a = test_noisy_a["label"].to_numpy()

    # NCM with noise features
    means_noisy_a = train_noisy_a.groupby("label")[feature_cols].mean()
    centers_a = means_noisy_a.values
    labels_a = means_noisy_a.index.to_numpy()
    dists_a = np.sum((X_test_a[:, None, :] - centers_a[None, :, :]) ** 2, axis=2)
    y_pred_ncm_a = labels_a[np.argmin(dists_a, axis=1)]
    acc_ncm_a = (y_test_a == y_pred_ncm_a).mean()
    results_a['ncm'].append(acc_ncm_a)

    # QDA with noise features
    try:
        qda_noisy_a = QDAClassifier().fit(X_train_a, y_train_a)
        y_pred_qda_a = qda_noisy_a.predict(X_test_a)
        acc_qda_a = (y_test_a == y_pred_qda_a).mean()
    except:
        acc_qda_a = np.nan  # May fail with too many features
    results_a['qda'].append(acc_qda_a)

    # --- Dataset B ---
    train_noisy_b = add_noise_features(train_df_b, n_noise)
    test_noisy_b = add_noise_features(test_df_b, n_noise)

    X_train_b = train_noisy_b[feature_cols].to_numpy()
    y_train_b = train_noisy_b["label"].to_numpy()
    X_test_b = test_noisy_b[feature_cols].to_numpy()
    y_test_b = test_noisy_b["label"].to_numpy()

    # NCM with noise features
    means_noisy_b = train_noisy_b.groupby("label")[feature_cols].mean()
    centers_b = means_noisy_b.values
    labels_b = means_noisy_b.index.to_numpy()
    dists_b = np.sum((X_test_b[:, None, :] - centers_b[None, :, :]) ** 2, axis=2)
    y_pred_ncm_b = labels_b[np.argmin(dists_b, axis=1)]
    acc_ncm_b = (y_test_b == y_pred_ncm_b).mean()
    results_b['ncm'].append(acc_ncm_b)

    # QDA with noise features
    try:
        qda_noisy_b = QDAClassifier().fit(X_train_b, y_train_b)
        y_pred_qda_b = qda_noisy_b.predict(X_test_b)
        acc_qda_b = (y_test_b == y_pred_qda_b).mean()
    except:
        acc_qda_b = np.nan
    results_b['qda'].append(acc_qda_b)

    print(f"Noise features: {n_noise:3d} | Dataset A: NCM={acc_ncm_a:.4f}, QDA={acc_qda_a:.4f} | "
          f"Dataset B: NCM={acc_ncm_b:.4f}, QDA={acc_qda_b:.4f}")

# -----------------------------------------------------------------------------
# PLOT: ACCURACY vs NUMBER OF NOISE FEATURES
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dataset A
ax1 = axes[0]
ax1.plot(noise_counts, results_a['ncm'], 'b-o', linewidth=2, markersize=8, label='NCM')
ax1.plot(noise_counts, results_a['qda'], 'r-s', linewidth=2, markersize=8, label='QDA')
ax1.axhline(y=accuracy, color='blue', linestyle='--', alpha=0.5, label='NCM baseline (no noise)')
ax1.set_xlabel('Number of Noise Features', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Dataset A: Robustness to Noise Features', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.4, 1.05)

# Dataset B
ax2 = axes[1]
ax2.plot(noise_counts, results_b['ncm'], 'b-o', linewidth=2, markersize=8, label='NCM')
ax2.plot(noise_counts, results_b['qda'], 'r-s', linewidth=2, markersize=8, label='QDA')
ax2.axhline(y=accuracy_b, color='blue', linestyle='--', alpha=0.5, label='NCM baseline (no noise)')
ax2.set_xlabel('Number of Noise Features', fontsize=12)
ax2.set_ylabel('Test Accuracy', fontsize=12)
ax2.set_title('Dataset B: Robustness to Noise Features', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.4, 1.05)

plt.tight_layout()
plt.savefig("extension3_noise_robustness.png", dpi=150)
plt.show()

print("\nPlot saved to: extension3_noise_robustness.png")

# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

analysis = """
KEY FINDINGS:

1. EFFECT OF NOISE FEATURES:
   - Noise features add irrelevant dimensions to the distance calculation
   - They increase the "noise" in the Euclidean distance
   - As noise increases, the signal (x1, x2) gets drowned out

2. NCM ROBUSTNESS:
   - NCM uses simple Euclidean distance
   - Each noise feature adds random variation to distances
   - With many noise features, NCM accuracy degrades

3. QDA ROBUSTNESS:
   - QDA estimates covariance matrices that include noise features
   - With many features, covariance estimation becomes unreliable
   - QDA may fail or become unstable with high-dimensional noise

4. THE CURSE OF DIMENSIONALITY:
   - Both classifiers suffer as dimensions increase
   - More features = more parameters to estimate
   - With limited training data, estimates become poor

5. PRACTICAL IMPLICATIONS:
   - Feature selection is important! Remove irrelevant features
   - Dimensionality reduction (PCA) can help
   - Simple classifiers may be more robust to noise than complex ones
"""
print(analysis)
