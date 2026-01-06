# =============================================================================
# EXTENSION 2: VARY COVARIANCES AND PLOT ACCURACY VS SCATTER
# =============================================================================
# Generate synthetic datasets with varying covariance ratios
# and measure how NCM accuracy degrades as covariances become more unequal.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("EXTENSION 2: ACCURACY vs COVARIANCE RATIO")
print("=" * 70)

# -----------------------------------------------------------------------------
# GENERATE SYNTHETIC DATA WITH VARYING COVARIANCE
# -----------------------------------------------------------------------------

def generate_dataset(n_samples=500, cov_ratio=1.0, seed=42):
    """
    Generate a 2-class dataset where class 1 has cov_ratio times the variance of class 0.

    Parameters:
    - n_samples: samples per class
    - cov_ratio: ratio of class 1 variance to class 0 variance (1.0 = equal)
    - seed: random seed for reproducibility

    Returns:
    - df: DataFrame with x1, x2, label columns
    """
    rng = np.random.default_rng(seed)

    # Class 0: fixed covariance
    cov0 = [[0.5, 0], [0, 0.5]]
    mean0 = [2, 3]

    # Class 1: covariance scaled by cov_ratio
    cov1 = [[0.5 * cov_ratio, 0], [0, 0.5 * cov_ratio]]
    mean1 = [6, 7]

    # Generate samples
    X0 = rng.multivariate_normal(mean0, cov0, n_samples)
    X1 = rng.multivariate_normal(mean1, cov1, n_samples)

    # Combine into DataFrame
    df = pd.DataFrame({
        'x1': np.concatenate([X0[:, 0], X1[:, 0]]),
        'x2': np.concatenate([X0[:, 1], X1[:, 1]]),
        'label': np.concatenate([np.zeros(n_samples), np.ones(n_samples)]).astype(int)
    })

    return df

# -----------------------------------------------------------------------------
# TEST NCM ACROSS DIFFERENT COVARIANCE RATIOS
# -----------------------------------------------------------------------------

# Covariance ratios to test (1 = equal, >1 = class 1 more spread)
cov_ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
accuracies_ncm = []
accuracies_qda = []

print("\nTesting NCM and QDA across covariance ratios...")
print("-" * 50)

for ratio in cov_ratios:
    # Generate dataset with this covariance ratio
    df_syn = generate_dataset(n_samples=500, cov_ratio=ratio, seed=42)

    # Split
    train_syn, test_syn = stratified_train_test_split(df_syn, test_frac=0.2)

    # Prepare arrays
    X_train_syn = train_syn[["x1", "x2"]].to_numpy()
    y_train_syn = train_syn["label"].to_numpy()
    X_test_syn = test_syn[["x1", "x2"]].to_numpy()
    y_test_syn = test_syn["label"].to_numpy()

    # NCM
    means_syn = train_syn.groupby("label")[["x1", "x2"]].mean()
    y_pred_syn = nearest_mean_predict(test_syn, means_syn)
    acc_ncm = (y_test_syn == y_pred_syn).mean()
    accuracies_ncm.append(acc_ncm)

    # QDA
    qda_syn = QDAClassifier().fit(X_train_syn, y_train_syn)
    y_pred_qda_syn = qda_syn.predict(X_test_syn)
    acc_qda = (y_test_syn == y_pred_qda_syn).mean()
    accuracies_qda.append(acc_qda)

    print(f"Cov ratio {ratio:5.2f}: NCM = {acc_ncm:.4f}, QDA = {acc_qda:.4f}")

# -----------------------------------------------------------------------------
# PLOT: ACCURACY vs COVARIANCE RATIO
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(cov_ratios, accuracies_ncm, 'b-o', linewidth=2, markersize=8, label='NCM (linear)')
ax.plot(cov_ratios, accuracies_qda, 'r-s', linewidth=2, markersize=8, label='QDA (quadratic)')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect accuracy')
ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='Equal covariance')

ax.set_xscale('log', base=2)
ax.set_xlabel('Covariance Ratio (Class 1 / Class 0)', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('NCM vs QDA Accuracy as Covariance Ratio Varies', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.05)

# Add annotations
ax.annotate('Equal covariance\n(NCM optimal)', xy=(1.0, 0.95), fontsize=9, ha='center')

plt.tight_layout()
plt.savefig("extension2_accuracy_vs_covariance.png", dpi=150)
plt.show()

print("\nPlot saved to: extension2_accuracy_vs_covariance.png")

# -----------------------------------------------------------------------------
# VISUALISE DATASETS AT DIFFERENT RATIOS
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
sample_ratios = [0.25, 1.0, 4.0, 16.0]

for ax, ratio in zip(axes, sample_ratios):
    df_vis = generate_dataset(n_samples=200, cov_ratio=ratio, seed=42)

    ax.scatter(df_vis[df_vis['label'] == 0]['x1'],
               df_vis[df_vis['label'] == 0]['x2'],
               c='blue', alpha=0.5, label='Class 0')
    ax.scatter(df_vis[df_vis['label'] == 1]['x1'],
               df_vis[df_vis['label'] == 1]['x2'],
               c='red', alpha=0.5, label='Class 1')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Cov Ratio: {ratio}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("extension2_varying_covariance_scatter.png", dpi=150)
plt.show()

print("Plot saved to: extension2_varying_covariance_scatter.png")

# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

analysis = """
KEY FINDINGS:

1. When covariance ratio = 1.0 (equal):
   - Both NCM and QDA achieve similar high accuracy
   - Linear boundary is optimal

2. As covariance ratio increases (class 1 spreads out):
   - NCM accuracy may decrease as the linear boundary becomes suboptimal
   - QDA maintains high accuracy by adapting its quadratic boundary

3. As covariance ratio decreases (class 1 becomes tighter):
   - Similar pattern - NCM struggles, QDA adapts

4. CONCLUSION:
   - NCM is best when covariances are roughly equal
   - QDA is more robust to varying covariance structures
   - The gap between NCM and QDA grows as covariance imbalance increases
"""
print(analysis)
