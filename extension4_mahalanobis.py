# =============================================================================
# EXTENSION 4: MAHALANOBIS DISTANCE WITH POOLED COVARIANCE
# =============================================================================
# Replace Euclidean distance in NCM with Mahalanobis distance.
# Mahalanobis distance accounts for feature correlations and scales.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("EXTENSION 4: MAHALANOBIS DISTANCE CLASSIFIER")
print("=" * 70)

# -----------------------------------------------------------------------------
# WHAT IS MAHALANOBIS DISTANCE?
# -----------------------------------------------------------------------------

explanation = """
EUCLIDEAN vs MAHALANOBIS DISTANCE:

EUCLIDEAN DISTANCE:
- Formula: d(x, μ) = sqrt((x - μ)^T (x - μ))
- Treats all features equally
- Assumes features are uncorrelated and have same scale
- A circle in 2D represents equal distances

MAHALANOBIS DISTANCE:
- Formula: d(x, μ) = sqrt((x - μ)^T Σ^{-1} (x - μ))
- Accounts for covariance structure
- Scales distances by feature variance
- Accounts for correlations between features
- An ellipse in 2D represents equal distances

WHY USE MAHALANOBIS?
- If one feature has high variance, Euclidean distance is dominated by it
- Mahalanobis "normalizes" by the covariance
- Better when features have different scales or are correlated
"""
print(explanation)

# -----------------------------------------------------------------------------
# IMPLEMENTATION: NCM WITH MAHALANOBIS DISTANCE
# -----------------------------------------------------------------------------

class MahalanobisNCM:
    """
    Nearest-Class-Mean classifier using Mahalanobis distance
    with a pooled covariance estimate.
    """
    def __init__(self):
        self.means = None
        self.cov_inv = None  # Inverse of pooled covariance
        self.classes = None

    def fit(self, X, y):
        """
        Fit the classifier: compute class means and pooled covariance.
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Compute class means
        self.means = {}
        for c in self.classes:
            self.means[c] = X[y == c].mean(axis=0)

        # Compute POOLED covariance matrix
        cov = np.zeros((n_features, n_features))
        for c in self.classes:
            X_c = X[y == c]
            X_centered = X_c - self.means[c]
            cov += X_centered.T @ X_centered

        cov /= (n_samples - len(self.classes))

        # Store inverse for efficient distance computation
        self.cov_inv = np.linalg.inv(cov)

        return self

    def mahalanobis_distance(self, X, mu):
        """
        Compute Mahalanobis distance from X to mu.

        d(x, μ) = sqrt((x - μ)^T Σ^{-1} (x - μ))
        """
        diff = X - mu
        # Efficient computation: sum of (diff @ cov_inv) * diff along axis 1
        return np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))

    def predict(self, X):
        """
        Predict class labels using nearest mean with Mahalanobis distance.
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, len(self.classes)))

        for i, c in enumerate(self.classes):
            distances[:, i] = self.mahalanobis_distance(X, self.means[c])

        return self.classes[np.argmin(distances, axis=1)]

# -----------------------------------------------------------------------------
# COMPARE: EUCLIDEAN NCM vs MAHALANOBIS NCM
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("COMPARISON: EUCLIDEAN NCM vs MAHALANOBIS NCM")
print("-" * 70)

# Prepare data
X_train_a = train_df[["x1", "x2"]].to_numpy()
y_train_a = train_df["label"].to_numpy()
X_test_a = test_df[["x1", "x2"]].to_numpy()
y_test_a = test_df["label"].to_numpy()

X_train_b = train_df_b[["x1", "x2"]].to_numpy()
y_train_b = train_df_b["label"].to_numpy()
X_test_b = test_df_b[["x1", "x2"]].to_numpy()
y_test_b = test_df_b["label"].to_numpy()

# Train Mahalanobis NCM
maha_ncm_a = MahalanobisNCM().fit(X_train_a, y_train_a)
maha_ncm_b = MahalanobisNCM().fit(X_train_b, y_train_b)

# Predictions
y_pred_maha_a = maha_ncm_a.predict(X_test_a)
y_pred_maha_b = maha_ncm_b.predict(X_test_b)

# Accuracies
acc_maha_a = (y_test_a == y_pred_maha_a).mean()
acc_maha_b = (y_test_b == y_pred_maha_b).mean()

print("\nDATASET A (Equal Covariance):")
print(f"  Euclidean NCM:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"  Mahalanobis NCM:  {acc_maha_a:.4f} ({acc_maha_a * 100:.2f}%)")

print("\nDATASET B (Unequal Covariance):")
print(f"  Euclidean NCM:    {accuracy_b:.4f} ({accuracy_b * 100:.2f}%)")
print(f"  Mahalanobis NCM:  {acc_maha_b:.4f} ({acc_maha_b * 100:.2f}%)")

# -----------------------------------------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY: ALL CLASSIFIERS")
print("=" * 70)
print(f"{'Classifier':<20} {'Dataset A':<12} {'Dataset B':<12}")
print("-" * 44)
print(f"{'Euclidean NCM':<20} {accuracy:<12.4f} {accuracy_b:<12.4f}")
print(f"{'Mahalanobis NCM':<20} {acc_maha_a:<12.4f} {acc_maha_b:<12.4f}")
print(f"{'LDA':<20} {acc_lda_a:<12.4f} {acc_lda_b:<12.4f}")
print(f"{'QDA':<20} {acc_qda_a:<12.4f} {acc_qda_b:<12.4f}")
print("-" * 44)

# -----------------------------------------------------------------------------
# VISUALISATION: EUCLIDEAN vs MAHALANOBIS CONTOURS
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Function to plot distance contours
def plot_distance_contours(ax, X, y, means, cov_inv, title, distance_type='mahalanobis'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Compute distances for each class
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if distance_type == 'mahalanobis':
        # Mahalanobis distance
        diff0 = grid_points - means[0]
        dist0 = np.sqrt(np.sum(diff0 @ cov_inv * diff0, axis=1)).reshape(xx.shape)
        diff1 = grid_points - means[1]
        dist1 = np.sqrt(np.sum(diff1 @ cov_inv * diff1, axis=1)).reshape(xx.shape)
    else:
        # Euclidean distance
        dist0 = np.sqrt(np.sum((grid_points - means[0]) ** 2, axis=1)).reshape(xx.shape)
        dist1 = np.sqrt(np.sum((grid_points - means[1]) ** 2, axis=1)).reshape(xx.shape)

    # Decision boundary: where distances are equal
    decision = dist0 - dist1

    # Plot
    ax.contour(xx, yy, dist0, levels=5, colors='blue', alpha=0.5, linestyles='--')
    ax.contour(xx, yy, dist1, levels=5, colors='red', alpha=0.5, linestyles='--')
    ax.contour(xx, yy, decision, levels=[0], colors='green', linewidths=2)

    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.5, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.5, label='Class 1')
    ax.scatter(means[0][0], means[0][1], c='blue', marker='X', s=200, edgecolor='black')
    ax.scatter(means[1][0], means[1][1], c='red', marker='X', s=200, edgecolor='black')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Dataset A - Euclidean
means_arr_a = np.array([means.loc[0].values, means.loc[1].values])
plot_distance_contours(axes[0, 0], X_test_a, y_test_a, means_arr_a, np.eye(2),
                       f'Dataset A - Euclidean NCM\nAcc: {accuracy:.2%}', 'euclidean')

# Dataset A - Mahalanobis
plot_distance_contours(axes[0, 1], X_test_a, y_test_a,
                       [maha_ncm_a.means[0], maha_ncm_a.means[1]], maha_ncm_a.cov_inv,
                       f'Dataset A - Mahalanobis NCM\nAcc: {acc_maha_a:.2%}', 'mahalanobis')

# Dataset B - Euclidean
means_arr_b = np.array([means_b.loc[0].values, means_b.loc[1].values])
plot_distance_contours(axes[1, 0], X_test_b, y_test_b, means_arr_b, np.eye(2),
                       f'Dataset B - Euclidean NCM\nAcc: {accuracy_b:.2%}', 'euclidean')

# Dataset B - Mahalanobis
plot_distance_contours(axes[1, 1], X_test_b, y_test_b,
                       [maha_ncm_b.means[0], maha_ncm_b.means[1]], maha_ncm_b.cov_inv,
                       f'Dataset B - Mahalanobis NCM\nAcc: {acc_maha_b:.2%}', 'mahalanobis')

plt.tight_layout()
plt.savefig("extension4_mahalanobis_comparison.png", dpi=150)
plt.show()

print("\nPlot saved to: extension4_mahalanobis_comparison.png")

# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

analysis = """
KEY FINDINGS:

1. MAHALANOBIS vs EUCLIDEAN:
   - Mahalanobis accounts for the covariance structure of the data
   - Euclidean treats all directions equally

2. EFFECT ON DECISION BOUNDARY:
   - Euclidean: Circular contours, straight perpendicular bisector boundary
   - Mahalanobis: Elliptical contours aligned with covariance
   - With POOLED covariance, both still produce LINEAR boundaries!

3. WHY POOLED COVARIANCE?
   - Uses average covariance across all classes
   - Assumes classes have similar covariance (like LDA)
   - More robust than per-class covariance with limited data

4. MAHALANOBIS NCM vs LDA:
   - Mathematically equivalent! Both use:
     * Class means
     * Pooled covariance
     * Linear decision boundary
   - LDA adds class priors (proportions)

5. WHEN MAHALANOBIS HELPS:
   - When features have very different scales
   - When features are correlated
   - Doesn't help with unequal covariance (still assumes shared covariance)

6. FOR UNEQUAL COVARIANCE:
   - Need per-class covariance (like QDA)
   - Or use a different distance metric altogether
"""
print(analysis)
