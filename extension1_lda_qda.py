# =============================================================================
# EXTENSION 1: LDA & QDA COMPARISON WITH NCM
# =============================================================================
# Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
# are classical classifiers that model class distributions as Gaussians.

# LDA: Assumes ALL classes share the SAME covariance matrix (like NCM)
#      -> Produces LINEAR decision boundaries
#
# QDA: Each class has its OWN covariance matrix
#      -> Produces QUADRATIC decision boundaries (curves, ellipses)

# =============================================================================
# IMPLEMENTATION FROM SCRATCH (no sklearn)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# LDA IMPLEMENTATION
# -----------------------------------------------------------------------------
# LDA models each class as a Gaussian with:
# - Its own mean vector (μ_k)
# - A SHARED covariance matrix (Σ) pooled across all classes

class LDAClassifier:
    """
    Linear Discriminant Analysis classifier.
    Assumes equal covariance matrices across classes.
    """
    def __init__(self):
        self.means = None      # Mean vector for each class
        self.cov = None        # Shared (pooled) covariance matrix
        self.priors = None     # Prior probability of each class
        self.classes = None    # Unique class labels

    def fit(self, X, y):
        """
        Fit the LDA model to training data.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,) with class labels
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Calculate mean for each class
        self.means = {}
        for c in self.classes:
            self.means[c] = X[y == c].mean(axis=0)

        # Calculate POOLED covariance matrix (shared across classes)
        # Pooled covariance = weighted average of within-class covariances
        self.cov = np.zeros((n_features, n_features))
        for c in self.classes:
            X_c = X[y == c]
            # Center the data for this class
            X_centered = X_c - self.means[c]
            # Add contribution to pooled covariance
            self.cov += X_centered.T @ X_centered

        # Divide by (n_samples - n_classes) for unbiased estimate
        self.cov /= (n_samples - len(self.classes))

        # Calculate prior probabilities (proportion of each class)
        self.priors = {}
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / n_samples

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Uses discriminant function: δ_k(x) = x^T Σ^{-1} μ_k - 0.5 μ_k^T Σ^{-1} μ_k + log(π_k)
        """
        # Compute inverse of covariance matrix
        cov_inv = np.linalg.inv(self.cov)

        # Calculate discriminant score for each class
        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            mu = self.means[c]
            # Linear discriminant function
            scores[:, i] = (X @ cov_inv @ mu
                          - 0.5 * mu @ cov_inv @ mu
                          + np.log(self.priors[c]))

        # Predict class with highest discriminant score
        return self.classes[np.argmax(scores, axis=1)]

# -----------------------------------------------------------------------------
# QDA IMPLEMENTATION
# -----------------------------------------------------------------------------
# QDA models each class as a Gaussian with:
# - Its own mean vector (μ_k)
# - Its OWN covariance matrix (Σ_k) - different for each class!

class QDAClassifier:
    """
    Quadratic Discriminant Analysis classifier.
    Each class has its own covariance matrix.
    """
    def __init__(self):
        self.means = None      # Mean vector for each class
        self.covs = None       # Covariance matrix for EACH class
        self.priors = None     # Prior probability of each class
        self.classes = None    # Unique class labels

    def fit(self, X, y):
        """
        Fit the QDA model to training data.
        """
        self.classes = np.unique(y)
        n_samples = X.shape[0]

        # Calculate mean and covariance for EACH class separately
        self.means = {}
        self.covs = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)

            # Calculate covariance for THIS class only
            X_centered = X_c - self.means[c]
            self.covs[c] = (X_centered.T @ X_centered) / (len(X_c) - 1)

            # Prior probability
            self.priors[c] = len(X_c) / n_samples

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Uses quadratic discriminant function.
        """
        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            mu = self.means[c]
            cov = self.covs[c]
            cov_inv = np.linalg.inv(cov)

            # Quadratic discriminant function
            # δ_k(x) = -0.5 log|Σ_k| - 0.5 (x-μ_k)^T Σ_k^{-1} (x-μ_k) + log(π_k)
            diff = X - mu

            # Mahalanobis distance squared: (x-μ)^T Σ^{-1} (x-μ)
            mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)

            # Log determinant of covariance
            log_det = np.log(np.linalg.det(cov))

            scores[:, i] = -0.5 * log_det - 0.5 * mahal_sq + np.log(self.priors[c])

        return self.classes[np.argmax(scores, axis=1)]

# =============================================================================
# COMPARISON: NCM vs LDA vs QDA
# =============================================================================

print("=" * 70)
print("EXTENSION 1: COMPARING NCM, LDA, AND QDA CLASSIFIERS")
print("=" * 70)

# Prepare data as numpy arrays
X_train = train_df[["x1", "x2"]].to_numpy()
y_train = train_df["label"].to_numpy()
X_test = test_df[["x1", "x2"]].to_numpy()
y_test = test_df["label"].to_numpy()

X_train_b = train_df_b[["x1", "x2"]].to_numpy()
y_train_b = train_df_b["label"].to_numpy()
X_test_b = test_df_b[["x1", "x2"]].to_numpy()
y_test_b = test_df_b["label"].to_numpy()

# -----------------------------------------------------------------------------
# DATASET A (Equal Covariance)
# -----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("DATASET A (Equal Covariance)")
print("-" * 70)

# Train classifiers
lda_a = LDAClassifier().fit(X_train, y_train)
qda_a = QDAClassifier().fit(X_train, y_train)

# Predictions
y_pred_lda_a = lda_a.predict(X_test)
y_pred_qda_a = qda_a.predict(X_test)

# Accuracies
acc_ncm_a = accuracy  # Already computed earlier
acc_lda_a = (y_test == y_pred_lda_a).mean()
acc_qda_a = (y_test == y_pred_qda_a).mean()

print(f"NCM Accuracy: {acc_ncm_a:.4f} ({acc_ncm_a * 100:.2f}%)")
print(f"LDA Accuracy: {acc_lda_a:.4f} ({acc_lda_a * 100:.2f}%)")
print(f"QDA Accuracy: {acc_qda_a:.4f} ({acc_qda_a * 100:.2f}%)")

# -----------------------------------------------------------------------------
# DATASET B (Unequal Covariance)
# -----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("DATASET B (Unequal Covariance)")
print("-" * 70)

# Train classifiers
lda_b = LDAClassifier().fit(X_train_b, y_train_b)
qda_b = QDAClassifier().fit(X_train_b, y_train_b)

# Predictions
y_pred_lda_b = lda_b.predict(X_test_b)
y_pred_qda_b = qda_b.predict(X_test_b)

# Accuracies
acc_ncm_b = accuracy_b  # Already computed earlier
acc_lda_b = (y_test_b == y_pred_lda_b).mean()
acc_qda_b = (y_test_b == y_pred_qda_b).mean()

print(f"NCM Accuracy: {acc_ncm_b:.4f} ({acc_ncm_b * 100:.2f}%)")
print(f"LDA Accuracy: {acc_lda_b:.4f} ({acc_lda_b * 100:.2f}%)")
print(f"QDA Accuracy: {acc_qda_b:.4f} ({acc_qda_b * 100:.2f}%)")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: CLASSIFIER COMPARISON")
print("=" * 70)
print(f"{'Classifier':<15} {'Dataset A':<15} {'Dataset B':<15} {'Notes'}")
print("-" * 70)
print(f"{'NCM':<15} {acc_ncm_a:<15.4f} {acc_ncm_b:<15.4f} Linear boundary (perpendicular bisector)")
print(f"{'LDA':<15} {acc_lda_a:<15.4f} {acc_lda_b:<15.4f} Linear boundary (shared covariance)")
print(f"{'QDA':<15} {acc_qda_a:<15.4f} {acc_qda_b:<15.4f} Quadratic boundary (per-class covariance)")
print("-" * 70)

# =============================================================================
# ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

analysis = """
KEY OBSERVATIONS:

1. DATASET A (Equal Covariance):
   - All three classifiers perform similarly well
   - This is expected because the covariance IS equal across classes
   - Linear boundaries (NCM, LDA) work just as well as quadratic (QDA)

2. DATASET B (Unequal Covariance):
   - QDA should perform better because it models per-class covariances
   - QDA's quadratic/oval boundary can adapt to the different spreads
   - NCM and LDA are limited by their equal-covariance assumption

3. NCM vs LDA:
   - Both produce linear boundaries
   - LDA is more principled (uses full covariance information)
   - NCM only uses class means, ignoring covariance entirely

4. WHEN TO USE EACH:
   - NCM: Simple, fast, when classes are well-separated
   - LDA: When equal covariance assumption holds, need probabilistic output
   - QDA: When classes have different spreads/shapes (unequal covariance)
"""
print(analysis)

# =============================================================================
# VISUALISATION: DECISION BOUNDARIES
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Create mesh grid for decision boundary plotting
def plot_decision_boundary(ax, clf, X, y, title, clf_type="custom"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Get predictions for mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='green', linewidths=2)

    # Plot data points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', alpha=0.6, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', alpha=0.6, label='Class 1')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Wrap NCM in a class for consistent interface
class NCMWrapper:
    def __init__(self, means):
        self.means = means

    def predict(self, X):
        centers = self.means.values
        labels = self.means.index.to_numpy()
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return labels[np.argmin(dists, axis=1)]

# Dataset A plots
ncm_wrapper_a = NCMWrapper(means)
plot_decision_boundary(axes[0, 0], ncm_wrapper_a, X_test, y_test,
                       f'Dataset A - NCM\nAcc: {acc_ncm_a:.2%}')
plot_decision_boundary(axes[0, 1], lda_a, X_test, y_test,
                       f'Dataset A - LDA\nAcc: {acc_lda_a:.2%}')
plot_decision_boundary(axes[0, 2], qda_a, X_test, y_test,
                       f'Dataset A - QDA\nAcc: {acc_qda_a:.2%}')

# Dataset B plots
ncm_wrapper_b = NCMWrapper(means_b)
plot_decision_boundary(axes[1, 0], ncm_wrapper_b, X_test_b, y_test_b,
                       f'Dataset B - NCM\nAcc: {acc_ncm_b:.2%}')
plot_decision_boundary(axes[1, 1], lda_b, X_test_b, y_test_b,
                       f'Dataset B - LDA\nAcc: {acc_lda_b:.2%}')
plot_decision_boundary(axes[1, 2], qda_b, X_test_b, y_test_b,
                       f'Dataset B - QDA\nAcc: {acc_qda_b:.2%}')

plt.tight_layout()
plt.savefig("extension1_lda_qda_comparison.png", dpi=150)
plt.show()

print("\nPlot saved to: extension1_lda_qda_comparison.png")
