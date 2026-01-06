# =============================================================================
# STEP 7: VISUALISATION
# =============================================================================

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# =============================================================================
# PLOT 1: DATASET A (EQUAL COVARIANCE) WITH DECISION BOUNDARY
# =============================================================================

# Create a figure with 1 row, 2 columns (side by side plots)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left plot: Training data with centroids ---
ax1 = axes[0]

# Scatter plot of training data, colored by class
# Class 0 = blue, Class 1 = red
train_class0 = train_df[train_df["label"] == 0]
train_class1 = train_df[train_df["label"] == 1]

ax1.scatter(train_class0["x1"], train_class0["x2"], c="blue", alpha=0.5, label="Class 0 (train)", s=30)
ax1.scatter(train_class1["x1"], train_class1["x2"], c="red", alpha=0.5, label="Class 1 (train)", s=30)

# Plot the centroids (mean vectors) as large markers
mu0 = means.loc[0].values
mu1 = means.loc[1].values
ax1.scatter(mu0[0], mu0[1], c="blue", marker="X", s=200, edgecolor="black", linewidth=2, label="μ0 (Class 0 mean)")
ax1.scatter(mu1[0], mu1[1], c="red", marker="X", s=200, edgecolor="black", linewidth=2, label="μ1 (Class 1 mean)")

ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title("Dataset A - Training Data with Class Centroids")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Right plot: Test data with decision boundary ---
ax2 = axes[1]

# Scatter plot of test data, colored by class
test_class0 = test_df[test_df["label"] == 0]
test_class1 = test_df[test_df["label"] == 1]

ax2.scatter(test_class0["x1"], test_class0["x2"], c="blue", alpha=0.5, label="Class 0 (test)", s=30)
ax2.scatter(test_class1["x1"], test_class1["x2"], c="red", alpha=0.5, label="Class 1 (test)", s=30)

# Plot centroids
ax2.scatter(mu0[0], mu0[1], c="blue", marker="X", s=200, edgecolor="black", linewidth=2, label="μ0")
ax2.scatter(mu1[0], mu1[1], c="red", marker="X", s=200, edgecolor="black", linewidth=2, label="μ1")

# =============================================================================
# NCM DECISION BOUNDARY (for equal covariance)
# =============================================================================
# The decision boundary is the perpendicular bisector of the line connecting μ0 and μ1
# Points on this line are equidistant from both centroids
#
# Math: The boundary is where distance to μ0 = distance to μ1
# This simplifies to a straight line perpendicular to the line joining the centroids

# Midpoint between the two centroids
midpoint = (mu0 + mu1) / 2

# Direction vector from μ0 to μ1
direction = mu1 - mu0

# Perpendicular direction (rotate 90 degrees by swapping and negating)
perpendicular = np.array([-direction[1], direction[0]])

# Create points along the decision boundary line
# Line equation: midpoint + t * perpendicular (for various values of t)
t_values = np.linspace(-3, 3, 100)
boundary_x = midpoint[0] + t_values * perpendicular[0]
boundary_y = midpoint[1] + t_values * perpendicular[1]

# Plot the decision boundary
ax2.plot(boundary_x, boundary_y, "g--", linewidth=2, label="Decision boundary")

# Set axis limits to match the data range
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())

ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_title("Dataset A - Test Data with NCM Decision Boundary")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("datasetA_visualisation.png", dpi=150)
plt.show()

print("Plot saved to: datasetA_visualisation.png")

# =============================================================================
# PLOT 2: DATASET B (UNEQUAL COVARIANCE)
# =============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# --- Left plot: Training data ---
ax3 = axes2[0]

train_class0_b = train_df_b[train_df_b["label"] == 0]
train_class1_b = train_df_b[train_df_b["label"] == 1]

ax3.scatter(train_class0_b["x1"], train_class0_b["x2"], c="blue", alpha=0.5, label="Class 0 (train)", s=30)
ax3.scatter(train_class1_b["x1"], train_class1_b["x2"], c="red", alpha=0.5, label="Class 1 (train)", s=30)

# Plot centroids
mu0_b = means_b.loc[0].values
mu1_b = means_b.loc[1].values
ax3.scatter(mu0_b[0], mu0_b[1], c="blue", marker="X", s=200, edgecolor="black", linewidth=2, label="μ0")
ax3.scatter(mu1_b[0], mu1_b[1], c="red", marker="X", s=200, edgecolor="black", linewidth=2, label="μ1")

ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_title("Dataset B - Training Data with Class Centroids")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Right plot: Test data ---
ax4 = axes2[1]

test_class0_b = test_df_b[test_df_b["label"] == 0]
test_class1_b = test_df_b[test_df_b["label"] == 1]

ax4.scatter(test_class0_b["x1"], test_class0_b["x2"], c="blue", alpha=0.5, label="Class 0 (test)", s=30)
ax4.scatter(test_class1_b["x1"], test_class1_b["x2"], c="red", alpha=0.5, label="Class 1 (test)", s=30)

# Plot centroids
ax4.scatter(mu0_b[0], mu0_b[1], c="blue", marker="X", s=200, edgecolor="black", linewidth=2, label="μ0")
ax4.scatter(mu1_b[0], mu1_b[1], c="red", marker="X", s=200, edgecolor="black", linewidth=2, label="μ1")

ax4.set_xlabel("x1")
ax4.set_ylabel("x2")
ax4.set_title("Dataset B - Test Data with Class Centroids")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("datasetB_visualisation.png", dpi=150)
plt.show()

print("Plot saved to: datasetB_visualisation.png")
