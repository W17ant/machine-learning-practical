# =============================================================================
# STEP 8: COMPARISON AND DISCUSSION
# =============================================================================

# =============================================================================
# ACCURACY COMPARISON
# =============================================================================

print("=" * 60)
print("ACCURACY COMPARISON: DATASET A vs DATASET B")
print("=" * 60)
print(f"Dataset A (equal covariance):   {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Dataset B (unequal covariance): {accuracy_b:.4f} ({accuracy_b * 100:.2f}%)")
print("=" * 60)

# Calculate the difference
diff = accuracy - accuracy_b
print(f"\nDifference: {abs(diff):.4f} ({abs(diff) * 100:.2f}%)")
if diff > 0:
    print("Dataset A has HIGHER accuracy")
else:
    print("Dataset B has HIGHER accuracy")

# =============================================================================
# DISCUSSION: WHY DOES ACCURACY DIFFER?
# =============================================================================

print("\n" + "=" * 60)
print("DISCUSSION")
print("=" * 60)

discussion = """
KEY CONCEPT: NCM uses a LINEAR decision boundary

The Nearest-Class-Mean classifier draws a straight line (in 2D) between classes.
This line is the perpendicular bisector of the line connecting the two centroids.

DATASET A (Equal Covariance):
- Both classes have the same "spread" or variance
- Data points are distributed similarly around each centroid
- A straight line boundary works well because the classes are symmetrical
- NCM is well-suited for this type of data

DATASET B (Unequal Covariance):
- Classes have DIFFERENT spreads (one tight, one wide)
- Class 0: TIGHT cluster (std ~0.36) centered around (2, 3)
- Class 1: SPREAD OUT (std ~1.56) centered around (6, 7) - 4x wider!
- A straight line cannot optimally separate classes with different shapes
- Some points from the "wider" Class 1 may extend towards Class 0's territory
- NCM accuracy typically DECREASES with unequal covariance

WHY AN OVAL BOUNDARY WOULD WORK BETTER FOR DATASET B:
- Class 0 is a tight, compact cluster
- An OVAL/ELLIPSE boundary around Class 0 would capture its shape
- Points inside the oval = Class 0, points outside = Class 1
- This accounts for Class 1's larger spread without misclassifying tight Class 0 points
- A linear boundary ignores the different spreads and cuts through both classes poorly

WHY THIS HAPPENS:
- NCM assumes equal covariance (same spread for all classes)
- When this assumption is violated, the linear boundary is suboptimal
- A QUADRATIC boundary (oval/ellipse) would be needed for optimal separation
"""
print(discussion)

# =============================================================================
# LINEAR vs QUADRATIC DECISION BOUNDARIES
# =============================================================================

print("=" * 60)
print("LINEAR vs QUADRATIC DECISION BOUNDARIES")
print("=" * 60)

boundaries = """
NCM DECISION BOUNDARY (Linear):
- Shape: Straight line (2D) or flat plane (higher dimensions)
- Equation: All points equidistant from both centroids
- Works best when: Classes have EQUAL covariance (same spread/shape)
- Limitation: Cannot curve around data with different spreads

    Class 0        |        Class 1
       o o         |           x x
      o o o        |          x x x
       o o         |           x x
                   |
           (straight line)

QUADRATIC DECISION BOUNDARY (e.g., QDA - Quadratic Discriminant Analysis):
- Shape: Curved line (ellipse, parabola, hyperbola)
- Takes into account DIFFERENT covariances for each class
- Works best when: Classes have UNEQUAL covariance
- Can "wrap around" a class with larger spread

    Class 0           )      Class 1
       o o           )         x x x x
      o o o         )        x x x x x x
       o o         )           x x x x
                  )
           (curved boundary)

PRACTICAL IMPLICATION:
- If Dataset B accuracy is lower than Dataset A, it suggests the classes
  have unequal covariance and NCM's linear boundary is not optimal
- To improve Dataset B results, you would need a classifier that accounts
  for different covariances (like QDA or other non-linear classifiers)
"""
print(boundaries)

# =============================================================================
# VISUALISATION: LINEAR vs QUADRATIC BOUNDARY COMPARISON
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Equal covariance (linear boundary works) ---
ax1 = axes[0]

# Generate example data for illustration
np.random.seed(42)
# Equal covariance - both classes have same spread
eq_class0 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 100)
eq_class1 = np.random.multivariate_normal([4, 4], [[0.5, 0], [0, 0.5]], 100)

ax1.scatter(eq_class0[:, 0], eq_class0[:, 1], c="blue", alpha=0.6, label="Class 0")
ax1.scatter(eq_class1[:, 0], eq_class1[:, 1], c="red", alpha=0.6, label="Class 1")

# Linear boundary (perpendicular bisector)
ax1.plot([0, 6], [6, 0], "g-", linewidth=2, label="NCM boundary (linear)")

ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_title("Equal Covariance:\nLinear Boundary Works Well")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 6)

# --- Plot 2: Unequal covariance (curved boundary needed) ---
ax2 = axes[1]

# Unequal covariance - class 1 has larger spread
uneq_class0 = np.random.multivariate_normal([2, 3], [[0.3, 0], [0, 0.3]], 100)
uneq_class1 = np.random.multivariate_normal([4, 3], [[1.5, 0.3], [0.3, 1.5]], 100)

ax2.scatter(uneq_class0[:, 0], uneq_class0[:, 1], c="blue", alpha=0.6, label="Class 0 (tight)")
ax2.scatter(uneq_class1[:, 0], uneq_class1[:, 1], c="red", alpha=0.6, label="Class 1 (spread)")

# Linear boundary (suboptimal)
ax2.axvline(x=3, color="green", linestyle="-", linewidth=2, label="NCM boundary (linear)")

# Quadratic boundary (optimal) - approximate with curved line
theta = np.linspace(-1.5, 1.5, 100)
curve_x = 3 + 0.8 * np.cosh(theta) - 0.8
curve_y = 3 + 1.5 * np.sinh(theta) * 0.5
ax2.plot(curve_x, curve_y, "m--", linewidth=2, label="Optimal boundary (curved)")

ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_title("Unequal Covariance:\nCurved Boundary Needed")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 7)
ax2.set_ylim(0, 6)

# --- Plot 3: OVAL/ELLIPTICAL BOUNDARY (class surrounded by another) ---
ax3 = axes[2]

# One class is tight cluster in center, other class surrounds it
# This creates an OVAL/ELLIPSE decision boundary
center_class = np.random.multivariate_normal([3, 3], [[0.3, 0], [0, 0.3]], 80)

# Surrounding class - ring of points around the center
angles = np.random.uniform(0, 2 * np.pi, 150)
radii = np.random.normal(2, 0.4, 150)
outer_class_x = 3 + radii * np.cos(angles)
outer_class_y = 3 + radii * np.sin(angles)
outer_class = np.column_stack([outer_class_x, outer_class_y])

ax3.scatter(center_class[:, 0], center_class[:, 1], c="blue", alpha=0.6, label="Class 0 (inner)")
ax3.scatter(outer_class[:, 0], outer_class[:, 1], c="red", alpha=0.6, label="Class 1 (outer)")

# Linear boundary (completely wrong for this data!)
ax3.axvline(x=3, color="green", linestyle="-", linewidth=2, label="NCM boundary (linear)")

# OVAL/ELLIPTICAL boundary (optimal for this case)
# Draw an ellipse around the inner class
theta_ellipse = np.linspace(0, 2 * np.pi, 100)
ellipse_x = 3 + 1.2 * np.cos(theta_ellipse)  # x radius
ellipse_y = 3 + 1.2 * np.sin(theta_ellipse)  # y radius
ax3.plot(ellipse_x, ellipse_y, "m--", linewidth=2, label="Optimal boundary (OVAL)")

ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_title("Surrounded Class:\nOVAL Boundary Needed")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 6)
ax3.set_ylim(0, 6)

plt.tight_layout()
plt.savefig("boundary_comparison.png", dpi=150)
plt.show()

print("\nPlot saved to: boundary_comparison.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
summary = f"""
1. Dataset A accuracy: {accuracy:.2%}
2. Dataset B accuracy: {accuracy_b:.2%}

3. NCM uses a LINEAR decision boundary (straight line)

4. Linear boundaries work well when classes have EQUAL covariance
   (same spread/shape) - this is Dataset A

5. Linear boundaries are SUBOPTIMAL when classes have UNEQUAL covariance
   (different spreads) - this is Dataset B

6. For unequal covariance data, a QUADRATIC boundary (curved line)
   would provide better separation and higher accuracy

7. Classifiers like QDA (Quadratic Discriminant Analysis) can learn
   quadratic boundaries by modelling each class's covariance separately
"""
print(summary)
