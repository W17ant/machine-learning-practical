# Machine Learning Practical - Notebook Guide

## Machine-Learning-Unit1-Practical.ipynb

A cell-by-cell walkthrough of the NCM classification practical.

---

## How to Use This Guide

1. Open `Machine-Learning-Unit1-Practical.ipynb` in Jupyter
2. Run each cell in order using `Shift+Enter`
3. Read this guide alongside to understand what each cell does

---

## Cell 1: Load Dataset A

**What happens:**
- Imports the pandas library (used for handling tabular data)
- Loads `datasetA.csv` into a variable called `df`
- Prints the number of rows (should be 1000)
- Prints the column names (should be x1, x2, label)
- Shows the first 5 rows of data

**Why:**
- We need to load the data before we can work with it
- Checking rows and columns confirms the file loaded correctly
- Dataset A has "equal covariance" - both classes have similar spread

**You should see:**
```
Number of rows: 1000
Column names: ['x1', 'x2', 'label']
```
Plus a table showing x1, x2 values and labels (0 or 1).

---

## Cell 2: Stratified Train/Test Split

**What happens:**
- Imports numpy (for numerical operations)
- Defines a function `stratified_train_test_split()`
- Splits the data: 80% for training, 20% for testing
- Prints the sizes of each set
- Shows how many samples of each class are in train and test

**Why:**
- We train the model on training data
- We test it on separate data it has never seen
- "Stratified" means both classes are split proportionally
- This prevents accidentally putting all of one class in train or test

**You should see:**
```
Training set size: 800 rows (80%)
Test set size: 200 rows (20%)

Training set class distribution:
label
0    400
1    400

Test set class distribution:
label
0    100
1    100
```

---

## Cell 3: Compute Class Means

**What happens:**
- Calculates the average position (centroid) of each class
- Uses ONLY training data (never test data)
- Stores results in a variable called `means`
- Prints μ0 (class 0 centroid) and μ1 (class 1 centroid)

**Why:**
- The centroid is the "center point" of each class
- NCM classifies new points based on which centroid is closer
- We must only use training data - using test data would be cheating

**Key variables created:**
- `means` - DataFrame with mean x1, x2 for each class
- `mu0` - Array [x1_mean, x2_mean] for class 0
- `mu1` - Array [x1_mean, x2_mean] for class 1

**You should see:**
```
Class mean vectors (centroids):
             x1        x2
label
0      1.994434  2.948663
1      5.883596  6.946654
```

---

## Cell 4: NCM Classifier

**What happens:**
- Defines the `nearest_mean_predict()` function
- For each test point, calculates distance to both centroids
- Assigns the point to whichever class centroid is closer
- Runs predictions on all test data
- Shows first 10 predictions vs actual labels

**Why:**
- This is the core classification algorithm
- Euclidean distance measures straight-line distance between points
- The class with the nearest mean "wins"

**How the distance calculation works:**
```
For a test point at (3.5, 4.0):
  Distance to μ0 at (2, 3) = sqrt((3.5-2)² + (4-3)²) = 1.80
  Distance to μ1 at (6, 7) = sqrt((3.5-6)² + (4-7)²) = 3.91

  1.80 < 3.91, so predict Class 0
```

**Key variables created:**
- `y_pred` - Array of predicted labels for test set

---

## Cell 5: Evaluate Accuracy

**What happens:**
- Compares predictions (`y_pred`) to actual labels (`y_true`)
- Calculates accuracy = correct predictions / total predictions
- Prints a results summary
- Saves predictions to `datasetA_predictions.csv`
- Shows any incorrect predictions

**Why:**
- Accuracy tells us how well the classifier performs
- Saving to CSV lets you review individual predictions later
- Seeing incorrect predictions helps understand where the model fails

**You should see:**
```
==================================================
DATASET A - NCM CLASSIFIER RESULTS
==================================================
Total test samples: 200
Correct predictions: 200
Incorrect predictions: 0
Test Accuracy: 1.0000 (100.00%)
==================================================
```

100% accuracy means NCM perfectly separates Dataset A's classes.

---

## Cell 6: Dataset B Pipeline

**What happens:**
- Loads `datasetB.csv` (unequal covariance data)
- Repeats the entire process: split, compute means, predict, evaluate
- Dataset B has different class spreads (Class 0 tight, Class 1 spread out)

**Why:**
- We want to compare NCM performance on equal vs unequal covariance
- Dataset B tests whether a linear boundary still works when classes have different shapes

**Key variables created:**
- `df_b`, `train_df_b`, `test_df_b` - Dataset B data
- `means_b` - Centroids for Dataset B
- `y_pred_b` - Predictions for Dataset B
- `accuracy_b` - Accuracy on Dataset B

**You should see:**
```
==================================================
DATASET B - NCM CLASSIFIER RESULTS
==================================================
Total test samples: 200
Correct predictions: 197
Incorrect predictions: 3
Test Accuracy: 0.9850 (98.50%)
==================================================
```

Lower than Dataset A because the linear boundary is not optimal for unequal covariance.

---

## Cell 7: Visualisation

**What happens:**
- Creates scatter plots for both datasets
- Shows training data colored by class (blue = 0, red = 1)
- Marks centroids with X symbols
- Draws the NCM decision boundary (green dashed line)
- Saves plots as PNG files

**Why:**
- Visualisation helps you understand what the classifier is doing
- You can see how the decision boundary separates the classes
- Comparing Dataset A and B plots shows why accuracy differs

**The decision boundary:**
- It's the perpendicular bisector of the line between μ0 and μ1
- Points on one side are closer to μ0 → predict Class 0
- Points on the other side are closer to μ1 → predict Class 1

**Files created:**
- `datasetA_visualisation.png`
- `datasetB_visualisation.png`

---

## Cell 8: Comparison and Discussion

**What happens:**
- Compares accuracy: Dataset A vs Dataset B
- Explains WHY accuracy differs
- Discusses linear vs quadratic decision boundaries
- Creates a 3-panel plot showing different boundary types
- Summarises key findings

**Key concepts explained:**

**Linear boundary (NCM, LDA):**
- Straight line separating classes
- Works well when classes have EQUAL covariance (same spread)
- Dataset A: equal covariance → 100% accuracy

**Quadratic boundary (QDA):**
- Curved line (ellipse, oval, hyperbola)
- Works well when classes have UNEQUAL covariance
- Can "wrap around" a tight class

**Why Dataset B has lower accuracy:**
- Class 0 is tight (std ~0.36)
- Class 1 is spread out (std ~1.56) - 4x wider
- A straight line cannot optimally separate them
- An OVAL around Class 0 would work better

**Files created:**
- `boundary_comparison.png`

---

## Cell 9: Troubleshooting Checklist

**What happens:**
- Verifies the implementation is correct
- Checks 5 common issues that cause wrong results
- Prints array shapes to debug any problems

**The 5 checks:**

1. **Means from training only** - Did we use `train_df` not `df`?
2. **Stratified split** - Are classes balanced in train and test?
3. **Both dimensions used** - Does distance use x1 AND x2?
4. **Labels aligned** - Do prediction indices match class labels?
5. **Correct shapes** - Are arrays the right dimensions?

**When to use:**
- If your accuracy is unexpectedly low
- If you get errors about array shapes
- To verify your implementation matches the reference

---

## Cell 10: Extension 1 - LDA & QDA Classifiers

**What happens:**
- Implements Linear Discriminant Analysis (LDA) from scratch
- Implements Quadratic Discriminant Analysis (QDA) from scratch
- Trains both on Dataset A and Dataset B
- Compares accuracy: NCM vs LDA vs QDA

**LDA (Linear Discriminant Analysis):**
- Models each class as a Gaussian distribution
- Assumes ALL classes share the SAME covariance matrix
- Produces LINEAR decision boundaries
- More principled than NCM (uses covariance info)

**QDA (Quadratic Discriminant Analysis):**
- Each class has its OWN covariance matrix
- Can model different spreads per class
- Produces QUADRATIC boundaries (curves, ellipses)
- Better for unequal covariance data

**Expected results:**
- Dataset A: All three similar (equal covariance suits linear boundary)
- Dataset B: QDA may perform better (can adapt to different spreads)

---

## Cell 11: Extension 1 Visualisation

**What happens:**
- Creates a 2x3 grid of plots
- Top row: Dataset A with NCM, LDA, QDA boundaries
- Bottom row: Dataset B with NCM, LDA, QDA boundaries
- Shows how each classifier draws its decision boundary

**What to look for:**
- NCM and LDA produce similar LINEAR boundaries
- QDA produces a CURVED boundary
- On Dataset B, QDA's curve may better separate the classes

**Files created:**
- `extension1_lda_qda_comparison.png`

---

## Cell 12: Extension 2 - Varying Covariance

**What happens:**
- Generates synthetic datasets with different covariance ratios
- Ratio 1.0 = equal covariance
- Ratio 4.0 = Class 1 is 4x more spread than Class 0
- Tests NCM and QDA at each ratio
- Plots accuracy vs covariance ratio

**Why this matters:**
- Shows exactly when NCM starts to fail
- Demonstrates QDA's robustness to unequal covariance
- Helps you understand the equal covariance assumption

**What to look for in the plot:**
- At ratio 1.0 (equal): Both NCM and QDA perform well
- As ratio increases or decreases: NCM accuracy may drop
- QDA maintains high accuracy across different ratios

**Files created:**
- `extension2_accuracy_vs_covariance.png`

---

## Cell 13: Extension 3 - Gaussian Noise Robustness

**What happens:**
- Adds random noise features to the data (0, 1, 2, 5, 10, 20, 50 features)
- Noise features have NO predictive power - pure random values
- Tests how accuracy degrades as noise increases
- Plots accuracy vs number of noise features

**Why this matters:**
- Real datasets often have irrelevant features
- This tests classifier robustness to "junk" features
- Demonstrates the "curse of dimensionality"

**The curse of dimensionality:**
- More features = more parameters to estimate
- With limited training data, estimates become unreliable
- Noise features add random variation to distances

**What to look for in the plot:**
- Accuracy decreases as noise features increase
- The useful signal (x1, x2) gets drowned out by noise
- Some classifiers may be more robust than others

**Files created:**
- `extension3_noise_robustness.png`

---

## Cell 14: Extension 4 - Mahalanobis Distance

**What happens:**
- Implements NCM using Mahalanobis distance instead of Euclidean
- Mahalanobis accounts for covariance structure
- Compares Euclidean NCM vs Mahalanobis NCM vs LDA vs QDA
- Prints final summary table of all classifiers

**Euclidean vs Mahalanobis:**

| Euclidean | Mahalanobis |
|-----------|-------------|
| Treats all directions equally | Accounts for covariance |
| Equal distance = circle | Equal distance = ellipse |
| Ignores feature scales | Normalizes by variance |
| Simple | More sophisticated |

**Mahalanobis formula:**
```
d = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
```
The Σ⁻¹ (inverse covariance) is what makes it different from Euclidean.

**When Mahalanobis helps:**
- Features have very different scales (e.g., age 0-100 vs salary 0-1000000)
- Features are correlated with each other
- Still uses pooled covariance, so still produces linear boundary

**Final summary table shows:**
- All classifiers compared on both datasets
- Helps you see which performs best where

---

## Summary: What You've Learned

1. **NCM Classification**
   - Uses class centroids and Euclidean distance
   - Simple but effective for well-separated classes

2. **Train/Test Split**
   - Always stratify to balance classes
   - Never use test data during training

3. **Equal vs Unequal Covariance**
   - Equal: Linear boundary works (NCM, LDA)
   - Unequal: Need quadratic boundary (QDA)

4. **Decision Boundaries**
   - Linear: Straight line (NCM, LDA, Mahalanobis NCM)
   - Quadratic: Curved line (QDA)

5. **Classifier Comparison**
   - NCM: Simplest, ignores covariance
   - LDA: Uses shared covariance, linear boundary
   - QDA: Uses per-class covariance, quadratic boundary
   - Mahalanobis NCM: Accounts for covariance in distance

6. **Robustness**
   - Noise features hurt all classifiers
   - Feature selection is important

---

## Reference Results

These are the expected results for the provided datasets:

| Dataset | Classifier | Accuracy |
|---------|------------|----------|
| A (equal cov) | NCM | 100.00% |
| A (equal cov) | LDA | ~100.00% |
| A (equal cov) | QDA | ~100.00% |
| B (unequal cov) | NCM | 98.50% |
| B (unequal cov) | LDA | ~98.50% |
| B (unequal cov) | QDA | ~98.50-99% |

If your results differ significantly, run the troubleshooting cell (Cell 9).

---

## Files Created by the Notebook

| File | Description |
|------|-------------|
| `datasetA_predictions.csv` | Test predictions for Dataset A |
| `datasetB_predictions.csv` | Test predictions for Dataset B |
| `datasetA_visualisation.png` | Scatter plot with boundary for A |
| `datasetB_visualisation.png` | Scatter plot with boundary for B |
| `boundary_comparison.png` | Linear vs quadratic boundary comparison |
| `extension1_lda_qda_comparison.png` | NCM vs LDA vs QDA boundaries |
| `extension2_accuracy_vs_covariance.png` | Accuracy vs covariance ratio |
| `extension3_noise_robustness.png` | Accuracy vs noise features |

---

Good luck with your revision!
