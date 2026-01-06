# =============================================================================
# STEP 1: LOAD DATASET A
# =============================================================================

# Import pandas - a library for working with tabular data (like spreadsheets)
import pandas as pd

# Load the CSV file into a DataFrame (a table structure)
# Dataset A has "equal covariance" - meaning the spread of data is similar for each class
df = pd.read_csv("datasetA.csv")

# =============================================================================
# STEP 2: VERIFY THE DATASET
# =============================================================================

# Check the number of rows using .shape[0]
# .shape returns (rows, columns), so [0] gets just the row count
# Expected: 1000 rows
print(f"Number of rows: {df.shape[0]}")

# Check the column names to ensure data loaded correctly
# Expected: x1, x2 (the two features) and label (the class: 0 or 1)
print(f"Column names: {list(df.columns)}")

# Display the first 5 rows to visually inspect the data
# .head() shows a preview without printing the entire dataset
df.head()
