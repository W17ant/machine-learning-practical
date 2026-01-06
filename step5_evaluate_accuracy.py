# =============================================================================
# STEP 6: EVALUATE TEST ACCURACY
# =============================================================================

# What is accuracy?
# - The proportion of correct predictions out of all predictions
# - Formula: accuracy = (number correct) / (total predictions)
# - Example: 180 correct out of 200 = 180/200 = 0.90 = 90%

# Get the actual labels from test data
y_true = test_df["label"].to_numpy()

# Calculate accuracy by comparing predictions to actual labels
# (y_true == y_pred) creates array of True/False for each prediction
# .mean() calculates proportion of True values (correct predictions)
accuracy = (y_true == y_pred).mean()

# Display the results
print("=" * 50)
print("DATASET A - NCM CLASSIFIER RESULTS")
print("=" * 50)
print(f"Total test samples: {len(y_true)}")
print(f"Correct predictions: {(y_true == y_pred).sum()}")
print(f"Incorrect predictions: {(y_true != y_pred).sum()}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 50)

# =============================================================================
# OPTIONAL: SAVE PREDICTIONS TO CSV
# =============================================================================

# Create a DataFrame with test data and predictions for review
results_df = test_df.copy()
results_df["predicted"] = y_pred
results_df["correct"] = (y_true == y_pred)

# Save to CSV file
results_df.to_csv("datasetA_predictions.csv", index=False)
print("\nPredictions saved to: datasetA_predictions.csv")

# Show a few incorrect predictions for analysis
incorrect = results_df[results_df["correct"] == False]
print(f"\nIncorrect predictions ({len(incorrect)} total):")
print(incorrect.head(10))
