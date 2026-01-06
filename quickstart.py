import numpy as np
import pandas as pd

# Reproducibility
rng = np.random.default_rng(42)

# --- Load your dataset (replace with the file you want) ---
df = pd.read_csv("datasetA.csv")  # or datasetB.csv
assert set(df.columns) == {"x1","x2","label"}
print(df.head())
# --- Stratified 80/20 split ---
def stratified_train_test_split(df, test_frac=0.2, random_state=42):
    rng_local = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    for cls, group in df.groupby("label"):
        idx = group.index.to_numpy()
        rng_local.shuffle(idx)
        n_test = int(len(idx) * test_frac)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    test_df  = df.loc[sorted(test_idx)].reset_index(drop=True)
    return train_df, test_df

train_df, test_df = stratified_train_test_split(df)
# --- Compute training means per class ---
means = train_df.groupby("label")[["x1","x2"]].mean()
print(means)
# --- Nearest-class-mean prediction ---
def nearest_mean_predict(test_df, means):
    centers = means.values  # rows correspond to class labels in means.index order
    labels = means.index.to_numpy()
    X = test_df[["x1","x2"]].to_numpy()
    dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
    pred_idx = np.argmin(dists, axis=1)
    return labels[pred_idx]

y_pred = nearest_mean_predict(test_df, means)
# --- Accuracy ---
def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

acc = accuracy(test_df["label"].to_numpy(), y_pred)
print(f"Test accuracy: {acc:.4f}")