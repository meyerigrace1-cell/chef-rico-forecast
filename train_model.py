import pandas as pd
import numpy as np

df = pd.read_csv("chef_rico_features.csv")

features = ["is_weekend", "is_payday", "is_rainy", "is_event", "ntsa_crackdown", "month", "day_of_month"]
X = df[features].values.astype(float)
y = df["burgers_sold"].values.astype(float)

X_design = np.column_stack([np.ones(len(X)), X])  # add intercept term

np.random.seed(42)
indices = np.random.permutation(len(X_design))
split = int(len(X_design) * 0.8)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X_design[train_idx], X_design[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Closed-form Ordinary Least Squares regression (no scipy/sklearn needed)
coefficients, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
y_pred = X_test @ coefficients

mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("Model: Multiple Linear Regression (numpy, no scipy needed)")
print(f"MAE={mae:.2f}, RMSE={rmse:.2f}")

print("\nFeature coefficients:")
for name, coef in zip(["intercept"] + features, coefficients):
    print(f"  {name}: {coef:.2f}")

np.savez("demand_model.npz", coefficients=coefficients, features=features)
print("\nSaved trained model -> demand_model.npz")