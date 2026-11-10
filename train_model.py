"""
Step 3 - Preprocessing split, model training, tuning and evaluation
(Specific Objectives 2 and 3).

* Chronological split: 70% train / 15% validation / 15% test (no shuffling,
  because we are forecasting the future from the past).
* Models compared: naive baseline, Linear Regression, tuned Random Forest.
* The deployed model is the Linear Regression (interpretable, tiny, fast).
  It is saved as demand_model.json (a versioned, human-readable artefact).

Run:  python train_model.py
"""
import json
import os
import datetime as dt
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CHARTS_OK = True
except Exception as _e:          # e.g. blocked by Windows App Control
    CHARTS_OK = False
    print(f"[note] Charts will be skipped ({type(_e).__name__}: matplotlib is blocked or missing). "
          "The numbers and the model are still produced.\n")

    class _NoPlot:
        def __getattr__(self, name):
            return lambda *a, **k: None
    plt = _NoPlot()
from features import FEATURES, feature_dict

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN = True
except Exception as e:                      # e.g. blocked / not installed
    SKLEARN = False
    print(f"[note] scikit-learn unavailable ({type(e).__name__}); "
          "falling back to a numpy-only linear regression.\n")

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- data ------
df = pd.read_csv("chef_rico_sales_data.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
X = np.array([[feature_dict(r.date.date(), r.is_promo, r.is_school_holiday)[f]
               for f in FEATURES] for r in df.itertuples()], dtype=float)
y = df["daily_sales"].values.astype(float)

days_without_data = [d for d in range(1, 8) if (df["day_of_week"] == d).sum() == 0]
if days_without_data:
    print(f"[note] No trading data for weekday number(s) {days_without_data} (store closed) - forecasts for those days are flagged.")
n = len(df)
i_val, i_test = int(n * 0.70), int(n * 0.85)
X_tr, y_tr = X[:i_val], y[:i_val]
X_va, y_va = X[i_val:i_test], y[i_val:i_test]
X_te, y_te = X[i_test:], y[i_test:]
X_trva, y_trva = X[:i_test], y[:i_test]
print(f"Rows: {n}  ->  train {len(X_tr)} | validation {len(X_va)} | test {len(X_te)}")
print(f"Test period starts {df['date'].iloc[i_test].date()}\n")


def metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res, ss_tot = float(np.sum(err ** 2)), float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    mape = float(np.mean(np.abs(err) / y_true) * 100)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "mape": round(mape, 2)}


def ols_fit(Xa, ya):
    """Closed-form least squares. Returns (intercept, coefficient array)."""
    design = np.column_stack([np.ones(len(Xa)), Xa])
    beta, *_ = np.linalg.lstsq(design, ya, rcond=None)
    return float(beta[0]), beta[1:]


results = []

# 1) naive baseline: always predict the training average
base_pred = np.full(len(y_te), y_trva.mean())
base = metrics(y_te, base_pred)
results.append({"model": "Baseline (training mean)", **base})

# 2) linear regression (fit on train + validation, scored on the untouched test set)
if SKLEARN:
    lr = LinearRegression().fit(X_trva, y_trva)
    intercept, coefs = float(lr.intercept_), lr.coef_
else:
    intercept, coefs = ols_fit(X_trva, y_trva)
lin_pred = intercept + X_te @ coefs
lin = metrics(y_te, lin_pred)
results.append({"model": "Linear Regression", **lin})

# 3) random forest with a small hyper-parameter search on the validation set
rf_pred, best_params = None, None
if SKLEARN:
    grid = [(n_est, depth, leaf) for n_est in (100, 300) for depth in (4, 6, 8, None) for leaf in (1, 3, 5)]
    best_mae = float("inf")
    for n_est, depth, leaf in grid:
        m = RandomForestRegressor(n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf, random_state=42)
        m.fit(X_tr, y_tr)
        mae_va = float(np.mean(np.abs(y_va - m.predict(X_va))))
        if mae_va < best_mae:
            best_mae, best_params = mae_va, {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": leaf}
    rf = RandomForestRegressor(random_state=42, **best_params).fit(X_trva, y_trva)
    rf_pred = rf.predict(X_te)
    results.append({"model": "Random Forest (tuned)", **metrics(y_te, rf_pred)})
    print(f"Best Random Forest settings (chosen on validation data): {best_params}\n")

print(f"{'Model':<28}{'MAE':>10}{'RMSE':>10}{'R2':>8}{'MAPE %':>9}")
for r in results:
    print(f"{r['model']:<28}{r['mae']:>10}{r['rmse']:>10}{r['r2']:>8}{r['mape']:>9}")

print("\nLinear Regression coefficients (effect on daily sales, other factors held equal):")
print(f"  {'intercept':<20}{intercept:>10.1f}")
for f, c in zip(FEATURES, coefs):
    print(f"  {f:<20}{c:>10.1f}")

# ------------------------------------------------------------ save model ----
artifact = {
    "model_name": "Multiple Linear Regression (OLS)",
    "version": "1.0.0",
    "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "features": FEATURES,
    "intercept": round(intercept, 4),
    "coefficients": {f: round(float(c), 4) for f, c in zip(FEATURES, coefs)},
    "metrics": lin,
    "baseline_mae": base["mae"],
    "comparison": results,
    "random_forest_params": best_params,
    "days_without_data": days_without_data,
    "split": {"train": len(X_tr), "validation": len(X_va), "test": len(X_te)},
    "dataset": {
        "source": "Kaggle - Rossmann Store Sales (Store 1, trading days)",
        "rows": n,
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
    },
}
with open("demand_model.json", "w") as f:
    json.dump(artifact, f, indent=2)
with open("figures/model_metrics.json", "w") as f:
    json.dump({"comparison": results, "split": artifact["split"], "rf_params": best_params}, f, indent=2)
print("\nSaved model artefact -> demand_model.json")

# ---------------------------------------------------------------- charts ----
test_dates = df["date"].iloc[i_test:]
plt.figure(figsize=(11, 4))
plt.plot(test_dates, y_te, label="Actual", linewidth=1.5)
plt.plot(test_dates, lin_pred, label="Linear Regression", linewidth=1.5)
if rf_pred is not None:
    plt.plot(test_dates, rf_pred, label="Random Forest", linewidth=1.5, alpha=0.8)
plt.title("Actual vs predicted daily sales (unseen test period)")
plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
plt.savefig("figures/05_actual_vs_predicted.png", dpi=150); plt.close()

names = [r["model"] for r in results]
plt.figure(figsize=(8, 4))
plt.bar(names, [r["mae"] for r in results], color=["#9aa5b1", "#b5451b", "#534ab7"][:len(results)])
plt.ylabel("Mean absolute error (lower is better)"); plt.title("Model comparison on the test set")
plt.xticks(rotation=15); plt.tight_layout(); plt.savefig("figures/06_model_comparison.png", dpi=150); plt.close()

order = np.argsort(np.abs(coefs))
plt.figure(figsize=(7, 5))
plt.barh([FEATURES[i] for i in order], [coefs[i] for i in order],
         color=["#b5451b" if coefs[i] >= 0 else "#185fa5" for i in order])
plt.title("Linear model: effect of each factor on daily sales"); plt.xlabel("Change in sales")
plt.tight_layout(); plt.savefig("figures/07_feature_effects.png", dpi=150); plt.close()
print("Saved charts -> figures/05..07 *.png" if CHARTS_OK else "Charts skipped (matplotlib unavailable).")
