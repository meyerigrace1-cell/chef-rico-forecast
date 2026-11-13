"""
OPTIONAL - Check the method on REAL restaurant data.

The Chef Rico dataset is simulated (no real burger records exist). To show that the
same method also works on real sales, run this script on any real restaurant daily-sales
file, for example the open dataset
    Taigel, F. (2019). Restaurant Sales Data with Features. Mendeley Data, V1,
    doi:10.17632/xv5kp6mxv7.1   (daily servings per product + local weather, CC BY 4.0).

Step 1 - see the columns:      python validate_on_real_data.py yourfile.csv
Step 2 - run the check:        python validate_on_real_data.py yourfile.csv --date-col DATE --target-col KOEFTE
With weather columns:          ... --weather-cols temperature,rain
Several products added up:     ... --target-col HAEHNCHEN,KOEFTE
Other options:                 --sep ";"  --decimal ","  --dayfirst  --aggregate sum|count|none

It compares, on the LAST 20% of days (never seen in training):
  1. training average   2. average for that weekday   3. linear regression (weekday + month-end)
  4. linear regression + weather columns (only if you give them)   5. random forest (if scikit-learn works)
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--date-col")
ap.add_argument("--target-col", help="column name, or several separated by commas (they are added together)")
ap.add_argument("--weather-cols", default="", help="comma-separated numeric weather columns")
ap.add_argument("--sep", default=None)
ap.add_argument("--decimal", default=".")
ap.add_argument("--dayfirst", action="store_true")
ap.add_argument("--aggregate", choices=["none", "sum", "count"], default="none",
                help="use sum/count if the file has one row per sale instead of one row per day")
a = ap.parse_args()

df = pd.read_csv(a.file, sep=a.sep, engine="python", decimal=a.decimal)
df.columns = [str(c).strip() for c in df.columns]

if not a.date_col or not a.target_col:
    print(f"File has {len(df)} rows and these columns:\n")
    print(df.dtypes.to_string())
    print("\nFirst rows:")
    print(df.head(5).to_string())
    print("\nNow run again with --date-col and --target-col (see the top of this file for examples).")
    sys.exit(0)

targets = [t.strip() for t in a.target_col.split(",")]
weather = [w.strip() for w in a.weather_cols.split(",") if w.strip()]
for col in [a.date_col] + targets + weather:
    if col not in df.columns:
        sys.exit(f"Column '{col}' not found. Available columns: {list(df.columns)}")

df[a.date_col] = pd.to_datetime(df[a.date_col], dayfirst=a.dayfirst, errors="coerce")
df = df.dropna(subset=[a.date_col])
for col in targets + weather:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if a.aggregate == "count":
    daily = df.groupby(df[a.date_col].dt.date).size().rename("y").to_frame()
    for w in weather:
        daily[w] = df.groupby(df[a.date_col].dt.date)[w].mean()
else:
    df["y"] = df[targets].sum(axis=1, min_count=1)
    agg = {"y": "sum"}
    agg.update({w: "mean" for w in weather})
    daily = df.groupby(df[a.date_col].dt.date).agg(agg)
daily.index = pd.to_datetime(daily.index)
daily = daily.sort_index().dropna(subset=["y"])
for w in weather:
    miss = int(daily[w].isna().sum())
    if miss:
        print(f"[note] {w}: {miss} missing values filled with the median")
        daily[w] = daily[w].fillna(daily[w].median())

print(f"Days: {len(daily)}   {daily.index.min().date()} -> {daily.index.max().date()}")
print(f"Target per day: mean {daily['y'].mean():.1f}, min {daily['y'].min():.0f}, max {daily['y'].max():.0f}")
if len(daily) < 100:
    print("[warning] fewer than 100 days - results will be unstable.")

# ---------------------------------------------------------------- features --
dow = daily.index.dayofweek
X_cal = pd.DataFrame({f"dow_{d}": (dow == d).astype(float) for d in range(1, 7)}, index=daily.index)
X_cal["month_end_window"] = daily.index.day.isin([28, 29, 30, 31, 1, 2]).astype(float)
X_wx = pd.concat([X_cal, daily[weather]], axis=1) if weather else None
y = daily["y"].values.astype(float)

cut = int(len(daily) * 0.8)
y_tr, y_te = y[:cut], y[cut:]


def metrics(true, pred):
    err = true - pred
    mask = true != 0
    return {"MAE": round(float(np.mean(np.abs(err))), 2),
            "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 2),
            "R2": round(float(1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)), 3),
            "MAPE%": round(float(np.mean(np.abs(err[mask]) / true[mask]) * 100), 1)}


def ols(Xdf):
    X = np.column_stack([np.ones(len(Xdf)), Xdf.values.astype(float)])
    beta, *_ = np.linalg.lstsq(X[:cut], y_tr, rcond=None)
    return X @ beta, beta


results = {}
results["Training average"] = metrics(y_te, np.full(len(y_te), y_tr.mean()))
wk_mean = pd.Series(y_tr).groupby(dow[:cut]).mean()
results["Average for that weekday"] = metrics(y_te, np.array([wk_mean.get(d, y_tr.mean()) for d in dow[cut:]]))
pred_cal, _ = ols(X_cal)
results["Linear: weekday + month-end"] = metrics(y_te, pred_cal[cut:])
pred_best = pred_cal
if X_wx is not None:
    pred_wx, beta_wx = ols(X_wx)
    results["Linear: + weather"] = metrics(y_te, pred_wx[cut:])
    pred_best = pred_wx
try:
    from sklearn.ensemble import RandomForestRegressor
    Xrf = X_wx if X_wx is not None else X_cal
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42).fit(Xrf.values[:cut], y_tr)
    results["Random forest"] = metrics(y_te, rf.predict(Xrf.values[cut:]))
except Exception as e:
    print(f"[note] random forest skipped ({type(e).__name__})")

print(f"\nResults on the last {len(y_te)} days (not used for training):")
print(f"{'Model':<30}{'MAE':>9}{'RMSE':>9}{'R2':>8}{'MAPE%':>8}")
for name, m in results.items():
    print(f"{name:<30}{m['MAE']:>9}{m['RMSE']:>9}{m['R2']:>8}{m['MAPE%']:>8}")

if X_wx is not None:
    print("\nWeather coefficients (change in target per unit of the weather column):")
    names = ["intercept"] + list(X_wx.columns)
    for n, b in zip(names, beta_wx):
        if n in weather:
            print(f"  {n:<24}{b:>9.3f}")

os.makedirs("figures", exist_ok=True)
with open("figures/real_validation.json", "w") as f:
    json.dump({"file": os.path.basename(a.file), "days": len(daily), "target": a.target_col,
               "weather_cols": weather, "results": results}, f, indent=2)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4))
    plt.plot(daily.index[cut:], y_te, label="Actual", linewidth=1.4)
    plt.plot(daily.index[cut:], pred_best[cut:], label="Linear model", linewidth=1.4)
    plt.title("Real restaurant data: actual vs predicted (unseen days)")
    plt.legend(); plt.tight_layout(); plt.savefig("figures/real_validation.png", dpi=150)
    print("\nSaved figures/real_validation.png and figures/real_validation.json")
except Exception as e:
    print(f"\nSaved figures/real_validation.json (chart skipped: {type(e).__name__})")
