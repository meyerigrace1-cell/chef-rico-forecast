"""
Step 2 - Exploratory data analysis (Specific Objective 1).
Prints the key patterns and saves charts into the figures/ folder.

Run:  python explore_data.py
"""
import json
import os
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
from features import DAY_NAMES

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("chef_rico_sales_data.csv", parse_dates=["date"])
df["month"] = df["date"].dt.month
df["rolling_7"] = df["daily_sales"].rolling(7, min_periods=1).mean()

print(f"Rows: {len(df)}   Period: {df['date'].min().date()} -> {df['date'].max().date()}")
print("\nMissing values per column:")
print(df.isna().sum().to_string())
print("\nDescriptive statistics (daily_sales):")
print(df["daily_sales"].describe().round(1).to_string())

summary = {"rows": int(len(df)), "mean_sales": round(float(df["daily_sales"].mean()), 1)}

def effect(col, label):
    g = df.groupby(col)["daily_sales"].mean().round(1)
    lo, hi = float(g.get(0, float("nan"))), float(g.get(1, float("nan")))
    pct = (hi - lo) / lo * 100 if lo else float("nan")
    print(f"\n{label}:  no = {lo:.1f}   yes = {hi:.1f}   ({pct:+.1f}%)")
    summary[col] = {"no": lo, "yes": hi, "pct_change": round(pct, 1)}

effect("is_promo", "Promotion running")
effect("is_school_holiday", "School holiday")
effect("is_payday", "Payday window (28th-2nd)")

by_dow = df.groupby("day_of_week")["daily_sales"].mean().round(1)
print("\nAverage sales by day of week:")
for d, v in by_dow.items():
    print(f"  {DAY_NAMES[d]:<10} {v}")
summary["by_day_of_week"] = {DAY_NAMES[d]: float(v) for d, v in by_dow.items()}

corr = df[["daily_sales", "day_of_week", "is_payday", "is_promo",
           "is_school_holiday", "month"]].corr()["daily_sales"].drop("daily_sales")
print("\nCorrelation with daily_sales:")
print(corr.round(3).to_string())
summary["correlation"] = {k: round(float(v), 3) for k, v in corr.items()}

with open("figures/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---- charts ----------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(df["date"], df["daily_sales"], label="Daily sales", alpha=0.5)
plt.plot(df["date"], df["rolling_7"], label="7-day average", linewidth=2)
plt.title("Daily sales over time"); plt.xlabel("Date"); plt.ylabel("Sales")
plt.legend(); plt.tight_layout(); plt.savefig("figures/01_sales_trend.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
plt.bar([DAY_NAMES[d] for d in by_dow.index], by_dow.values, color="#b5451b")
plt.title("Average sales by day of week"); plt.ylabel("Average sales")
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig("figures/02_sales_by_weekday.png", dpi=150); plt.close()

labels = ["Promo", "School holiday", "Payday window"]
cols = ["is_promo", "is_school_holiday", "is_payday"]
no = [summary[c]["no"] for c in cols]; yes = [summary[c]["yes"] for c in cols]
x = range(len(cols))
plt.figure(figsize=(8, 4))
plt.bar([i - 0.2 for i in x], no, width=0.4, label="No", color="#9aa5b1")
plt.bar([i + 0.2 for i in x], yes, width=0.4, label="Yes", color="#b5451b")
plt.xticks(list(x), labels); plt.ylabel("Average sales"); plt.legend()
plt.title("Effect of external factors on average sales")
plt.tight_layout(); plt.savefig("figures/03_factor_effects.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
c = corr.sort_values()
plt.barh(c.index, c.values, color="#534ab7")
plt.title("Correlation with daily sales"); plt.xlabel("Pearson r")
plt.tight_layout(); plt.savefig("figures/04_correlation.png", dpi=150); plt.close()

print("\nSaved charts and eda_summary.json in the figures/ folder." if CHARTS_OK else "\nSaved eda_summary.json (charts skipped).")
