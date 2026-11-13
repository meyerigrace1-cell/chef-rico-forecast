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

from features import DAY_NAMES, RAINY_THRESHOLD_MM

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("chef_rico_sales_data.csv", parse_dates=["date"])
df["month"] = df["date"].dt.month
df["is_rainy"] = (df["rain_mm"] >= RAINY_THRESHOLD_MM).astype(int)
df["rolling_7"] = df["burgers_sold"].rolling(7, min_periods=1).mean()

print(f"Nights: {len(df)}   Period: {df['date'].min().date()} -> {df['date'].max().date()}")
print("\nMissing values per column:")
print(df.drop(columns=["rolling_7"]).isna().sum().to_string())
print("\nDescriptive statistics (burgers_sold):")
print(df["burgers_sold"].describe().round(1).to_string())

summary = {"nights": int(len(df)), "mean_burgers": round(float(df["burgers_sold"].mean()), 1),
           "median_burgers": float(df["burgers_sold"].median()),
           "std_burgers": round(float(df["burgers_sold"].std()), 1),
           "min_burgers": int(df["burgers_sold"].min()), "max_burgers": int(df["burgers_sold"].max())}

FACTORS = [("is_rainy", "Rainy night (>= 2 mm)"), ("is_event", "Nairobi event nearby"),
           ("is_ntsa_crackdown", "NTSA crackdown"), ("is_payday", "Pay-day window (28th-2nd)"),
           ("is_public_holiday", "Public holiday")]
for col, label in FACTORS:
    g = df.groupby(col)["burgers_sold"].mean()
    lo, hi = float(g.get(0, float("nan"))), float(g.get(1, float("nan")))
    pct = (hi - lo) / lo * 100
    print(f"\n{label}:  no = {lo:.1f}   yes = {hi:.1f}   ({pct:+.1f}%)   nights with factor = {int(df[col].sum())}")
    summary[col] = {"label": label, "no": round(lo, 1), "yes": round(hi, 1),
                    "pct_change": round(pct, 1), "nights": int(df[col].sum())}

by_dow = df.groupby("day_of_week")["burgers_sold"].mean().round(1)
print("\nAverage burgers sold by day of week:")
for d, v in by_dow.items():
    print(f"  {DAY_NAMES[d]:<10} {v}")
summary["by_day_of_week"] = {DAY_NAMES[d]: float(v) for d, v in by_dow.items()}

corr = df[["burgers_sold", "day_of_week", "is_payday", "is_public_holiday", "is_event",
           "is_ntsa_crackdown", "rain_mm"]].corr()["burgers_sold"].drop("burgers_sold")
print("\nCorrelation with burgers_sold:")
print(corr.round(3).to_string())
summary["correlation"] = {k: round(float(v), 3) for k, v in corr.items()}

with open("figures/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---- charts ----------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(df["date"], df["burgers_sold"], label="Nightly burgers sold", alpha=0.45)
plt.plot(df["date"], df["rolling_7"], label="7-night average", linewidth=2)
plt.title("Burgers sold per night over time"); plt.xlabel("Date"); plt.ylabel("Burgers")
plt.legend(); plt.tight_layout(); plt.savefig("figures/01_sales_trend.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
plt.bar([DAY_NAMES[d] for d in by_dow.index], by_dow.values, color="#b5451b")
plt.title("Average burgers sold by day of week"); plt.ylabel("Average burgers")
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig("figures/02_sales_by_weekday.png", dpi=150); plt.close()

labels = [l for _, l in FACTORS]
no = [summary[c]["no"] for c, _ in FACTORS]; yes = [summary[c]["yes"] for c, _ in FACTORS]
x = range(len(FACTORS))
plt.figure(figsize=(10, 4.5))
plt.bar([i - 0.2 for i in x], no, width=0.4, label="Factor absent", color="#9aa5b1")
plt.bar([i + 0.2 for i in x], yes, width=0.4, label="Factor present", color="#b5451b")
plt.xticks(list(x), labels, rotation=15, ha="right"); plt.ylabel("Average burgers sold"); plt.legend()
plt.title("Effect of external factors on average nightly demand")
plt.tight_layout(); plt.savefig("figures/03_factor_effects.png", dpi=150); plt.close()

plt.figure(figsize=(6.5, 4))
c = corr.sort_values()
plt.barh(c.index, c.values, color="#534ab7")
plt.title("Correlation with burgers sold"); plt.xlabel("Pearson r")
plt.tight_layout(); plt.savefig("figures/04_correlation.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
plt.scatter(df["rain_mm"], df["burgers_sold"], s=10, alpha=0.4, color="#185fa5")
plt.title("Rainfall vs burgers sold"); plt.xlabel("Rainfall that night (mm)"); plt.ylabel("Burgers sold")
plt.tight_layout(); plt.savefig("figures/08_rain_vs_sales.png", dpi=150); plt.close()

print("\nSaved charts and eda_summary.json in the figures/ folder." if CHARTS_OK else "\nSaved eda_summary.json (charts skipped).")
