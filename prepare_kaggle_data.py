"""
Step 1 - Turn the raw Kaggle file into the project's dataset.

Source : Kaggle "Rossmann Store Sales" (file: train.csv)
Used as : a real-world daily retail sales dataset standing in for Chef Rico's
          food-truck records (no real POS records exist for the case study).
Output : chef_rico_sales_data.csv

Run:  python prepare_kaggle_data.py
"""
import os
import sys
import datetime as dt
import pandas as pd
from features import derive

STORE_ID = 1                                   # one store = one business
CANDIDATES = ["train.csv", "kaggle_train.csv"]

path = next((p for p in CANDIDATES if os.path.exists(p)), None)
if path is None:
    sys.exit("Could not find train.csv in this folder. Download it from Kaggle "
             "(Rossmann Store Sales -> Data tab) and put it next to this script.")

raw = pd.read_csv(path, low_memory=False)
print(f"Loaded {len(raw):,} raw rows from {path}")
print("Columns:", list(raw.columns))

# --- cleaning --------------------------------------------------------------
df = raw[raw["Store"] == STORE_ID].copy()
print(f"Rows for store {STORE_ID}: {len(df):,}")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

holiday_days = int((df["StateHoliday"].astype(str).str.strip() != "0").sum())
before = len(df)
df = df[(df["Open"] == 1) & (df["Sales"] > 0)]          # closed days (Sundays, state holidays) carry no demand signal
print(f"State-holiday days in raw data for this store: {holiday_days} (the store is closed on them)")
print(f"Removed {before - len(df)} closed / zero-sales days -> {len(df)} trading days")

df = df.drop_duplicates(subset="Date")
assert df[["Date", "Sales", "Promo", "StateHoliday", "SchoolHoliday"]].isna().sum().sum() == 0, \
    "Unexpected missing values"

# --- build the project dataset --------------------------------------------
out = pd.DataFrame()
out["date"] = df["Date"].dt.strftime("%Y-%m-%d")
out["day_of_week"] = [derive(d.date())[0] for d in df["Date"]]
out["is_payday"] = [derive(d.date())[2] for d in df["Date"]]
out["is_promo"] = df["Promo"].astype(int).values
out["is_school_holiday"] = df["SchoolHoliday"].astype(int).values
out["daily_sales"] = df["Sales"].astype(int).values

out.to_csv("chef_rico_sales_data.csv", index=False)

print(f"\nSaved {len(out)} rows -> chef_rico_sales_data.csv")
print(f"Date range: {out['date'].min()} to {out['date'].max()}")
print(out.head(8).to_string(index=False))
print("\nDaily sales summary:")
print(out["daily_sales"].describe().round(1))
