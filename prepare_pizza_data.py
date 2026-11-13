"""
OPTIONAL - Turn the Kaggle "Pizza Place Sales" files into one row per day.

Source (real data from one small restaurant, 2015) - search Kaggle for "Pizza Place Sales"
(by Mysar Ahmad Bhat) and download it. Put these two files in this folder:
    orders.csv          (order_id, date, time)
    order_details.csv   (order_details_id, order_id, pizza_id, quantity)

Run:   python prepare_pizza_data.py
Then:  python validate_on_real_data.py real_pizza_daily.csv --date-col date --target-col pizzas_sold

If you downloaded a single merged file instead (columns such as order_date and quantity), run:
       python prepare_pizza_data.py merged_file.csv
"""
import os
import sys
import pandas as pd

if len(sys.argv) > 1:                                     # single merged file
    df = pd.read_csv(sys.argv[1])
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in ("order_date", "date") if c in df.columns), None)
    if date_col is None or "quantity" not in df.columns:
        sys.exit(f"Need a date column (order_date/date) and a quantity column. Found: {list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    daily = df.dropna(subset=["date"]).groupby(df["date"].dt.date)["quantity"].sum()
else:                                                     # the two separate Kaggle files
    for f in ("orders.csv", "order_details.csv"):
        if not os.path.exists(f):
            sys.exit(f"{f} not found in this folder. Download 'Pizza Place Sales' from Kaggle and copy the CSV files here.")
    orders = pd.read_csv("orders.csv")
    details = pd.read_csv("order_details.csv")
    orders.columns = [c.strip().lower() for c in orders.columns]
    details.columns = [c.strip().lower() for c in details.columns]
    merged = details.merge(orders[["order_id", "date"]], on="order_id", how="inner")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    daily = merged.dropna(subset=["date"]).groupby(merged["date"].dt.date)["quantity"].sum()

out = daily.rename("pizzas_sold").reset_index()
out.columns = ["date", "pizzas_sold"]
out.to_csv("real_pizza_daily.csv", index=False)
print(f"Saved {len(out)} days -> real_pizza_daily.csv ({out['date'].min()} to {out['date'].max()})")
print(f"Pizzas sold per day: mean {out['pizzas_sold'].mean():.1f}, min {out['pizzas_sold'].min()}, max {out['pizzas_sold'].max()}")
print(out.head(5).to_string(index=False))
