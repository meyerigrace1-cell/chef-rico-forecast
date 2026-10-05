"""
Generates synthetic historical sales data for Chef Rico's Food Truck.
Simulates 9 months of nightly burger sales, with demand affected by:
  - weekends (Fri/Sat nightlife nights)
  - paydays (end/start of month)
  - rain (reduces nightlife turnout)
  - Nairobi events (boost turnout)
  - NTSA crackdowns (reduce turnout)

Run with:  python generate_data.py
Output:    chef_rico_sales_data.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

start_date = datetime(2025, 10, 1)
days = 270  # ~9 months of nightly data

records = []
for i in range(days):
    date = start_date + timedelta(days=i)
    day_of_week = date.weekday()  # Monday=0 ... Sunday=6

    is_weekend = day_of_week in [4, 5]              # Fri, Sat
    is_payday = date.day in [28, 29, 30, 31, 1, 2]   # month-end/start
    is_rainy = np.random.random() < 0.25             # ~25% of nights
    is_event = np.random.random() < 0.10              # ~10% of nights
    ntsa_crackdown = np.random.random() < 0.07        # ~7% of nights

    demand = 80  # baseline burgers sold on an average night

    if is_weekend:
        demand += 40
    if is_payday:
        demand += 50
    if is_event:
        demand += 60
    if is_rainy:
        demand -= 45
    if ntsa_crackdown:
        demand -= 30

    demand += np.random.normal(0, 10)  # natural day-to-day noise
    demand = max(0, round(demand))

    records.append({
        "date": date.strftime("%Y-%m-%d"),
        "day_of_week": date.strftime("%A"),
        "is_weekend": int(is_weekend),
        "is_payday": int(is_payday),
        "is_rainy": int(is_rainy),
        "is_event": int(is_event),
        "ntsa_crackdown": int(ntsa_crackdown),
        "burgers_sold": demand,
    })

df = pd.DataFrame(records)
df.to_csv("chef_rico_sales_data.csv", index=False)

print(f"Generated {len(df)} days of data -> chef_rico_sales_data.csv")
print(df.head(10))
print("\nQuick stats:")
print(df["burgers_sold"].describe())
