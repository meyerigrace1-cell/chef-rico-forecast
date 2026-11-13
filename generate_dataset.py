"""
Step 1 - Build the case-study dataset for Chef Rico's food truck.

WHY THIS EXISTS
No real point-of-sale records exist for the case study, and no public (Kaggle) dataset
contains burgers sold together with Nairobi weather, NTSA crackdowns and Nairobi events.
So the nightly demand is SIMULATED from a transparent rule set that mirrors the problem
statement (rainy nights collapse demand, pay-day Fridays surge, events boost it, NTSA
crackdowns suppress it). The ground-rules are printed and saved to
figures/data_assumptions.json so they can be documented and defended.
Real records captured through the dashboard can replace this file later.

The script is deterministic (fixed random seed): everybody gets the same file.

Run:  python generate_dataset.py
Output: chef_rico_sales_data.csv
"""
import json
import os
import datetime as dt
import numpy as np
import pandas as pd
from features import derive

SEED = 2026
START, END = dt.date(2024, 1, 1), dt.date(2026, 10, 31)

# Approximate Nairobi climate: probability that a night is rainy, by month
# (long rains Mar-May, short rains Oct-Dec).
P_RAIN_BY_MONTH = {1: .18, 2: .17, 3: .32, 4: .55, 5: .45, 6: .14,
                   7: .10, 8: .12, 9: .13, 10: .28, 11: .48, 12: .30}

ASSUMPTIONS = {
    "baseline_burgers_monday": 70,
    "weekday_effect": {"Tuesday": 3, "Wednesday": 8, "Thursday": 20, "Friday": 45, "Saturday": 55, "Sunday": 10},
    "payday_window_28th_to_2nd": 25,
    "payday_friday_extra": 20,
    "public_holiday": 15,
    "nairobi_event_nearby": 50,
    "ntsa_crackdown": -35,
    "rain_per_mm_capped_at_20mm": -2.2,
    "probability_event": {"Fri_Sat": 0.22, "other_nights": 0.07},
    "probability_ntsa_crackdown": {"Fri_Sat": 0.12, "Thu_Sun": 0.06, "other_nights": 0.02},
    "rain_amount_when_raining": "gamma(shape=1.2, scale=8) mm, capped at 60 mm",
    "noise": "normal, standard deviation = 8% of expected demand + 4 burgers",
    "seed": SEED,
}

rng = np.random.default_rng(SEED)
DOW_EFFECT = {1: 0, 2: 3, 3: 8, 4: 20, 5: 45, 6: 55, 7: 10}
rows = []
day = START
while day <= END:
    dow, payday, holiday = derive(day)
    fri_sat = dow in (5, 6)
    event = int(rng.random() < (0.22 if fri_sat else 0.07))
    p_ntsa = 0.12 if fri_sat else (0.06 if dow in (4, 7) else 0.02)
    ntsa = int(rng.random() < p_ntsa)
    raining = rng.random() < P_RAIN_BY_MONTH[day.month]
    rain_mm = round(float(min(60.0, rng.gamma(1.2, 8.0))), 1) if raining else 0.0

    mean = 70 + DOW_EFFECT[dow]
    mean += 25 * payday + (20 if (payday and dow == 5) else 0)
    mean += 15 * holiday + 50 * event - 35 * ntsa
    mean -= 2.2 * min(rain_mm, 20.0)
    burgers = int(max(10, round(rng.normal(mean, 0.08 * max(mean, 0) + 4))))

    rows.append({"date": day.isoformat(), "day_of_week": dow, "is_payday": payday,
                 "is_public_holiday": holiday, "is_event": event, "is_ntsa_crackdown": ntsa,
                 "rain_mm": rain_mm, "burgers_sold": burgers})
    day += dt.timedelta(days=1)

df = pd.DataFrame(rows)
df.to_csv("chef_rico_sales_data.csv", index=False)
os.makedirs("figures", exist_ok=True)
with open("figures/data_assumptions.json", "w") as f:
    json.dump(ASSUMPTIONS, f, indent=2)

print(f"Generated {len(df)} nights ({df['date'].min()} to {df['date'].max()}) -> chef_rico_sales_data.csv")
print(df.head(8).to_string(index=False))
print("\nBurgers sold per night:")
print(df["burgers_sold"].describe().round(1).to_string())
