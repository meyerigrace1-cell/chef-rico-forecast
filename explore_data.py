import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("chef_rico_sales_data.csv")
df["date"] = pd.to_datetime(df["date"])

# --- Pattern checks (Specific Objective 1) ---
print("Average burgers sold: weekend vs weekday")
print(df.groupby("is_weekend")["burgers_sold"].mean())

print("\nAverage burgers sold: payday vs non-payday")
print(df.groupby("is_payday")["burgers_sold"].mean())

print("\nAverage burgers sold: rainy vs not")
print(df.groupby("is_rainy")["burgers_sold"].mean())

print("\nAverage burgers sold: event vs no event")
print(df.groupby("is_event")["burgers_sold"].mean())

# --- Feature engineering (Specific Objective 2) ---
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day
df["rolling_7day_avg"] = df["burgers_sold"].rolling(window=7, min_periods=1).mean()

df.to_csv("chef_rico_features.csv", index=False)
print("\nSaved engineered dataset -> chef_rico_features.csv")

# --- Visualization ---
plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["burgers_sold"], label="Daily sales")
plt.plot(df["date"], df["rolling_7day_avg"], label="7-day average", linewidth=2)
plt.legend()
plt.title("Chef Rico's Food Truck — Daily Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Burgers Sold")
plt.tight_layout()
plt.savefig("demand_trend.png")
print("Saved chart -> demand_trend.png")