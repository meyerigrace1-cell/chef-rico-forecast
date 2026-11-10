"""
Step 4 - Load the prepared dataset into the cloud database (Neon PostgreSQL).
Creates the table if needed, clears old rows, then inserts the new data.

Run:  python load_to_db.py
"""
import pandas as pd
from database import SessionLocal, engine, Base, Sale

Base.metadata.create_all(bind=engine)

df = pd.read_csv("chef_rico_sales_data.csv", parse_dates=["date"])
records = [
    {
        "date": r.date.date(),
        "day_of_week": int(r.day_of_week),
        "is_payday": int(r.is_payday),
        "is_promo": int(r.is_promo),
        "is_school_holiday": int(r.is_school_holiday),
        "daily_sales": int(r.daily_sales),
    }
    for r in df.itertuples()
]

with SessionLocal() as db:
    removed = db.query(Sale).delete()
    db.bulk_insert_mappings(Sale, records)
    db.commit()
    total = db.query(Sale).count()

print(f"Removed {removed} old rows, inserted {len(records)} rows. Table now holds {total} rows.")
