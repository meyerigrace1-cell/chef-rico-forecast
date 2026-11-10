"""
Shared feature logic.

Both the training script and the live API import this file, so the model is
always fed features in exactly the same way it was trained (no train/serve mismatch).
"""
import datetime as dt
import math

PAYDAY_DAYS = {28, 29, 30, 31, 1, 2}   # month-end / month-start pay window
DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
             5: "Friday", 6: "Saturday", 7: "Sunday"}

# Monday (dow_1) is the baseline day, so it has no column of its own.
# season_sin / season_cos describe the time of year as a smooth yearly cycle.
FEATURES = [
    "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "dow_7",
    "is_payday", "is_promo", "is_school_holiday", "season_sin", "season_cos",
]


def derive(date: dt.date):
    """Return (day_of_week 1-7, month 1-12, is_payday 0/1) for a calendar date."""
    dow = date.isoweekday()
    return dow, date.month, 1 if date.day in PAYDAY_DAYS else 0


def feature_dict(date: dt.date, is_promo: int, is_school_holiday: int) -> dict:
    dow, _, is_payday = derive(date)
    angle = 2 * math.pi * date.timetuple().tm_yday / 365.25
    values = {f"dow_{d}": 1 if dow == d else 0 for d in range(2, 8)}
    values.update({
        "is_payday": is_payday,
        "is_promo": int(is_promo),
        "is_school_holiday": int(is_school_holiday),
        "season_sin": round(math.sin(angle), 6),
        "season_cos": round(math.cos(angle), 6),
    })
    return values
