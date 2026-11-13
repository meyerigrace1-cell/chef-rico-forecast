"""
Shared feature logic - imported by the training scripts AND the live API so the
model always receives features exactly as it was trained (no train/serve mismatch).
Uses only the Python standard library.
"""
import datetime as dt

PAYDAY_DAYS = {28, 29, 30, 31, 1, 2}     # month-end / month-start pay window
DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
             5: "Friday", 6: "Saturday", 7: "Sunday"}

# Monday (dow_1) is the baseline day, so it has no column of its own.
FEATURES = [
    "dow_2", "dow_3", "dow_4", "dow_5", "dow_6", "dow_7",
    "is_payday", "is_public_holiday", "is_event", "is_ntsa_crackdown", "rain_mm",
]

RAIN_PRESETS_MM = {"dry": 0.0, "light": 3.0, "heavy": 15.0}   # used by the dashboard
RAINY_THRESHOLD_MM = 2.0                                      # "rainy night" for charts


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def is_public_holiday(date: dt.date) -> int:
    """Kenyan public holidays that fall on fixed dates, plus Good Friday / Easter Monday."""
    fixed = {(1, 1), (5, 1), (6, 1), (10, 10), (10, 20), (12, 12), (12, 25), (12, 26)}
    if (date.month, date.day) in fixed:
        return 1
    easter = easter_sunday(date.year)
    return 1 if date in (easter - dt.timedelta(days=2), easter + dt.timedelta(days=1)) else 0


def derive(date: dt.date):
    """Return (day_of_week 1-7, is_payday 0/1, is_public_holiday 0/1)."""
    return date.isoweekday(), (1 if date.day in PAYDAY_DAYS else 0), is_public_holiday(date)


def feature_dict(date: dt.date, is_event: int, is_ntsa_crackdown: int, rain_mm: float) -> dict:
    dow, payday, holiday = derive(date)
    values = {f"dow_{d}": 1 if dow == d else 0 for d in range(2, 8)}
    values.update({
        "is_payday": payday,
        "is_public_holiday": holiday,
        "is_event": int(is_event),
        "is_ntsa_crackdown": int(is_ntsa_crackdown),
        "rain_mm": float(rain_mm),
    })
    return values
