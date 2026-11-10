"""
Chef Rico Demand Forecast API (FastAPI).

Endpoints
  GET  /            web dashboard
  GET  /health      monitoring: API, database and model status
  GET  /model-info  model details and accuracy
  POST /predict     forecast daily sales for a date
  POST /sales       store a daily sales record in the cloud database
  GET  /sales       list stored sales records
  GET  /docs        interactive API documentation (Swagger UI)
"""
import datetime as dt
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from database import SessionLocal, engine, Base, Sale
from features import DAY_NAMES, derive, feature_dict

BASE_DIR = Path(__file__).parent

try:
    Base.metadata.create_all(bind=engine)          # creates the table on first run
except Exception as exc:                           # keep the API up; /health reports it
    print(f"[warning] could not reach the database at startup: {exc}")


def load_model():
    path = BASE_DIR / "demand_model.json"
    if not path.exists():
        print("[warning] demand_model.json not found - run train_model.py")
        return None
    return json.loads(path.read_text())


MODEL = load_model()

app = FastAPI(title="Chef Rico Demand Forecast API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictIn(BaseModel):
    date: dt.date
    is_promo: int = Field(0, ge=0, le=1)
    is_school_holiday: int = Field(0, ge=0, le=1)


class SaleIn(BaseModel):
    date: dt.date
    is_promo: int = Field(0, ge=0, le=1)
    is_school_holiday: int = Field(0, ge=0, le=1)
    daily_sales: int = Field(..., ge=0)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(BASE_DIR / "dashboard.html")


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        database = "connected"
    except Exception:
        database = "unreachable"
    return {
        "status": "ok" if database == "connected" and MODEL else "degraded",
        "database": database,
        "model_loaded": MODEL is not None,
        "model_version": MODEL["version"] if MODEL else None,
    }


@app.get("/model-info")
def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not found. Run train_model.py and redeploy.")
    return MODEL


@app.post("/predict")
def predict(body: PredictIn):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not found. Run train_model.py and redeploy.")
    values = feature_dict(body.date, body.is_promo, body.is_school_holiday)
    prediction = MODEL["intercept"] + sum(MODEL["coefficients"][f] * values[f] for f in MODEL["features"])
    prediction = max(0.0, prediction)
    mae = MODEL["metrics"]["mae"]
    dow, _, _ = derive(body.date)
    warning = None
    if dow in MODEL.get("days_without_data", []):
        warning = f"The training data has no trading on {DAY_NAMES[dow]}s, so treat this forecast with caution."
    return {
        "date": str(body.date),
        "day": DAY_NAMES[dow],
        "predicted_sales": round(prediction),
        "likely_range_low": round(max(0.0, prediction - mae)),
        "likely_range_high": round(prediction + mae),
        "typical_error_mae": mae,
        "model": MODEL["model_name"],
        "warning": warning,
    }


@app.post("/sales")
def add_sale(body: SaleIn):
    dow, _, payday = derive(body.date)
    with SessionLocal() as db:
        record = Sale(
            date=body.date, day_of_week=dow, is_payday=payday,
            is_promo=body.is_promo, is_school_holiday=body.is_school_holiday,
            daily_sales=body.daily_sales,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return {"message": "Sale recorded", "id": record.id}


@app.get("/sales")
def list_sales(limit: int = 2000):
    with SessionLocal() as db:
        rows = db.query(Sale).order_by(Sale.date).limit(limit).all()
    return [
        {
            "id": r.id, "date": str(r.date), "day": DAY_NAMES[r.day_of_week],
            "is_payday": r.is_payday, "is_promo": r.is_promo,
            "is_school_holiday": r.is_school_holiday,
            "daily_sales": r.daily_sales,
        }
        for r in rows
    ]
