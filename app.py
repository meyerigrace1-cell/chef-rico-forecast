"""
Chef Rico Demand Forecast API (FastAPI).

Endpoints
  GET  /            web dashboard
  GET  /health      monitoring: API, database and model status
  GET  /model-info  model details and accuracy
  POST /predict     forecast burgers for a night
  POST /sales       store a night's actual sales in the cloud database
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

LABELS = {"is_payday": "Pay-day window", "is_public_holiday": "Public holiday",
          "is_event": "Nairobi event nearby", "is_ntsa_crackdown": "NTSA crackdown",
          "rain_mm": "Rainfall"}

app = FastAPI(title="Chef Rico Demand Forecast API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictIn(BaseModel):
    date: dt.date
    is_event: int = Field(0, ge=0, le=1)
    is_ntsa_crackdown: int = Field(0, ge=0, le=1)
    rain_mm: float = Field(0.0, ge=0, le=300)


class SaleIn(BaseModel):
    date: dt.date
    is_event: int = Field(0, ge=0, le=1)
    is_ntsa_crackdown: int = Field(0, ge=0, le=1)
    rain_mm: float = Field(0.0, ge=0, le=300)
    burgers_sold: int = Field(..., ge=0, le=5000)


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
    values = feature_dict(body.date, body.is_event, body.is_ntsa_crackdown, body.rain_mm)
    contributions = {f: MODEL["coefficients"][f] * values[f] for f in MODEL["features"]}
    prediction = max(0.0, MODEL["intercept"] + sum(contributions.values()))
    mae = MODEL["metrics"]["mae"]
    dow, _, _ = derive(body.date)

    # "Why this forecast": the factors that are switched on, with their effect in burgers
    drivers = [{"factor": LABELS[f], "effect": round(contributions[f])}
               for f in LABELS if abs(contributions.get(f, 0)) >= 0.5]
    day_effect = sum(contributions[f] for f in MODEL["features"] if f.startswith("dow_"))
    if abs(day_effect) >= 0.5:
        drivers.insert(0, {"factor": f"{DAY_NAMES[dow]} (vs Monday)", "effect": round(day_effect)})

    return {
        "date": str(body.date),
        "day": DAY_NAMES[dow],
        "predicted_burgers": round(prediction),
        "likely_range_low": round(max(0.0, prediction - mae)),
        "likely_range_high": round(prediction + mae),
        "typical_error_mae": mae,
        "drivers": drivers,
        "model": MODEL["model_name"],
    }


@app.post("/sales")
def add_sale(body: SaleIn):
    dow, payday, holiday = derive(body.date)
    with SessionLocal() as db:
        record = Sale(
            date=body.date, day_of_week=dow, is_payday=payday, is_public_holiday=holiday,
            is_event=body.is_event, is_ntsa_crackdown=body.is_ntsa_crackdown,
            rain_mm=body.rain_mm, burgers_sold=body.burgers_sold,
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
            "is_payday": r.is_payday, "is_public_holiday": r.is_public_holiday,
            "is_event": r.is_event, "is_ntsa_crackdown": r.is_ntsa_crackdown,
            "rain_mm": r.rain_mm, "burgers_sold": r.burgers_sold,
        }
        for r in rows
    ]
