import os
from datetime import date
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    day_of_week = Column(String)
    is_weekend = Column(Integer)
    is_payday = Column(Integer)
    is_rainy = Column(Integer)
    is_event = Column(Integer)
    ntsa_crackdown = Column(Integer)
    burgers_sold = Column(Integer)

app = FastAPI(title="Chef Rico Demand Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_data = np.load("demand_model.npz", allow_pickle=True)
coefficients = model_data["coefficients"]

class SaleIn(BaseModel):
    date: date
    day_of_week: str
    is_weekend: int
    is_payday: int
    is_rainy: int
    is_event: int
    ntsa_crackdown: int
    burgers_sold: int

class PredictIn(BaseModel):
    is_weekend: int
    is_payday: int
    is_rainy: int
    is_event: int
    ntsa_crackdown: int
    month: int
    day_of_month: int

from fastapi.responses import FileResponse

@app.get("/")
def root():
    return FileResponse("dashboard.html")

@app.post("/sales")
def add_sale(sale: SaleIn):
    db = SessionLocal()
    db_sale = Sale(**sale.dict())
    db.add(db_sale)
    db.commit()
    db.refresh(db_sale)
    db.close()
    return {"message": "Sale recorded", "id": db_sale.id}

@app.get("/sales")
def list_sales():
    db = SessionLocal()
    sales = db.query(Sale).order_by(Sale.date).all()
    db.close()
    return [
        {
            "date": str(s.date), "day_of_week": s.day_of_week,
            "is_weekend": s.is_weekend, "is_payday": s.is_payday,
            "is_rainy": s.is_rainy, "is_event": s.is_event,
            "ntsa_crackdown": s.ntsa_crackdown, "burgers_sold": s.burgers_sold,
        } for s in sales
    ]

@app.post("/predict")
def predict(input: PredictIn):
    x = np.array([1, input.is_weekend, input.is_payday, input.is_rainy,
                  input.is_event, input.ntsa_crackdown, input.month,
                  input.day_of_month], dtype=float)
    prediction = float(np.dot(x, coefficients))
    return {"predicted_burgers_sold": round(prediction, 1)}