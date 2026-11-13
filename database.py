"""Database connection (Neon PostgreSQL) and the sales table definition."""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, Date
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Create a .env file (see .env.example) "
        "or set it as an environment variable on your hosting platform."
    )

# Accept whatever Neon gives you and make sure SQLAlchemy uses the psycopg2 driver.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql+psycopg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Sale(Base):
    __tablename__ = "burger_sales"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    day_of_week = Column(Integer, nullable=False)        # 1 = Monday ... 7 = Sunday
    is_payday = Column(Integer, nullable=False)
    is_public_holiday = Column(Integer, nullable=False)
    is_event = Column(Integer, nullable=False)
    is_ntsa_crackdown = Column(Integer, nullable=False)
    rain_mm = Column(Float, nullable=False)
    burgers_sold = Column(Integer, nullable=False)
