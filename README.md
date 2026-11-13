# Chef Rico's Food Truck - Cloud-Based Machine Learning Demand Forecasting System

Forecasts how many "Midnight Burgers" Chef Rico should prepare each night, using weather (rainfall), Nairobi events,
NTSA crackdowns, pay-days, public holidays and day of week. Forecasts are served by a cloud API and shown on a web dashboard.

**Pipeline:** case-study dataset -> cleaning and feature engineering (Python, Pandas) -> model training and comparison
(scikit-learn) -> FastAPI backend -> Neon PostgreSQL (cloud database) -> web dashboard (HTML, CSS, JavaScript, live Open-Meteo weather)
-> deployed on Render via GitHub.

## Project files
| File | Purpose |
|---|---|
| `generate_dataset.py` | Builds the case-study dataset (`chef_rico_sales_data.csv`) from documented assumptions |
| `explore_data.py` | Exploratory data analysis, charts saved to `figures/` |
| `train_model.py` | Chronological train/validation/test split, model comparison, saves `demand_model.json` |
| `load_to_db.py` | Loads the dataset into the cloud PostgreSQL database |
| `app.py`, `database.py`, `features.py` | FastAPI backend, database model, shared feature logic |
| `dashboard.html` | Web dashboard served by the API |
| `load_test.py` | Simple load test (latency and throughput) |
| `prepare_pizza_data.py`, `validate_on_real_data.py` | Optional: check the same method on a real Kaggle restaurant dataset (Pizza Place Sales) |
| `Dockerfile` | Container image for the API |

## Run it locally
```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python load_to_db.py
python -m uvicorn app:app --reload
```
Open http://127.0.0.1:8000 (dashboard) or http://127.0.0.1:8000/docs (API documentation).
Create a `.env` file from `.env.example` with your own database connection string first.
To rebuild everything from scratch: `python generate_dataset.py`, `python explore_data.py`, `python train_model.py`.

## API
`GET /health` | `GET /model-info` | `POST /predict` | `POST /sales` | `GET /sales`

## Data note
No real point-of-sale records exist for the case study and no public dataset combines burgers sold with Nairobi weather,
NTSA crackdowns and events, so nightly demand is simulated from transparent, documented assumptions (`figures/data_assumptions.json`).
Records captured through the dashboard can replace the simulated history. `prepare_pizza_data.py` and `validate_on_real_data.py` check the same method on a real Kaggle restaurant dataset (Pizza Place Sales).
