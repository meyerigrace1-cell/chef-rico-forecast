# Chef Rico's Food Truck - Cloud-Based Machine Learning Demand Forecasting System

A cloud-based machine learning system that forecasts daily demand for a small food business and presents the forecasts in a web dashboard.

**Pipeline:** Kaggle dataset -> cleaning and feature engineering (Python, Pandas) -> model training and evaluation (scikit-learn) -> FastAPI backend -> Neon PostgreSQL (cloud database) -> web dashboard (HTML, CSS, JavaScript) -> deployed on Render via GitHub.

## Project files
| File | Purpose |
|---|---|
| `prepare_kaggle_data.py` | Cleans the raw Kaggle `train.csv` into `chef_rico_sales_data.csv` |
| `explore_data.py` | Exploratory data analysis, charts saved to `figures/` |
| `train_model.py` | Chronological train/validation/test split, model comparison, saves `demand_model.json` |
| `load_to_db.py` | Loads the dataset into the cloud PostgreSQL database |
| `app.py`, `database.py`, `features.py` | FastAPI backend, database model, shared feature logic |
| `dashboard.html` | Web dashboard served by the API |
| `load_test.py` | Simple load test (latency and throughput) |
| `Dockerfile` | Container image for the API |

## Run it locally
```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python prepare_kaggle_data.py
python explore_data.py
python train_model.py
python load_to_db.py
uvicorn app:app --reload
```
Then open http://127.0.0.1:8000 (dashboard) or http://127.0.0.1:8000/docs (API documentation).
Create a `.env` file from `.env.example` with your own database connection string first.

## API
`GET /health` | `GET /model-info` | `POST /predict` | `POST /sales` | `GET /sales`

## Data note
No real point-of-sale records exist for the case-study business, so the public Kaggle *Rossmann Store Sales* dataset (daily sales for one store) is used as a proxy demand time series.
