# AQI Web App (Feature Store + Dashboard)

## Overview

1. **Feature Store** (`feature_store.py`): Loads the trained model, scaler, imputer, and feature list from `models/` (and config). Used by the API.
2. **FastAPI backend** (`api_backend.py`): Exposes the model and features via REST. Endpoints: `/health`, `/features`, `/predict`, `/predict/single`, `/predict/next_3_days`.
3. **Streamlit dashboard** (`dashboard.py`): Simple, descriptive UI that calls the API to show next 3 days forecast and custom predictions.
4. **Gradio app** (`gradio_app.py`): Alternative UI (single prediction + next 3 days) that also calls the API.

## Run order

1. Train the model once (if not already done):
   ```bash
   python eda_and_train.py
   ```

2. Start the **API** (loads model from Feature Store):
   ```bash
   uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000
   ```

3. Start the **Streamlit** dashboard:
   ```bash
   streamlit run dashboard.py
   ```
   Open the URL shown (default http://localhost:8501). Use the sidebar to change API URL if needed.

4. Optional – **Gradio** (API must be running):
   ```bash
   gradio gradio_app.py
   ```
   Or: `python gradio_app.py`. Default: http://localhost:7860.

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and model name |
| GET | `/features` | Feature list and model metadata from Feature Store |
| POST | `/predict` | Body: `{"features": [ {...}, ... ]}` → AQI predictions |
| POST | `/predict/single` | Body: `{"features": {...}}` → single AQI prediction |
| GET | `/predict/next_3_days` | Fetches Open-Meteo forecast and returns hourly AQI for next 3 days |

## Dashboard tabs

- **Next 3 days forecast**: Button to fetch and display hourly AQI table, chart, and daily summary.
- **Custom prediction**: Input key features (PM2.5, PM10, weather, time) and get one AQI prediction.
- **Feature list**: Shows feature names and model info from the Feature Store.
