"""
FastAPI backend: loads model and features from Feature Store,
exposes /features, /predict, and /predict/next_3_days.
Run: uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from feature_store import load_store, get_features, predict

app = FastAPI(
    title="AQI Prediction API",
    description="Loads model from Feature Store and serves AQI (1-5) predictions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load Feature Store at startup
@app.on_event("startup")
def startup():
    try:
        load_store()
    except FileNotFoundError as e:
        print(f"Warning: {e}. Run eda_and_train.py first.")


class PredictRequest(BaseModel):
    features: List[dict]  # list of dicts with feature names as keys


class PredictSingleRequest(BaseModel):
    features: dict  # single row


@app.get("/health")
def health():
    """Health check."""
    try:
        store = load_store()
        return {"status": "ok", "model": store["metadata"].get("best_model_name")}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/features")
def features():
    """Return feature schema and model info from Feature Store."""
    try:
        return get_features()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict_endpoint(body: PredictRequest):
    """Compute AQI predictions for one or more feature rows."""
    try:
        results = predict(body.features)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/single")
def predict_single(body: PredictSingleRequest):
    """Predict AQI for a single feature row."""
    try:
        results = predict(body.features)
        return {"prediction": results[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _fetch_next_3_days():
    """Fetch forecast from Open-Meteo and return predictions using Feature Store."""
    from predict_next_3days import fetch_forecast_hourly, build_features_from_forecast

    aq_data, w_data = fetch_forecast_hourly()
    if not aq_data or "hourly" not in aq_data:
        return None
    X, timestamps = build_features_from_forecast(
        aq_data["hourly"],
        w_data.get("hourly", {}) if w_data else {},
    )
    if X is None or X.empty:
        return None

    rows = X.to_dict("records")
    predictions = predict(rows)
    out = []
    for ts, pred in zip(timestamps, predictions):
        out.append({
            "timestamp_utc": ts.isoformat(),
            "aqi": pred["aqi"],
            "aqi_label": pred["aqi_label"],
            "aqi_continuous": round(pred["aqi_continuous"], 2),
        })
    return out


@app.get("/predict/next_3_days")
def predict_next_3_days_endpoint():
    """Fetch next 3 days hourly forecast from Open-Meteo and return AQI predictions."""
    try:
        result = _fetch_next_3_days()
        if result is None:
            raise HTTPException(status_code=502, detail="Failed to fetch forecast from Open-Meteo.")
        return {"predictions": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
