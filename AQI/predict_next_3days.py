"""
Fetch next 3 days hourly data from Open-Meteo, build features, and predict AQI using trained model.
Run eda_and_train.py first to generate models/scaler and models/best_model.
"""
import os
import math
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib

try:
    from config import (
        LAT, LON, FEATURE_COLS,
        MODEL_DIR, SCALER_PATH, BEST_MODEL_PATH, MODEL_METADATA_PATH
    )
except ImportError:
    LAT = 24.8607
    LON = 67.0011
    FEATURE_COLS = [
        "pm25", "pm10", "pm_ratio", "pm_sum",
        "no2", "so2", "co", "o3",
        "temperature", "humidity", "wind_speed",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "month_sin", "month_cos"
    ]
    MODEL_DIR = "models"
    SCALER_PATH = "models/scaler.joblib"
    BEST_MODEL_PATH = "models/best_model.joblib"
    MODEL_METADATA_PATH = "models/model_metadata.json"


def fetch_forecast_hourly():
    """Fetch next 3 days hourly air quality + weather from Open-Meteo forecast APIs."""
    # Air quality forecast (next 3 days)
    aq_url = "https://air-quality.api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "timezone": "UTC",
        "forecast_days": 3,
    }
    # Weather forecast (next 3 days)
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "UTC",
        "forecast_days": 3,
    }
    
    try:
        r_aq = requests.get(aq_url, params=aq_params, timeout=15)
        r_w = requests.get(w_url, params=w_params, timeout=15)
        aq = r_aq.json() if r_aq.status_code == 200 else None
        w = r_w.json() if r_w.status_code == 200 else None
    except Exception as e:
        print(f"API request failed: {e}")
        return None, None
    
    return aq, w


def build_features_from_forecast(aq_hourly, w_hourly):
    """
    Build same feature matrix as training: one row per hour.
    Uses same column order as FEATURE_COLS.
    """
    times = aq_hourly.get("time", [])
    if not times:
        return None, []
    
    n = len(times)
    
    def get_arr(data, key, default=None):
        if not data or key not in data:
            return [default] * n
        return data[key]
    
    pm25 = get_arr(aq_hourly, "pm2_5")
    pm10 = get_arr(aq_hourly, "pm10")
    no2 = get_arr(aq_hourly, "nitrogen_dioxide")
    so2 = get_arr(aq_hourly, "sulphur_dioxide")
    co = get_arr(aq_hourly, "carbon_monoxide")
    o3 = get_arr(aq_hourly, "ozone")
    
    temp = get_arr(w_hourly, "temperature_2m") if w_hourly else [None] * n
    humidity = get_arr(w_hourly, "relative_humidity_2m") if w_hourly else [None] * n
    wind = get_arr(w_hourly, "wind_speed_10m") if w_hourly else [None] * n
    
    rows = []
    timestamps = []
    
    for i in range(n):
        try:
            ts_str = times[i].replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_str)
        except Exception:
            continue
        
        pm25_i = pm25[i] if i < len(pm25) else None
        pm10_i = pm10[i] if i < len(pm10) else None
        
        pm_ratio = None
        pm_sum = None
        if pm25_i is not None and pm10_i is not None and pm10_i != 0:
            pm_ratio = pm25_i / pm10_i
            pm_sum = pm25_i + pm10_i
        
        hour = dt.hour
        day_w = dt.weekday()
        month = dt.month
        
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        day_sin = math.sin(2 * math.pi * day_w / 7)
        day_cos = math.cos(2 * math.pi * day_w / 7)
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        
        row = {
            "pm25": pm25_i,
            "pm10": pm10_i,
            "pm_ratio": pm_ratio,
            "pm_sum": pm_sum,
            "no2": no2[i] if i < len(no2) else None,
            "so2": so2[i] if i < len(so2) else None,
            "co": co[i] if i < len(co) else None,
            "o3": o3[i] if i < len(o3) else None,
            "temperature": temp[i] if i < len(temp) else None,
            "humidity": humidity[i] if i < len(humidity) else None,
            "wind_speed": wind[i] if i < len(wind) else None,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
        rows.append(row)
        timestamps.append(dt)
    
    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    return X, timestamps


def main():
    print("=" * 60)
    print("NEXT 3 DAYS AQI PREDICTION")
    print("=" * 60)
    
    if not os.path.isfile(SCALER_PATH):
        print("Models not found. Run: python eda_and_train.py")
        return

    with open(MODEL_METADATA_PATH) as f:
        meta = json.load(f)

    print("\n1. Loading scaler, imputer, and model...")
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.joblib"))

    best_is_keras = meta.get("best_is_keras", False)
    best_is_pytorch = meta.get("best_is_pytorch", False)
    NN_MODEL_PATH = os.path.join(MODEL_DIR, "neural_net.keras")

    if best_is_keras and os.path.isfile(NN_MODEL_PATH):
        import tensorflow as tf
        model = tf.keras.models.load_model(NN_MODEL_PATH)
    elif best_is_pytorch and os.path.isfile(os.path.join(MODEL_DIR, "pytorch_model.joblib")):
        model = joblib.load(os.path.join(MODEL_DIR, "pytorch_model.joblib"))
    else:
        model = joblib.load(BEST_MODEL_PATH)

    print(f"   Model: {meta.get('best_model_name', 'unknown')} (regression)")
    
    print("\n2. Fetching next 3 days hourly data from Open-Meteo...")
    aq_data, w_data = fetch_forecast_hourly()
    if not aq_data or "hourly" not in aq_data:
        print("   Failed to fetch air quality forecast.")
        return
    
    aq_hourly = aq_data["hourly"]
    w_hourly = w_data.get("hourly", {}) if w_data else {}
    
    print("\n3. Building features...")
    X, timestamps = build_features_from_forecast(aq_hourly, w_hourly)
    if X is None or X.empty:
        print("   No rows to predict.")
        return
    
    # Same pipeline as training: impute then scale
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    print("\n4. Predicting AQI (1-5) for each hour...")
    # Regression models output continuous; clip and round to 1-5
    if best_is_pytorch:
        import torch
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            raw = model(X_t).numpy().ravel()
    else:
        raw = np.asarray(model.predict(X_scaled)).ravel()
    preds_continuous = raw
    preds = np.clip(np.round(preds_continuous), 1, 5).astype(int)

    aqi_labels = {1: "Good", 2: "Moderate", 3: "Unhealthy (Sensitive)", 4: "Unhealthy", 5: "Very Unhealthy"}

    print("\n" + "=" * 60)
    print("HOURLY PREDICTIONS (next 3 days)")
    print("=" * 60)

    current_date = None
    day_count = 0
    for i, (ts, aqi) in enumerate(zip(timestamps, preds)):
        d = ts.date()
        if d != current_date:
            current_date = d
            day_count += 1
            print(f"\n--- Day {day_count}: {d} ---")
        print(f"  {ts.strftime('%Y-%m-%d %H:%M')} UTC  ->  AQI {int(aqi)} ({aqi_labels.get(int(aqi), '?')})")

    # Save to CSV
    out = pd.DataFrame({
        "timestamp_utc": timestamps,
        "aqi_predicted_continuous": preds_continuous,
        "aqi_predicted": preds,
        "aqi_label": [aqi_labels.get(int(p), "?") for p in preds],
    })
    out_path = "predictions_next_3days.csv"
    out.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
