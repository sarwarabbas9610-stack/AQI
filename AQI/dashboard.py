"""
Streamlit dashboard: loads model and features from API (Feature Store),
computes predictions and shows them on a simple, descriptive dashboard.
Run: streamlit run dashboard.py
Ensure API is running: uvicorn api_backend:app --host 0.0.0.0 --port 8000
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Default API base URL (backend must be running)
API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AQI Prediction Dashboard", page_icon="🌬️", layout="wide")

st.title("🌬️ AQI Prediction Dashboard")
st.caption("Model and features loaded from Feature Store via API. Predictions for Karachi.")

# Sidebar: API URL and model info
with st.sidebar:
    api_url = st.text_input("API base URL", value=API_BASE, help="FastAPI backend URL")
    st.divider()
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            d = r.json()
            st.success(f"API: {d.get('status', 'ok')}")
            if d.get("model"):
                st.info(f"Model: {d['model']}")
        else:
            st.error("API returned an error")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.warning("Start the backend: uvicorn api_backend:app --port 8000")

    st.divider()
    try:
        r = requests.get(f"{api_url}/features", timeout=3)
        if r.status_code == 200:
            fe = r.json()
            st.subheader("Feature Store")
            st.write(f"Features: {len(fe.get('feature_columns', []))} columns")
            st.write(f"Best R²: {fe.get('best_r2', 'N/A')}")
    except Exception:
        pass

# Tabs: Next 3 days forecast | Custom prediction | Feature list
tab1, tab2, tab3 = st.tabs(["📅 Next 3 days forecast", "🔢 Custom prediction", "📋 Feature list"])

with tab1:
    st.subheader("Next 3 days AQI forecast (hourly)")
    st.markdown("Fetches live data from Open-Meteo and runs the trained model. **Start the API first.**")
    if st.button("Fetch and predict next 3 days", type="primary"):
        with st.spinner("Calling API and Open-Meteo..."):
            try:
                r = requests.get(f"{api_url}/predict/next_3_days", timeout=30)
                if r.status_code != 200:
                    st.error(f"API error: {r.status_code} - {r.text}")
                else:
                    data = r.json()
                    preds = data.get("predictions", [])
                    if not preds:
                        st.warning("No predictions returned.")
                    else:
                        df = pd.DataFrame(preds)
                        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        # Simple line chart: AQI over time
                        st.line_chart(df.set_index("timestamp_utc")[["aqi", "aqi_continuous"]])
                        # Summary by day
                        df["date"] = df["timestamp_utc"].dt.date
                        daily = df.groupby("date").agg({"aqi": ["mean", "min", "max"], "timestamp_utc": "count"}).round(2)
                        daily.columns = ["AQI mean", "AQI min", "AQI max", "Hours"]
                        st.subheader("Daily summary")
                        st.dataframe(daily, use_container_width=True, hide_index=True)
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Is it running on " + api_url + "?")
            except Exception as e:
                st.error(str(e))

with tab2:
    st.subheader("Custom prediction (single row)")
    st.markdown("Enter feature values; missing values will be imputed by the model pipeline.")
    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.number_input("PM2.5", value=35.0, min_value=0.0, max_value=500.0, step=1.0)
        pm10 = st.number_input("PM10", value=50.0, min_value=0.0, max_value=500.0, step=1.0)
        no2 = st.number_input("NO2", value=1.0, min_value=0.0, max_value=100.0, step=0.1)
        so2 = st.number_input("SO2", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
        co = st.number_input("CO", value=200.0, min_value=0.0, max_value=1000.0, step=1.0)
        o3 = st.number_input("O3", value=80.0, min_value=0.0, max_value=300.0, step=1.0)
    with col2:
        temperature = st.number_input("Temperature (°C)", value=25.0, min_value=-20.0, max_value=50.0, step=0.5)
        humidity = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, step=1.0)
        wind_speed = st.number_input("Wind speed (m/s)", value=3.0, min_value=0.0, max_value=50.0, step=0.1)
        hour = st.slider("Hour (for cyclic encoding)", 0, 23, 12)
        day_week = st.slider("Day of week (0=Mon)", 0, 6, 2)
        month = st.slider("Month", 1, 12, 6)

    # Cyclic encoding
    import math
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    day_sin = math.sin(2 * math.pi * day_week / 7)
    day_cos = math.cos(2 * math.pi * day_week / 7)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    pm_ratio = pm25 / pm10 if pm10 else None
    pm_sum = pm25 + pm10

    if st.button("Predict AQI"):
        row = {
            "pm25": pm25, "pm10": pm10, "pm_ratio": pm_ratio, "pm_sum": pm_sum,
            "no2": no2, "so2": so2, "co": co, "o3": o3,
            "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "day_sin": day_sin, "day_cos": day_cos,
            "month_sin": month_sin, "month_cos": month_cos,
        }
        try:
            r = requests.post(f"{api_url}/predict/single", json={"features": row}, timeout=5)
            if r.status_code == 200:
                pred = r.json().get("prediction", {})
                aqi = pred.get("aqi", 0)
                label = pred.get("aqi_label", "")
                st.success(f"**Predicted AQI: {aqi}** — {label}")
                st.metric("AQI (continuous)", round(pred.get("aqi_continuous", 0), 2))
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

with tab3:
    st.subheader("Feature list (from Feature Store)")
    try:
        r = requests.get(f"{api_url}/features", timeout=3)
        if r.status_code == 200:
            fe = r.json()
            cols = fe.get("feature_columns", [])
            st.write("Model:", fe.get("best_model_name"), "| R²:", fe.get("best_r2"))
            st.json(cols)
        else:
            st.error("Could not load features")
    except Exception as e:
        st.error(str(e))
