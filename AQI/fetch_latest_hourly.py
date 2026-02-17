"""
Fetch latest hourly data only (last 2 days). For CI/CD: run every hour to keep DB updated.
Uses same Open-Meteo APIs and document shape as api.py; duplicates are skipped.
"""
import requests
from datetime import datetime, timedelta
from api import connect_mongodb, process_data_point, LAT, LON


def fetch_api(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def fetch_last_n_days(collection, days=2):
    """Fetch last N days of hourly data and upsert into MongoDB."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "timezone": "UTC",
    }
    w_url = "https://archive-api.open-meteo.com/v1/archive"
    w_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "UTC",
    }

    aq_data = fetch_api(aq_url, aq_params)
    w_data = fetch_api(w_url, w_params)

    if not aq_data or "hourly" not in aq_data:
        print("No air quality data received.")
        return 0

    hourly = aq_data["hourly"].copy()
    if w_data and "hourly" in w_data:
        hourly.update(w_data["hourly"])

    times = hourly.get("time", [])
    stored = 0
    for idx, time_str in enumerate(times):
        try:
            ts = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            doc = process_data_point(hourly, idx, ts)
            if not doc:
                continue
            unix = doc["timestamp_unix"]
            if collection.find_one({"timestamp_unix": unix}) is None:
                collection.insert_one(doc)
                stored += 1
        except Exception as e:
            print(f"Skip {time_str}: {e}")
    return stored


def main():
    try:
        collection = connect_mongodb()
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        raise SystemExit(1)
    n = fetch_last_n_days(collection, days=2)
    print(f"Stored {n} new hourly records.")


if __name__ == "__main__":
    main()
