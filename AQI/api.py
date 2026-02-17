import requests
from datetime import datetime, timedelta
import math
from pymongo import MongoClient
import time

try:
    from config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME, LAT, LON
except ImportError:
    LAT = 24.8607
    LON = 67.0011
    MONGO_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "aqi_data"
    COLLECTION_NAME = "karachi_air_quality"

def fetch_api(url, params=None):
    """Fetch API data safely with error handling"""
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"Warning: API returned status code {response.status_code}")
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Warning: Request failed: {e}")
        return None

def connect_mongodb():
    """Connect to MongoDB and return collection"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        client.admin.command('ping')
        print(f"Connected to MongoDB: {DATABASE_NAME}.{COLLECTION_NAME}")
        return collection
    except Exception as e:
        raise Exception(f"Failed to connect to MongoDB: {e}")

def calculate_aqi_from_pm25(pm25):
    """Calculate AQI category (1-5) from PM2.5 value based on US EPA standards"""
    if pm25 is None:
        return None
    if pm25 <= 12:
        return 1  # Good
    elif pm25 <= 35.4:
        return 2  # Moderate
    elif pm25 <= 55.4:
        return 3  # Unhealthy for Sensitive Groups
    elif pm25 <= 150.4:
        return 4  # Unhealthy
    else:
        return 5  # Very Unhealthy

def process_data_point(hourly_data, index, timestamp):
    """Process a single data point from Open-Meteo hourly data"""
    if not hourly_data or index >= len(hourly_data.get("time", [])):
        return None
    
    # Extract air quality data
    pm25 = hourly_data.get("pm2_5", [None])[index] if "pm2_5" in hourly_data and index < len(hourly_data.get("pm2_5", [])) else None
    pm10 = hourly_data.get("pm10", [None])[index] if "pm10" in hourly_data and index < len(hourly_data.get("pm10", [])) else None
    no2 = hourly_data.get("nitrogen_dioxide", [None])[index] if "nitrogen_dioxide" in hourly_data and index < len(hourly_data.get("nitrogen_dioxide", [])) else None
    so2 = hourly_data.get("sulphur_dioxide", [None])[index] if "sulphur_dioxide" in hourly_data and index < len(hourly_data.get("sulphur_dioxide", [])) else None
    co = hourly_data.get("carbon_monoxide", [None])[index] if "carbon_monoxide" in hourly_data and index < len(hourly_data.get("carbon_monoxide", [])) else None
    o3 = hourly_data.get("ozone", [None])[index] if "ozone" in hourly_data and index < len(hourly_data.get("ozone", [])) else None
    
    # Calculate AQI from PM2.5
    aqi = calculate_aqi_from_pm25(pm25)
    
    # Extract weather data if available
    temperature = hourly_data.get("temperature_2m", [None])[index] if "temperature_2m" in hourly_data and index < len(hourly_data.get("temperature_2m", [])) else None
    humidity = hourly_data.get("relative_humidity_2m", [None])[index] if "relative_humidity_2m" in hourly_data and index < len(hourly_data.get("relative_humidity_2m", [])) else None
    wind_speed = hourly_data.get("wind_speed_10m", [None])[index] if "wind_speed_10m" in hourly_data and index < len(hourly_data.get("wind_speed_10m", [])) else None
    
    # Skip if no air quality data
    if pm25 is None and pm10 is None:
        return None
    
    # Extract raw data
    raw_data = {
        "timestamp_utc": timestamp,
        "aqi": aqi,
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "so2": so2,
        "co": co,
        "o3": o3,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed
    }
    
    # Time features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    
    # Cyclic encoding
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    day_sin = math.sin(2 * math.pi * day_of_week / 7)
    day_cos = math.cos(2 * math.pi * day_of_week / 7)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    
    # Engineered features
    pm_ratio = None
    pm_sum = None
    if raw_data["pm10"] is not None and raw_data["pm25"] is not None:
        pm_ratio = raw_data["pm25"] / raw_data["pm10"]
        pm_sum = raw_data["pm25"] + raw_data["pm10"]
    
    # Features
    features = {
        "pm25": raw_data["pm25"],
        "pm10": raw_data["pm10"],
        "pm_ratio": pm_ratio,
        "pm_sum": pm_sum,
        "no2": raw_data["no2"],
        "so2": raw_data["so2"],
        "co": raw_data["co"],
        "o3": raw_data["o3"],
        "temperature": raw_data["temperature"],
        "humidity": raw_data["humidity"],
        "wind_speed": raw_data["wind_speed"],
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "month_sin": month_sin,
        "month_cos": month_cos
    }
    
    # Document for MongoDB
    document = {
        "timestamp_utc": timestamp,
        "timestamp_unix": int(timestamp.timestamp()),
        "location": {
            "lat": LAT,
            "lon": LON,
            "city": "Karachi"
        },
        "raw_data": raw_data,
        "features": features,
        "target": aqi,
        "created_at": datetime.utcnow()
    }
    
    return document

def fetch_historical_data(start_date, end_date, collection):
    """Fetch historical data using Open-Meteo API"""
    successful = 0
    failed = 0
    
    print(f"\nFetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    # Fetch in 30-day chunks
    chunk_days = 30
    current_start = start_date
    
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        
        print(f"\nFetching chunk: {start_str} to {end_str}")
        
        # Air quality API
        air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_quality_params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
            "timezone": "UTC"
        }
        
        # Weather API
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        weather_params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "UTC"
        }
        
        print("  Fetching air quality data...")
        air_quality_json = fetch_api(air_quality_url, air_quality_params)
        
        print("  Fetching weather data...")
        weather_json = fetch_api(weather_url, weather_params)
        
        if not air_quality_json or "hourly" not in air_quality_json:
            print(f"  [FAIL] No air quality data")
            current_start = current_end + timedelta(days=1)
            continue
        
        # Merge weather data
        hourly_data = air_quality_json["hourly"].copy()
        if weather_json and "hourly" in weather_json:
            hourly_data.update(weather_json["hourly"])
            print("  [OK] Weather data merged")
        else:
            print("  [WARN] No weather data available")
        
        # Process hourly records
        time_strings = hourly_data.get("time", [])
        print(f"  Processing {len(time_strings)} hourly records...")
        
        chunk_successful = 0
        for idx, time_str in enumerate(time_strings):
            try:
                timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                document = process_data_point(hourly_data, idx, timestamp)
                
                if document:
                    existing = collection.find_one({"timestamp_unix": document["timestamp_unix"]})
                    if not existing:
                        collection.insert_one(document)
                        successful += 1
                        chunk_successful += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  [ERROR] {e}")
        
        print(f"  Chunk complete: {chunk_successful} stored, {len(time_strings) - chunk_successful} skipped/failed")
        current_start = current_end + timedelta(days=1)
        time.sleep(1)
    
    print("=" * 60)
    print(f"\nSummary: {successful} successful, {failed} failed")
    return successful, failed

if __name__ == "__main__":
    try:
        collection = connect_mongodb()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    # Calculate REAL past 3 months
    today = datetime.now()
    end_date = datetime(today.year, today.month, today.day)
    start_date = end_date - timedelta(days=90)
    
    print(f"\nStarting data fetch for past 3 months...")
    print(f"Today: {today.strftime('%Y-%m-%d')}")
    print(f"Start Date: {start_date.strftime('%Y-%m-%d')} (90 days ago)")
    print(f"End Date: {end_date.strftime('%Y-%m-%d')} (today)")
    
    successful, failed = fetch_historical_data(start_date, end_date, collection)
    
    total_documents = collection.count_documents({})
    print(f"\nTotal documents in MongoDB: {total_documents}")
    print("\nDone!")

