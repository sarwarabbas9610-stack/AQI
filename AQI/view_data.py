from pymongo import MongoClient
from datetime import datetime, timedelta
import json

try:
    from config import MONGO_URI, DATABASE_NAME, COLLECTION_NAME
except ImportError:
    MONGO_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "aqi_data"
    COLLECTION_NAME = "karachi_air_quality"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def print_record(record):
    """Print a record in readable format"""
    print("\n" + "="*70)
    print(f"RECORD ID: {record['_id']}")
    print("="*70)
    
    print(f"\n📅 TIMESTAMP:")
    print(f"   UTC: {record['timestamp_utc']}")
    print(f"   Unix: {record['timestamp_unix']}")
    
    print(f"\n📍 LOCATION:")
    loc = record.get('location', {})
    print(f"   Latitude: {loc.get('lat')}")
    print(f"   Longitude: {loc.get('lon')}")
    print(f"   City: {loc.get('city')}")
    
    print(f"\n🌬️ RAW DATA (Air Quality & Weather):")
    raw = record.get('raw_data', {})
    print(f"   AQI Category: {raw.get('aqi')} (1=Good, 5=Very Unhealthy)")
    print(f"   PM2.5: {raw.get('pm25')} μg/m³")
    print(f"   PM10: {raw.get('pm10')} μg/m³")
    print(f"   NO2: {raw.get('no2')} μg/m³")
    print(f"   SO2: {raw.get('so2')} μg/m³")
    print(f"   CO: {raw.get('co')} μg/m³")
    print(f"   O3: {raw.get('o3')} μg/m³")
    print(f"   Temperature: {raw.get('temperature', 'N/A')}°C")
    print(f"   Humidity: {raw.get('humidity', 'N/A')}%")
    print(f"   Wind Speed: {raw.get('wind_speed', 'N/A')} m/s")
    
    print(f"\n🔬 ML FEATURES:")
    features = record.get('features', {})
    print(f"   PM2.5: {features.get('pm25')}")
    print(f"   PM10: {features.get('pm10')}")
    print(f"   PM Ratio: {features.get('pm_ratio', 'N/A')}")
    print(f"   PM Sum: {features.get('pm_sum', 'N/A')}")
    print(f"   NO2: {features.get('no2')}")
    print(f"   SO2: {features.get('so2')}")
    print(f"   CO: {features.get('co')}")
    print(f"   O3: {features.get('o3')}")
    print(f"   Temperature: {features.get('temperature', 'N/A')}")
    print(f"   Humidity: {features.get('humidity', 'N/A')}")
    print(f"   Wind Speed: {features.get('wind_speed', 'N/A')}")
    print(f"   Hour (sin): {features.get('hour_sin', 'N/A'):.4f}" if features.get('hour_sin') is not None else f"   Hour (sin): N/A")
    print(f"   Hour (cos): {features.get('hour_cos', 'N/A'):.4f}" if features.get('hour_cos') is not None else f"   Hour (cos): N/A")
    print(f"   Day (sin): {features.get('day_sin', 'N/A'):.4f}" if features.get('day_sin') is not None else f"   Day (sin): N/A")
    print(f"   Day (cos): {features.get('day_cos', 'N/A'):.4f}" if features.get('day_cos') is not None else f"   Day (cos): N/A")
    print(f"   Month (sin): {features.get('month_sin', 'N/A'):.4f}" if features.get('month_sin') is not None else f"   Month (sin): N/A")
    print(f"   Month (cos): {features.get('month_cos', 'N/A'):.4f}" if features.get('month_cos') is not None else f"   Month (cos): N/A")
    
    print(f"\n🎯 TARGET (AQI): {record.get('target')}")
    print(f"📝 Created At: {record.get('created_at')}")
    print("="*70)

# Main menu
print("\n" + "="*70)
print("MONGODB DATA VIEWER - 3 MONTHS OF AQI DATA")
print("="*70)

total = collection.count_documents({})
print(f"\nTotal records: {total}")

# Get date range
oldest = collection.find_one(sort=[("timestamp_unix", 1)])
newest = collection.find_one(sort=[("timestamp_unix", -1)])

if oldest and newest:
    print(f"Date range: {oldest['timestamp_utc']} to {newest['timestamp_utc']}")
    print(f"Coverage: ~{total/24:.0f} days of hourly data")

print("\n" + "="*70)
print("OPTIONS:")
print("="*70)
print("1. View recent 5 records")
print("2. View records by date")
print("3. View records by AQI category")
print("4. Export one record as JSON")
print("5. Show statistics")
print("="*70)

choice = input("\nEnter choice (1-5): ").strip()

if choice == "1":
    print("\n📋 RECENT 5 RECORDS:")
    for record in collection.find().sort("timestamp_unix", -1).limit(5):
        print_record(record)

elif choice == "2":
    date_str = input("Enter date (YYYY-MM-DD): ").strip()
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_unix = int(target_date.timestamp())
        end_unix = int((target_date + timedelta(days=1)).timestamp())
        
        records = collection.find({
            "timestamp_unix": {"$gte": start_unix, "$lt": end_unix}
        }).sort("timestamp_unix", 1)
        
        count = collection.count_documents({
            "timestamp_unix": {"$gte": start_unix, "$lt": end_unix}
        })
        
        print(f"\n📋 RECORDS FOR {date_str} ({count} records):")
        for record in records:
            print_record(record)
    except ValueError:
        print("Invalid date format")

elif choice == "3":
    aqi = input("Enter AQI category (1-5): ").strip()
    if aqi.isdigit() and 1 <= int(aqi) <= 5:
        count = collection.count_documents({"target": int(aqi)})
        print(f"\n📋 RECORDS WITH AQI = {aqi} ({count} total):")
        for record in collection.find({"target": int(aqi)}).sort("timestamp_unix", -1).limit(5):
            print_record(record)
    else:
        print("Invalid AQI")

elif choice == "4":
    date_str = input("Enter date and time (YYYY-MM-DD HH:MM): ").strip()
    try:
        target_dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        unix_ts = int(target_dt.timestamp())
        record = collection.find_one({"timestamp_unix": unix_ts})
        
        if record:
            # Convert to JSON-serializable format
            record_json = record.copy()
            record_json["_id"] = str(record_json["_id"])
            for key in ["timestamp_utc", "created_at"]:
                if key in record_json and isinstance(record_json[key], datetime):
                    record_json[key] = record_json[key].isoformat()
            
            filename = f"record_{date_str.replace(' ', '_')}.json"
            with open(filename, 'w') as f:
                json.dump(record_json, f, indent=2, default=str)
            print(f"\n✅ Exported to {filename}")
            print_record(record)
        else:
            print("Record not found")
    except ValueError:
        print("Invalid date format")

elif choice == "5":
    print("\n📊 STATISTICS:")
    print("="*70)
    
    # By AQI
    print("\nRecords by AQI Category:")
    aqi_names = {1: "Good", 2: "Moderate", 3: "Unhealthy (Sensitive)", 4: "Unhealthy", 5: "Very Unhealthy"}
    for aqi in range(1, 6):
        count = collection.count_documents({"target": aqi})
        print(f"   AQI {aqi} ({aqi_names[aqi]}): {count} records ({count/total*100:.1f}%)")
    
    # By month
    from collections import defaultdict
    monthly = defaultdict(int)
    for record in collection.find({}, {"timestamp_utc": 1}):
        ts = record.get('timestamp_utc')
        if ts:
            month_key = ts.strftime('%Y-%m') if isinstance(ts, datetime) else str(ts)[:7]
            monthly[month_key] += 1
    
    print("\nRecords by Month:")
    for month in sorted(monthly.keys()):
        print(f"   {month}: {monthly[month]} records")
    
    # Weather data coverage
    with_weather = collection.count_documents({"raw_data.temperature": {"$ne": None}})
    print(f"\nWeather Data Coverage:")
    print(f"   Records with temperature: {with_weather} ({with_weather/total*100:.1f}%)")
    print(f"   Records without temperature: {total - with_weather} ({(total-with_weather)/total*100:.1f}%)")

else:
    print("Invalid choice")

client.close()


