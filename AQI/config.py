# Shared config for EDA/training and prediction
# CI/CD: set MONGODB_URI secret in GitHub Actions (e.g. MongoDB Atlas connection string)
import os
MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "aqi_data"
COLLECTION_NAME = "karachi_air_quality"

LAT = 24.8607
LON = 67.0011

# Feature columns (must match order used in training)
FEATURE_COLS = [
    "pm25", "pm10", "pm_ratio", "pm_sum",
    "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month_sin", "month_cos"
]

TARGET_COL = "target"

# Paths for saved artifacts
MODEL_DIR = "models"
SCALER_PATH = "models/scaler.joblib"
BEST_MODEL_PATH = "models/best_model.joblib"
MODEL_METADATA_PATH = "models/model_metadata.json"
NN_MODEL_PATH = "models/neural_net.keras"
