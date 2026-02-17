"""
EDA (cleaning, normalization) + Train and evaluate best ML model for AQI prediction.
Regression: predict AQI (1-5 as continuous), evaluate with RMSE, MAE, R2.
Models: Scikit-learn (Ridge Regression, Random Forest) + TensorFlow/PyTorch neural network.
Saves scaler, imputer, and best model for prediction script.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Config
try:
    from config import (
        MONGO_URI, DATABASE_NAME, COLLECTION_NAME,
        FEATURE_COLS, TARGET_COL,
        MODEL_DIR, SCALER_PATH, BEST_MODEL_PATH, MODEL_METADATA_PATH
    )
except ImportError:
    MONGO_URI = "mongodb://localhost:27017/"
    DATABASE_NAME = "aqi_data"
    COLLECTION_NAME = "karachi_air_quality"
    FEATURE_COLS = [
        "pm25", "pm10", "pm_ratio", "pm_sum",
        "no2", "so2", "co", "o3",
        "temperature", "humidity", "wind_speed",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "month_sin", "month_cos"
    ]
    TARGET_COL = "target"
    MODEL_DIR = "models"
    SCALER_PATH = "models/scaler.joblib"
    BEST_MODEL_PATH = "models/best_model.joblib"
    MODEL_METADATA_PATH = "models/model_metadata.json"

NN_MODEL_PATH = os.path.join(MODEL_DIR, "neural_net.keras")


def load_data_from_mongodb():
    """Load all documents from MongoDB into a DataFrame."""
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    coll = db[COLLECTION_NAME]
    cursor = coll.find({}).sort("timestamp_unix", 1)
    rows = []
    for doc in cursor:
        f = doc.get("features", {})
        row = {c: f.get(c) for c in FEATURE_COLS}
        row[TARGET_COL] = doc.get("target")
        row["timestamp_utc"] = doc.get("timestamp_utc")
        row["timestamp_unix"] = doc.get("timestamp_unix")
        rows.append(row)
    client.close()
    return pd.DataFrame(rows)


def run_eda(df):
    """Exploratory Data Analysis: summary, missing, outliers."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n1. Shape:", df.shape)
    print("\n2. Target (AQI 1-5) distribution:")
    print(df[TARGET_COL].value_counts().sort_index())

    print("\n3. Missing values per feature:")
    missing = df[FEATURE_COLS].isnull().sum()
    for c in FEATURE_COLS:
        pct = 100 * missing[c] / len(df)
        print(f"   {c}: {missing[c]} ({pct:.1f}%)")

    print("\n4. Numeric summary (before cleaning):")
    print(df[FEATURE_COLS].describe().round(2))

    print("\n5. Outlier counts (outside 1.5*IQR):")
    for col in FEATURE_COLS:
        if df[col].dtype in (np.float64, np.int64):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out = ((df[col] < low) | (df[col] > high)).sum()
            print(f"   {col}: {n_out}")

    print("=" * 60)


def clean_and_prepare(df):
    """
    Cleaning: impute missing, cap outliers, drop rows with invalid target.
    Returns X (features), y (target numeric for regression), and fitted imputer.
    """
    print("\n" + "=" * 60)
    print("CLEANING & PREPARATION")
    print("=" * 60)

    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    df = df[df[TARGET_COL].between(1, 5)]

    imputer = SimpleImputer(strategy="median")
    X = df[FEATURE_COLS].copy()
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=FEATURE_COLS, index=X.index)

    for col in FEATURE_COLS:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            X[col] = X[col].clip(low, high)

    y = df[TARGET_COL]
    print(f"Final samples: {len(X)}, target (AQI) stats: mean={y.mean():.2f}, std={y.std():.2f}")
    print("=" * 60)

    return X, y, imputer


def normalize_and_split(X, y, test_size=0.2, random_state=42):
    """Train/test split then fit StandardScaler on train only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (
        X_train_scaled, X_test_scaled,
        y_train.values, y_test.values,
        scaler,
    )


def regression_metrics(y_true, y_pred, name="Model"):
    """Compute and print RMSE, MAE, R2."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Clip predictions to AQI range for fair metrics
    y_pred_clip = np.clip(y_pred, 1, 5)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clip))
    mae = mean_absolute_error(y_true, y_pred_clip)
    r2 = r2_score(y_true, y_pred_clip)

    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_neural_network(input_dim, output_dim=1):
    """Build a small MLP for regression. Prefer TensorFlow, fallback to PyTorch."""
    # Try TensorFlow / Keras first
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(output_dim),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model, "tensorflow"
    except ImportError:
        pass

    # Try PyTorch
    try:
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self, input_dim, hidden=(64, 32, 16), out_dim=1):
                super().__init__()
                layers_list = []
                prev = input_dim
                for h in hidden:
                    layers_list.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
                    prev = h
                layers_list.append(nn.Linear(prev, out_dim))
                self.net = nn.Sequential(*layers_list)

            def forward(self, x):
                return self.net(x)

        model = MLP(input_dim)
        return model, "pytorch"
    except ImportError:
        pass

    return None, None


def train_pytorch_model(model, X_train, y_train, X_test, y_test, epochs=80, batch_size=32):
    """Train PyTorch model and return predictions + metrics."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_t).cpu().numpy().ravel()
    return y_pred


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train Ridge, Random Forest, and (if available) Neural Network.
    Evaluate with RMSE, MAE, R2. Return best model by R2 and metadata.
    """
    print("\n" + "=" * 60)
    print("TRAINING MODELS (Regression: RMSE, MAE, R2)")
    print("=" * 60)

    n_features = X_train.shape[1]
    results = []
    best_r2 = -np.inf
    best_name = None
    best_model = None
    best_is_keras = False
    best_is_pytorch = False

    # ----- Ridge Regression -----
    print("\n--- Ridge Regression ---")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    m_ridge = regression_metrics(y_test, y_pred_ridge, "Ridge")
    results.append({"model": "Ridge Regression", **m_ridge})
    if m_ridge["r2"] > best_r2:
        best_r2 = m_ridge["r2"]
        best_name = "Ridge Regression"
        best_model = ridge

    # ----- Random Forest Regressor -----
    print("\n--- Random Forest Regressor ---")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    m_rf = regression_metrics(y_test, y_pred_rf, "Random Forest")
    results.append({"model": "Random Forest", **m_rf})
    if m_rf["r2"] > best_r2:
        best_r2 = m_rf["r2"]
        best_name = "Random Forest"
        best_model = rf

    # ----- Neural Network (TensorFlow or PyTorch) -----
    nn_model, backend = build_neural_network(n_features)
    if nn_model is not None and backend == "tensorflow":
        print("\n--- Neural Network (TensorFlow/Keras) ---")
        history = nn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=80,
            batch_size=32,
            verbose=0,
        )
        y_pred_nn = nn_model.predict(X_test, verbose=0).ravel()
        m_nn = regression_metrics(y_test, y_pred_nn, "Neural Network")
        results.append({"model": "Neural Network", **m_nn})
        if m_nn["r2"] > best_r2:
            best_r2 = m_nn["r2"]
            best_name = "Neural Network"
            best_model = nn_model
            best_is_keras = True
    elif nn_model is not None and backend == "pytorch":
        print("\n--- Neural Network (PyTorch) ---")
        y_pred_nn = train_pytorch_model(nn_model, X_train, y_train, X_test, y_test)
        m_nn = regression_metrics(y_test, y_pred_nn, "Neural Network")
        results.append({"model": "Neural Network", **m_nn})
        if m_nn["r2"] > best_r2:
            best_r2 = m_nn["r2"]
            best_name = "Neural Network"
            best_model = nn_model
            best_is_pytorch = True
    else:
        print("\n--- Neural Network: skipped (install tensorflow or torch) ---")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY (Regression metrics)")
    print("=" * 60)
    for r in results:
        print(f"  {r['model']}: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}, R2={r['r2']:.4f}")
    print(f"\nBEST MODEL: {best_name} (R2={best_r2:.4f})")
    print("=" * 60)

    return best_model, best_name, best_r2, results, best_is_keras, best_is_pytorch


def save_artifacts(scaler, imputer, best_model, best_name, best_r2, feature_cols, best_is_keras, best_is_pytorch):
    """Save scaler, imputer, best model, and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.joblib"))

    if best_is_keras:
        best_model.save(NN_MODEL_PATH)
        joblib.dump(None, BEST_MODEL_PATH)  # placeholder; predictor loads from NN_MODEL_PATH
    elif best_is_pytorch:
        joblib.dump(best_model, os.path.join(MODEL_DIR, "pytorch_model.joblib"))
        joblib.dump(best_model, BEST_MODEL_PATH)  # PyTorch model can be joblib'd
    else:
        joblib.dump(best_model, BEST_MODEL_PATH)

    metadata = {
        "best_model_name": best_name,
        "best_r2": float(best_r2),
        "feature_columns": feature_cols,
        "regression": True,
        "saved_at": datetime.utcnow().isoformat(),
        "best_is_keras": best_is_keras,
        "best_is_pytorch": best_is_pytorch,
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved: {SCALER_PATH}, imputer, best model, {MODEL_METADATA_PATH}")


def main():
    print("Loading data from MongoDB...")
    df = load_data_from_mongodb()
    if df.empty or len(df) < 100:
        raise SystemExit("Not enough data in MongoDB. Need at least 100 records.")

    run_eda(df)
    X, y, imputer = clean_and_prepare(df)

    X_train, X_test, y_train, y_test, scaler = normalize_and_split(X, y)

    best_model, best_name, best_r2, results, best_is_keras, best_is_pytorch = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    save_artifacts(
        scaler, imputer, best_model, best_name, best_r2, FEATURE_COLS,
        best_is_keras, best_is_pytorch
    )
    print("\nDone. Run predict_next_3days.py to predict AQI for the next 3 days.")


if __name__ == "__main__":
    main()
