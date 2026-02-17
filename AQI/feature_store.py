"""
Feature Store: loads model, scaler, imputer, and feature schema from disk.
Single source of truth for the ML pipeline used by the API and dashboard.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

try:
    from config import (
        FEATURE_COLS,
        MODEL_DIR,
        SCALER_PATH,
        BEST_MODEL_PATH,
        MODEL_METADATA_PATH,
    )
except ImportError:
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

NN_MODEL_PATH = os.path.join(MODEL_DIR, "neural_net.keras")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.joblib")

_store = None


def load_store():
    """Load scaler, imputer, model, and metadata from disk. Cached in _store."""
    global _store
    if _store is not None:
        return _store

    if not os.path.isfile(SCALER_PATH) or not os.path.isfile(MODEL_METADATA_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Run: python eda_and_train.py"
        )

    with open(MODEL_METADATA_PATH) as f:
        meta = json.load(f)

    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    best_is_keras = meta.get("best_is_keras", False)
    best_is_pytorch = meta.get("best_is_pytorch", False)

    if best_is_keras and os.path.isfile(NN_MODEL_PATH):
        import tensorflow as tf
        model = tf.keras.models.load_model(NN_MODEL_PATH)
    elif best_is_pytorch and os.path.isfile(os.path.join(MODEL_DIR, "pytorch_model.joblib")):
        model = joblib.load(os.path.join(MODEL_DIR, "pytorch_model.joblib"))
    else:
        model = joblib.load(BEST_MODEL_PATH)

    _store = {
        "scaler": scaler,
        "imputer": imputer,
        "model": model,
        "metadata": meta,
        "feature_columns": meta.get("feature_columns", FEATURE_COLS),
        "best_is_keras": best_is_keras,
        "best_is_pytorch": best_is_pytorch,
    }
    return _store


def get_features():
    """Return list of feature names and model metadata."""
    store = load_store()
    return {
        "feature_columns": store["feature_columns"],
        "best_model_name": store["metadata"].get("best_model_name"),
        "best_r2": store["metadata"].get("best_r2"),
        "saved_at": store["metadata"].get("saved_at"),
    }


def predict(features_list):
    """
    Compute AQI predictions (1-5) for one or more feature rows.
    features_list: list of dicts with keys in FEATURE_COLS, or single dict.
    Returns: list of {"aqi_continuous": float, "aqi": int, "aqi_label": str}.
    """
    store = load_store()
    scaler = store["scaler"]
    imputer = store["imputer"]
    model = store["model"]
    cols = store["feature_columns"]
    best_is_pytorch = store["best_is_pytorch"]

    if isinstance(features_list, dict):
        features_list = [features_list]

    X = pd.DataFrame(features_list)
    for c in cols:
        if c not in X.columns:
            X[c] = None
    X = X[cols]

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    if best_is_pytorch:
        import torch
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            raw = model(X_t).numpy().ravel()
    else:
        raw = np.asarray(model.predict(X_scaled)).ravel()

    aqi_labels = {
        1: "Good",
        2: "Moderate",
        3: "Unhealthy (Sensitive)",
        4: "Unhealthy",
        5: "Very Unhealthy",
    }

    out = []
    preds = np.clip(np.round(raw), 1, 5).astype(int)
    for i, r in enumerate(raw):
        aqi_int = int(preds[i]) if i < len(preds) else int(np.clip(round(r), 1, 5))
        out.append({
            "aqi_continuous": float(r),
            "aqi": aqi_int,
            "aqi_label": aqi_labels.get(aqi_int, "Unknown"),
        })
    return out


def reset_store():
    """Clear cached store (e.g. for tests or reload)."""
    global _store
    _store = None
