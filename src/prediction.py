"""
Prediction module for Network Intrusion Detection System.
Handles inference using trained models.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest


def predict_supervised(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> np.ndarray:
    """Run supervised (Random Forest) predictions. Returns class labels."""
    return model.predict(X)


def predict_anomaly(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """
    Run Isolation Forest anomaly detection.
    Returns: 1 = normal (inlier), -1 = anomaly (outlier).
    """
    return model.predict(X)


def predict_single(
    rf_model: RandomForestClassifier,
    iso_model: IsolationForest,
    X: np.ndarray,
) -> dict:
    """
    Predict a single sample through both models.
    Returns detailed result with timing.
    """
    start = time.time()

    rf_pred = rf_model.predict(X)[0]
    iso_pred = iso_model.predict(X)[0]

    elapsed_ms = (time.time() - start) * 1000

    return {
        "supervised_prediction": int(rf_pred),
        "anomaly_prediction": int(iso_pred),
        "processing_time_ms": round(elapsed_ms, 2),
    }


def batch_predict(
    rf_model: RandomForestClassifier,
    iso_model: IsolationForest,
    X: np.ndarray,
) -> dict:
    """
    Run both models on a batch of samples.
    Returns arrays for supervised and anomaly results.
    """
    start = time.time()

    rf_preds = predict_supervised(rf_model, X)
    iso_preds = predict_anomaly(iso_model, X)

    elapsed_ms = (time.time() - start) * 1000

    return {
        "supervised_predictions": rf_preds,
        "anomaly_predictions": iso_preds,
        "processing_time_ms": round(elapsed_ms, 2),
        "total_samples": len(X),
    }
