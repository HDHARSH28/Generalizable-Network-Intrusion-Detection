"""
Prediction module for Network Intrusion Detection System.
Handles inference using trained models with confidence scores.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest


def predict_supervised(
    model: RandomForestClassifier,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run supervised (Random Forest) predictions.
    Returns: (class_labels, confidence_scores)
    """
    labels = model.predict(X)
    probas = model.predict_proba(X)
    # Confidence = probability of the predicted class
    confidences = np.max(probas, axis=1)
    return labels, confidences


def predict_anomaly(
    model: IsolationForest,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Isolation Forest anomaly detection.
    Returns: (predictions, anomaly_scores)
        predictions: 1 = normal (inlier), -1 = anomaly (outlier)
        anomaly_scores: lower = more anomalous
    """
    predictions = model.predict(X)
    scores = model.decision_function(X)
    return predictions, scores


def predict_single(
    rf_model: RandomForestClassifier,
    iso_model: IsolationForest,
    X: np.ndarray,
) -> dict:
    """
    Predict a single sample through both models.
    Returns detailed result with timing, confidence, and anomaly score.
    """
    start = time.time()

    rf_labels, rf_confidences = predict_supervised(rf_model, X)
    iso_preds, iso_scores = predict_anomaly(iso_model, X)

    elapsed_ms = (time.time() - start) * 1000

    return {
        "supervised_prediction": int(rf_labels[0]),
        "supervised_confidence": float(rf_confidences[0]),
        "anomaly_prediction": int(iso_preds[0]),
        "anomaly_score": float(iso_scores[0]),
        "processing_time_ms": round(elapsed_ms, 2),
    }


def batch_predict(
    rf_model: RandomForestClassifier,
    iso_model: IsolationForest,
    X: np.ndarray,
) -> dict:
    """
    Run both models on a batch of samples.
    Returns arrays for supervised and anomaly results with scores.
    """
    start = time.time()

    rf_labels, rf_confidences = predict_supervised(rf_model, X)
    iso_preds, iso_scores = predict_anomaly(iso_model, X)

    elapsed_ms = (time.time() - start) * 1000

    return {
        "supervised_predictions": rf_labels,
        "supervised_confidences": rf_confidences,
        "anomaly_predictions": iso_preds,
        "anomaly_scores": iso_scores,
        "processing_time_ms": round(elapsed_ms, 2),
        "total_samples": len(X),
    }
