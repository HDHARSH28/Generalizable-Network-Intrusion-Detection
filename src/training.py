"""
Training module for Network Intrusion Detection System.
Trains Random Forest (supervised) and Isolation Forest (anomaly detection).
"""

import os
import json
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# ─────────────────────── Random Forest ───────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=20,
        min_samples_split=5,
    )
    clf.fit(X_train, y_train)
    return clf


# ─────────────────────── Isolation Forest ────────────────────────────────

def train_isolation_forest(
    X_normal: np.ndarray,
    contamination: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> IsolationForest:
    """
    Train an Isolation Forest on NORMAL traffic only.
    Anomalies (attacks / zero-day) should deviate from this baseline.
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    iso.fit(X_normal)
    return iso


# ─────────────────────── Evaluation ──────────────────────────────────────

def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: dict | None = None,
) -> dict:
    """Evaluate the supervised model and return all metrics."""
    y_pred = model.predict(X_test)

    # Build target names aligned with actual classes
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = None
    if label_names:
        # Handle both int and string keys in label_names
        target_names = []
        for c in classes:
            name = label_names.get(int(c),
                   label_names.get(str(int(c)), str(c)))
            target_names.append(name)

    cm = confusion_matrix(y_test, y_pred).tolist()

    # Calculate FPR for binary classification
    fpr = None
    if len(classes) == 2:
        tn = cm[0][0]
        fp = cm[0][1]
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "false_positive_rate": fpr,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True,
        ),
    }
    return metrics


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 10,
) -> list[dict]:
    """Extract top-N feature importances from the Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    result = []
    for i in indices:
        name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
        result.append({"feature": name, "importance": float(importances[i])})
    return result


# ─────────────────────── Save / Load ─────────────────────────────────────

def save_models(
    rf_model: RandomForestClassifier,
    iso_model: IsolationForest,
    metrics: dict,
    models_dir: str = MODELS_DIR,
):
    """Persist trained models and evaluation metrics."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))
    joblib.dump(iso_model, os.path.join(models_dir, "isolation_forest.pkl"))

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def load_models(models_dir: str = MODELS_DIR) -> dict:
    """
    Load saved models and metrics with version compatibility check.
    Raises a clear error if models are incompatible.
    """
    from src.preprocessing import load_model_metadata

    # Check version compatibility via metadata
    metadata = load_model_metadata(models_dir)
    if metadata:
        saved_sklearn = metadata.get("sklearn_version", "unknown")
        current_sklearn = sklearn.__version__
        # Compare major.minor version (patch differences are usually safe)
        saved_parts = saved_sklearn.split(".")[:2]
        current_parts = current_sklearn.split(".")[:2]
        if saved_parts != current_parts and saved_sklearn != "unknown":
            raise RuntimeError(
                f"Model version mismatch: models were trained with "
                f"scikit-learn {saved_sklearn}, but the current version is "
                f"{current_sklearn}. Retrain using: python train.py"
            )

    rf = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    iso = joblib.load(os.path.join(models_dir, "isolation_forest.pkl"))
    metrics = {}
    metrics_path = os.path.join(models_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return {"random_forest": rf, "isolation_forest": iso, "metrics": metrics}


def models_exist(models_dir: str = MODELS_DIR) -> bool:
    """Check whether all required model files exist."""
    required = [
        "random_forest.pkl",
        "isolation_forest.pkl",
        "scaler.pkl",
        "feature_selector.pkl",
        "selected_features.json",
    ]
    return all(
        os.path.exists(os.path.join(models_dir, f)) for f in required
    )
