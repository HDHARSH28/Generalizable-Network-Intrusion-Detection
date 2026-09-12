"""
Preprocessing module for Network Intrusion Detection System.
Handles data loading, cleaning, feature selection, and scaling.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────── Constants ───────────────────────────────

CIC_IDS_LABEL_COLUMN = "Label"
RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# ─────────────────────────────── Dataset Loading ─────────────────────────

def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load CSV files from the data directory.
    Combines compatible CSV files and returns a single DataFrame.
    """
    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in sorted(csv_files):
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(path)}: {e}")

    if not frames:
        raise ValueError("Could not read any CSV file successfully.")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return basic statistics about the dataset."""
    label_col = _find_label_column(df)
    stats = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "label_column": label_col,
        "label_distribution": {},
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if label_col:
        stats["label_distribution"] = df[label_col].value_counts().to_dict()
    return stats


# ─────────────────────────── Label Helpers ───────────────────────────────

def _find_label_column(df: pd.DataFrame) -> str | None:
    """Identify the label column in the dataset."""
    candidates = ["Label", "label", "LABEL", "class", "Class", "CLASS",
                   "attack_cat", "Attack", "attack"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback – last column if it has few unique values
    last = df.columns[-1]
    if df[last].nunique() < 50:
        return last
    return None


def _binarize_labels(series: pd.Series) -> pd.Series:
    """Convert multiclass labels to binary (BENIGN=0, else=1)."""
    benign_patterns = ["BENIGN", "benign", "Normal", "normal", "NORMAL"]
    return series.apply(lambda x: 0 if str(x).strip() in benign_patterns else 1)


# ─────────────────────────── Cleaning ────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Remove empty columns
    - Handle missing / infinite values
    - Remove duplicates
    - Convert features to numeric
    """
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Replace infinities with NaN, then fill NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def select_numeric_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Keep only numeric columns + the label column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col and label_col not in numeric_cols:
        numeric_cols.append(label_col)
    return df[numeric_cols]


# ─────────────────────── Full Preprocessing Pipeline ─────────────────────

def preprocess_dataset(
    df: pd.DataFrame,
    n_features: int = 20,
    test_size: float = 0.2,
    binary: bool = True,
) -> dict:
    """
    Full preprocessing pipeline:
    1. Clean data
    2. Select numeric features
    3. Binarize labels (optional)
    4. Train/test split
    5. Scale features
    6. Feature selection (SelectKBest)

    Returns a dict with all artefacts needed for training.
    """
    label_col = _find_label_column(df)
    if label_col is None:
        raise ValueError("Cannot find a label column in the dataset.")

    df = clean_data(df)
    df = select_numeric_features(df, label_col)

    # Separate features / labels
    y_raw = df[label_col].copy()
    X = df.drop(columns=[label_col])

    # Encode labels
    if binary:
        y = _binarize_labels(y_raw)
        label_names = {0: "BENIGN", 1: "ATTACK"}
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
        label_names = dict(enumerate(le.classes_))

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Clip extreme values to prevent overflow during scaling
    X = X.clip(-1e10, 1e10)

    # Train / test split  (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y,
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection — SelectKBest
    n_features = min(n_features, X_train_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Identify selected feature names
    mask = selector.get_support()
    selected_feature_names = X.columns[mask].tolist()

    return {
        "X_train": X_train_selected,
        "X_test": X_test_selected,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "scaler": scaler,
        "selector": selector,
        "selected_features": selected_feature_names,
        "all_feature_names": X.columns.tolist(),
        "label_names": label_names,
        "label_col": label_col,
    }


# ─────────────────── Inference-time Preprocessing ────────────────────────

def preprocess_for_prediction(
    df: pd.DataFrame,
    scaler: StandardScaler,
    selector: SelectKBest,
    selected_features: list[str],
    all_feature_names: list[str] | None = None,
) -> np.ndarray:
    """
    Preprocess new data for prediction using saved artefacts.
    """
    df = clean_data(df)

    # Keep only numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]

    # Ensure same columns as training
    if all_feature_names is not None:
        for col in all_feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in all_feature_names if c in df.columns]]
        # Pad any still-missing columns
        for col in all_feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[all_feature_names]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.clip(-1e10, 1e10)

    scaled = scaler.transform(df)
    selected = selector.transform(scaled)
    return selected


# ─────────────────── Demo Dataset Generator ──────────────────────────────

def generate_demo_dataset(n_samples: int = 2000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a small synthetic dataset that mimics CIC-IDS2017 features.
    Clearly labelled as demonstration data.
    """
    rng = np.random.RandomState(random_state)

    # Typical CIC-IDS2017-like feature names
    feature_names = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
        "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
        "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd Header Length", "Bwd Header Length", "Packet Length Mean",
    ]

    n_normal = int(n_samples * 0.7)
    n_attack = n_samples - n_normal

    # Normal traffic — low variance, moderate values
    normal_data = rng.exponential(scale=500, size=(n_normal, len(feature_names)))
    normal_data = np.abs(normal_data)

    # Attack traffic — higher variance, different distribution
    attack_data = rng.exponential(scale=2000, size=(n_attack, len(feature_names)))
    attack_data[:, 0] *= 3  # longer flow duration
    attack_data[:, 1] *= 5  # more packets
    attack_data[:, 9] *= 10  # higher bytes/s
    attack_data = np.abs(attack_data)

    data = np.vstack([normal_data, attack_data])
    labels = ["BENIGN"] * n_normal + ["ATTACK"] * n_attack

    # Shuffle
    idx = rng.permutation(n_samples)
    data = data[idx]
    labels = [labels[i] for i in idx]

    df = pd.DataFrame(data, columns=feature_names)
    df["Label"] = labels
    return df


# ─────────────────── Save / Load Helpers ─────────────────────────────────

def save_preprocessing_artefacts(result: dict, models_dir: str = MODELS_DIR):
    """Save scaler, selector, and feature info."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(result["scaler"], os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(result["selector"], os.path.join(models_dir, "feature_selector.pkl"))

    meta = {
        "selected_features": result["selected_features"],
        "all_feature_names": result["all_feature_names"],
        "label_names": {str(k): v for k, v in result["label_names"].items()},
        "label_col": result["label_col"],
    }
    with open(os.path.join(models_dir, "selected_features.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_preprocessing_artefacts(models_dir: str = MODELS_DIR) -> dict:
    """Load saved preprocessing objects."""
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    selector = joblib.load(os.path.join(models_dir, "feature_selector.pkl"))
    with open(os.path.join(models_dir, "selected_features.json"), "r") as f:
        meta = json.load(f)
    return {
        "scaler": scaler,
        "selector": selector,
        "selected_features": meta["selected_features"],
        "all_feature_names": meta["all_feature_names"],
        "label_names": meta["label_names"],
        "label_col": meta["label_col"],
    }
