# 🛡️ Network Intrusion Detection System

**A Hybrid Machine Learning Approach for Generalizable Network Intrusion Detection with Improved Zero-Day Attack Detection**

> B.Tech Semester Project — Functional Prototype

---

## Overview

This prototype demonstrates a hybrid intrusion detection system that combines:

| Model | Type | Purpose |
|---|---|---|
| **Random Forest** | Supervised Classifier | Detect known attack patterns |
| **Isolation Forest** | Anomaly Detector | Detect unseen / zero-day anomalies |

### Hybrid Detection Logic

```
Incoming Traffic → Random Forest → Known Attack? 
                                      ├── YES → 🔴 KNOWN ATTACK
                                      └── NO  → Isolation Forest → Anomaly?
                                                                    ├── YES → 🟠 ANOMALOUS
                                                                    └── NO  → 🟢 NORMAL
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

**Option A — With CIC-IDS2017 data** (recommended):

Place CSV files in the `data/` directory, then:

```bash
python train.py
```

**Option B — Demo mode** (quick testing):

```bash
python train.py --demo
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Project Structure

```
network-ids/
│
├── app.py                  # Streamlit dashboard (5 pages)
├── train.py                # Training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # Place CIC-IDS2017 CSV files here
│   └── README.md
│
├── models/                 # Trained models (auto-generated)
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── feature_selector.pkl
│   ├── selected_features.json
│   └── metrics.json
│
├── src/                    # Source modules
│   ├── __init__.py
│   ├── preprocessing.py    # Data loading, cleaning, feature selection
│   ├── training.py         # Model training and evaluation
│   ├── prediction.py       # Inference (single + batch)
│   └── hybrid_detector.py  # Hybrid decision logic
│
└── results/                # Output directory
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| **📊 Dashboard** | Overview with metrics, model status, and architecture |
| **📂 Analyze CSV** | Upload CSV → preprocess → predict → hybrid results |
| **🔍 Single Prediction** | Enter feature values for one-sample analysis |
| **📈 Model Performance** | Accuracy, precision, recall, F1, confusion matrix |
| **⚙️ Train Models** | Train/retrain models from the UI |

---

## Dataset

This prototype is designed for the **CIC-IDS2017** dataset.

- Download from: https://www.unb.ca/cic/datasets/ids-2017.html
- Place the CSV files (GeneratedLabelledFlows) in the `data/` directory
- The system handles multiple CSV files, combining them automatically

### Demo Mode

If no dataset is available, use `python train.py --demo` to generate synthetic data for UI testing.

> ⚠️ Demo mode results must NOT be used as research results.

---

## Configuration

| Parameter | Default | Flag |
|---|---|---|
| Number of features | 20 | `--features N` |
| Data directory | `data/` | `--data-dir PATH` |
| Models directory | `models/` | `--models-dir PATH` |
| Demo mode | Off | `--demo` |

Example:

```bash
python train.py --demo --features 15
```

---

## Technology Stack

- **Python 3.11+**
- **Streamlit** — Web dashboard
- **scikit-learn** — Random Forest, Isolation Forest, SelectKBest
- **pandas / NumPy** — Data processing
- **Plotly** — Interactive charts
- **joblib** — Model serialization

---

## Note

This is a **prototype** for semester demonstration purposes. The full research implementation (cross-dataset generalization, autoencoders, SMOTE, etc.) will be developed in the next phase.

---

*B.Tech Semester Project — Network Intrusion Detection System*
