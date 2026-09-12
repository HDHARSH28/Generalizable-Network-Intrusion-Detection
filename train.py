#!/usr/bin/env python3
"""
train.py — Network Intrusion Detection System
Train Random Forest + Isolation Forest on CIC-IDS2017 data (or demo data).

Usage:
    python train.py                  # Train on data/ directory (CIC-IDS2017 CSVs)
    python train.py --demo           # Train on synthetic demo data
    python train.py --features 20    # Select top-20 features (default)
"""

import os
import sys
import argparse
import json
import time

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    load_dataset,
    preprocess_dataset,
    generate_demo_dataset,
    save_preprocessing_artefacts,
    get_dataset_stats,
)
from src.training import (
    train_random_forest,
    train_isolation_forest,
    evaluate_model,
    get_feature_importance,
    save_models,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train the Hybrid IDS models (Random Forest + Isolation Forest)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use a synthetic demo dataset instead of real CIC-IDS2017 data.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"),
        help="Path to the directory containing CIC-IDS2017 CSV files.",
    )
    parser.add_argument(
        "--features", type=int, default=20,
        help="Number of top features to select (default: 20).",
    )
    parser.add_argument(
        "--models-dir", type=str, default=os.path.join(PROJECT_ROOT, "models"),
        help="Directory to save trained models.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Network Intrusion Detection System — Model Training")
    print("=" * 65)
    print()

    # ── Step 1: Load dataset ──────────────────────────────────────────
    if args.demo:
        print("[1/9] Generating DEMO dataset (2000 samples) ...")
        df = generate_demo_dataset(n_samples=2000)
        print(f"      → Generated {len(df)} records with {len(df.columns)} columns.")
    else:
        print(f"[1/9] Loading dataset from: {args.data_dir}")
        try:
            df = load_dataset(args.data_dir)
            print(f"      → Loaded {len(df)} records with {len(df.columns)} columns.")
        except FileNotFoundError:
            print()
            print("  ✗ No CSV files found in data/ directory.")
            print()
            print("  OPTIONS:")
            print("    1. Place CIC-IDS2017 CSV files inside the data/ folder.")
            print("    2. Run with --demo flag to use synthetic data:")
            print("       python train.py --demo")
            print()
            sys.exit(1)

    # ── Step 2: Dataset statistics ────────────────────────────────────
    print()
    print("[2/9] Dataset Statistics:")
    stats = get_dataset_stats(df)
    print(f"      Total records   : {stats['total_records']:,}")
    print(f"      Total features  : {stats['total_features']}")
    print(f"      Label column    : {stats['label_column']}")
    print(f"      Missing values  : {stats['missing_values']:,}")
    print(f"      Duplicate rows  : {stats['duplicate_rows']:,}")
    print(f"      Label distribution:")
    for label, count in stats["label_distribution"].items():
        print(f"        {label}: {count:,}")

    # ── Step 3: Preprocess ────────────────────────────────────────────
    print()
    print(f"[3/9] Preprocessing (selecting top {args.features} features) ...")
    t0 = time.time()
    result = preprocess_dataset(df, n_features=args.features)
    print(f"      → Done in {time.time() - t0:.1f}s")
    print(f"      Train samples : {result['X_train'].shape[0]:,}")
    print(f"      Test samples  : {result['X_test'].shape[0]:,}")
    print(f"      Features used : {result['X_train'].shape[1]}")
    print(f"      Selected features: {result['selected_features']}")

    # ── Step 4: Train/Test split done inside preprocess ───────────────
    print()
    print("[4/9] Train/Test split ✓ (80/20, stratified, random_state=42)")

    # ── Step 5: Train Random Forest ───────────────────────────────────
    print()
    print("[5/9] Training Random Forest classifier ...")
    t0 = time.time()
    rf_model = train_random_forest(result["X_train"], result["y_train"])
    print(f"      → Done in {time.time() - t0:.1f}s")

    # ── Step 6: Train Isolation Forest ────────────────────────────────
    print()
    print("[6/9] Training Isolation Forest anomaly detector ...")
    # Train on NORMAL traffic only (label 0 = benign)
    normal_mask = result["y_train"] == 0
    X_normal = result["X_train"][normal_mask]
    print(f"      Using {X_normal.shape[0]:,} normal samples for training.")
    t0 = time.time()
    iso_model = train_isolation_forest(X_normal)
    print(f"      → Done in {time.time() - t0:.1f}s")

    # ── Step 7: Evaluate Random Forest ────────────────────────────────
    print()
    print("[7/9] Evaluating Random Forest on test set ...")
    metrics = evaluate_model(rf_model, result["X_test"], result["y_test"], result["label_names"])
    print(f"      Accuracy  : {metrics['accuracy']:.4f}")
    print(f"      Precision : {metrics['precision']:.4f}")
    print(f"      Recall    : {metrics['recall']:.4f}")
    print(f"      F1 Score  : {metrics['f1_score']:.4f}")

    # Feature importance
    feat_imp = get_feature_importance(rf_model, result["selected_features"])
    print()
    print("      Top 10 Important Features:")
    for i, fi in enumerate(feat_imp[:10], 1):
        print(f"        {i:2d}. {fi['feature']:<30s} {fi['importance']:.4f}")

    metrics["feature_importance"] = feat_imp

    # ── Step 8: Save models ───────────────────────────────────────────
    print()
    print(f"[8/9] Saving models to {args.models_dir}/ ...")
    save_models(rf_model, iso_model, metrics, args.models_dir)
    print("      ✓ random_forest.pkl")
    print("      ✓ isolation_forest.pkl")
    print("      ✓ metrics.json")

    # ── Step 9: Save preprocessing artefacts ──────────────────────────
    print()
    print("[9/9] Saving preprocessing artefacts ...")
    save_preprocessing_artefacts(result, args.models_dir)
    print("      ✓ scaler.pkl")
    print("      ✓ feature_selector.pkl")
    print("      ✓ selected_features.json")

    # ── Done ──────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    if args.demo:
        print("  ⚠  DEMO MODE — Models trained on synthetic data.")
        print("  ⚠  These results must NOT be used as research results.")
    else:
        print("  ✓  Training complete on CIC-IDS2017 data.")
    print()
    print("  Next step → streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
