"""
Hybrid Detection module for Network Intrusion Detection System.

Combines supervised (Random Forest) and unsupervised (Isolation Forest)
results into a final hybrid decision.

Logic:
    1. Random Forest classifies traffic → Known Attack or Benign.
    2. If Benign → Isolation Forest checks for anomaly.
    3. Final decision:
        - KNOWN ATTACK   (RF says attack)
        - ANOMALOUS      (RF says benign BUT IF says anomaly → potential zero-day)
        - NORMAL          (both models agree it's benign)
"""

import numpy as np
import pandas as pd


# Result codes
NORMAL = 0
KNOWN_ATTACK = 1
ANOMALOUS = 2

RESULT_LABELS = {
    NORMAL: "NORMAL",
    KNOWN_ATTACK: "KNOWN ATTACK",
    ANOMALOUS: "ANOMALOUS",
}

RESULT_EMOJI = {
    NORMAL: "🟢",
    KNOWN_ATTACK: "🔴",
    ANOMALOUS: "🟠",
}

RESULT_COLORS = {
    NORMAL: "#00e676",
    KNOWN_ATTACK: "#ff1744",
    ANOMALOUS: "#ff9100",
}


def hybrid_decision_single(
    supervised_pred: int,
    anomaly_pred: int,
    attack_label: int = 1,
) -> int:
    """
    Compute hybrid decision for a single sample.

    Args:
        supervised_pred: Predicted class from Random Forest (0=benign, 1=attack).
        anomaly_pred: Isolation Forest result (1=inlier/normal, -1=outlier/anomaly).
        attack_label: The label value that represents "attack" in supervised output.

    Returns:
        One of NORMAL, KNOWN_ATTACK, ANOMALOUS.
    """
    if supervised_pred == attack_label:
        return KNOWN_ATTACK

    # Supervised says benign → check anomaly detector
    if anomaly_pred == -1:
        return ANOMALOUS

    return NORMAL


def hybrid_decision_batch(
    supervised_preds: np.ndarray,
    anomaly_preds: np.ndarray,
    attack_label: int = 1,
) -> np.ndarray:
    """
    Compute hybrid decisions for a batch of samples.
    """
    results = np.zeros(len(supervised_preds), dtype=int)

    for i in range(len(supervised_preds)):
        results[i] = hybrid_decision_single(
            supervised_preds[i], anomaly_preds[i], attack_label,
        )

    return results


def build_results_dataframe(
    supervised_preds: np.ndarray,
    anomaly_preds: np.ndarray,
    hybrid_preds: np.ndarray,
    label_names: dict | None = None,
) -> pd.DataFrame:
    """
    Build a human-readable results DataFrame.
    """
    n = len(supervised_preds)

    sup_labels = []
    for p in supervised_preds:
        if label_names and str(int(p)) in label_names:
            sup_labels.append(label_names[str(int(p))])
        else:
            sup_labels.append("BENIGN" if p == 0 else "ATTACK")

    ano_labels = ["Normal" if a == 1 else "Anomalous" for a in anomaly_preds]

    final_labels = [RESULT_LABELS.get(h, "UNKNOWN") for h in hybrid_preds]
    final_emojis = [RESULT_EMOJI.get(h, "❓") for h in hybrid_preds]

    df = pd.DataFrame({
        "Record": range(1, n + 1),
        "Supervised Result": sup_labels,
        "Anomaly Result": ano_labels,
        "Final Decision": final_labels,
        "Status": final_emojis,
    })
    return df


def get_summary_counts(hybrid_preds: np.ndarray) -> dict:
    """Return counts for each hybrid category."""
    unique, counts = np.unique(hybrid_preds, return_counts=True)
    summary = {RESULT_LABELS.get(u, "UNKNOWN"): int(c) for u, c in zip(unique, counts)}
    # Ensure all keys present
    for code, label in RESULT_LABELS.items():
        if label not in summary:
            summary[label] = 0
    return summary
