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
    supervised_confidences: np.ndarray | None = None,
    anomaly_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Build a human-readable results DataFrame with confidence and anomaly scores.
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

    data = {
        "Record": range(1, n + 1),
        "Supervised Result": sup_labels,
    }

    # Add confidence if available
    if supervised_confidences is not None:
        data["Confidence"] = [f"{c:.2%}" for c in supervised_confidences]

    data["Anomaly Result"] = ano_labels

    # Add anomaly score if available
    if anomaly_scores is not None:
        data["Anomaly Score"] = [f"{s:.4f}" for s in anomaly_scores]

    data["Final Decision"] = final_labels
    data["Status"] = final_emojis

    df = pd.DataFrame(data)
    return df


def get_summary_counts(hybrid_preds: np.ndarray) -> dict:
    """
    Return counts for each hybrid category.
    Returns an OrderedDict-like dict with consistent key order:
    NORMAL, KNOWN ATTACK, ANOMALOUS.
    """
    unique, counts = np.unique(hybrid_preds, return_counts=True)
    count_map = {int(u): int(c) for u, c in zip(unique, counts)}

    # Always return in this fixed order
    summary = {
        "NORMAL": count_map.get(NORMAL, 0),
        "KNOWN ATTACK": count_map.get(KNOWN_ATTACK, 0),
        "ANOMALOUS": count_map.get(ANOMALOUS, 0),
    }
    return summary
