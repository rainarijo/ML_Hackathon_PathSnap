"""Evaluation metrics module for PathSnap Part B.

Calculates:
1. Top-1 Candidate Ranking Accuracy (grouped by (trip_id, point_idx))
2. Macro F1-score
3. Binary F1-score (positive class)
4. Precision
5. Recall
6. Log Loss
"""
import logging
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def calculate_top1_accuracy(
    df_meta: pd.DataFrame,
    probas: np.ndarray,
) -> float:
    """Calculate Top-1 Candidate Ranking Accuracy grouped by (trip_id, point_idx).

    For each GPS point (trip_id, point_idx), we identify the candidate road arc
    with the highest predicted probability of label 1. If that candidate's ground
    truth label is 1, the point is correctly matched. Top-1 accuracy is the
    fraction of correctly matched GPS points across the dataset.

    Parameters
    ----------
    df_meta : pd.DataFrame
        DataFrame containing at least ['trip_id', 'point_idx', 'label'].
    probas : np.ndarray
        1D array of predicted probabilities P(label=1) corresponding to rows in df_meta.

    Returns
    -------
    float
        Top-1 candidate ranking accuracy in [0.0, 1.0].
    """
    if len(df_meta) != len(probas):
        raise ValueError(
            f"Length mismatch: df_meta has {len(df_meta)} rows but probas has {len(probas)} elements."
        )

    # Use a lightweight slice to prevent side-effects on caller DataFrame
    df_eval = pd.DataFrame(
        {
            "trip_id": df_meta["trip_id"].to_numpy(),
            "point_idx": df_meta["point_idx"].to_numpy(),
            "label": df_meta["label"].to_numpy(),
            "prob": np.asarray(probas, dtype=np.float64),
        }
    )

    # Pick the candidate index with the maximum predicted probability for each (trip_id, point_idx)
    idx_max = df_eval.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()
    top1_correct = (df_eval.loc[idx_max, "label"] == 1).to_numpy()

    return float(np.mean(top1_correct))


def evaluate_predictions(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    probas: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all required classification and ranking metrics.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing ['trip_id', 'point_idx', 'label'].
    y_true : np.ndarray
        True binary labels (0 or 1).
    probas : np.ndarray
        Predicted probabilities for positive class (label=1).
    threshold : float, default=0.5
        Decision threshold for converting probabilities into binary class predictions.

    Returns
    -------
    Dict[str, float]
        Dictionary with Top-1 accuracy, Macro F1, Binary F1, Precision, Recall, and Log Loss.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probas = np.asarray(probas, dtype=np.float64)

    # Clamp probabilities for numerical stability in log_loss
    eps = 1e-15
    probas_clamped = np.clip(probas, eps, 1.0 - eps)

    y_pred = (probas >= threshold).astype(np.int64)

    # Calculate Top-1 candidate ranking accuracy
    top1 = calculate_top1_accuracy(df_meta, probas)

    # Calculate classification metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    loss = float(log_loss(y_true, probas_clamped))
    row_accuracy = float(accuracy_score(y_true, y_pred))

    metrics = {
        "top1_accuracy": top1,
        "macro_f1": macro_f1,
        "f1": binary_f1,
        "precision": precision,
        "recall": recall,
        "log_loss": loss,
        "row_accuracy": row_accuracy,
    }
    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary into a readable single-line or multi-line string."""
    p = f"{prefix} " if prefix else ""
    return (
        f"{p}Top-1 Acc: {metrics.get('top1_accuracy', 0.0):.4f} | "
        f"Macro F1: {metrics.get('macro_f1', 0.0):.4f} | "
        f"F1 (pos): {metrics.get('f1', 0.0):.4f} | "
        f"Precision: {metrics.get('precision', 0.0):.4f} | "
        f"Recall: {metrics.get('recall', 0.0):.4f} | "
        f"Log Loss: {metrics.get('log_loss', 0.0):.4f}"
    )


if __name__ == "__main__":
    # Self-test using mock candidate data
    print("Testing evaluate_models module...")
    mock_meta = pd.DataFrame(
        {
            "trip_id": ["01", "01", "01", "02", "02"],
            "point_idx": [0, 0, 0, 0, 0],
            "label": [0, 1, 0, 1, 0],
        }
    )
    # Candidate 1 correctly identified for point 1; Candidate 0 correctly for point 2
    mock_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    res = evaluate_predictions(mock_meta, mock_meta["label"].to_numpy(), mock_probs)
    print("Self-test results:")
    print(format_metrics(res))
    assert res["top1_accuracy"] == 1.0, "Expected 1.0 Top-1 accuracy in mock test!"
    print("Self-test passed successfully!")
