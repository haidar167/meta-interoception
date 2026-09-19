"""Evaluation metrics: ROC-AUC for error detection, calibration, and risk-coverage analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def compute_auc(y_true_error: Sequence[int], y_score_error: Sequence[float]) -> float:
    """Compute ROC-AUC for error detection.

    Args:
        y_true_error: Binary array where 1 indicates an incorrect prediction (error) and 0 indicates correct.
        y_score_error: Predicted score/probability of error (higher score = more likely error).

    Returns:
        ROC-AUC in [0.0, 1.0].
    """
    y_true = np.asarray(y_true_error)
    y_score = np.asarray(y_score_error)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }


def compute_risk_coverage(
    y_correct: Sequence[int],
    confidence_scores: Sequence[float],
    min_coverage: float = 0.05,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage curve for selective prediction.

    Args:
        y_correct: Binary array (1 = model prediction was correct, 0 = model prediction was an error).
        confidence_scores: Confidence score (higher score = more certain to be correct).
        min_coverage: Lowest coverage fraction to evaluate.
        num_points: Number of coverage threshold points.

    Returns:
        (coverages, accuracies)
    """
    y_arr = np.asarray(y_correct)
    conf_arr = np.asarray(confidence_scores)

    # Sort descending by confidence
    sorted_indices = np.argsort(-conf_arr)
    sorted_correct = y_arr[sorted_indices]

    n_total = len(sorted_correct)
    coverages = np.linspace(min_coverage, 1.0, num_points)
    accuracies = []

    for cov in coverages:
        k = max(1, int(round(cov * n_total)))
        acc = float(np.mean(sorted_correct[:k]))
        accuracies.append(acc)

    return coverages, np.array(accuracies)


def find_coverage_at_accuracy(
    coverages: np.ndarray,
    accuracies: np.ndarray,
    target_accuracy: float = 0.99,
) -> float:
    """Find the maximum coverage that achieves at least target_accuracy."""
    valid_indices = np.where(accuracies >= target_accuracy)[0]
    if len(valid_indices) == 0:
        return 0.0
    return float(np.max(coverages[valid_indices]))
