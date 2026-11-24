"""
Model evaluation metrics module for RockWatch.
Calculates threshold-independent (AUC-ROC, PR-AUC) and threshold-dependent metrics (F1, MCC, etc.).
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Find optimal classification threshold on validation probabilities.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities of class 1.
    metric : str
        Metric to optimize: 'f1' or 'mcc'.
    thresholds : Optional[np.ndarray]
        Array of candidate thresholds (default 0.10 to 0.90 in steps of 0.01).
        
    Returns
    -------
    Tuple[float, float]
        (optimal_threshold, best_metric_score)
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 81)

    best_thresh = 0.50
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_thresh = t

    return float(best_thresh), float(best_score)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluate binary predictions across comprehensive metrics.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with AUC-ROC, PR-AUC, Accuracy, Precision, Recall, F1, MCC, Brier Score, and Confusion Matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Check if single class
    has_both_classes = len(np.unique(y_true)) > 1

    auc_roc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": float(threshold),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "brier_score": brier,
        "confusion_matrix": cm,
    }
