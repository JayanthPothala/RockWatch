"""Evaluation metrics and validation routines for RockWatch."""
from .metrics import evaluate_predictions, find_optimal_threshold
from .spatial_cv import SpatialBlockKFold
from .calibration import evaluate_calibration, compute_expected_calibration_error

__all__ = [
    "evaluate_predictions",
    "find_optimal_threshold",
    "SpatialBlockKFold",
    "evaluate_calibration",
    "compute_expected_calibration_error",
]
