"""
Baseline models module for RockWatch.
Implements:
1. DummyClassifier (sanity lower-bound benchmark)
2. LogisticRegression (linear, interpretable baseline)
3. DecisionTreeClassifier (interpretable nonlinear tree baseline)
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Default feature subsets
ALL_FEATURES = [
    "sin_aspect",
    "cos_aspect",
    "Dist_River_z_corr",
    "Elevation_z_corr",
    "FlowAccumulation_z_corr",
    "NDVI_z_corr",
    "Rainfall_z_corr",
    "Slope_z_corr",
    "LULC_10",
    "LULC_20",
    "LULC_30",
    "LULC_40",
    "LULC_50",
    "LULC_60",
    "LULC_70",
    "LULC_80",
    "LULC_100",
]

# For Logistic Regression: drop LULC_100 to avoid dummy variable trap
LR_FEATURES = [f for f in ALL_FEATURES if f != "LULC_100"]


def create_dummy_baseline(strategy: str = "prior") -> DummyClassifier:
    """
    Lower-bound benchmark model.
    'prior' outputs class prior probability (e.g. ~0.49).
    """
    return DummyClassifier(strategy=strategy)


def create_logistic_regression(
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Linear baseline with L2 regularization.
    Evaluated with 16 predictors (LULC_100 dropped as reference).
    """
    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )


def create_decision_tree(
    max_depth: int = 5,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Interpretable shallow decision tree baseline with 17 predictors.
    Constrained to max_depth=5 to prevent overfitting to local training clusters.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
