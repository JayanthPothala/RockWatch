"""
Advanced machine learning models for RockWatch.
Implements:
1. RandomForestClassifier (Bagged ensemble of de-correlated decision trees)
2. GradientBoostingClassifier (Boosted trees with learning rate shrinkage)
3. SVC with RBF kernel (Radial Basis Function Support Vector Machine)
"""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def create_random_forest(
    n_estimators: int = 300,
    max_depth: Optional[int] = 12,
    min_samples_leaf: int = 10,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Random Forest ensemble.
    Tuned with min_samples_leaf=10 and max_depth=12 to prevent spatial overfitting.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def create_gradient_boosting(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    min_samples_leaf: int = 15,
    subsample: float = 0.85,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """
    Gradient Boosting ensemble.
    Tuned with conservative learning rate (0.05) and subsampling (0.85)
    to promote generalization across geographic covariate shifts.
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
    )


def create_svm_rbf(
    C: float = 1.0,
    gamma: str = "scale",
    probability: bool = True,
    random_state: int = 42,
) -> SVC:
    """
    Support Vector Machine with Radial Basis Function (RBF) kernel.
    Outputs Platt-calibrated susceptibility probabilities.
    """
    return SVC(
        C=C,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
    )
