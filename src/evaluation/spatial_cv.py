"""
Spatial Cross-Validation module for RockWatch.
Implements spatial block partitioning using K-Means clustering on geographic coordinates
to evaluate hyperparameter configurations without spatial data leakage.
"""
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class SpatialBlockKFold:
    """
    Spatial Block K-Fold cross-validator.
    Partitions geographical coordinates into K spatially contiguous spatial clusters.
    """
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray, coords: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets by spatial clusters.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        coords : np.ndarray
            Spatial coordinates array of shape (N, 2), containing [latitude, longitude].
            
        Yields
        ------
        train_idx, val_idx : Tuple[np.ndarray, np.ndarray]
        """
        kmeans = KMeans(n_clusters=self.n_splits, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)

        for fold in range(self.n_splits):
            val_mask = cluster_labels == fold
            train_mask = ~val_mask
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(train_mask)[0]
            yield train_idx, val_idx
