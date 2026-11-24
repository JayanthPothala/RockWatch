"""
Spatial splitting module for RockWatch.
Implements latitude-band blocking with spatial exclusion buffers.
"""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def compute_latitude_cuts(
    df: pd.DataFrame,
    q1: float = 0.70,
    q2: float = 0.85,
) -> Tuple[float, float]:
    """
    Compute deterministic latitude cut points from deduplicated dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Deduplicated dataset.
    q1 : float
        Quantile for train/val boundary (default 0.70).
    q2 : float
        Quantile for val/test boundary (default 0.85).
        
    Returns
    -------
    Tuple[float, float]
        (cut1, cut2) in degrees North.
    """
    lat_sorted = np.sort(df["latitude"].values)
    n = len(lat_sorted)
    idx_q1 = int(np.floor(n * q1))
    idx_q2 = int(np.floor(n * q2))
    
    cut1 = float(lat_sorted[idx_q1 - 1])
    cut2 = float(lat_sorted[idx_q2 - 1])
    return cut1, cut2


def apply_spatial_split(
    df: pd.DataFrame,
    cut1: float,
    cut2: float,
    buffer_m: float = 500.0,
    km_per_deg_lat: float = 111.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Apply latitude-based spatial split with a buffer exclusion zone around boundary cuts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Deduplicated dataset containing 'latitude'.
    cut1 : float
        Latitude boundary between Train and Val.
    cut2 : float
        Latitude boundary between Val and Test.
    buffer_m : float
        Buffer distance in meters around each cut point.
    km_per_deg_lat : float
        Distance in km per degree latitude (~111.0 km).
        
    Returns
    -------
    Tuple containing:
        - train_df (pd.DataFrame)
        - val_df (pd.DataFrame)
        - test_df (pd.DataFrame)
        - excluded_df (pd.DataFrame)
        - metadata (Dict[str, Any])
    """
    buffer_deg = buffer_m / (km_per_deg_lat * 1000.0)
    
    in_cut1_buffer = (df["latitude"] >= cut1 - buffer_deg) & (df["latitude"] <= cut1 + buffer_deg)
    in_cut2_buffer = (df["latitude"] >= cut2 - buffer_deg) & (df["latitude"] <= cut2 + buffer_deg)
    excl_mask = in_cut1_buffer | in_cut2_buffer
    
    excluded_df = df[excl_mask].copy()
    df_work = df[~excl_mask].copy()
    
    train_df = df_work[df_work["latitude"] < cut1].copy()
    val_df = df_work[(df_work["latitude"] >= cut1) & (df_work["latitude"] < cut2)].copy()
    test_df = df_work[df_work["latitude"] >= cut2].copy()
    
    metadata = {
        "cut1": cut1,
        "cut2": cut2,
        "buffer_m": buffer_m,
        "buffer_deg": buffer_deg,
        "n_excluded_cut1": int(in_cut1_buffer.sum()),
        "n_excluded_cut2": int(in_cut2_buffer.sum()),
        "n_excluded_total": int(excl_mask.sum()),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    
    return train_df, val_df, test_df, excluded_df, metadata
