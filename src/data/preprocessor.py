"""
Data preprocessing module for RockWatch.
Handles deduplication, feature re-standardisation fitted strictly on training data,
and construction of final feature matrices.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCALED_PREDICTOR_COLS = [
    "Dist_River_z",
    "Elevation_z",
    "FlowAccumulation_z",
    "NDVI_z",
    "Rainfall_z",
    "Slope_z",
]

FINAL_FEATURE_COLUMNS = [
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

TARGET_COLUMN = "class"


def deduplicate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate coordinates by retaining the row with the minimum sample_id.
    All 409 duplicate groups (1,081 rows) are verified to be field-identical
    in class labels, z-scores, LULC categories, and inventory identifiers.
    
    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe (11,737 rows).
    """
    df_dedup = df.loc[df.groupby(["latitude", "longitude"])["sample_id"].idxmin()].copy()
    if df_dedup.duplicated(subset=["latitude", "longitude"]).sum() != 0:
        raise ValueError("Duplicate coordinates still remain after deduplication.")
    return df_dedup


def re_standardise_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = SCALED_PREDICTOR_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fit StandardScaler strictly on training split and transform train, validation, and test splits.
    This creates '*_z_corr' columns.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        Updated (train_df, val_df, test_df) with '*_z_corr' columns, plus scaler parameter metadata.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    scaler_dict: Dict[str, Dict[str, float]] = {}
    for col, mean_val, std_val in zip(feature_cols, scaler.mean_, scaler.scale_):
        scaler_dict[col] = {
            "train_mean": float(mean_val),
            "train_std": float(std_val),
        }
        train_df[f"{col}_corr"] = (train_df[col] - mean_val) / std_val
        val_df[f"{col}_corr"] = (val_df[col] - mean_val) / std_val
        test_df[f"{col}_corr"] = (test_df[col] - mean_val) / std_val

    scaler_params = {
        "fitted_on": "TRAIN split (after deduplication and exclusion zone)",
        "n_train_rows": len(train_df),
        "features": scaler_dict,
        "excluded": {
            "Aspect_z": "Replaced by sin_aspect and cos_aspect; not scaled"
        },
    }

    return train_df, val_df, test_df, scaler_params


def build_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FINAL_FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract the 17 predictor columns and target column to produce feature matrices.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (features_train, features_val, features_test)
    """
    all_cols = feature_cols + [target_col]
    features_train = train_df[all_cols].copy()
    features_val = val_df[all_cols].copy()
    features_test = test_df[all_cols].copy()

    return features_train, features_val, features_test
