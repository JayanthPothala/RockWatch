"""
Feature engineering module for RockWatch.
Implements circular aspect transformation and diagnostic flags.
"""
import numpy as np
import pandas as pd


def add_water_body_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add diagnostic flag for landslide inventory events mapped to water body land cover.
    
    flag_water_lulc = 1 for rows where (LULC_80 == 1) and (class == 1)
    flag_water_lulc = 0 otherwise
    """
    df = df.copy()
    water_landslide_mask = (df["LULC_80"] == 1) & (df["class"] == 1)
    df["flag_water_lulc"] = water_landslide_mask.astype(int)
    return df


def compute_aspect_trig(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform circular transformation on aspect_deg:
    sin_aspect = sin(aspect_deg * pi / 180)
    cos_aspect = cos(aspect_deg * pi / 180)
    
    For flat terrain (slope_deg == 0), aspect is physically undefined.
    Set sin_aspect = 0.0 and cos_aspect = 0.0 for slope_deg == 0.
    """
    df = df.copy()
    if "aspect_deg" not in df.columns:
        raise ValueError("Column 'aspect_deg' is required for circular aspect transformation.")
    
    rad = np.radians(df["aspect_deg"])
    df["sin_aspect"] = np.sin(rad)
    df["cos_aspect"] = np.cos(rad)
    
    if "slope_deg" in df.columns:
        flat_mask = df["slope_deg"] == 0
        df.loc[flat_mask, "sin_aspect"] = 0.0
        df.loc[flat_mask, "cos_aspect"] = 0.0
        
    return df
