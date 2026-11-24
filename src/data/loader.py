"""
Data loader module for RockWatch.
Provides functions to load raw datasets (read-only) and processed datasets.
"""
from pathlib import Path
from typing import Optional
import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "himachal_landslide"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_raw_modelling_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load original modelling dataset from data/raw/himachal_landslide/landslide_modelling_dataset_12409.csv.
    Does not modify the original file.
    """
    base_dir = Path(data_dir) if data_dir else DEFAULT_RAW_DIR
    file_path = base_dir / "landslide_modelling_dataset_12409.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Raw modelling file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def load_processed_data(
    split: str = "train",
    features_only: bool = False,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load processed dataset splits.
    
    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test', or 'dedup' (for modelling_deduped).
    features_only : bool
        If True, loads features_{split}.csv. If False, loads {split}_spatial.csv.
    processed_dir : Optional[Path]
        Directory where processed files reside.
        
    Returns
    -------
    pd.DataFrame
    """
    base_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    if split == "dedup":
        filename = "modelling_deduped.csv"
    elif features_only:
        filename = f"features_{split}.csv"
    else:
        filename = f"{split}_spatial.csv"
        
    file_path = base_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found at: {file_path}")
    return pd.read_csv(file_path)
