"""
Spatial visualization module for RockWatch.
Plots geographic distributions, boundary cuts, and spatial split layouts.
"""
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spatial_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cut1: float = 31.713044,
    cut2: float = 32.144235,
    buffer_m: float = 500.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot geographic scatter of Train, Validation, and Test splits in Himachal Pradesh,
    showing latitude boundary cuts and exclusion buffer zones.
    """
    buffer_deg = buffer_m / 111000.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Subplot 1: Splits
    ax1 = axes[0]
    ax1.scatter(train_df["longitude"], train_df["latitude"], c="#2b83ba", s=6, alpha=0.5, label=f"TRAIN Split (n={len(train_df):,})")
    ax1.scatter(val_df["longitude"], val_df["latitude"], c="#fdae61", s=8, alpha=0.6, label=f"VAL Split (n={len(val_df):,})")
    ax1.scatter(test_df["longitude"], test_df["latitude"], c="#d7191c", s=8, alpha=0.6, label=f"TEST Split (n={len(test_df):,})")

    # Draw CUT lines
    for cut, label in [(cut1, "CUT 1 (31.713°N)"), (cut2, "CUT 2 (32.144°N)")]:
        ax1.axhline(cut, color="black", linestyle="--", linewidth=1.2, label=label)
        ax1.axhspan(cut - buffer_deg, cut + buffer_deg, color="gray", alpha=0.35)

    ax1.set_title("Geographical Block Partitioning (70% / 15% / 15%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Longitude (°E)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Latitude (°N)", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5, facecolor="white")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Landslide vs Pseudo-absence
    ax2 = axes[1]
    all_df = pd.concat([train_df, val_df, test_df])
    c0 = all_df[all_df["class"] == 0]
    c1 = all_df[all_df["class"] == 1]

    ax2.scatter(c0["longitude"], c0["latitude"], c="#1b9e77", s=5, alpha=0.4, label=f"Pseudo-absence (class=0, n={len(c0):,})")
    ax2.scatter(c1["longitude"], c1["latitude"], c="#d95f02", s=6, alpha=0.5, label=f"Landslide (class=1, n={len(c1):,})")

    ax2.set_title("Spatial Distribution of Landslide Inventory vs Pseudo-Absences", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitude (°E)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Latitude (°N)", fontsize=10, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5, facecolor="white")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("RockWatch - Himachal Pradesh Spatial Structure & Validation Split", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
