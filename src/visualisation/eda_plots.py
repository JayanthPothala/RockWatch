"""
Exploratory Data Analysis (EDA) plotting module for RockWatch.
Generates publication-quality charts using Matplotlib and Seaborn.
"""
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot class distribution across geographic splits.
    """
    splits_data = []
    for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
        c1 = (df["class"] == 1).sum()
        c0 = (df["class"] == 0).sum()
        total = len(df)
        splits_data.append({"Split": name, "Class": "Landslide (1)", "Count": c1, "Pct": c1 / total * 100})
        splits_data.append({"Split": name, "Class": "Pseudo-Absence (0)", "Count": c0, "Pct": c0 / total * 100})

    summary_df = pd.DataFrame(splits_data)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    palette = {"Landslide (1)": "#d95f02", "Pseudo-Absence (0)": "#1b9e77"}
    bars = sns.barplot(
        data=summary_df,
        x="Split",
        y="Count",
        hue="Class",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    for p in bars.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height):,}\n({height / 8188 * 100:.1f}%)" if height > 3000 else f"{int(height):,}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

    ax.set_title("Class Distribution Across Spatial Splits (RockWatch)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Spatial Split", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.legend(title="Class Label", frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "class",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot feature histograms / KDE split by class label.
    """
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.8 * n_rows), dpi=300)
    axes = axes.flatten()

    palette = {0: "#1b9e77", 1: "#d95f02"}
    labels = {0: "Pseudo-absence (0)", 1: "Landslide (1)"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for cls in [0, 1]:
            data_cls = df[df[target_col] == cls][col].dropna()
            sns.kdeplot(
                data=data_cls,
                ax=ax,
                color=palette[cls],
                label=labels[cls],
                fill=True,
                alpha=0.3,
                linewidth=1.8,
            )
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Standardized Value" if "_corr" in col else "Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        if i == 0:
            ax.legend(frameon=True, fontsize=8)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Feature Distributions by Landslide Susceptibility Class", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    cols: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot Spearman correlation heatmap for continuous predictors and target.
    """
    corr = df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation"},
        annot_kws={"size": 8, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_title("Predictor Correlation Heatmap (Spearman Rank)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_lulc_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot landslide prevalence across LULC categories.
    """
    lulc_labels = {
        "LULC_10": "Trees (10)",
        "LULC_20": "Shrubland (20)",
        "LULC_30": "Grassland (30)",
        "LULC_40": "Cropland (40)",
        "LULC_50": "Built-up (50)",
        "LULC_60": "Bare/Sparse (60)",
        "LULC_70": "Snow/Ice (70)",
        "LULC_80": "Water Bodies (80)",
        "LULC_100": "Moss/Lichen (100)",
    }

    records = []
    for col, name in lulc_labels.items():
        if col in df.columns:
            subset = df[df[col] == 1]
            total = len(subset)
            if total > 0:
                c1 = (subset["class"] == 1).sum()
                rate = c1 / total * 100
                records.append({
                    "LULC": name,
                    "Total": total,
                    "Landslides": c1,
                    "Susceptibility Rate (%)": rate,
                })

    res_df = pd.DataFrame(records).sort_values("Susceptibility Rate (%)", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    ax2 = ax1.twinx()

    x = np.arange(len(res_df))
    width = 0.4

    ax1.bar(x - width / 2, res_df["Total"], width=width, color="#4575b4", label="Total Samples", edgecolor="black", linewidth=0.5)
    ax1.bar(x - width / 2, res_df["Landslides"], width=width, color="#d73027", label="Landslide Events", edgecolor="black", linewidth=0.5)

    ax2.plot(x, res_df["Susceptibility Rate (%)"], color="#2b83ba", marker="o", linewidth=2.2, label="Prevalence Rate (%)")
    ax2.axhline(50.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="Baseline (50%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(res_df["LULC"], rotation=30, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Landslide Prevalence Rate (%)", fontsize=11, fontweight="bold", color="#2b83ba")
    ax2.set_ylim(0, 105)

    ax1.legend(loc="upper left", frameon=True)
    ax2.legend(loc="upper right", frameon=True)

    plt.title("Landslide Prevalence Across Land Use / Land Cover (LULC) Classes", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_elevation_slope_profile(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot 2D distribution of Elevation vs Slope for landslides vs pseudo-absences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharex=True, sharey=True)

    elev_col = "elevation_m" if "elevation_m" in df.columns else "Elevation_z_corr"
    slope_col = "slope_deg" if "slope_deg" in df.columns else "Slope_z_corr"

    c0 = df[df["class"] == 0]
    c1 = df[df["class"] == 1]

    sns.kdeplot(
        data=c0,
        x=elev_col,
        y=slope_col,
        cmap="Greens",
        fill=True,
        thresh=0.05,
        levels=10,
        ax=axes[0],
    )
    axes[0].set_title(f"Pseudo-Absences (class=0) - n={len(c0):,}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Elevation (m)" if "elevation_m" in df.columns else "Elevation (std)", fontsize=10)
    axes[0].set_ylabel("Slope (degrees)" if "slope_deg" in df.columns else "Slope (std)", fontsize=10)

    sns.kdeplot(
        data=c1,
        x=elev_col,
        y=slope_col,
        cmap="Oranges",
        fill=True,
        thresh=0.05,
        levels=10,
        ax=axes[1],
    )
    axes[1].set_title(f"Landslides (class=1) - n={len(c1):,}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Elevation (m)" if "elevation_m" in df.columns else "Elevation (std)", fontsize=10)

    fig.suptitle("Elevation vs Slope Joint Density Distribution", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
