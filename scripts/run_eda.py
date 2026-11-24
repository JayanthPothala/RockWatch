"""
RockWatch - Phase 2: Exploratory Data Analysis (EDA) Script
===========================================================
Executes comprehensive exploratory analysis on the processed Himachal Pradesh dataset:
1. Generates publication-ready figures in reports/figures/:
   - class_distribution.png
   - spatial_split_map.png
   - feature_distributions.png
   - correlation_heatmap.png
   - lulc_susceptibility.png
   - elevation_slope_profile.png
2. Computes statistical significance & effect size (Cohen's d, Mann-Whitney U).
3. Computes univariate ROC-AUC per predictor to rank predictive power.
4. Calculates Variance Inflation Factors (VIF) to assess multicollinearity.
5. Quantifies geographic covariate shift across Train, Validation, and Test splits.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_data
from src.visualisation.eda_plots import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_matrix,
    plot_lulc_distribution,
    plot_elevation_slope_profile,
)
from src.visualisation.spatial_plots import plot_spatial_splits


def compute_vif(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) using OLS R-squared for each feature."""
    vif_data = []
    from sklearn.linear_model import LinearRegression
    for i, col in enumerate(feature_cols):
        X = df[feature_cols].drop(columns=[col])
        y = df[col]
        r_sq = LinearRegression().fit(X, y).score(X, y)
        vif = 1.0 / (1.0 - r_sq) if r_sq < 0.9999 else 999.0
        vif_data.append({"Feature": col, "VIF": vif, "R2_with_others": r_sq})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def run_eda():
    print("=" * 80)
    print("ROCKWATCH - PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # 1. Load Processed Data
    print("\n[1] Loading processed dataset splits...")
    train_spatial = load_processed_data("train", features_only=False)
    val_spatial = load_processed_data("val", features_only=False)
    test_spatial = load_processed_data("test", features_only=False)

    f_train = load_processed_data("train", features_only=True)
    f_val = load_processed_data("val", features_only=True)
    f_test = load_processed_data("test", features_only=True)
    all_spatial = pd.concat([train_spatial, val_spatial, test_spatial], ignore_index=True)
    all_features = pd.concat([f_train, f_val, f_test], ignore_index=True)

    print(f"  Train: {len(train_spatial):,} | Val: {len(val_spatial):,} | Test: {len(test_spatial):,}")
    print(f"  Total modelling samples: {len(all_spatial):,}")

    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate Plots
    print("\n[2] Generating figures in reports/figures/...")

    # Fig 1: Class distribution
    p1 = fig_dir / "class_distribution.png"
    plot_class_distribution(train_spatial, val_spatial, test_spatial, save_path=p1)
    print(f"  Saved: {p1.name}")

    # Fig 2: Spatial split map
    p2 = fig_dir / "spatial_split_map.png"
    plot_spatial_splits(train_spatial, val_spatial, test_spatial, save_path=p2)
    print(f"  Saved: {p2.name}")

    # Fig 3: Feature distributions by class
    p3 = fig_dir / "feature_distributions.png"
    continuous_features = [
        "Slope_z_corr", "Elevation_z_corr", "NDVI_z_corr", "Rainfall_z_corr",
        "Dist_River_z_corr", "FlowAccumulation_z_corr", "sin_aspect", "cos_aspect"
    ]
    plot_feature_distributions(f_train, continuous_features, target_col="class", save_path=p3)
    print(f"  Saved: {p3.name}")

    # Fig 4: Correlation heatmap
    p4 = fig_dir / "correlation_heatmap.png"
    plot_correlation_matrix(f_train, continuous_features + ["class"], save_path=p4)
    print(f"  Saved: {p4.name}")

    # Fig 5: LULC prevalence
    p5 = fig_dir / "lulc_susceptibility.png"
    plot_lulc_distribution(train_spatial, save_path=p5)
    print(f"  Saved: {p5.name}")

    # Fig 6: Elevation vs Slope profile
    p6 = fig_dir / "elevation_slope_profile.png"
    plot_elevation_slope_profile(train_spatial, save_path=p6)
    print(f"  Saved: {p6.name}")

    # 3. Statistical Analysis of Feature Separation Power
    print("\n" + "=" * 80)
    print("FEATURE PREDICTIVE POWER & SEPARATION ANALYSIS (TRAIN SPLIT)")
    print("=" * 80)

    stats_list = []
    all_predictors = [c for c in f_train.columns if c != "class"]

    c1 = f_train[f_train["class"] == 1]
    c0 = f_train[f_train["class"] == 0]

    for feat in all_predictors:
        vals1 = c1[feat].dropna()
        vals0 = c0[feat].dropna()

        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(vals1, vals0, alternative="two-sided")

        # Cohen's d
        mean_diff = vals1.mean() - vals0.mean()
        pooled_std = np.sqrt(((len(vals1) - 1) * vals1.var() + (len(vals0) - 1) * vals0.var()) / (len(vals1) + len(vals0) - 2))
        cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Univariate ROC-AUC
        # Check direction
        auc = roc_auc_score(f_train["class"], f_train[feat])
        if auc < 0.5:
            auc = 1.0 - auc  # direction-invariant predictive power

        stats_list.append({
            "Feature": feat,
            "Mean_Class1": float(vals1.mean()),
            "Mean_Class0": float(vals0.mean()),
            "Delta": float(mean_diff),
            "Cohen_d": float(cohen_d),
            "P_Value": float(p_val),
            "Univariate_AUC": float(auc),
        })

    stats_df = pd.DataFrame(stats_list).sort_values("Univariate_AUC", ascending=False)
    print(stats_df.to_string(index=False, formatters={
        "Mean_Class1": "{:+.3f}".format,
        "Mean_Class0": "{:+.3f}".format,
        "Delta": "{:+.3f}".format,
        "Cohen_d": "{:+.3f}".format,
        "P_Value": "{:.2e}".format,
        "Univariate_AUC": "{:.4f}".format,
    }))

    # 4. Multicollinearity Assessment (VIF)
    print("\n" + "=" * 80)
    print("MULTICOLLINEARITY ASSESSMENT (VIF ON CONTINUOUS PREDICTORS)")
    print("=" * 80)
    vif_df = compute_vif(f_train, continuous_features)
    print(vif_df.to_string(index=False, formatters={"VIF": "{:.2f}".format, "R2_with_others": "{:.3f}".format}))

    # 5. Geographic Domain Shift Across Splits
    print("\n" + "=" * 80)
    print("GEOGRAPHIC COVARIATE SHIFT ANALYSIS (FEATURE MEANS ACROSS SPLITS)")
    print("=" * 80)
    shift_data = []
    for feat in continuous_features:
        m_train = f_train[feat].mean()
        m_val = f_val[feat].mean()
        m_test = f_test[feat].mean()
        shift_data.append({
            "Feature": feat,
            "Train_Mean": m_train,
            "Val_Mean": m_val,
            "Test_Mean": m_test,
            "Max_Shift_Std": max(abs(m_val - m_train), abs(m_test - m_train)),
        })
    shift_df = pd.DataFrame(shift_data).sort_values("Max_Shift_Std", ascending=False)
    print(shift_df.to_string(index=False, formatters={
        "Train_Mean": "{:+.3f}".format,
        "Val_Mean": "{:+.3f}".format,
        "Test_Mean": "{:+.3f}".format,
        "Max_Shift_Std": "{:.3f}".format,
    }))

    print("\nEDA analysis complete. Figures and statistical outputs ready.")


if __name__ == "__main__":
    run_eda()
