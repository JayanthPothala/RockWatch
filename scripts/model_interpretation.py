"""
RockWatch - Phase 6: Model Interpretation & Partial Dependence Analysis
========================================================================
Analyzes the decision logic of the Champion Gradient Boosting model:
1. Generates 1D Partial Dependence Plots (PDP) for top environmental features:
   - Elevation_z_corr, Slope_z_corr, Rainfall_z_corr, NDVI_z_corr, Dist_River_z_corr, cos_aspect
2. Generates 2D Partial Dependence Interaction Contour (Elevation vs Slope)
3. Identifies critical geomorphological tipping points / failure thresholds
4. Saves plots to reports/figures/
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_data
from src.models.baseline import ALL_FEATURES


def run_model_interpretation():
    print("=" * 80)
    print("ROCKWATCH - PHASE 6: MODEL INTERPRETATION & PARTIAL DEPENDENCE ANALYSIS")
    print("=" * 80)

    # 1. Load Data & Model
    print("\n[1] Loading processed training data and champion model...")
    f_train = load_processed_data("train", features_only=True)
    X_train = f_train[ALL_FEATURES].values

    models_dir = PROJECT_ROOT / "models" / "trained"
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    eval_dir = PROJECT_ROOT / "models" / "evaluation"

    gb_model = joblib.load(models_dir / "gradient_boosting_v1.joblib")
    print(f"  Loaded model: gradient_boosting_v1.joblib")

    # 2. Compute 1D Partial Dependence for Top 6 Predictors
    top_features = [
        "Elevation_z_corr",
        "Slope_z_corr",
        "Rainfall_z_corr",
        "NDVI_z_corr",
        "Dist_River_z_corr",
        "cos_aspect",
    ]

    print("\n[2] Computing 1D Partial Dependence across top features...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    feature_units = {
        "Elevation_z_corr": "Standardized Elevation (σ)",
        "Slope_z_corr": "Standardized Slope (σ)",
        "Rainfall_z_corr": "Standardized Rainfall (σ)",
        "NDVI_z_corr": "Standardized NDVI (σ)",
        "Dist_River_z_corr": "Distance to River (σ)",
        "cos_aspect": "Cosine Aspect (North +1, South -1)",
    }

    pdp_summary = {}

    for i, feat_name in enumerate(top_features):
        feat_idx = ALL_FEATURES.index(feat_name)
        pdp_res = partial_dependence(gb_model, X_train, features=[feat_idx], kind="average")
        grid_vals = pdp_res["grid_values"][0]
        avg_preds = pdp_res["average"][0]

        # Convert log-odds marginal to probability scale via sigmoid
        prob_preds = 1.0 / (1.0 + np.exp(-avg_preds))

        ax = axes[i]
        ax.plot(grid_vals, prob_preds, color="#d7191c", linewidth=2.4)
        ax.axhline(0.50, color="gray", linestyle="--", alpha=0.5, label="Prior Balance (0.50)")

        # Mark threshold tipping points
        grad = np.gradient(prob_preds, grid_vals)
        tipping_idx = np.argmax(np.abs(grad))
        tipping_val = grid_vals[tipping_idx]
        ax.axvline(tipping_val, color="#2b83ba", linestyle=":", alpha=0.8, label=f"Max Gradient ({tipping_val:+.2f})")

        ax.set_title(f"{feat_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feature_units.get(feat_name, feat_name), fontsize=9)
        ax.set_ylabel("Marginal Susceptibility Probability", fontsize=9)
        ax.set_ylim(0.15, 0.85)
        ax.legend(loc="best", frameon=True, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

        pdp_summary[feat_name] = {
            "min_prob": float(np.min(prob_preds)),
            "max_prob": float(np.max(prob_preds)),
            "tipping_point": float(tipping_val),
        }
        print(f"  {feat_name:22s}: Probability Range = [{np.min(prob_preds):.3f}, {np.max(prob_preds):.3f}], Max Tipping Point = {tipping_val:+.2f}σ")

    fig.suptitle("RockWatch - Partial Dependence Plots (Gradient Boosting)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    p_pdp = fig_dir / "partial_dependence_plots.png"
    fig.savefig(p_pdp, bbox_inches="tight")
    print(f"\n  Saved PDP figure: {p_pdp.name}")

    # 3. 2D Interaction PDP (Elevation vs Slope)
    print("\n[3] Computing 2D Interaction Partial Dependence (Elevation vs Slope)...")
    elev_idx = ALL_FEATURES.index("Elevation_z_corr")
    slope_idx = ALL_FEATURES.index("Slope_z_corr")

    pdp_2d = partial_dependence(gb_model, X_train, features=[(elev_idx, slope_idx)], kind="average")
    grid_elev = pdp_2d["grid_values"][0]
    grid_slope = pdp_2d["grid_values"][1]
    avg_2d = pdp_2d["average"][0]
    prob_2d = 1.0 / (1.0 + np.exp(-avg_2d))

    fig2, ax2 = plt.subplots(figsize=(9, 7), dpi=300)
    ELEV, SLOPE = np.meshgrid(grid_elev, grid_slope, indexing="ij")

    contour = ax2.contourf(ELEV, SLOPE, prob_2d, levels=15, cmap="YlOrRd", alpha=0.85)
    cbar = fig2.colorbar(contour, ax=ax2)
    cbar.set_label("Marginal Landslide Probability", fontsize=10, fontweight="bold")

    # Add contour line at the 0.50 threshold
    lines = ax2.contour(ELEV, SLOPE, prob_2d, levels=[0.37, 0.50, 0.65], colors=["black", "blue", "darkred"], linestyles=["--", "-", "-."], linewidths=[1.2, 1.8, 1.5])
    ax2.clabel(lines, inline=True, fontsize=9, fmt="P=%.2f")

    ax2.set_title("2D Partial Dependence Interaction: Elevation vs Slope Angle", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Standardized Elevation (Elevation_z_corr)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Standardized Slope (Slope_z_corr)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    p_2d = fig_dir / "elevation_slope_interaction_pdp.png"
    fig2.savefig(p_2d, bbox_inches="tight")
    print(f"  Saved 2D Interaction figure: {p_2d.name}")

    # Save summary
    with open(eval_dir / "interpretation_summary.json", "w", encoding="utf-8") as f:
        json.dump(pdp_summary, f, indent=2)

    print("\nModel interpretation analysis complete.")


if __name__ == "__main__":
    run_model_interpretation()
