"""
RockWatch - Phase 6: Regional Landslide Susceptibility Mapping
==============================================================
Generates susceptibility maps and spatial predictions for Himachal Pradesh:
1. Applies Champion Gradient Boosting model to generate continuous susceptibility scores P in [0, 1].
2. Classifies locations into 4 geotechnical susceptibility tiers:
   - Low Susceptibility:       P < 0.20
   - Moderate Susceptibility:  0.20 <= P < 0.37
   - High Susceptibility:      0.37 <= P < 0.70
   - Very High Susceptibility: P >= 0.70
3. Generates GeoJSON and CSV prediction tables:
   - data/processed/himachal_susceptibility_predictions.csv
   - data/processed/himachal_susceptibility_points.geojson
4. Produces publication-quality spatial maps:
   - reports/figures/susceptibility_map_himachal.png
   - reports/figures/district_hazard_ranking.png
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_data
from src.models.baseline import ALL_FEATURES
from src.data.preprocessor import SCALED_PREDICTOR_COLS


def generate_susceptibility_map():
    print("=" * 80)
    print("ROCKWATCH - PHASE 6: REGIONAL LANDSLIDE SUSCEPTIBILITY MAPPING")
    print("=" * 80)

    # 1. Load Data, Scaler & Champion Model
    print("\n[1] Loading processed deduplicated dataset, scaler parameters, and model...")
    dedup_path = PROJECT_ROOT / "data" / "processed" / "modelling_deduped.csv"
    scaler_path = PROJECT_ROOT / "data" / "processed" / "scaler_params.json"
    model_path = PROJECT_ROOT / "models" / "trained" / "gradient_boosting_v1.joblib"

    df_dedup = pd.read_csv(dedup_path)
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_info = json.load(f)
    gb_model = joblib.load(model_path)

    print(f"  Deduplicated samples: {len(df_dedup):,} rows across Himachal Pradesh")

    # 2. Re-standardize 6 predictors using training scaler parameters
    scaled_cols = ["Dist_River_z", "Elevation_z", "FlowAccumulation_z", "NDVI_z", "Rainfall_z", "Slope_z"]
    for col in scaled_cols:
        mean_val = scaler_info["features"][col]["train_mean"]
        std_val = scaler_info["features"][col]["train_std"]
        df_dedup[f"{col}_corr"] = (df_dedup[col] - mean_val) / std_val

    # 3. Predict Susceptibility Probabilities
    print("\n[2] Computing continuous landslide susceptibility probabilities P in [0, 1]...")
    feature_cols = [
        "sin_aspect", "cos_aspect", "Dist_River_z_corr", "Elevation_z_corr",
        "FlowAccumulation_z_corr", "NDVI_z_corr", "Rainfall_z_corr", "Slope_z_corr",
        "LULC_10", "LULC_20", "LULC_30", "LULC_40", "LULC_50", "LULC_60",
        "LULC_70", "LULC_80", "LULC_100",
    ]
    X_all = df_dedup[feature_cols].values
    susceptibility_scores = gb_model.predict_proba(X_all)[:, 1]
    df_dedup["susceptibility_score"] = susceptibility_scores

    # 4. Classify into 4 Susceptibility Tiers
    # Standard civil defense categories:
    # Low: < 0.20, Moderate: 0.20-0.37, High: 0.37-0.70, Very High: >= 0.70
    conditions = [
        (df_dedup["susceptibility_score"] < 0.20),
        (df_dedup["susceptibility_score"] >= 0.20) & (df_dedup["susceptibility_score"] < 0.37),
        (df_dedup["susceptibility_score"] >= 0.37) & (df_dedup["susceptibility_score"] < 0.70),
        (df_dedup["susceptibility_score"] >= 0.70),
    ]
    tier_names = ["Low", "Moderate", "High", "Very High"]
    tier_codes = [1, 2, 3, 4]
    df_dedup["susceptibility_class"] = np.select(conditions, tier_names, default="Moderate")
    df_dedup["susceptibility_tier_code"] = np.select(conditions, tier_codes, default=2)

    tier_counts = df_dedup["susceptibility_class"].value_counts()[tier_names]
    print("\n  === SUSCEPTIBILITY TIER DISTRIBUTION ===")
    for name in tier_names:
        cnt = tier_counts[name]
        pct = cnt / len(df_dedup) * 100
        print(f"    {name:12s}: {cnt:5d} locations ({pct:.1f}%)")

    # 5. Save Prediction Tables
    out_csv = PROJECT_ROOT / "data" / "processed" / "himachal_susceptibility_predictions.csv"
    export_cols = [
        "sample_id", "latitude", "longitude", "class", "susceptibility_score",
        "susceptibility_class", "susceptibility_tier_code", "elevation_m", "slope_deg"
    ]
    df_dedup[export_cols].to_csv(out_csv, index=False)
    print(f"\n  Saved susceptibility predictions CSV: {out_csv.name}")

    # Generate GeoJSON
    out_geojson = PROJECT_ROOT / "data" / "processed" / "himachal_susceptibility_points.geojson"
    features = []
    for _, row in df_dedup.iterrows():
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            "properties": {
                "sample_id": int(row["sample_id"]),
                "ground_truth_class": int(row["class"]),
                "susceptibility_score": round(float(row["susceptibility_score"]), 4),
                "susceptibility_class": str(row["susceptibility_class"]),
                "elevation_m": round(float(row["elevation_m"]), 1),
                "slope_deg": round(float(row["slope_deg"]), 1),
            },
        }
        features.append(feat)

    geojson_obj = {
        "type": "FeatureCollection",
        "name": "RockWatch_Himachal_Susceptibility",
        "features": features,
    }
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)
    print(f"  Saved susceptibility GeoJSON: {out_geojson.name}")

    # 6. Generate Publication-Quality Susceptibility Map
    print("\n[3] Generating High-Resolution Regional Susceptibility Map...")
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tier_colors = {
        "Low": "#2ca25f",        # Green
        "Moderate": "#ffbc12",   # Gold/Yellow
        "High": "#fa8c16",       # Orange
        "Very High": "#d7191c",  # Vivid Red
    }

    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)

    # Plot points by susceptibility tier (ordered so high risk is plotted on top)
    for tier in ["Low", "Moderate", "High", "Very High"]:
        subset = df_dedup[df_dedup["susceptibility_class"] == tier]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            c=tier_colors[tier],
            s=12 if tier in ["High", "Very High"] else 8,
            alpha=0.65 if tier in ["High", "Very High"] else 0.45,
            label=f"{tier} Risk ({len(subset):,} locations)",
            edgecolors="none",
        )

    # Annotate key geographical regions / district hubs
    hubs = [
        ("Shimla (2,276 m)", 77.1734, 31.1048),
        ("Mandi (760 m)", 76.9317, 31.7087),
        ("Kullu (1,279 m)", 77.1095, 31.9579),
        ("Dharamshala (1,457 m)", 76.3234, 32.2190),
        ("Chamba (1,006 m)", 76.1258, 32.5534),
        ("Kinnaur / Kalpa", 78.2563, 31.5385),
        ("Lahaul / Keylong", 77.0320, 32.5710),
    ]
    for label, lon, lat in hubs:
        ax.plot(lon, lat, marker="^", color="black", markersize=7)
        ax.annotate(
            label,
            (lon, lat),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
        )

    ax.set_title("RockWatch – Landslide Susceptibility Map of Himachal Pradesh, India\n(Champion Model: Gradient Boosting Classifier)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.legend(title="Susceptibility Tier", loc="lower right", frameon=True, fontsize=9, facecolor="white")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    p_map = fig_dir / "susceptibility_map_himachal.png"
    fig.savefig(p_map, bbox_inches="tight")
    print(f"  Saved regional susceptibility map: {p_map.name}")

    # 7. Regional Hazard Ranking by Latitude Zone
    print("\n[4] Generating Geographic Latitude Zone Hazard Analysis...")
    df_dedup["lat_zone"] = pd.cut(
        df_dedup["latitude"],
        bins=[30.0, 31.0, 31.5, 32.0, 32.5, 33.5],
        labels=["< 31.0°N (Sirmaur/Solan)", "31.0°-31.5°N (Shimla/Bilaspur)", "31.5°-32.0°N (Mandi/Kullu S)", "32.0°-32.5°N (Kangra/Kullu N)", "> 32.5°N (Chamba/Lahaul)"],
    )

    zone_stats = df_dedup.groupby("lat_zone", observed=False)["susceptibility_class"].value_counts(normalize=True).unstack()[tier_names] * 100

    fig2, ax2 = plt.subplots(figsize=(10, 5.5), dpi=300)
    zone_stats.plot(
        kind="barh",
        stacked=True,
        color=[tier_colors[t] for t in tier_names],
        edgecolor="black",
        linewidth=0.5,
        ax=ax2,
    )
    ax2.set_title("Landslide Hazard Breakdown Across Regional Latitude Corridors", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Percentage of Sampled Locations (%)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Geographic Region", fontsize=10, fontweight="bold")
    ax2.legend(title="Hazard Tier", loc="lower right", frameon=True)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    p_rank = fig_dir / "district_hazard_ranking.png"
    fig2.savefig(p_rank, bbox_inches="tight")
    print(f"  Saved regional hazard ranking plot: {p_rank.name}")

    print("\nPhase 6 susceptibility mapping execution complete.")


if __name__ == "__main__":
    generate_susceptibility_map()
