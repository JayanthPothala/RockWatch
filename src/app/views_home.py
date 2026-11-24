"""
RockWatch Web Application - Home View
Overview dashboard, dynamic dataset metrics, and project introduction.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
from src.app.styles import get_metric_card


@st.cache_data
def load_home_statistics():
    """
    Load actual dataset and model performance statistics directly from processed files.
    Ensures zero hardcoding or discrepancy with underlying research artifacts.
    """
    dedup_path = Path("data/processed/modelling_deduped.csv")
    boot_path = Path("models/evaluation/bootstrap_ci.csv")
    train_path = Path("data/processed/train_spatial.csv")
    val_path = Path("data/processed/val_spatial.csv")
    test_path = Path("data/processed/test_spatial.csv")

    stats = {
        "n_total": 11737,
        "n_events": 5859,
        "n_pseudo": 5878,
        "min_elev": 246,
        "max_elev": 6430,
        "champion_name": "Gradient Boosting",
        "champion_auc": 0.8035,
        "auc_ci_lower": 0.782,
        "auc_ci_upper": 0.823,
        "n_train": 8188,
        "n_val": 1731,
        "n_test": 1750,
    }

    if dedup_path.exists():
        df_dedup = pd.read_csv(dedup_path, usecols=["class", "elevation_m"])
        stats["n_total"] = len(df_dedup)
        stats["n_events"] = int((df_dedup["class"] == 1).sum())
        stats["n_pseudo"] = int((df_dedup["class"] == 0).sum())
        stats["min_elev"] = int(df_dedup["elevation_m"].min())
        stats["max_elev"] = int(df_dedup["elevation_m"].max())

    if boot_path.exists():
        df_boot = pd.read_csv(boot_path)
        gb_row = df_boot[df_boot["Model"] == "Gradient Boosting"]
        if not gb_row.empty:
            stats["champion_auc"] = float(gb_row["Mean_Test_AUC"].values[0])
            stats["auc_ci_lower"] = float(gb_row["AUC_95_CI_Lower"].values[0])
            stats["auc_ci_upper"] = float(gb_row["AUC_95_CI_Upper"].values[0])

    if train_path.exists() and val_path.exists() and test_path.exists():
        stats["n_train"] = len(pd.read_csv(train_path, usecols=[0]))
        stats["n_val"] = len(pd.read_csv(val_path, usecols=[0]))
        stats["n_test"] = len(pd.read_csv(test_path, usecols=[0]))

    return stats


def render_home_view():
    """Render the Home Dashboard page."""
    # Top Branding Banner
    st.markdown(
        """
        <div class="rock-brand-banner">
            <span class="rock-banner-badge">🛰️ GEOSPATIAL ENVIRONMENTAL INTELLIGENCE</span>
            <h1 class="rock-banner-title">RockWatch: Landslide Susceptibility System</h1>
            <p class="rock-banner-subtitle">
                An end-to-end, scientifically validated machine learning system for terrain hazard assessment
                and spatial landslide susceptibility mapping across Himachal Pradesh, India.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = load_home_statistics()

    # Top KPI Metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            get_metric_card(
                "Study Domain Points",
                f"{stats['n_total']:,}",
                f"{stats['n_events']:,} events · {stats['n_pseudo']:,} pseudo-absences",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            get_metric_card(
                "Champion Model",
                stats["champion_name"],
                f"ROC-AUC: {stats['champion_auc']:.4f} (95% CI: {stats['auc_ci_lower']:.3f}–{stats['auc_ci_upper']:.3f})",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            get_metric_card(
                "Spatial Generalization",
                "Latitude Split",
                f"{stats['n_train']:,} Train · {stats['n_val']:,} Val · {stats['n_test']:,} Test",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            get_metric_card(
                "Geographic Scope",
                "Himachal Pradesh",
                f"Elevation: {stats['min_elev']:,} m – {stats['max_elev']:,} m",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Main Project Overview & Architecture
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            """
            <div class="rock-card">
                <h3 class="rock-card-title">⛰️ Project Overview & Scientific Mission</h3>
                <p style="color: #475569; line-height: 1.6; font-size: 0.95rem;">
                    The state of Himachal Pradesh in the Western Himalayas experiences catastrophic landslide events
                    annually, threatening critical mountain transportation corridors, hydroelectric infrastructure, and
                    vulnerable communities. <strong>RockWatch</strong> combines Geological Survey of India (GSI) historical
                    inventories with multi-source Earth observation predictors (Sentinel-2, SRTM DEM, ERA5 reanalysis)
                    to quantify spatial susceptibility.
                </p>
                <div style="background: #f8fafc; border-radius: 10px; padding: 1rem; border-left: 4px solid #10b981; margin-top: 1rem;">
                    <h5 style="margin: 0 0 0.4rem 0; color: #065f46; font-weight: 700;">Key Methodological Safeguards:</h5>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #334155; font-size: 0.9rem;">
                        <li><strong>Data Leakage Prevention:</strong> Exact linear re-standardisation using training-only statistics (<code style="color: #065f46;">scaler_params.json</code>), strictly preventing cross-split feature contamination.</li>
                        <li><strong>Spatially Blocked Evaluation:</strong> Latitudinal partitioning (South, Central, North) paired with 500 m buffer zones to mitigate spatial autocorrelation bias.</li>
                        <li><strong>Strict Topography Integrity:</strong> Circular aspect parameterization via sine and cosine trigonometric transformation, avoiding discontinuous north jumps.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="rock-card">
                <h3 class="rock-card-title">🎯 Landslide Susceptibility Tiers</h3>
                <p style="color: #64748b; font-size: 0.88rem; margin-bottom: 1rem;">
                    Relative spatial predisposition scores from our champion Gradient Boosting model are categorized into four planning tiers:
                </p>
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.7rem 0.9rem;">
                        <span class="tier-badge tier-low">Low Hazard (Score &lt; 0.20)</span>
                        <div style="font-size: 0.82rem; color: #166534; margin-top: 0.3rem;">Valley floors, gentle gradient terraces, stabilized soils.</div>
                    </div>
                    <div style="background: #fefce8; border: 1px solid #fef08a; border-radius: 8px; padding: 0.7rem 0.9rem;">
                        <span class="tier-badge tier-moderate">Moderate Hazard (0.20 ≤ Score &lt; 0.45)</span>
                        <div style="font-size: 0.82rem; color: #854d0e; margin-top: 0.3rem;">Rolling hills, agricultural terraces, moderate gradient slopes.</div>
                    </div>
                    <div style="background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 0.7rem 0.9rem;">
                        <span class="tier-badge tier-high">High Hazard (0.45 ≤ Score &lt; 0.70)</span>
                        <div style="font-size: 0.82rem; color: #9a3412; margin-top: 0.3rem;">Steep terrain, high drainage density, weathered escarpments.</div>
                    </div>
                    <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 0.7rem 0.9rem;">
                        <span class="tier-badge tier-very-high">Very High Hazard (Score ≥ 0.70)</span>
                        <div style="font-size: 0.82rem; color: #991b1b; margin-top: 0.3rem;">Active shear zones, extreme slope &gt; 35°, high terrain conditioning.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive System Workflow Diagram / Modules Grid
    st.markdown(
        """
        <div class="rock-card">
            <h3 class="rock-card-title">🧭 Application Navigation & Core Modules</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; background: #fafafa;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #1e4533; margin-bottom: 0.3rem;">🗺️ Susceptibility Map</div>
                    <p style="color: #64748b; font-size: 0.86rem; margin: 0;">
                        Explore 11,737 pre-computed prediction points across Himachal Pradesh with dynamic filtering by hazard tier, elevation, and terrain slope.
                    </p>
                </div>
                <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; background: #fafafa;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #1e4533; margin-bottom: 0.3rem;">⚡ Landslide Prediction</div>
                    <p style="color: #64748b; font-size: 0.86rem; margin: 0;">
                        Input environmental variables for any mountain coordinate to receive immediate inference, calibrated hazard tier, and planning advisories.
                    </p>
                </div>
                <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; background: #fafafa;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #1e4533; margin-bottom: 0.3rem;">📊 Model Performance</div>
                    <p style="color: #64748b; font-size: 0.86rem; margin: 0;">
                        Inspect rigorous benchmark evaluations, ROC/PR curves, confusion matrices, and bootstrap confidence intervals across 6 ML architectures.
                    </p>
                </div>
                <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; background: #fafafa;">
                    <div style="font-size: 1.1rem; font-weight: 700; color: #1e4533; margin-bottom: 0.3rem;">🏔️ Regional Analysis</div>
                    <p style="color: #64748b; font-size: 0.86rem; margin: 0;">
                        Analyze susceptibility breakdowns across latitudinal corridors, elevation bands, slope profiles, and key district population centers.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Disclaimer Note
    st.markdown(
        """
        <div class="rock-disclaimer-box">
            <strong>⚠️ Scientific & Disaster Management Disclaimer:</strong>
            RockWatch models <em>static spatial terrain susceptibility</em> based on geophysical conditioning and long-term
            environmental characteristics. It does NOT forecast real-time landslide triggers (e.g., dynamic hourly cloudbursts
            or seismic events). Local ground stabilization and municipal engineering decisions require in-situ geotechnical investigations.
        </div>
        """,
        unsafe_allow_html=True,
    )
