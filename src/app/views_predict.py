"""
RockWatch Web Application - Landslide Prediction View
Single-point interactive environmental parameter input and real-time susceptibility inference.
"""

from datetime import datetime
import json
import math
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from src.app.styles import get_metric_card, get_tier_badge
from src.app.report_generator import generate_markdown_report, generate_html_report
from src.app.explainability import compute_local_shap, create_shap_waterfall_plot, generate_shap_narrative

# Model and scaler constants
FEATURE_NAMES = [
    "sin_aspect", "cos_aspect", "Dist_River_z_corr", "Elevation_z_corr",
    "FlowAccumulation_z_corr", "NDVI_z_corr", "Rainfall_z_corr", "Slope_z_corr",
    "LULC_10", "LULC_20", "LULC_30", "LULC_40", "LULC_50", "LULC_60",
    "LULC_70", "LULC_80", "LULC_100"
]

LULC_LABELS = {
    10: "Trees / Forest Cover (LULC 10)",
    20: "Shrubland (LULC 20)",
    30: "Grassland / Meadow (LULC 30)",
    40: "Cropland / Agriculture (LULC 40)",
    50: "Built-up / Urban Settlement (LULC 50)",
    60: "Bare Ground / Sparse Rock (LULC 60)",
    70: "Snow and Glacial Ice (LULC 70)",
    80: "Permanent Water Bodies (LULC 80)",
    100: "Moss and Lichen (LULC 100)",
}

# Original global normalization constants from the full 12,409 dataset
GLOBAL_PARAMS = {
    "elevation_m": {"mean": 1780.143605, "std": 1263.610134},
    "slope_deg": {"mean": 26.793842, "std": 13.434014},
    "aspect_deg": {"mean": 178.187856, "std": 98.745557},
}

# Representative scenario presets
PRESETS = {
    "monsoon_steep": {
        "name": "🌧️ Steep Monsoon Slope\n(Kangra / Dharamshala)",
        "elevation": 1650,
        "slope": 38.0,
        "aspect": 210.0,
        "rainfall_index": 1.6,
        "ndvi": 0.38,
        "dist_river": 180,
        "flow_accum": 850,
        "lulc": 60,
    },
    "urban_ridge": {
        "name": "🏙️ Urbanized Ridge\n(Shimla Ridge)",
        "elevation": 2200,
        "slope": 26.0,
        "aspect": 160.0,
        "rainfall_index": 0.3,
        "ndvi": 0.25,
        "dist_river": 1200,
        "flow_accum": 120,
        "lulc": 50,
    },
    "arid_alpine": {
        "name": "❄️ High Alpine Valley\n(Spiti / Kinnaur)",
        "elevation": 3600,
        "slope": 34.0,
        "aspect": 45.0,
        "rainfall_index": -1.5,
        "ndvi": -0.05,
        "dist_river": 2500,
        "flow_accum": 80,
        "lulc": 70,
    },
    "valley_floor": {
        "name": "🌾 Valley Floor Basin\n(Kullu River Plain)",
        "elevation": 950,
        "slope": 8.0,
        "aspect": 90.0,
        "rainfall_index": -0.2,
        "ndvi": 0.52,
        "dist_river": 60,
        "flow_accum": 4500,
        "lulc": 40,
    },
}

DEFAULT_VALUES = {
    "pred_elevation": 1850,
    "pred_slope": 31.0,
    "pred_aspect": 190.0,
    "pred_rainfall_index": 0.5,
    "pred_ndvi": 0.45,
    "pred_dist_river": 300,
    "pred_flow_accum": 250,
    "pred_lulc": 10,
    "latest_prediction": None,
    "latest_shap": None,
    "active_preset_name": None,
}


@st.cache_resource
def load_prediction_artifacts():
    """Load trained Gradient Boosting champion model and training scaler parameters."""
    model_path = Path("models/trained/gradient_boosting_v1.joblib")
    scaler_path = Path("data/processed/scaler_params.json")

    if not model_path.exists():
        st.error(f"Champion model not found at {model_path}!")
        return None, None

    model = joblib.load(model_path)
    with open(scaler_path, "r") as f:
        scaler_params = json.load(f)

    return model, scaler_params


def apply_preset(preset_key: str):
    """Callback to update session state when a scenario preset is selected."""
    preset = PRESETS[preset_key]
    st.session_state["pred_elevation"] = preset["elevation"]
    st.session_state["pred_slope"] = float(preset["slope"])
    st.session_state["pred_aspect"] = float(preset["aspect"])
    st.session_state["pred_rainfall_index"] = float(preset["rainfall_index"])
    st.session_state["pred_ndvi"] = float(preset["ndvi"])
    st.session_state["pred_dist_river"] = preset["dist_river"]
    st.session_state["pred_flow_accum"] = preset["flow_accum"]
    st.session_state["pred_lulc"] = preset["lulc"]
    st.session_state["active_preset_name"] = preset["name"].split("\n")[0]


def render_predict_view():
    """Render the Interactive Landslide Prediction Page."""
    st.markdown(
        """
        <div class="rock-brand-banner">
            <span class="rock-banner-badge">⚡ REAL-TIME INFERENCE ENGINE</span>
            <h1 class="rock-banner-title">Single-Point Susceptibility Assessment</h1>
            <p class="rock-banner-subtitle">
                Evaluate relative terrain susceptibility using the trained Gradient Boosting champion model
                with training-only standardisation and scientifically grounded planning advisories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, scaler_params = load_prediction_artifacts()
    if model is None or scaler_params is None:
        st.stop()

    # Initialize Session State Keys
    for key, val in DEFAULT_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Presets Selection Bar
    st.markdown("#### ⚡ Quick Presets (Representative Himalayan Terrains)")
    preset_cols = st.columns(4)

    for col, (pkey, pdata) in zip(preset_cols, PRESETS.items()):
        with col:
            st.button(
                pdata["name"],
                key=f"btn_{pkey}",
                on_click=apply_preset,
                args=(pkey,),
                use_container_width=True,
            )

    if st.session_state.get("active_preset_name"):
        st.info(f"Loaded Scenario Preset: **{st.session_state['active_preset_name']}**")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Input Form
    with st.form("prediction_form"):
        col_topo, col_hydro = st.columns(2)

        with col_topo:
            st.markdown("##### ⛰️ 1. Topographic & Geomorphic Parameters")
            elevation_m = st.number_input(
                "Elevation (meters above sea level):",
                min_value=150,
                max_value=7000,
                step=50,
                key="pred_elevation",
                help="SRTM Digital Elevation Model height in meters (Himachal range: 246m - 6,430m).",
            )

            slope_deg = st.slider(
                "Slope Angle (degrees):",
                min_value=0.0,
                max_value=75.0,
                step=0.5,
                key="pred_slope",
                help="Terrain gradient in degrees. Slopes above 25°–35° exhibit elevated hazard conditioning.",
            )

            aspect_deg = st.slider(
                "Aspect (compass orientation in degrees):",
                min_value=0.0,
                max_value=360.0,
                step=5.0,
                key="pred_aspect",
                help="0° = North, 90° = East, 180° = South, 270° = West. If slope = 0°, aspect is flat.",
            )

            lulc_options = list(LULC_LABELS.keys())
            selected_lulc = st.selectbox(
                "Land Use / Land Cover (LULC):",
                options=lulc_options,
                format_func=lambda x: LULC_LABELS[x],
                key="pred_lulc",
                help="ESA WorldCover classification. High vegetative canopy enhances root cohesion.",
            )

        with col_hydro:
            st.markdown("##### 💧 2. Hydrological & Environmental Parameters")
            rainfall_index = st.slider(
                "Precipitation Anomaly / Monsoon Intensity Index:",
                min_value=-2.5,
                max_value=2.5,
                step=0.1,
                key="pred_rainfall_index",
                help="Precipitation standard score: negative = rain-shadow (Spiti); positive = monsoonal front (Kangra).",
            )

            ndvi_val = st.slider(
                "NDVI (Normalized Difference Vegetation Index):",
                min_value=-0.20,
                max_value=0.90,
                step=0.02,
                key="pred_ndvi",
                help="Sentinel-2 vegetation vigour: <0 = rock/water/snow; 0.2-0.5 = shrub/grass; >0.6 = dense forest.",
            )

            dist_river_m = st.number_input(
                "Distance to Nearest Stream / River Channel (meters):",
                min_value=0,
                max_value=6000,
                step=50,
                key="pred_dist_river",
                help="Proximity to drainage line. Fluvial toe erosion frequently triggers slope destabilization.",
            )

            flow_accum_cells = st.number_input(
                "Upstream Flow Accumulation (cell count):",
                min_value=0,
                max_value=25000,
                step=50,
                key="pred_flow_accum",
                help="Number of upslope contributing pixels draining through this coordinate.",
            )

        submitted = st.form_submit_button("🚀 Compute Landslide Susceptibility", use_container_width=True)

    # Submission and Inference Handling
    if submitted:
        # Validate Inputs
        validation_errors = []
        if elevation_m < 150 or elevation_m > 7000:
            validation_errors.append("Elevation must be between 150 m and 7,000 m.")
        if slope_deg < 0.0 or slope_deg > 90.0:
            validation_errors.append("Slope angle must be between 0° and 90°.")
        if aspect_deg < 0.0 or aspect_deg > 360.0:
            validation_errors.append("Aspect must be between 0° and 360°.")
        if not (-1.0 <= ndvi_val <= 1.0):
            validation_errors.append("NDVI must be between -1.0 and 1.0.")
        if dist_river_m < 0:
            validation_errors.append("Distance to stream cannot be negative.")
        if flow_accum_cells < 0:
            validation_errors.append("Flow accumulation cannot be negative.")

        if validation_errors:
            for err in validation_errors:
                st.error(f"❌ Input Validation Error: {err}")
        else:
            # 1. Circular aspect engineering
            if slope_deg == 0.0:
                sin_aspect = 0.0
                cos_aspect = 0.0
            else:
                rad = math.radians(aspect_deg)
                sin_aspect = math.sin(rad)
                cos_aspect = math.cos(rad)

            # 2. Reconstruct original global z-scores
            elev_z = (elevation_m - GLOBAL_PARAMS["elevation_m"]["mean"]) / GLOBAL_PARAMS["elevation_m"]["std"]
            slope_z = (round(slope_deg) - GLOBAL_PARAMS["slope_deg"]["mean"]) / GLOBAL_PARAMS["slope_deg"]["std"]

            # Hydrological and environmental standard score transformations
            dist_river_z = -0.1314 + (dist_river_m / 4389.0) * 0.8
            ndvi_z = (ndvi_val - 0.45) / 0.18
            rainfall_z = rainfall_index
            flow_accum_z = -0.5783 + (math.log1p(flow_accum_cells) / 5.0) * 0.8

            # 3. Apply training-only scaler from scaler_params.json
            sp = scaler_params["features"]
            elev_z_corr = (elev_z - sp["Elevation_z"]["train_mean"]) / sp["Elevation_z"]["train_std"]
            slope_z_corr = (slope_z - sp["Slope_z"]["train_mean"]) / sp["Slope_z"]["train_std"]
            dist_river_z_corr = (dist_river_z - sp["Dist_River_z"]["train_mean"]) / sp["Dist_River_z"]["train_std"]
            ndvi_z_corr = (ndvi_z - sp["NDVI_z"]["train_mean"]) / sp["NDVI_z"]["train_std"]
            rainfall_z_corr = (rainfall_z - sp["Rainfall_z"]["train_mean"]) / sp["Rainfall_z"]["train_std"]
            flow_accum_z_corr = (flow_accum_z - sp["FlowAccumulation_z"]["train_mean"]) / sp["FlowAccumulation_z"]["train_std"]

            # 4. One-hot encode LULC
            lulc_vector = [1 if selected_lulc == code else 0 for code in [10, 20, 30, 40, 50, 60, 70, 80, 100]]

            # 5. Assemble exact feature vector
            input_vector = [
                sin_aspect,
                cos_aspect,
                dist_river_z_corr,
                elev_z_corr,
                flow_accum_z_corr,
                ndvi_z_corr,
                rainfall_z_corr,
                slope_z_corr,
            ] + lulc_vector

            feature_df = pd.DataFrame([input_vector], columns=FEATURE_NAMES)

            # 6. Execute fresh Model Inference
            y_prob = float(model.predict_proba(feature_df)[0, 1])

            # Assign Hazard Tier
            if y_prob < 0.20:
                tier_name = "Low"
                tier_class = "tier-low"
                tier_color = "#10b981"
                tier_desc = "Terrain exhibits minimal static predisposition to slope failure under baseline conditions."
            elif y_prob < 0.45:
                tier_name = "Moderate"
                tier_class = "tier-moderate"
                tier_color = "#f59e0b"
                tier_desc = "Moderate terrain conditioning; slope failure possible during prolonged high-intensity precipitation."
            elif y_prob < 0.70:
                tier_name = "High"
                tier_class = "tier-high"
                tier_color = "#f97316"
                tier_desc = "Steep terrain and high drainage conditioning indicate substantial predisposition to failure."
            else:
                tier_name = "Very High"
                tier_class = "tier-very-high"
                tier_color = "#ef4444"
                tier_desc = "Severe geomorphic predisposition; active shear planes or steep escarpments highly susceptible to triggering."

            # Store fresh prediction in Session State
            st.session_state["latest_prediction"] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "score": y_prob,
                "tier_name": tier_name,
                "tier_class": tier_class,
                "tier_color": tier_color,
                "tier_desc": tier_desc,
                "elevation_m": elevation_m,
                "slope_deg": slope_deg,
                "aspect_deg": aspect_deg,
                "selected_lulc": selected_lulc,
                "rainfall_index": rainfall_index,
                "ndvi_val": ndvi_val,
                "dist_river_m": dist_river_m,
                "flow_accum_cells": flow_accum_cells,
            }

            # 7. Compute Local Model Explainability via SHAP TreeExplainer
            raw_inputs = {
                "elevation_m": elevation_m,
                "slope_deg": slope_deg,
                "aspect_deg": aspect_deg,
                "selected_lulc": selected_lulc,
                "rainfall_index": rainfall_index,
                "ndvi_val": ndvi_val,
                "dist_river_m": dist_river_m,
                "flow_accum_cells": flow_accum_cells,
            }
            shap_data, shap_err = compute_local_shap(model, feature_df, raw_inputs)
            if shap_err:
                st.session_state["latest_shap"] = None
                st.session_state["shap_error"] = shap_err
            else:
                st.session_state["latest_shap"] = shap_data
                st.session_state["shap_error"] = None

    # Render Result Display
    latest = st.session_state.get("latest_prediction")

    if latest is None:
        st.markdown(
            """
            <div class="rock-card" style="text-align: center; padding: 2.2rem 1.5rem; margin-top: 1rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">⛰️</div>
                <h4 style="color: #1e4533; margin: 0 0 0.5rem 0;">Awaiting Parameter Submission</h4>
                <p style="color: #64748b; font-size: 0.92rem; max-width: 600px; margin: 0 auto;">
                    Adjust the topographic and environmental parameters above or choose a representative regional preset,
                    then click <strong>"Compute Landslide Susceptibility"</strong> to run inference with the champion Gradient Boosting model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        y_prob = latest["score"]
        tier_name = latest["tier_name"]
        tier_class = latest["tier_class"]
        tier_color = latest["tier_color"]
        tier_desc = latest["tier_desc"]

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Prediction Assessment Summary")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Gauge Chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=y_prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "<b>Relative Susceptibility Index</b>", "font": {"size": 18, "color": "#0f172a"}},
                    number={"suffix": "%", "valueformat": ".1f", "font": {"size": 36, "color": tier_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                        "bar": {"color": tier_color, "thickness": 0.3},
                        "bgcolor": "#f8fafc",
                        "borderwidth": 1,
                        "bordercolor": "#cbd5e1",
                        "steps": [
                            {"range": [0, 20], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [20, 45], "color": "rgba(245, 158, 11, 0.15)"},
                            {"range": [45, 70], "color": "rgba(249, 115, 22, 0.15)"},
                            {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "#0f172a", "width": 3},
                            "thickness": 0.8,
                            "value": y_prob * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=280, margin=dict(l=25, r=25, t=40, b=25))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown(
                f"""
                <div class="rock-card" style="height: 280px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Classified Susceptibility Tier</div>
                    <div style="margin-top: 0.4rem; margin-bottom: 0.8rem;">
                        <span class="tier-badge {tier_class}" style="font-size: 1.2rem; padding: 0.4rem 1.1rem;">
                            {tier_name.upper()} HAZARD ({y_prob*100:.1f}%)
                        </span>
                    </div>
                    <p style="color: #334155; font-size: 0.95rem; line-height: 1.5; margin-bottom: 0.8rem;">
                        {tier_desc}
                    </p>
                    <div style="font-size: 0.82rem; color: #64748b; border-top: 1px solid #f1f5f9; padding-top: 0.6rem;">
                        Model: <strong>Gradient Boosting (Champion)</strong> · Calibration Reference: 0.37 (Balanced F1: 0.737)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Scientific Planning Advisory Tiers
        st.markdown("#### 🏛️ Geospatial Planning & Screening Advisories")
        st.markdown(
            """
            <p style="color: #64748b; font-size: 0.86rem; margin-bottom: 0.9rem;">
                Advisory tiers represent relative spatial conditioning derived from model calibration.
                <strong>Susceptibility scores do NOT denote verified temporal event probabilities or real-time warnings.</strong>
                These screening advisories provide regional planning guidance and do not constitute independent statutory authority
                for evacuations, construction bans, or emergency civil protection declarations.
            </p>
            """,
            unsafe_allow_html=True,
        )

        is_tier1_active = y_prob >= 0.16
        is_tier2_active = y_prob >= 0.37
        is_tier3_active = y_prob >= 0.86

        pol1, pol2, pol3 = st.columns(3)

        with pol1:
            st.markdown(
                f"""
                <div style="background: {'#fefce8' if is_tier1_active else '#f8fafc'}; border: 1px solid {'#fef08a' if is_tier1_active else '#e2e8f0'}; border-radius: 10px; padding: 1.1rem; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Advisory Level 1</span>
                        <span style="font-size: 0.75rem; background: {'#eab308' if is_tier1_active else '#94a3b8'}; color: white; border-radius: 6px; padding: 0.15rem 0.5rem; font-weight: 600;">
                            {'APPLICABLE' if is_tier1_active else 'BASELINE'}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;"><strong>Routine Geomorphic Screening</strong> (Index ≥ 0.16)</div>
                    <p style="font-size: 0.85rem; color: #334155; margin-top: 0.6rem; margin-bottom: 0; line-height: 1.5;">
                        Planning action: Periodic inspection of roadside culverts, monitoring cross-drainage channels, and maintaining stabilizing vegetative cover. Does not warrant emergency or evacuation action.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with pol2:
            st.markdown(
                f"""
                <div style="background: {'#fff7ed' if is_tier2_active else '#f8fafc'}; border: 1px solid {'#fdba74' if is_tier2_active else '#e2e8f0'}; border-radius: 10px; padding: 1.1rem; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Advisory Level 2</span>
                        <span style="font-size: 0.75rem; background: {'#f97316' if is_tier2_active else '#94a3b8'}; color: white; border-radius: 6px; padding: 0.15rem 0.5rem; font-weight: 600;">
                            {'APPLICABLE' if is_tier2_active else 'BASELINE'}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;"><strong>Enhanced Terrain Surveillance</strong> (Index ≥ 0.37)</div>
                    <p style="font-size: 0.85rem; color: #334155; margin-top: 0.6rem; margin-bottom: 0; line-height: 1.5;">
                        Planning action: Recommended for inclusion in municipal hazard screening registers. Preliminary engineering geological reconnaissance advised prior to significant cut-slope grading or road expansion.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with pol3:
            st.markdown(
                f"""
                <div style="background: {'#fef2f2' if is_tier3_active else '#f8fafc'}; border: 1px solid {'#fca5a5' if is_tier3_active else '#e2e8f0'}; border-radius: 10px; padding: 1.1rem; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Advisory Level 3</span>
                        <span style="font-size: 0.75rem; background: {'#ef4444' if is_tier3_active else '#94a3b8'}; color: white; border-radius: 6px; padding: 0.15rem 0.5rem; font-weight: 600;">
                            {'RECOMMENDED' if is_tier3_active else 'BASELINE'}
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;"><strong>Priority Geotechnical Investigation</strong> (Index ≥ 0.86)</div>
                    <p style="font-size: 0.85rem; color: #334155; margin-top: 0.6rem; margin-bottom: 0; line-height: 1.5;">
                        Planning action: High terrain conditioning detected. All proposed infrastructure corridors, deep cuts, or building foundations must undergo detailed in-situ geotechnical drilling, shear strength testing, and certified structural FoS analysis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Local SHAP Model Explainability Section
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 🔬 Local Model Explainability (SHAP TreeExplainer)")
        st.markdown(
            """
            <p style="color: #64748b; font-size: 0.88rem; margin-bottom: 1.1rem;">
                SHAP (SHapley Additive exPlanations) isolates the exact mathematical attribution of each terrain parameter
                to this specific prediction. The baseline represents the average predisposition across Himachal Pradesh
                prior (<strong>E[f(X)] = -0.291 log-odds</strong>, ~42.8%), while individual features push the hazard index up (red) or down (green).
            </p>
            """,
            unsafe_allow_html=True,
        )

        shap_data = st.session_state.get("latest_shap")
        shap_err = st.session_state.get("shap_error")

        if shap_err:
            st.warning(
                f"⚠️ SHAP explanation could not be rendered: {shap_err}. "
                f"Model prediction and planning advisories remain valid."
            )
        elif shap_data:
            # Interactive Waterfall Plot
            fig_waterfall = create_shap_waterfall_plot(shap_data, max_features=10)
            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Natural Language Narrative Synthesis
            st.markdown("##### 📝 Geomorphic Attribution Synthesis")
            narrative_text = generate_shap_narrative(shap_data)
            st.markdown(
                f"""
                <div class="rock-card" style="padding: 1.2rem; margin-top: 0.5rem; line-height: 1.6; font-size: 0.92rem; color: #334155;">
                    {narrative_text.replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Detailed Feature Attribution Breakdown Table
            with st.expander("🔍 View Complete Feature Attribution Breakdown (All 17 Factors)"):
                df_contrib = pd.DataFrame([
                    {
                        "Conditioning Feature": c["display_name"],
                        "Evaluated Input": c["raw_val_str"] or "—",
                        "SHAP Impact (Log-Odds)": f"{c['shap_value']:+.4f}",
                        "Direction": "Increasing Risk ↗" if c["shap_value"] > 0 else "Buffering Risk ↘",
                        "Geomorphic Role": c["description"],
                    }
                    for c in shap_data["contributions_sorted"]
                ])
                st.dataframe(df_contrib, use_container_width=True, hide_index=True)

        # Downloadable Assessment Report Section
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 📥 Download Site Assessment Report")
        st.markdown(
            """
            <p style="color: #64748b; font-size: 0.88rem; margin-bottom: 1rem;">
                Export an official, university-standard landslide susceptibility assessment report documenting the evaluated
                terrain parameters, model classification, geomorphic influences, and planning advisories.
            </p>
            """,
            unsafe_allow_html=True,
        )

        report_data = {
            "timestamp": latest.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")),
            "score": y_prob,
            "tier_name": tier_name,
            "tier_color": tier_color,
            "site_name": st.session_state.get("active_preset_name") or f"Mountain Slope Site (Elev: {latest['elevation_m']}m, Slope: {latest['slope_deg']}°)",
            "inputs": {
                "elevation_m": latest["elevation_m"],
                "slope_deg": latest["slope_deg"],
                "aspect_deg": latest["aspect_deg"],
                "selected_lulc": latest["selected_lulc"],
                "lulc_label": LULC_LABELS.get(latest["selected_lulc"], f"LULC {latest['selected_lulc']}"),
                "rainfall_index": latest["rainfall_index"],
                "ndvi_val": latest["ndvi_val"],
                "dist_river_m": latest["dist_river_m"],
                "flow_accum_cells": latest["flow_accum_cells"],
            },
            "shap_contributions": shap_data.get("contributions_sorted") if shap_data else None,
        }

        md_report = generate_markdown_report(report_data)
        html_report = generate_html_report(report_data)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.download_button(
                "📄 Download Assessment Report (Markdown)",
                data=md_report,
                file_name=f"RockWatch_Susceptibility_Assessment_{int(latest['elevation_m'])}m.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with btn_col2:
            st.download_button(
                "🌐 Download Printable Assessment (HTML)",
                data=html_report,
                file_name=f"RockWatch_Susceptibility_Assessment_{int(latest['elevation_m'])}m.html",
                mime="text/html",
                use_container_width=True,
            )
