"""
RockWatch Web Application - Model Explainability Engine
Generates local SHAP explanations, interactive waterfall plots, and geomorphic syntheses.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go
from scipy.special import expit


# Human-friendly mapping for features and their units
FEATURE_METADATA = {
    "Slope_z_corr": {
        "display_name": "Slope Gradient",
        "unit": "°",
        "description": "Gravitational shear stress driver along inclined surfaces.",
    },
    "Elevation_z_corr": {
        "display_name": "Elevation",
        "unit": " m",
        "description": "Macro-climatic thermal, periglacial, and vegetative elevation zone.",
    },
    "Dist_River_z_corr": {
        "display_name": "Distance to Stream",
        "unit": " m",
        "description": "Fluvial toe-erosion vulnerability and bank scouring proximity.",
    },
    "FlowAccumulation_z_corr": {
        "display_name": "Flow Accumulation",
        "unit": " cells",
        "description": "Catchment drainage concentration and pore-water pressure potential.",
    },
    "Rainfall_z_corr": {
        "display_name": "Precipitation Anomaly",
        "unit": " index",
        "description": "Monsoon intensity predisposition relative to regional baseline.",
    },
    "NDVI_z_corr": {
        "display_name": "Vegetation Vigour (NDVI)",
        "unit": "",
        "description": "Vegetative canopy density and root cohesion stabilizing factor.",
    },
    "sin_aspect": {
        "display_name": "Aspect (East-West Sin)",
        "unit": "",
        "description": "Eastward vs. westward solar radiation and moisture retention.",
    },
    "cos_aspect": {
        "display_name": "Aspect (North-South Cos)",
        "unit": "",
        "description": "North-facing vs. south-facing insolation and freeze-thaw cycles.",
    },
    "LULC_10": {"display_name": "LULC: Forest / Tree Cover", "unit": "", "description": "High bio-stabilization via deep root systems."},
    "LULC_20": {"display_name": "LULC: Shrubland", "unit": "", "description": "Moderate ground cover and soil retention."},
    "LULC_30": {"display_name": "LULC: Grassland / Meadow", "unit": "", "description": "Surface turf protection without deep root anchoring."},
    "LULC_40": {"display_name": "LULC: Cropland / Agriculture", "unit": "", "description": "Terraced or cultivated slopes with seasonal saturation."},
    "LULC_50": {"display_name": "LULC: Built-up / Settlement", "unit": "", "description": "Impervious surface cut-slopes and municipal loading."},
    "LULC_60": {"display_name": "LULC: Bare Ground / Rock", "unit": "", "description": "Unconsolidated colluvium or exposed bedrock prone to raveling."},
    "LULC_70": {"display_name": "LULC: Snow / Glacial Ice", "unit": "", "description": "High periglacial cryosphere freeze-thaw conditioning."},
    "LULC_80": {"display_name": "LULC: Water Bodies", "unit": "", "description": "Permanent aquatic or lacustrine surface."},
    "LULC_100": {"display_name": "LULC: Moss and Lichen", "unit": "", "description": "High-altitude alpine ground layer."},
}


@st.cache_resource
def get_tree_explainer(_model):
    """
    Construct or retrieve cached SHAP TreeExplainer for the champion tree-based ensemble.
    Leading underscore prevents Streamlit from hashing the sklearn model object.
    """
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer, None
    except Exception as e:
        return None, str(e)


def compute_local_shap(
    model,
    feature_df: pd.DataFrame,
    raw_inputs: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Execute real SHAP TreeExplainer attribution for a single inference instance.
    Returns structured explanation data or an error message.
    """
    explainer, err = get_tree_explainer(model)
    if explainer is None:
        return None, f"TreeExplainer initialization failed: {err}"

    try:
        # Pass numpy array to avoid feature name mismatch warnings
        input_values = feature_df.values
        shap_values = explainer.shap_values(input_values)

        # GradientBoostingClassifier returns (1, n_features) or list
        if isinstance(shap_values, list):
            # Binary classification in some SHAP versions returns list [neg_class, pos_class]
            val_array = np.array(shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0])
        elif isinstance(shap_values, np.ndarray):
            val_array = shap_values[0] if shap_values.ndim == 2 else shap_values
        else:
            val_array = np.array(shap_values)

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        # Extract actual physical values for intuitive display
        feature_cols = list(feature_df.columns)
        contributions = []

        for idx, col in enumerate(feature_cols):
            shap_val = float(val_array[idx])
            meta = FEATURE_METADATA.get(col, {"display_name": col, "unit": "", "description": ""})

            # Derive display string for the raw physical input
            raw_val_str = ""
            if col == "Slope_z_corr":
                raw_val_str = f"{raw_inputs.get('slope_deg', 0):.1f}°"
            elif col == "Elevation_z_corr":
                raw_val_str = f"{raw_inputs.get('elevation_m', 0):,} m"
            elif col == "Dist_River_z_corr":
                raw_val_str = f"{raw_inputs.get('dist_river_m', 0):,} m"
            elif col == "FlowAccumulation_z_corr":
                raw_val_str = f"{raw_inputs.get('flow_accum_cells', 0):,} cells"
            elif col == "Rainfall_z_corr":
                raw_val_str = f"{raw_inputs.get('rainfall_index', 0):+.2f}"
            elif col == "NDVI_z_corr":
                raw_val_str = f"{raw_inputs.get('ndvi_val', 0):.2f}"
            elif col == "sin_aspect":
                raw_val_str = f"{math.sin(math.radians(raw_inputs.get('aspect_deg', 0))):.2f}"
            elif col == "cos_aspect":
                raw_val_str = f"{math.cos(math.radians(raw_inputs.get('aspect_deg', 0))):.2f}"
            elif col.startswith("LULC_"):
                code = int(col.split("_")[1])
                is_active = (raw_inputs.get("selected_lulc") == code)
                raw_val_str = "Active" if is_active else "Inactive"

            contributions.append({
                "feature": col,
                "display_name": meta["display_name"],
                "raw_val_str": raw_val_str,
                "shap_value": shap_val,
                "abs_shap": abs(shap_val),
                "description": meta["description"],
            })

        # Calculate exact model log-odds f(x) and predicted probability
        total_shap = float(np.sum(val_array))
        fx_log_odds = base_val + total_shap
        derived_prob = float(expit(fx_log_odds))

        # Sort contributions by absolute impact
        contributions_sorted = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)

        return {
            "base_value": base_val,
            "base_prob": float(expit(base_val)),
            "total_shap": total_shap,
            "fx_log_odds": fx_log_odds,
            "derived_prob": derived_prob,
            "contributions": contributions,
            "contributions_sorted": contributions_sorted,
        }, None

    except Exception as e:
        return None, f"Failed to compute SHAP values: {e}"


def create_shap_waterfall_plot(shap_data: Dict[str, Any], max_features: int = 10) -> go.Figure:
    """
    Construct an interactive, publication-quality horizontal Plotly waterfall chart.
    Displays base value -> individual feature attributions -> final model prediction.
    """
    base_val = shap_data["base_value"]
    contributions = shap_data["contributions_sorted"]

    # Filter to top N features and group remaining into 'Other features'
    top_items = contributions[:max_features]
    remaining_items = contributions[max_features:]

    plot_items = list(reversed(top_items))  # bottom to top for horizontal waterfall

    y_labels = ["Base Value (Regional Baseline)"]
    x_values = [base_val]
    measures = ["relative"]
    hover_texts = [f"Himachal Pradesh regional baseline prior: {base_val:+.3f} log-odds (Prob: {shap_data['base_prob']*100:.1f}%)"]

    for item in plot_items:
        label = f"{item['display_name']} ({item['raw_val_str']})" if item['raw_val_str'] else item['display_name']
        y_labels.append(label)
        x_values.append(item["shap_value"])
        measures.append("relative")
        direction = "INCREASING risk" if item["shap_value"] > 0 else "BUFFERING / reducing risk"
        hover_texts.append(
            f"<b>{item['display_name']}</b><br>"
            f"Value: {item['raw_val_str']}<br>"
            f"SHAP Impact: {item['shap_value']:+.4f} log-odds<br>"
            f"Effect: {direction}<br>"
            f"<i>{item['description']}</i>"
        )

    if remaining_items:
        rem_sum = sum(item["shap_value"] for item in remaining_items)
        y_labels.append(f"Other ({len(remaining_items)} features)")
        x_values.append(rem_sum)
        measures.append("relative")
        hover_texts.append(f"Aggregate contribution of {len(remaining_items)} lower-order conditioning features: {rem_sum:+.4f} log-odds")

    # Final model output bar
    y_labels.append("Model Output f(x)")
    x_values.append(shap_data["fx_log_odds"])
    measures.append("total")
    hover_texts.append(
        f"<b>Final Prediction</b><br>"
        f"Model Log-Odds: {shap_data['fx_log_odds']:+.3f}<br>"
        f"Susceptibility Index: <b>{shap_data['derived_prob']*100:.1f}%</b>"
    )

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures,
            y=y_labels,
            x=x_values,
            text=[f"{v:+.3f}" if m == "relative" else f"{v:+.2f}" for v, m in zip(x_values, measures)],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="text",
            connector={"line": {"color": "#94a3b8", "dash": "dot"}},
            decreasing={"marker": {"color": "#10b981", "line": {"color": "#059669", "width": 1.2}}},
            increasing={"marker": {"color": "#ef4444", "line": {"color": "#dc2626", "width": 1.2}}},
            totals={"marker": {"color": "#1e4533", "line": {"color": "#064e3b", "width": 1.5}}},
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Local SHAP Feature Attribution (TreeExplainer Log-Odds Contribution)</b>",
            "font": {"size": 15, "color": "#0f172a"},
            "x": 0.02,
        },
        xaxis=dict(
            title="Contribution to Susceptibility Log-Odds (ln[p / (1-p)])",
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1",
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            gridcolor="#f8fafc",
            tickfont={"size": 12, "color": "#1e293b"},
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=10, r=40, t=50, b=50),
        height=max(420, len(y_labels) * 36),
    )

    return fig


def generate_shap_narrative(shap_data: Dict[str, Any]) -> str:
    """
    Synthesize the SHAP attribution into understandable, scientifically rigorous text.
    """
    contributions = shap_data["contributions_sorted"]
    pos_contributors = [c for c in contributions if c["shap_value"] > 0]
    neg_contributors = [c for c in contributions if c["shap_value"] < 0]

    base_p = shap_data["base_prob"] * 100
    final_p = shap_data["derived_prob"] * 100

    lines = []
    lines.append(
        f"The regional baseline probability expectation across the Indian Western Himalayas prior is **{base_p:.1f}%** "
        f"(`{shap_data['base_value']:+.3f}` log-odds). Conditioning parameters for this specific terrain profile adjust the model "
        f"output by a net **{shap_data['total_shap']:+.3f}** log-odds to arrive at the final relative susceptibility index of **{final_p:.1f}%**."
    )

    if pos_contributors:
        top_pos = pos_contributors[:3]
        lines.append("\n**Primary Hazard Accelerators (Pushing Susceptibility UP):**")
        for item in top_pos:
            val_txt = f" (`{item['raw_val_str']}`)" if item['raw_val_str'] else ""
            lines.append(
                f"- **{item['display_name']}**{val_txt}: Contributed **`+{item['shap_value']:.3f}`** to log-odds. "
                f"{item['description']}"
            )

    if neg_contributors:
        top_neg = neg_contributors[:3]
        lines.append("\n**Primary Buffering & Mitigating Factors (Pushing Susceptibility DOWN):**")
        for item in top_neg:
            val_txt = f" (`{item['raw_val_str']}`)" if item['raw_val_str'] else ""
            lines.append(
                f"- **{item['display_name']}**{val_txt}: Contributed **`{item['shap_value']:.3f}`** to log-odds. "
                f"{item['description']}"
            )

    return "\n".join(lines)
