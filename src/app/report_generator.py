"""
RockWatch Web Application - Report Generator Engine
Reusable utility for generating structured landslide susceptibility assessment reports
in Markdown and HTML, as well as spatial GeoJSON and CSV data exports.
"""

from datetime import datetime
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """
    Generate a university-project-quality Markdown landslide susceptibility assessment report.
    Adheres strictly to scientific governance and decision-support guidelines.
    """
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    score = data.get("score", 0.0)
    tier_name = data.get("tier_name", "Unknown")
    inputs = data.get("inputs", {})

    # Topographic inputs
    elev = inputs.get("elevation_m", "N/A")
    slope = inputs.get("slope_deg", "N/A")
    aspect = inputs.get("aspect_deg", "N/A")
    lulc_desc = inputs.get("lulc_label", f"Code {inputs.get('selected_lulc', 'N/A')}")

    # Hydrological inputs
    dist_river = inputs.get("dist_river_m", "N/A")
    flow_accum = inputs.get("flow_accum_cells", "N/A")
    rainfall_idx = inputs.get("rainfall_index", "N/A")
    ndvi = inputs.get("ndvi_val", "N/A")

    site_name = data.get("site_name", "Individual Evaluated Mountain Slope")

    md = f"""# RockWatch: Landslide Susceptibility Assessment Report
**Geospatial Environmental Intelligence System for Himachal Pradesh, India**

---

### 1. Document & Evaluation Metadata
- **Report Title:** Site-Specific Landslide Susceptibility Assessment
- **Evaluation Timestamp:** {timestamp}
- **Assessed Location / Profile:** {site_name}
- **Analytical Model:** Gradient Boosting Classifier (Champion Ensemble v1.0)
- **Baseline Calibration Reference:** Threshold $T = 0.37$ (Balanced $F_1 = 0.737$, $\\text{{MCC}} = 0.453$)
- **Validation Standard:** Spatially Blocked Latitudinal Holdout (500 m Exclusion Buffer)

---

### 2. Terrain & Environmental Conditioning Profile
| Conditioning Parameter | Evaluated Input Value | Scientific Domain / Units | Geomorphic Role |
| :--- | :--- | :--- | :--- |
| **Elevation** | `{elev} m` | Meters above sea level | Macro-climatic thermal and periglacial gradient |
| **Slope Gradient** | `{slope}°` | Degrees inclination | Gravitational shear stress component |
| **Aspect Orientation** | `{aspect}°` | Compass bearing ($0°=\\text{{N}}, 180°=\\text{{S}}$) | Solar insolation and freeze-thaw cycles |
| **Land Cover (LULC)** | `{lulc_desc}` | ESA WorldCover 10 m category | Root cohesion and surface vegetative protection |
| **Stream Proximity** | `{dist_river} m` | Distance to nearest channel | Fluvial toe-erosion and undercut susceptibility |
| **Flow Accumulation** | `{flow_accum} cells` | Drainage catchment cell count | Subsurface hydrological pore-pressure build-up |
| **Precipitation Index** | `{rainfall_idx}` | Standardized monsoon anomaly | Regional monsoonal saturation predisposition |
| **Vegetation Index** | `{ndvi}` | Sentinel-2 NDVI ($-1.0$ to $+1.0$) | Canopy density and bio-stabilization factor |

---

### 3. Model Inference & Susceptibility Classification
- **Relative Susceptibility Index:** **`{score * 100:.2f}%`** (Model Score: `{score:.4f}` on range $[0.0, 1.0]$)
- **Classified Hazard Tier:** **`{tier_name.upper()} HAZARD`**
- **Classification Baseline:**
  - *Low Hazard:* Score $< 0.20$
  - *Moderate Hazard:* $0.20 \\le \\text{{Score}} < 0.45$
  - *High Hazard:* $0.45 \\le \\text{{Score}} < 0.70$
  - *Very High Hazard:* $\\text{{Score}} \\ge 0.70$

#### Key Conditioning Influences:
"""

    if isinstance(slope, (int, float)) and slope >= 30.0:
        md += f"- **High Slope Angle ({slope}°):** Exceeds critical geotechnical angle of repose for loose colluvium; major contributor to shear instability.\n"
    elif isinstance(slope, (int, float)) and slope < 15.0:
        md += f"- **Low Slope Angle ({slope}°):** Mild gravitational driving force; substantially suppresses mass wasting potential.\n"

    if isinstance(elev, (int, float)) and 1000 <= elev <= 3000:
        md += f"- **Critical Elevation Belt ({elev} m):** Falls within the active Himalayan monsoonal cloudburst and toe-cutting zone.\n"

    if isinstance(dist_river, (int, float)) and dist_river <= 200:
        md += f"- **Fluvial Proximity ({dist_river} m):** High vulnerability to channel bank scouring during peak river discharge.\n"

    shap_items = data.get("shap_contributions")
    if shap_items:
        md += "\n#### Local Model Explainability (SHAP TreeExplainer Attribution):\n"
        md += "| Feature | Evaluated Input | SHAP Contribution (Log-Odds) | Geomorphic Influence |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        for item in shap_items[:6]:
            direction = "Increasing Risk ↗" if item["shap_value"] > 0 else "Buffering Risk ↘"
            val_str = item.get("raw_val_str") or "—"
            md += f"| **{item['display_name']}** | `{val_str}` | `{item['shap_value']:+.4f}` ({direction}) | {item['description']} |\n"

    md += f"""
---

### 4. Planning & Screening Advisories
These guidelines represent regional planning decision-support filters derived from statistical model calibration:

1. **Screening Level 1 (Index $\\ge 0.16$ — Routine Maintenance):**
   - Regular pre-monsoon inspection and clearance of roadside culverts.
   - Preservation of existing native vegetative ground cover.
   - *Note: This screening level does NOT warrant emergency or evacuation procedures.*

2. **Screening Level 2 (Index $\\ge 0.37$ — Enhanced Terrain Surveillance):**
   - Recommended for inclusion in municipal and district hazard monitoring registers.
   - Preliminary engineering geological reconnaissance advised prior to major slope excavation or road widening.

3. **Screening Level 3 (Index $\\ge 0.86$ — Priority Geotechnical Characterization):**
   - Significant terrain conditioning detected.
   - All proposed infrastructure alignments, residential developments, or cut slopes should undergo in-situ geotechnical drilling, soil shear strength testing, and certified structural slope engineering prior to regulatory approvals.

---

### 5. Scientific Limitations & Governance Disclaimer
> **CRITICAL SCIENTIFIC & DISASTER MANAGEMENT NOTICE:**
> 1. **Susceptibility $\\neq$ Real-Time Forecasting:** This assessment evaluates **static spatial predisposition** based on long-term geophysical conditioning factors. It does **NOT** forecast the temporal occurrence, timing, or velocity of individual landslide events.
> 2. **Decision-Support Estimate:** Model outputs are computational estimates designed for macro-level spatial screening and regional planning.
> 3. **Non-Statutory Status:** This document is an academic and planning assessment tool. It does **NOT** constitute independent statutory authority for life-safety evacuations, building moratoria, construction bans, or emergency civil defense declarations.
> 4. **Site-Specific Investigations Required:** All structural, municipal, and civil engineering design decisions must involve certified geotechnical engineers and on-site subsurface core borehole sampling.

---
*RockWatch Landslide Intelligence System · Indian Western Himalayas Framework · Generated for Himachal Pradesh Research & Planning*
"""
    return md.strip()


def generate_html_report(data: Dict[str, Any]) -> str:
    """
    Generate a styled, printable HTML landslide susceptibility assessment report.
    """
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    score = data.get("score", 0.0)
    tier_name = data.get("tier_name", "Unknown")
    tier_color = data.get("tier_color", "#10b981")
    inputs = data.get("inputs", {})

    elev = inputs.get("elevation_m", "N/A")
    slope = inputs.get("slope_deg", "N/A")
    aspect = inputs.get("aspect_deg", "N/A")
    lulc_desc = inputs.get("lulc_label", f"Code {inputs.get('selected_lulc', 'N/A')}")
    dist_river = inputs.get("dist_river_m", "N/A")
    flow_accum = inputs.get("flow_accum_cells", "N/A")
    rainfall_idx = inputs.get("rainfall_index", "N/A")
    ndvi = inputs.get("ndvi_val", "N/A")

    site_name = data.get("site_name", "Individual Evaluated Mountain Slope")

    shap_html = ""
    shap_items = data.get("shap_contributions")
    if shap_items:
        shap_html = """
    <h3 style="color: #1e4533; margin-bottom: 8px; margin-top: 18px;">2. Local Model Explainability (SHAP TreeExplainer Attribution)</h3>
    <table>
        <tr>
            <th>Conditioning Feature</th>
            <th>Evaluated Input</th>
            <th>SHAP Contribution (Log-Odds)</th>
            <th>Geomorphic Influence</th>
        </tr>
        """
        for item in shap_items[:6]:
            dir_color = "#ef4444" if item["shap_value"] > 0 else "#10b981"
            dir_txt = "Increasing Risk ↗" if item["shap_value"] > 0 else "Buffering Risk ↘"
            val_str = item.get("raw_val_str") or "—"
            shap_html += f"""
        <tr>
            <td><strong>{item['display_name']}</strong></td>
            <td><code>{val_str}</code></td>
            <td style="color: {dir_color}; font-weight: 700;">{item['shap_value']:+.4f} ({dir_txt})</td>
            <td>{item['description']}</td>
        </tr>
            """
        shap_html += "    </table>\n"

    advisory_sec_num = "3" if shap_items else "2"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RockWatch - Landslide Susceptibility Assessment</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            background: #ffffff;
            margin: 40px auto;
            max-width: 850px;
            line-height: 1.6;
        }}
        .header {{
            border-bottom: 3px solid #1e4533;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            color: #1e4533;
            margin: 0 0 6px 0;
            font-size: 26px;
            font-weight: 800;
        }}
        .header p {{
            color: #64748b;
            margin: 0;
            font-size: 14px;
        }}
        .score-box {{
            background: #f8fafc;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .tier-badge {{
            display: inline-block;
            background: {tier_color}22;
            color: {tier_color};
            border: 1px solid {tier_color};
            font-weight: 800;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0 24px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 12px;
            border: 1px solid #e2e8f0;
            text-align: left;
        }}
        th {{
            background: #f1f5f9;
            color: #334155;
            font-weight: 700;
        }}
        .advisory-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            margin-bottom: 24px;
        }}
        .advisory-card {{
            background: #fafafa;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            font-size: 13px;
        }}
        .disclaimer {{
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            font-size: 13px;
            color: #7f1d1d;
            margin-top: 30px;
        }}
        .footer {{
            margin-top: 36px;
            border-top: 1px solid #e2e8f0;
            padding-top: 14px;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⛰️ RockWatch: Landslide Susceptibility Assessment</h1>
        <p>Scientific Geospatial Environmental Intelligence · Himachal Pradesh, India</p>
    </div>

    <div class="score-box">
        <div>
            <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #64748b;">Target Location Profile</div>
            <div style="font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 2px;">{site_name}</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Assessed: {timestamp} · Model: Gradient Boosting Champion v1</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #64748b;">Relative Susceptibility Index</div>
            <div style="font-size: 32px; font-weight: 800; color: {tier_color}; line-height: 1.1;">{score * 100:.1f}%</div>
            <div style="margin-top: 6px;"><span class="tier-badge">{tier_name.upper()} HAZARD</span></div>
        </div>
    </div>

    <h3 style="color: #1e4533; margin-bottom: 8px;">1. Environmental Conditioning Parameters</h3>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
            <th>Scientific Meaning</th>
        </tr>
        <tr><td><strong>Elevation (SRTM DEM)</strong></td><td>{elev} m</td><td>Macro-climatic thermal and periglacial gradient</td></tr>
        <tr><td><strong>Slope Gradient</strong></td><td>{slope}°</td><td>Gravitational shear driving stress</td></tr>
        <tr><td><strong>Aspect Orientation</strong></td><td>{aspect}°</td><td>Solar insolation and freeze-thaw weathering</td></tr>
        <tr><td><strong>Land Cover (ESA WorldCover)</strong></td><td>{lulc_desc}</td><td>Root cohesion and vegetative stabilization</td></tr>
        <tr><td><strong>Distance to Drainage</strong></td><td>{dist_river} m</td><td>Fluvial toe undercut potential</td></tr>
        <tr><td><strong>Flow Accumulation</strong></td><td>{flow_accum} cells</td><td>Subsurface hydrologic catchment area</td></tr>
        <tr><td><strong>Precipitation Anomaly</strong></td><td>{rainfall_idx}</td><td>Monsoon saturation predisposition index</td></tr>
        <tr><td><strong>Vegetation Index (NDVI)</strong></td><td>{ndvi}</td><td>Sentinel-2 canopy density factor</td></tr>
    </table>

    {shap_html}

    <h3 style="color: #1e4533; margin-bottom: 8px;">{advisory_sec_num}. Geospatial Planning Advisories</h3>
    <div class="advisory-grid">
        <div class="advisory-card">
            <strong style="color: #0f172a;">Level 1 (Index &ge; 0.16)</strong>
            <p style="margin: 6px 0 0 0; color: #475569;">Routine culvert clearance and drainage inspection. Routine vegetative maintenance; not an emergency alert.</p>
        </div>
        <div class="advisory-card">
            <strong style="color: #0f172a;">Level 2 (Index &ge; 0.37)</strong>
            <p style="margin: 6px 0 0 0; color: #475569;">Enhanced terrain surveillance. Recommended for municipal register tracking and slope toe checks.</p>
        </div>
        <div class="advisory-card">
            <strong style="color: #0f172a;">Level 3 (Index &ge; 0.86)</strong>
            <p style="margin: 6px 0 0 0; color: #475569;">Priority geotechnical investigation. In-situ core drilling and structural FoS analysis advised before structural works.</p>
        </div>
    </div>

    <div class="disclaimer">
        <strong>⚠️ SCIENTIFIC & DISASTER MANAGEMENT DISCLAIMER:</strong><br>
        This report reflects <em>static spatial terrain susceptibility</em> based on environmental conditioning factors.
        It does <strong>NOT</strong> forecast real-time failure events or dynamic storm-triggered occurrences.
        These outputs are computational decision-support estimates and do <strong>NOT</strong> constitute statutory authority
        for construction prohibitions, building bans, or civil protection evacuation orders. All engineering interventions
        require certified on-site geotechnical core drilling and ground stabilization analysis.
    </div>

    <div class="footer">
        RockWatch Landslide Susceptibility System · Western Himalayas Research Framework · Report Generated Automatically
    </div>
</body>
</html>
"""
    return html.strip()


def export_regional_summary_csv(df: pd.DataFrame) -> str:
    """
    Export regional and district summary statistics to valid CSV format.
    """
    df_clean = df.copy()
    return df_clean.to_csv(index=False)


def export_regional_geojson(
    df: pd.DataFrame,
    max_features: Optional[int] = None
) -> str:
    """
    Convert filtered prediction dataframe into valid GeoJSON FeatureCollection (RFC 7946).
    Preserves true coordinates and all associated prediction attributes without fabrication.
    """
    df_clean = df.dropna(subset=["latitude", "longitude", "susceptibility_score"]).copy()

    if max_features is not None and len(df_clean) > max_features:
        df_clean = df_clean.sample(n=max_features, random_state=42)

    features = []
    for _, row in df_clean.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        props = {
            "sample_id": int(row["sample_id"]) if "sample_id" in row and pd.notnull(row["sample_id"]) else None,
            "susceptibility_score": round(float(row["susceptibility_score"]), 4),
            "susceptibility_class": str(row.get("susceptibility_class", "Unknown")),
            "elevation_m": round(float(row.get("elevation_m", 0)), 1),
            "slope_deg": round(float(row.get("slope_deg", 0)), 2),
            "ground_truth_class": int(row["class"]) if "class" in row and pd.notnull(row["class"]) else None,
        }

        if "district" in row and pd.notnull(row["district"]):
            props["district"] = str(row["district"])
        if "corridor" in row and pd.notnull(row["corridor"]):
            props["corridor"] = str(row["corridor"])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],  # GeoJSON standard: [longitude, latitude]
            },
            "properties": props,
        }
        features.append(feature)

    geojson_dict = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "RockWatch Himachal Pradesh Landslide Susceptibility Export",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "feature_count": len(features),
            "coordinate_system": "WGS84 (EPSG:4326)",
        },
        "features": features,
    }

    return json.dumps(geojson_dict, indent=2)
