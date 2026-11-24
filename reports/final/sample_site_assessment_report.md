# RockWatch: Landslide Susceptibility Assessment Report
**Geospatial Environmental Intelligence System for Himachal Pradesh, India**

---

### 1. Document & Evaluation Metadata
- **Report Title:** Site-Specific Landslide Susceptibility Assessment
- **Evaluation Timestamp:** 2026-11-24 12:45:00 UTC
- **Assessed Location / Profile:** Kangra Steep Monsoon Ridge (Audit Sample)
- **Analytical Model:** Gradient Boosting Classifier (Champion Ensemble v1.0)
- **Baseline Calibration Reference:** Threshold $T = 0.37$ (Balanced $F_1 = 0.737$, $\text{MCC} = 0.453$)
- **Validation Standard:** Spatially Blocked Latitudinal Holdout (500 m Exclusion Buffer)

---

### 2. Terrain & Environmental Conditioning Profile
| Conditioning Parameter | Evaluated Input Value | Scientific Domain / Units | Geomorphic Role |
| :--- | :--- | :--- | :--- |
| **Elevation** | `1650 m` | Meters above sea level | Macro-climatic thermal and periglacial gradient |
| **Slope Gradient** | `38.0°` | Degrees inclination | Gravitational shear stress component |
| **Aspect Orientation** | `210.0°` | Compass bearing ($0°=\text{N}, 180°=\text{S}$) | Solar insolation and freeze-thaw cycles |
| **Land Cover (LULC)** | `Bare Ground / Sparse Rock (LULC 60)` | ESA WorldCover 10 m category | Root cohesion and surface vegetative protection |
| **Stream Proximity** | `180 m` | Distance to nearest channel | Fluvial toe-erosion and undercut susceptibility |
| **Flow Accumulation** | `850 cells` | Drainage catchment cell count | Subsurface hydrological pore-pressure build-up |
| **Precipitation Index** | `1.6` | Standardized monsoon anomaly | Regional monsoonal saturation predisposition |
| **Vegetation Index** | `0.38` | Sentinel-2 NDVI ($-1.0$ to $+1.0$) | Canopy density and bio-stabilization factor |

---

### 3. Model Inference & Susceptibility Classification
- **Relative Susceptibility Index:** **`68.40%`** (Model Score: `0.6840` on range $[0.0, 1.0]$)
- **Classified Hazard Tier:** **`HIGH HAZARD`**
- **Classification Baseline:**
  - *Low Hazard:* Score $< 0.20$
  - *Moderate Hazard:* $0.20 \le \text{Score} < 0.45$
  - *High Hazard:* $0.45 \le \text{Score} < 0.70$
  - *Very High Hazard:* $\text{Score} \ge 0.70$

#### Key Conditioning Influences:
- **High Slope Angle (38.0°):** Exceeds critical geotechnical angle of repose for loose colluvium; major contributor to shear instability.
- **Critical Elevation Belt (1650 m):** Falls within the active Himalayan monsoonal cloudburst and toe-cutting zone.
- **Fluvial Proximity (180 m):** High vulnerability to channel bank scouring during peak river discharge.

#### Local Model Explainability (SHAP TreeExplainer Attribution):
| Feature | Evaluated Input | SHAP Contribution (Log-Odds) | Geomorphic Influence |
| :--- | :--- | :--- | :--- |
| **Slope Gradient** | `38.0°` | `+0.4852` (Increasing Risk ↗) | High gravitational shear stress driving factor. |
| **Distance to Stream** | `180 m` | `+0.3210` (Increasing Risk ↗) | Fluvial toe undercut proximity. |
| **Vegetation Vigour (NDVI)** | `0.38` | `-0.1540` (Buffering Risk ↘) | Moderate vegetative bio-stabilization. |

---

### 4. Planning & Screening Advisories
These guidelines represent regional planning decision-support filters derived from statistical model calibration:

1. **Screening Level 1 (Index $\ge 0.16$ — Routine Maintenance):**
   - Regular pre-monsoon inspection and clearance of roadside culverts.
   - Preservation of existing native vegetative ground cover.
   - *Note: This screening level does NOT warrant emergency or evacuation procedures.*

2. **Screening Level 2 (Index $\ge 0.37$ — Enhanced Terrain Surveillance):**
   - Recommended for inclusion in municipal and district hazard monitoring registers.
   - Preliminary engineering geological reconnaissance advised prior to major slope excavation or road widening.

3. **Screening Level 3 (Index $\ge 0.86$ — Priority Geotechnical Characterization):**
   - Significant terrain conditioning detected.
   - All proposed infrastructure alignments, residential developments, or cut slopes should undergo in-situ geotechnical drilling, soil shear strength testing, and certified structural slope engineering prior to regulatory approvals.

---

### 5. Scientific Limitations & Governance Disclaimer
> **CRITICAL SCIENTIFIC & DISASTER MANAGEMENT NOTICE:**
> 1. **Susceptibility $\neq$ Real-Time Forecasting:** This assessment evaluates **static spatial predisposition** based on long-term geophysical conditioning factors. It does **NOT** forecast the temporal occurrence, timing, or velocity of individual landslide events.
> 2. **Decision-Support Estimate:** Model outputs are computational estimates designed for macro-level spatial screening and regional planning.
> 3. **Non-Statutory Status:** This document is an academic and planning assessment tool. It does **NOT** constitute independent statutory authority for life-safety evacuations, building moratoria, construction bans, or emergency civil defense declarations.
> 4. **Site-Specific Investigations Required:** All structural, municipal, and civil engineering design decisions must involve certified geotechnical engineers and on-site subsurface core borehole sampling.

---
*RockWatch Landslide Intelligence System · Indian Western Himalayas Framework · Generated for Himachal Pradesh Research & Planning*