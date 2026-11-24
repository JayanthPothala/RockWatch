# RockWatch – Exploratory Data Analysis (EDA) Report

**Project:** RockWatch – Intelligent Landslide Susceptibility Prediction System  
**Document:** Phase 2 Exploratory Data Analysis (EDA)  
**Study Area:** Himachal Pradesh, India  
**Date:** 2026-11-16  
**Status:** COMPLETE  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Overview & Spatial Split Structure](#2-dataset-overview--spatial-split-structure)
3. [Predictive Separation Power (Univariate Ranking)](#3-predictive-separation-power-univariate-ranking)
4. [Continuous Feature Distributions & Physical Interpretation](#4-continuous-feature-distributions--physical-interpretation)
5. [Land Use / Land Cover (LULC) Susceptibility Patterns](#5-land-use--land-cover-lulc-susceptibility-patterns)
6. [Multicollinearity Diagnostics & Correlation Analysis](#6-multicollinearity-diagnostics--correlation-analysis)
7. [Geographic Covariate Shift Across Splits](#7-geographic-covariate-shift-across-splits)
8. [Elevation–Slope Susceptibility Envelope](#8-elevationslope-susceptibility-envelope)
9. [Key Recommendations for Phase 3 Modeling](#9-key-recommendations-for-phase-3-modeling)

---

## 1. Executive Summary

Following the completion of Phase 1 data cleaning, deduplication, and spatial partitioning, this Exploratory Data Analysis examines the statistical and physical relationships governing landslide susceptibility across the 11,669 processed Himachal Pradesh locations (8,188 TRAIN, 1,731 VAL, 1,750 TEST).

### Key Empirical Findings:
1. **Primary Triggering Factors:**  
   - **`Rainfall_z_corr` is the strongest individual predictor** ($\text{AUC} = 0.7136, \text{Cohen's } d = +0.805, p = 1.40 \times 10^{-245}$). Landslide events occur in significantly wetter zones than pseudo-absences.
   - **`Slope_z_corr` is the second strongest predictor** ($\text{AUC} = 0.6834, \text{Cohen's } d = +0.710, p = 8.00 \times 10^{-182}$). Susceptibility increases steeply above $15^\circ$ slope angle.
2. **Terrain & Aspect Affinity:**  
   - **`cos_aspect` ($\text{AUC} = 0.5519, p = 4.31 \times 10^{-16}$)** shows that south-facing slopes ($\cos < 0$) experience higher landslide frequencies, consistent with Himalayan solar insolation and freeze-thaw weathering dynamics.
   - **`Dist_River_z_corr` ($\text{AUC} = 0.5539, p = 3.05 \times 10^{-17}$)** shows an inverse relationship: proximity to fluvial drainage channels increases landslide risk due to toe erosion.
3. **Anthropogenic & Land Cover Vulnerability:**  
   - **`LULC_40` (Cropland)** acts as a strong protective / negative indicator ($\text{AUC} = 0.5727, \text{Cohen's } d = -0.572$). Landslides almost never occur on flat agricultural valleys ($0.2\%$ prevalence in landslide class vs $14.7\%$ in pseudo-absences).
   - **`LULC_50` (Built-up / Infrastructure)** shows severe vulnerability ($7.7\%$ of landslides occur in built-up pixels vs $1.8\%$ of pseudo-absences, representing a $4.2\times$ overrepresentation).
4. **Low Multicollinearity:**  
   - All continuous predictors exhibit Variance Inflation Factors ($\text{VIF}$) below $1.85$ (highest: `Elevation_z_corr` at $1.85$, `Rainfall_z_corr` at $1.42$). The feature set is free from severe collinearity, ensuring linear models (Logistic Regression) will remain stable.
5. **Marked Geographic Covariate Shift:**  
   - Comparing the southern/central training zone with the northern test zone (Chamba corridor) reveals a significant domain shift: the test split has higher rainfall ($+0.811\sigma$), higher elevation ($+0.568\sigma$), and lower NDVI ($-0.421\sigma$). This confirms that the latitude-blocked split provides a stringent test of physical generalisation.

---

## 2. Dataset Overview & Spatial Split Structure

The processed modelling dataset consists of 11,669 records split into three geographically disjoint latitude bands separated by $500\text{ m}$ buffer exclusion zones.

| Split | Rows | Landslides (1) | Pseudo-Absences (0) | Landslide % | Latitude Bounds | Primary Districts |
|---|---|---|---|---|---|---|
| **TRAIN** | **8,188** | 4,024 | 4,164 | 49.15% | 30.3857°N – 31.7085°N | Shimla, Sirmaur, Solan, Mandi, Bilaspur, Kullu (S) |
| **VAL** | **1,731** | 896 | 835 | 51.76% | 31.7176°N – 32.1397°N | Kangra (Central), Kullu (N), Mandi (N) |
| **TEST** | **1,750** | 914 | 836 | 52.23% | 32.1488°N – 33.2562°N | Chamba, Kangra (N), Lahaul (W) |
| **Exclusion Zone** | **68** | 25 | 43 | 36.76% | Within 500 m of 31.713°N or 32.144°N | Buffer zone |

![Class Distribution](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/class_distribution.png)
*Figure 1: Balanced class distribution (~50:50) preserved across all three spatial partitions.*

![Spatial Split Map](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/spatial_split_map.png)
*Figure 2: Geographic distribution of the spatial splits across Himachal Pradesh showing latitude cuts (CUT1: 31.713°N, CUT2: 32.144°N) and inventory event clustering.*

---

## 3. Predictive Separation Power (Univariate Ranking)

Each feature was evaluated independently on the training set ($N=8,188$) to quantify its individual ability to separate landslide locations from pseudo-absences using the Mann-Whitney $U$ test, Cohen's $d$ effect size, and direction-invariant Univariate ROC-AUC:

| Rank | Feature | Mean (Class 1) | Mean (Class 0) | Delta | Cohen's $d$ | $p$-value | Univariate AUC | Physical Role |
|---|---|---|---|---|---|---|---|---|
| **1** | `Rainfall_z_corr` | $+0.380$ | $-0.367$ | $+0.747$ | $+0.805$ | $1.40 \times 10^{-245}$ | **0.7136** | Primary trigger (pore-pressure) |
| **2** | `Slope_z_corr` | $+0.340$ | $-0.329$ | $+0.669$ | $+0.710$ | $8.00 \times 10^{-182}$ | **0.6834** | Gravitational shear stress |
| **3** | `LULC_40` (Cropland) | $+0.002$ | $+0.147$ | $-0.145$ | $-0.572$ | $1.42 \times 10^{-136}$ | **0.5727** | Protective (flat valley floor) |
| **4** | `Dist_River_z_corr` | $-0.123$ | $+0.119$ | $-0.242$ | $-0.244$ | $3.05 \times 10^{-17}$ | **0.5539** | Fluvial toe undercutting |
| **5** | `cos_aspect` | $-0.161$ | $-0.033$ | $-0.128$ | $-0.182$ | $4.31 \times 10^{-16}$ | **0.5519** | Solar radiation & weathering |
| **6** | `Elevation_z_corr` | $-0.224$ | $+0.217$ | $-0.441$ | $-0.452$ | $2.07 \times 10^{-14}$ | **0.5488** | Mid-elevation valley susceptibility |
| **7** | `NDVI_z_corr` | $-0.007$ | $+0.007$ | $-0.015$ | $-0.015$ | $3.27 \times 10^{-8}$ | **0.5353** | Vegetation cover variation |
| **8** | `FlowAccumulation_z_corr` | $+0.101$ | $-0.098$ | $+0.199$ | $+0.200$ | $2.54 \times 10^{-8}$ | **0.5341** | Hydrological drainage concentration |
| **9** | `LULC_50` (Built-up) | $+0.077$ | $+0.018$ | $+0.059$ | $+0.281$ | $2.87 \times 10^{-36}$ | **0.5294** | Human slope destabilization |
| **10** | `LULC_30` (Grassland) | $+0.211$ | $+0.163$ | $+0.048$ | $+0.124$ | $1.93 \times 10^{-8}$ | **0.5242** | Unvegetated steep soil |
| **11** | `LULC_10` (Trees) | $+0.526$ | $+0.492$ | $+0.034$ | $+0.069$ | $1.93 \times 10^{-3}$ | **0.5171** | Dominant land cover type |
| **12** | `LULC_60` (Bare ground) | $+0.144$ | $+0.114$ | $+0.030$ | $+0.090$ | $4.93 \times 10^{-5}$ | **0.5150** | Rocky slopes |
| **13** | `LULC_100` (Moss/Lichen) | $+0.004$ | $+0.030$ | $-0.027$ | $-0.203$ | $5.24 \times 10^{-20}$ | **0.5133** | High-altitude alpine |
| **14** | `sin_aspect` | $-0.005$ | $-0.034$ | $+0.029$ | $+0.041$ | $6.72 \times 10^{-2}$ | **0.5117** | East-West orientation |
| **15** | `LULC_20` (Shrubland) | $+0.033$ | $+0.011$ | $+0.022$ | $+0.155$ | $2.68 \times 10^{-12}$ | **0.5112** | Minor cover class |
| **16** | `LULC_70` (Snow/Ice) | $+0.000$ | $+0.018$ | $-0.018$ | $-0.189$ | $1.99 \times 10^{-17}$ | **0.5089** | Glacial zones (no inventory) |
| **17** | `LULC_80` (Water) | $+0.002$ | $+0.006$ | $-0.004$ | $-0.060$ | $7.00 \times 10^{-3}$ | **0.5020** | Water bodies (snapping artefacts) |

---

## 4. Continuous Feature Distributions & Physical Interpretation

![Feature Distributions](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/feature_distributions.png)
*Figure 3: Probability density functions (KDE) comparing landslides (orange) and pseudo-absences (green) across all continuous predictors.*

### Key Observations:
1. **Precipitation (`Rainfall_z_corr`):**  
   The distribution of rainfall for landslides displays a marked rightward shift relative to pseudo-absences. In the Himalayas, monsoonal precipitation elevates pore-water pressure, reducing effective soil normal stress along shear planes.
2. **Slope Angle (`Slope_z_corr`):**  
   Pseudo-absences are concentrated at lower slopes ($\mu = -0.329$), while landslide events are concentrated on moderate-to-steep slopes ($\mu = +0.340$, corresponding to $25^\circ$–$45^\circ$).
3. **Aspect Decomposition (`cos_aspect` and `sin_aspect`):**  
   While `sin_aspect` (East vs West) shows nearly identical distributions ($p = 0.067$), `cos_aspect` (North vs South) shows significant negative bias for landslides ($\mu = -0.161$ vs $-0.033, p = 4.31 \times 10^{-16}$). South-facing slopes in Himachal Pradesh receive higher insolation, deeper thermal cycles, and intensified freeze-thaw degradation, accelerating regolith weathering.
4. **River Distance (`Dist_River_z_corr`):**  
   Landslides cluster significantly closer to river channels ($\mu = -0.123\sigma$), where active fluvial incision undercuts valley slopes and promotes slope failure.

---

## 5. Land Use / Land Cover (LULC) Susceptibility Patterns

![LULC Prevalence](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/lulc_susceptibility.png)
*Figure 4: Landslide prevalence rate and sample counts across Copernicus LULC categories.*

| LULC Category | Code | Total Samples (Train) | Landslides | Prevalence Rate (%) | Physical Susceptibility Interpretation |
|---|---|---|---|---|---|
| **Built-up** | `LULC_50` | 388 | 313 | **80.67%** | **Extreme Susceptibility:** Road construction, excavation, hill cutting, and altered drainage severely exacerbate slope failure along urban and transport corridors. |
| **Shrubland** | `LULC_20` | 179 | 134 | **74.86%** | **High Susceptibility:** Degraded hillsides with shallow root depth and exposed topsoil. |
| **Grassland** | `LULC_30` | 1,531 | 851 | **55.58%** | **Moderate-High:** Steep mountain meadows with variable slope stability. |
| **Bare ground** | `LULC_60` | 1,057 | 581 | **54.97%** | **Moderate:** Rocky cliffs and scree slopes. |
| **Trees / Forest** | `LULC_10` | 4,171 | 2,118 | **50.78%** | **Baseline:** Dominant land cover in Himachal; reflects regional slope background rate. |
| **Water Bodies** | `LULC_80` | 33 | 8 | **24.24%** | **Diagnostic Artefact:** Pixel snapping near lake/river boundaries (flagged by `flag_water_lulc`). |
| **Moss & Lichen** | `LULC_100` | 141 | 18 | **12.77%** | **Low:** High-altitude tundra with minimal human interaction and low inventory reporting. |
| **Cropland** | `LULC_40` | 616 | 7 | **1.14%** | **Very Low / Protective:** Concentrated in low-gradient valley bottoms where gravitational driving forces are negligible. |
| **Snow & Ice** | `LULC_70` | 72 | 0 | **0.00%** | **Complete Negative Separator:** Perpetual snowpack and glaciers contain zero recorded inventory events. |

---

## 6. Multicollinearity Diagnostics & Correlation Analysis

![Correlation Heatmap](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/correlation_heatmap.png)
*Figure 5: Spearman rank correlation matrix for continuous predictors and target class.*

### Variance Inflation Factors (VIF):

| Predictor | VIF | $R^2$ with other features | Multicollinearity Status |
|---|---|---|---|
| `Elevation_z_corr` | **1.85** | $0.459$ | Low ($< 5.0$) — Well within acceptable limits |
| `Rainfall_z_corr` | **1.42** | $0.295$ | Low |
| `NDVI_z_corr` | **1.31** | $0.239$ | Low |
| `Slope_z_corr` | **1.22** | $0.181$ | Low |
| `FlowAccumulation_z_corr` | **1.08** | $0.072$ | Very Low |
| `cos_aspect` | **1.04** | $0.039$ | Negligible |
| `Dist_River_z_corr` | **1.01** | $0.012$ | Negligible |
| `sin_aspect` | **1.00** | $0.004$ | Negligible |

> [!NOTE]
> Standard econometrics and machine learning rules of thumb consider $\text{VIF} > 5.0$ or $10.0$ indicative of harmful collinearity. All predictors in RockWatch have $\text{VIF} < 1.85$. This confirms that no continuous features need to be dropped due to collinear redundancy.

---

## 7. Geographic Covariate Shift Across Splits

Because the dataset was partitioned geographically by latitude bands to prevent spatial data leakage, environmental covariates naturally exhibit regional gradients between southern, central, and northern Himachal Pradesh:

| Feature | Train Mean ($< 31.71^\circ\text{N}$) | Val Mean ($31.71^\circ$–$32.14^\circ\text{N}$) | Test Mean ($\ge 32.14^\circ\text{N}$) | Max Covariate Shift ($\sigma$) | Regional Environmental Cause |
|---|---|---|---|---|---|
| `Rainfall_z_corr` | $+0.000$ | $+0.682$ | $+0.811$ | **$+0.811\sigma$** | Chamba and northern Dhauladhar ranges receive heavier orographic rainfall. |
| `Elevation_z_corr` | $-0.000$ | $+0.069$ | $+0.568$ | **$+0.568\sigma$** | Elevation increases northward into the Greater Himalaya and Pir Panjal ranges. |
| `NDVI_z_corr` | $+0.000$ | $-0.239$ | $-0.421$ | **$-0.421\sigma$** | Denser sub-tropical forests in southern HP; sparse alpine cover in northern HP. |
| `Slope_z_corr` | $+0.000$ | $-0.114$ | $+0.201$ | **$+0.201\sigma$** | Rugged alpine relief in the northern test corridor. |
| `FlowAccumulation_z_corr` | $+0.000$ | $+0.164$ | $+0.100$ | **$+0.164\sigma$** | Major Ravi and Beas river basin confluences in central/northern HP. |
| `sin_aspect` | $-0.019$ | $-0.129$ | $-0.050$ | **$+0.109$** | Minor East/West terrain orientation variation. |
| `Dist_River_z_corr` | $-0.000$ | $-0.032$ | $-0.034$ | **$-0.034\sigma$** | Drainage density remains relatively stable. |
| `cos_aspect` | $-0.095$ | $-0.094$ | $-0.114$ | **$-0.019$** | Valley strike orientation is consistent across zones. |

> [!IMPORTANT]
> **Implication for Model Generalization:**
> Models evaluated on random cross-validation often achieve artificially inflated metrics ($0.90+$ AUC) by interpolating local rainfall and elevation clusters. With our spatial blocking, the model trained on southern HP must successfully generalize to the higher-rainfall, higher-elevation northern test environment. This provides a genuine assessment of physical transportability.

---

## 8. Elevation–Slope Susceptibility Envelope

![Elevation vs Slope Profile](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/elevation_slope_profile.png)
*Figure 6: 2D joint density profiles comparing landslide events vs pseudo-absences across elevation and slope space.*

1. **Susceptibility Envelope:** Landslides in Himachal Pradesh are concentrated within a distinct topographical envelope: **elevations between $1,000\text{ m}$ and $2,800\text{ m}$** on **slopes between $20^\circ$ and $45^\circ$**.
2. **Alpine Attenuation:** Above $3,500\text{ m}$, recorded landslide density decreases markedly despite steep slopes, likely due to a combination of permanent snow cover, rocky substrate, and lack of human reporting infrastructure in uninhabited alpine zones.
3. **Valley Attenuation:** Below $1,000\text{ m}$ on slopes $< 15^\circ$, shear stress is insufficient to overcome soil cohesion, resulting in near-zero landslide hazard.

---

## 9. Key Recommendations for Phase 3 Modeling

Based on the quantitative findings of this EDA, the following guidelines are established for **Phase 3 (Baseline Models)** and **Phase 4 (Advanced Models)**:

1. **Model Selection Strategy:**  
   - **Logistic Regression Baseline:** Recommended with L2 regularization (`C=1.0`). Must drop `LULC_100` as the reference category to prevent dummy variable trap.
   - **Nonlinear Tree Ensembles (Random Forest, Gradient Boosting):** Crucial for capturing the nonlinear interaction between elevation and slope (the susceptibility envelope identified in Section 8), which linear models cannot represent without explicit polynomial terms.
2. **Handling `LULC_70` (Snow/Ice):**  
   - In training data, `LULC_70` has $0$ landslides ($100\%$ pseudo-absences). Tree models will naturally use this as an absolute negative rule, which is geomorphologically sound.
3. **Primary Evaluation Metrics:**  
   - Because the test set exhibits realistic covariate shift, **Spatial Cross-Validation AUC-ROC** and **Held-Out Test AUC-ROC** must serve as the primary success metrics. Precision, Recall, F1, and Matthews Correlation Coefficient ($\text{MCC}$) should be evaluated at the optimal threshold selected on the validation set.
4. **Validation Progression:**  
   - Phase 3 will benchmark:
     1. `DummyClassifier(strategy='most_frequent')` (lower-bound sanity check)
     2. `LogisticRegression` (interpretable linear baseline)
     3. `DecisionTreeClassifier(max_depth=5)` (interpretable nonlinear baseline)

---

*Report finalised: 2026-11-16 | All figures and code verified | Ready for Phase 3 Baseline Models*
