# RockWatch – Baseline Models Performance & Evaluation Report

**Project:** RockWatch – Intelligent Landslide Susceptibility Prediction System  
**Phase:** Phase 3 — Baseline Models Benchmark  
**Study Area:** Himachal Pradesh, India  
**Date:** 2026-11-16  
**Status:** COMPLETE  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Baseline Models Overview](#2-baseline-models-overview)
3. [Quantitative Performance Comparison](#3-quantitative-performance-comparison)
4. [Receiver Operating Characteristic (ROC) Analysis](#4-receiver-operating-characteristic-roc-analysis)
5. [Logistic Regression Coefficient & Odds Ratio Interpretation](#5-logistic-regression-coefficient--odds-ratio-interpretation)
6. [Decision Tree Top Split Decisions & Nonlinearity](#6-decision-tree-top-split-decisions--nonlinearity)
7. [Spatial Generalization & Covariate Shift Analysis](#7-spatial-generalization--covariate-shift-analysis)
8. [Artifacts Generated](#8-artifacts-generated)
9. [Recommendations for Phase 4 (Advanced Models)](#9-recommendations-for-phase-4-advanced-models)

---

## 1. Executive Summary

In Phase 3 of the RockWatch pipeline, three baseline models were implemented, trained strictly on the geographical training partition ($N=8,188$, southern Himachal Pradesh), tuned on the spatial validation partition ($N=1,731$, central Himachal Pradesh), and evaluated once on the held-out spatial test partition ($N=1,750$, northern Himachal Pradesh / Chamba):

1. **`DummyClassifier` (Prior baseline):** Established the trivial lower-bound benchmark ($\text{AUC} = 0.5000, \text{MCC} = 0.0000$).
2. **`LogisticRegression` (L2 Regularized linear model):** Achieved **Validation AUC = 0.8356** and **Test AUC = 0.7000** ($\text{Test F1} = 0.7070, \text{Test Accuracy} = 64.00\%$).
3. **`DecisionTreeClassifier` (Shallow nonlinear tree, $d=5$):** Achieved **Validation AUC = 0.8613** and **Test AUC = 0.7606** ($\text{Test F1} = 0.7197, \text{Test MCC} = 0.3899, \text{Test Accuracy} = 69.60\%$).

### Key Scientific Insights:
- **Tree-based advantage in geographic generalization:** The Decision Tree outperformed Logistic Regression by **$+0.0606$ in Held-out Test AUC** ($0.7606$ vs $0.7000$) and by **$+0.1024$ in Matthews Correlation Coefficient** ($0.3899$ vs $0.2875$). This demonstrates that nonlinear partitioning across elevation and slope thresholds enables significantly better transportability into the northern climate regime.
- **Physical interpretability of logistic coefficients:** The linear model verified all physical mechanisms documented in Phase 2: built-up human infrastructure ($\text{OR} = 5.59$), steep slopes ($\text{OR} = 2.84$), and heavy precipitation ($\text{OR} = 2.17$) act as strong risk multipliers, while cropland ($\text{OR} = 0.068$) and permanent snow/ice ($\text{OR} = 0.265$) act as protective envelopes.

---

## 2. Baseline Models Overview

All models were fitted on `features_train.csv` ($N=8,188$):

| Model | Architecture | Hyperparameters | Predictors Included | Purpose |
|---|---|---|---|---|
| **Dummy (Prior)** | Empirical prior probability | `strategy='prior'` | All 17 features (ignored) | Theoretical baseline sanity check |
| **Logistic Regression** | Linear with L2 penalty | $C=1.0$, `solver='lbfgs'`, `max_iter=1000` | 16 features (`LULC_100` dropped as reference) | Interpretable linear benchmark |
| **Decision Tree** | CART binary decision tree | `max_depth=5`, `min_samples_leaf=20` | All 17 features | Interpretable shallow nonlinear benchmark |

---

## 3. Quantitative Performance Comparison

Classification metrics across Train, Validation, and Held-out Spatial Test sets:

| Model | Train AUC | Val AUC | Held-Out Test AUC | Val PR-AUC | Test PR-AUC | Val Optimal Threshold | Val F1 (Opt) | Test F1 (Opt) | Test MCC (Opt) | Test Accuracy (Opt) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Dummy (Prior)** | $0.5000$ | $0.5000$ | **$0.5000$** | $0.5176$ | $0.5223$ | $0.10$ | $0.6821$ | $0.6862$ | $0.0000$ | $52.23\%$ |
| **Logistic Regression** | $0.8620$ | $0.8356$ | **$0.7000$** | $0.8067$ | $0.6984$ | $0.43$ | $0.7960$ | $0.7070$ | $0.2875$ | $64.00\%$ |
| **Decision Tree ($d=5$)** | $0.8849$ | $0.8613$ | **$0.7606$** | $0.8391$ | $0.7354$ | $0.22$ | $0.8255$ | $0.7197$ | $0.3899$ | $69.60\%$ |

> [!NOTE]
> The optimal classification threshold was calibrated on the validation set to maximize F1-score and applied out-of-sample to the test set, preventing threshold-selection leakage.

---

## 4. Receiver Operating Characteristic (ROC) Analysis

![Baseline ROC Curves](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/roc_curves_baselines.png)
*Figure 1: Receiver Operating Characteristic (ROC) curves on Validation and Held-out Spatial Test partitions.*

### Observations on Generalization Drop:
- **Logistic Regression:** Drops from $0.8356$ (Val) to $0.7000$ (Test) ($\Delta = -0.1356$). The linear hyperplane struggles with the northern shift where rainfall is uniformly higher.
- **Decision Tree ($d=5$):** Drops from $0.8613$ (Val) to $0.7606$ (Test) ($\Delta = -0.1007$). Retains significantly higher discriminative power because its step functions decouple high rainfall from high slope conditions.

---

## 5. Logistic Regression Coefficient & Odds Ratio Interpretation

Fitted model intercept: $\beta_0 = -0.9043$ (baseline log-odds corresponding to reference class `LULC_100` / Moss & Lichen at average predictor values).

| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | 95% Direction | Geomorphological Interpretation |
|---|---|---|---|---|
| `LULC_50` (Built-up) | **$+1.7201$** | **$5.5851$** | Positive ($p < 10^{-10}$) | **Extreme Risk Multiplier:** Buildings, roads, and cut slopes increase odds by $5.6\times$. |
| `Slope_z_corr` | **$+1.0438$** | **$2.8400$** | Positive | **Primary Gravity Driver:** Each $+1\sigma$ slope increases odds by $2.84\times$. |
| `LULC_60` (Bare ground) | **$+1.0085$** | **$2.7414$** | Positive | Steep rockfaces and scree slopes have $2.74\times$ higher odds. |
| `LULC_30` (Grassland) | **$+0.9863$** | **$2.6814$** | Positive | Mountain pastures on steep slopes experience shallow failures. |
| `LULC_10` (Forest / Trees) | **$+0.9797$** | **$2.6637$** | Positive | Natural hillside baseline in Himachal Pradesh. |
| `LULC_20` (Shrubland) | **$+0.9480$** | **$2.5806$** | Positive | Degraded shrub vegetation on hillside slopes. |
| `Rainfall_z_corr` | **$+0.7729$** | **$2.1660$** | Positive | **Primary Trigger:** Each $+1\sigma$ precipitation increases odds by $2.17\times$. |
| `sin_aspect` | $+0.0924$ | $1.0968$ | Weak positive | Minor East/West orientation difference. |
| `FlowAccumulation_z_corr` | $+0.0767$ | $1.0797$ | Weak positive | Minor drainage convergence effect. |
| `cos_aspect` | $-0.0249$ | $0.9754$ | Weak negative | South-facing slopes slightly elevated. |
| `Dist_River_z_corr` | **$-0.3302$** | **$0.7188$** | Negative | **Fluvial Undercutting:** Moving $+1\sigma$ away from rivers reduces odds by $28.1\%$. |
| `NDVI_z_corr` | **$-0.6894$** | **$0.5019$** | Negative | Dense canopy cover reduces odds by $49.8\%$. |
| `Elevation_z_corr` | **$-1.0226$** | **$0.3596$** | Negative | High alpine zones have reduced reporting / inventory presence. |
| `LULC_70` (Snow / Ice) | **$-1.3286$** | **$0.2649$** | Negative | Glacial regions have $73.5\%$ reduced landslide odds. |
| `LULC_80` (Water bodies) | **$-2.1712$** | **$0.1140$** | Negative | Surface water pixels have $88.6\%$ reduced odds. |
| `LULC_40` (Cropland) | **$-2.6934$** | **$0.0677$** | Negative | **Protective Valley:** Flat agricultural terraces have $93.2\%$ reduced odds. |

---

## 6. Decision Tree Top Split Decisions & Nonlinearity

Analyzing the fitted shallow tree ($d=5$) reveals the hierarchically discovered physical rules:
1. **Root Split ($d=0$):** `Rainfall_z_corr <= +0.076`
   - Immediately separates the drier southern lowlands from high-precipitation monsoonal belts.
2. **Second-Level Splits ($d=1$):**
   - In wet regions: splits on `Slope_z_corr <= +0.023` (identifies steep wet slopes as the primary high-hazard cluster).
   - In dry regions: splits on `Elevation_z_corr <= -0.584` (identifies low-elevation river valleys).
3. **Third-Level Splits ($d=2$):**
   - Incorporates `LULC_40` (Cropland as zero-landslide leaf) and `Dist_River_z_corr` (fluvial proximity).

---

## 7. Spatial Generalization & Covariate Shift Analysis

The held-out test partition represents the Chamba/northern corridor ($> 32.14^\circ\text{N}$), which has:
- **$+0.811\sigma$ higher mean rainfall** than the training set.
- **$+0.568\sigma$ higher mean elevation**.
- **$-0.421\sigma$ lower vegetation density (NDVI)**.

Because of this domain shift, linear models trained with an unconstrained positive weight on rainfall overestimate susceptibility in the north unless calibrated with nonlinear interaction terms. The decision tree mitigates this by conditioning rainfall on slope and elevation boundaries.

---

## 8. Artifacts Generated

1. **Serialized Models (`models/trained/`):**
   - [`dummy_baseline.joblib`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/models/trained/dummy_baseline.joblib)
   - [`logistic_regression_v1.joblib`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/models/trained/logistic_regression_v1.joblib)
   - [`decision_tree_v1.joblib`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/models/trained/decision_tree_v1.joblib)
2. **Benchmark Tables & Metrics (`models/evaluation/`):**
   - [`baseline_results.csv`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/models/evaluation/baseline_results.csv)
   - [`baseline_metrics.json`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/models/evaluation/baseline_metrics.json)
3. **Plots & Reports:**
   - [`roc_curves_baselines.png`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/figures/roc_curves_baselines.png)
   - [`04_baseline_models.ipynb`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/notebooks/04_baseline_models.ipynb)
   - [`baseline_models_report.md`](file:///c:/Users/potha/OneDrive/Desktop/AI%20PROJECTS/RockWatch/reports/models/baseline_models_report.md)

---

## 9. Recommendations for Phase 4 (Advanced Models)

1. **Random Forest Classifier:**
   - Ensembling multiple de-correlated trees with bootstrap aggregation (`bagging`) should reduce variance and improve the test AUC beyond the Decision Tree's $0.7606$.
   - Hyperparameters to optimize: `n_estimators` ($100$–$500$), `max_depth` ($8$–$20$), `min_samples_leaf` ($5$–$20$).
2. **Gradient Boosting Classifier (HistGradientBoosting / LightGBM / XGBoost):**
   - Sequential boosting will focus on hard-to-classify examples near valley-slope interfaces.
   - Hyperparameters to optimize: `learning_rate` ($0.03$–$0.1$), `max_iter` ($100$–$300$), `max_leaf_nodes` ($15$–$31$).
3. **Support Vector Machine (SVM with RBF Kernel):**
   - Radial basis function mapping to test smooth nonlinear boundaries in continuous space.
4. **Target Benchmark:**
   - Beat the current top baseline (Decision Tree: **Test AUC = 0.7606, Test F1 = 0.7197, Test MCC = 0.3899**).

---

*Report finalised: 2026-11-16 | Phase 3 Complete | Awaiting approval for Phase 4 (Advanced Models)*
