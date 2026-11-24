# RockWatch – Dataset Audit Report

**Project:** RockWatch – Intelligent Landslide Susceptibility Prediction System  
**Audit Date:** 2026-11-16  
**Auditor:** Automated pre-training audit (Python 3.14, Pandas, NumPy)  
**Dataset Version:** Version 2 (October 2026)  
**Data Location:** `data/raw/himachal_landslide/`

> [!IMPORTANT]
> The original dataset files were **never modified** during this audit. All statistics are computed from the raw files as downloaded. No synthetic data or fabricated records appear anywhere in this report.

---

## Table of Contents

1. [File Inventory](#1-file-inventory)
2. [Main Modelling Dataset – Structural Overview](#2-main-modelling-dataset--structural-overview)
3. [Target Variable & Class Distribution](#3-target-variable--class-distribution)
4. [Predictor Feature Summary](#4-predictor-feature-summary)
5. [Missing Values](#5-missing-values)
6. [Duplicates](#6-duplicates)
7. [Data Quality – Outliers & Distributions](#7-data-quality--outliers--distributions)
8. [Coordinate Bounds Check](#8-coordinate-bounds-check)
9. [Suspicious Records](#9-suspicious-records)
10. [Redundant & Non-Predictor Columns](#10-redundant--non-predictor-columns)
11. [Cross-File Linkage Verification](#11-cross-file-linkage-verification)
12. [Supporting Files Audit](#12-supporting-files-audit)
13. [Scientific Assessment](#13-scientific-assessment)
14. [Preprocessing Leakage Analysis](#14-preprocessing-leakage-analysis)
15. [Spatial Split Strategy](#15-spatial-split-strategy)
16. [Limitations Summary](#16-limitations-summary)
17. [Recommendations Before Model Training](#17-recommendations-before-model-training)

---

## 1. File Inventory

| File | Size (bytes) | Rows | Columns | Purpose |
|------|-------------|------|---------|---------|
| `hp_landslide_inventory_clean.csv` | 424,028 | 6,539 | 14 | Georeferenced GSI landslide inventory (all label = 1). Cleaned from source. |
| `landslide_features_master_6503.csv` | 2,435,506 | 6,503 | 46 | 40 environmental features extracted at each landslide location via GEE. Landslides only. |
| `landslide_modelling_dataset_12409.csv` | 3,366,994 | 12,409 | 30 | **Primary modelling dataset.** 6,531 landslide + 5,878 pseudo-absence samples. 7 z-scored predictors + LULC one-hot encoding. |
| `inventory_records_without_features_36.csv` | 2,474 | 36 | 14 | 36 inventory records for which GEE feature extraction failed. |
| `data_dictionary.csv` | 11,067 | 90 | 6 | Column-level documentation for all files. |
| `figure_1_outlined.png` | 2,938,683 | — | — | Study area map figure. |
| `README.md` | 9,936 | — | — | Dataset description, file relationships, known limitations. |

All 7 files are present. All 5 CSV files were successfully loaded and parsed without errors.

---

## 2. Main Modelling Dataset – Structural Overview

**File:** `landslide_modelling_dataset_12409.csv`

| Property | Value |
|----------|-------|
| Total rows | **12,409** |
| Total columns | **30** |
| Row index | `sample_id` (1 to 12,409, sequential, no gaps) |
| Coordinate system | WGS84 / EPSG:4326 (decimal degrees) |
| Coordinate columns | `latitude`, `longitude` (30 m SRTM pixel centres) |
| Target variable | `class` (integer: 1 = landslide, 0 = non-landslide pseudo-absence) |

### Column List and Data Types

| # | Column | dtype | Role |
|---|--------|-------|------|
| 1 | `sample_id` | int64 | Row identifier – **not a predictor** |
| 2 | `gee_system_index` | str | GEE export key – **not a predictor** |
| 3 | `longitude` | float64 | Spatial coordinate – spatial split key only |
| 4 | `latitude` | float64 | Spatial coordinate – spatial split key only |
| 5 | `class` | int64 | **Target variable** (0/1) |
| 6 | `Aspect_z` | float64 | z-scored predictor |
| 7 | `Dist_River_z` | float64 | z-scored predictor |
| 8 | `Elevation_z` | float64 | z-scored predictor |
| 9 | `FlowAccumulation_z` | float64 | z-scored predictor |
| 10 | `NDVI_z` | float64 | z-scored predictor |
| 11 | `Rainfall_z` | float64 | z-scored predictor |
| 12 | `Slope_z` | float64 | z-scored predictor |
| 13 | `aspect_deg` | float64 | Raw terrain (degrees) – redundant with `Aspect_z` |
| 14 | `elevation_m` | int64 | Raw terrain (metres) – redundant with `Elevation_z` |
| 15 | `slope_deg` | float64 | Raw terrain (degrees) – redundant with `Slope_z` |
| 16–25 | `LULC_10` … `LULC_100` | int64 | One-hot land-cover flags (10 classes) |
| 26 | `lulc_code` | int64 | LULC code – **redundant** with one-hot columns |
| 27 | `lulc_name` | str | LULC name – **redundant** with one-hot columns |
| 28 | `nearest_inventory_sno` | float64 | Class-1-only link to inventory – **not a predictor** |
| 29 | `match_dist_m` | float64 | Distance to nearest inventory record – **not a predictor** |
| 30 | `n_samples_at_coord` | int64 | Co-location count – **not a predictor** |

---

## 3. Target Variable & Class Distribution

| Class | Label | Count | Percentage |
|-------|-------|-------|-----------|
| 1 | Landslide | **6,531** | 52.63 % |
| 0 | Non-landslide (pseudo-absence) | **5,878** | 47.37 % |
| **Total** | | **12,409** | 100.00 % |

**Class balance ratio (1:0):** 1.1111 — the dataset is **mildly imbalanced**, with landslide samples outnumbering pseudo-absences by ~11 %. This level of imbalance is mild and unlikely to cause severe class weighting issues, but should be tracked at evaluation time (use AUC-ROC and F1 alongside accuracy).

---

## 4. Predictor Feature Summary

### 4.1 Z-Scored Predictors (7 columns)

These are the **primary modelling features** in the dataset.

| Predictor | Raw source | Min (z) | 25th % | Median | 75th % | Max (z) | Corr. with class |
|-----------|-----------|---------|--------|--------|--------|---------|-----------------|
| `Aspect_z` | SRTM aspect (°) | −1.804 | −0.802 | 0.049 | 0.788 | 1.831 | +0.011 (very weak) |
| `Dist_River_z` | HydroSHEDS distance to river | −0.131 | −0.129 | −0.126 | −0.121 | 8.041 | −0.123 (weak) |
| `Elevation_z` | SRTM elevation (m) | −1.214 | −0.755 | −0.266 | 0.424 | 3.680 | −0.291 (moderate) |
| `FlowAccumulation_z` | HydroSHEDS log flow accum. | −0.578 | −0.578 | −0.389 | 0.008 | 5.005 | +0.133 (weak) |
| `NDVI_z` | Landsat NDVI median | −5.971 | −0.688 | 0.141 | 0.849 | 1.795 | +0.054 (very weak) |
| `Rainfall_z` | CHIRPS rainfall | −2.328 | −0.721 | 0.026 | 0.556 | 3.924 | +0.324 (**moderate-strong**) |
| `Slope_z` | SRTM slope (°) | −1.992 | −0.654 | 0.090 | 0.685 | 3.733 | +0.245 (moderate) |

**Strongest predictors** (by correlation with class): Rainfall_z (+0.324), Slope_z (+0.245), Elevation_z (−0.291).  
**Weakest predictor**: Aspect_z (+0.011) — may contribute negligible signal by itself.

### 4.2 Raw Terrain Columns (3 columns — available for reference)

| Column | Min | Mean | Median | Max | Notes |
|--------|-----|------|--------|-----|-------|
| `aspect_deg` | 0° | 178.19° | 183.21° | 358.95° | Circular variable; linear z-score may be problematic |
| `elevation_m` | 246 m | 1,780 m | 1,444 m | **6,430 m** | Right-skewed; max exceeds HP highest landslide elevations |
| `slope_deg` | 0° | 26.79° | 27.77° | 76.82° | Near-symmetric; raw values useful for interpretation |

> [!WARNING]
> `aspect_deg` is a **circular variable** (0° ≡ 360°). The z-score transformation applied in `Aspect_z` treats it as a linear variable, which is **mathematically incorrect**. The angular distance between 1° and 359° is 2°, not 358°. For modelling, consider sine/cosine decomposition: `sin(aspect_deg × π/180)` and `cos(aspect_deg × π/180)`.

### 4.3 LULC One-Hot Columns (10 columns)

| LULC Class | Code | Total | Class 1 | Class 0 |
|-----------|------|-------|---------|---------|
| Tree cover | LULC_10 | 6,252 | 3,522 | 2,730 |
| Shrubland | LULC_20 | 196 | 150 | 46 |
| Grassland | LULC_30 | 2,495 | 1,386 | 1,109 |
| Cropland | LULC_40 | 699 | 30 | 669 |
| Built-up | LULC_50 | 580 | 496 | 84 |
| Bare/sparse vegetation | LULC_60 | 1,761 | 901 | 860 |
| Snow and ice | LULC_70 | 110 | 0 | 110 |
| Permanent water bodies | LULC_80 | 78 | 28 | 50 |
| Herbaceous wetland | LULC_90 | 1 | 0 | 1 |
| Moss and lichen | LULC_100 | 237 | 18 | 219 |

**Notable observations:**
- `LULC_90` (Herbaceous wetland): only **1 row** — this column is effectively constant in modelling and should be dropped or monitored for one-hot encoding collisions.
- `LULC_70` (Snow and ice): all 110 samples are class 0 — creates a **perfect separator** for this sub-class. Tree models may overfit on this column.
- **Cropland (LULC_40):** strongly associated with class 0 (95.7 % are non-landslide) — this may reflect the sampling strategy's geographic bias.
- Since each row sums to exactly 1 across LULC columns, **perfect multicollinearity** exists. For linear models (logistic regression), drop one LULC column to avoid the dummy variable trap.

### 4.4 Inter-Predictor Correlations

| | Aspect_z | Dist_River_z | Elevation_z | FlowAccum_z | NDVI_z | Rainfall_z | Slope_z |
|---|----------|-------------|-------------|-------------|--------|------------|---------|
| **Elevation_z** | 0.046 | 0.074 | 1.000 | −0.109 | **−0.494** | **−0.430** | 0.364 |
| **NDVI_z** | 0.003 | −0.022 | −0.494 | −0.130 | 1.000 | 0.279 | −0.101 |
| **Rainfall_z** | 0.006 | −0.078 | −0.430 | 0.005 | 0.279 | 1.000 | −0.061 |

The strongest inter-predictor correlations are **Elevation ↔ NDVI (r = −0.494)** and **Elevation ↔ Rainfall (r = −0.430)**, reflecting the well-known orographic relationship: higher elevations receive less rainfall and have lower vegetation density in this region. These correlations are moderate, not severe enough to cause multicollinearity problems for tree-based models, but should be monitored for logistic regression.

---

## 5. Missing Values

### Modelling Dataset (`landslide_modelling_dataset_12409.csv`)

| Column | Missing Count | Missing % | Explanation |
|--------|-------------|-----------|-------------|
| `nearest_inventory_sno` | **5,878** | 47.37 % | Expected — class 0 (pseudo-absence) rows have no inventory link |
| `match_dist_m` | **5,878** | 47.37 % | Expected — same 5,878 class-0 rows |
| All 7 z-score predictors | 0 | 0.00 % | Complete |
| All LULC columns | 0 | 0.00 % | Complete |
| `latitude`, `longitude`, `class` | 0 | 0.00 % | Complete |

**Conclusion:** The only missing values are structurally expected (class-conditional). **No predictor column has any missing values.** The modelling dataset is complete for training.

### Master Features Dataset (`landslide_features_master_6503.csv`)

| Property | Value |
|----------|-------|
| Missing cells | **0** (confirmed) |
| Rows | 6,503 |
| Columns | 46 |

The master features file has no missing values. All 40 environmental features are complete for all 6,503 landslide records.

### Inventory Dataset (`hp_landslide_inventory_clean.csv`)

| Column | Missing Count | Notes |
|--------|-------------|-------|
| `slide_name` | 6,309 / 6,539 | Expected — most sites unnamed |
| `year` | 6,056 / 6,539 | Only 483 records have a known year (1970–2023) |
| `event_note` | 6,496 / 6,539 | Free text; sparse |
| `flag_note` | 6,523 / 6,539 | Only 16 records were modified during cleaning |
| `subdivision` | 5 | Negligible |
| `material` | 1 | Negligible |
| `movement` | 2 | Negligible |

---

## 6. Duplicates

### 6.1 Exact Row Duplicates

**Modelling dataset:** `0` exact duplicate rows.

### 6.2 Coordinate Duplicates (Co-located Samples)

| Metric | Value |
|--------|-------|
| Rows sharing a lat/lon with ≥1 other row | **1,081** (8.71 % of dataset) |
| Distinct coordinate pairs duplicated | **409** |
| Maximum rows at one coordinate | **15** |

Distribution of `n_samples_at_coord`:

| Count at coord | Number of rows |
|---------------|--------------|
| 1 (unique) | 11,328 |
| 2 | 566 |
| 3 | 177 |
| 4 | 148 |
| 5 | 90 |
| 6 | 18 |
| 7 | 21 |
| 8 | 16 |
| 9 | 18 |
| 12 | 12 |
| 15 | 15 |

> [!WARNING]
> **1,081 rows share coordinates with other rows.** This includes cases where the same 30 m pixel was associated with multiple inventory records. If a random train/test split is used, the same pixel location may appear in both train and test, **inflating performance metrics**. Spatial deduplication or spatial blocking is required before splitting.

### 6.3 Feature Vector Duplicates (Master File)

In `landslide_features_master_6503.csv`:
- **1,024 rows** (15.7 %) have an identical 42-feature vector as another row (`duplicate_feature_vector = True`)
- **986 rows** share coordinates with at least one other row

---

## 7. Data Quality – Outliers & Distributions

### 7.1 Outliers in Z-Score Predictors (beyond ±3 SD)

| Predictor | Outliers (|z| > 3) | % | Min | Max | Notes |
|-----------|-------------------|---|-----|-----|-------|
| `Aspect_z` | 0 | 0.00 % | −1.804 | +1.831 | Well-bounded |
| `Dist_River_z` | **189** | 1.52 % | −0.131 | +8.041 | **Severely right-skewed** |
| `Elevation_z` | 77 | 0.62 % | −1.214 | +3.680 | Mildly skewed |
| `FlowAccumulation_z` | **355** | 2.86 % | −0.578 | +5.005 | **Severely right-skewed** |
| `NDVI_z` | 20 | 0.16 % | **−5.971** | +1.795 | Extreme low values |
| `Rainfall_z` | 77 | 0.62 % | −2.328 | +3.924 | Acceptable |
| `Slope_z` | 16 | 0.13 % | −1.992 | +3.733 | Acceptable |

### 7.2 Skewness Analysis

| Predictor | Skewness | Kurtosis | Interpretation |
|-----------|---------|---------|----------------|
| `Aspect_z` | −0.084 | −1.031 | **Bimodal/uniform** (expected for circular aspect) |
| `Dist_River_z` | **+7.917** | **60.69** | Extremely right-skewed — most points very close to rivers |
| `Elevation_z` | +1.156 | +0.691 | Moderately right-skewed |
| `FlowAccumulation_z` | **+2.557** | +6.758 | Strongly right-skewed |
| `NDVI_z` | −0.557 | −0.348 | Mildly left-skewed |
| `Rainfall_z` | +0.433 | +0.422 | Near-normal |
| `Slope_z` | −0.071 | −0.311 | Near-normal |

### 7.3 Dist_River_z Distribution Issue

The raw minimum of `Dist_River_z` is **−0.131** and maximum is **+8.041**. The IQR spans only −0.131 to −0.121 (a range of **0.010 z-score units**), meaning 75 % of values are compressed into an extremely narrow band near zero. This confirms that the underlying raw `dist_river_m` is **strongly right-skewed** — the vast majority of points are within a very short distance of a river, with rare extreme-distance outliers pulling the max to z = +8.04. The z-score transformation has not corrected the skewness; outliers will still be influential.

### 7.4 NDVI_z Extreme Low Values

- **20 rows** have NDVI_z < −5.97 (extreme outliers)
- All 20 belong to `lulc_name = Permanent water bodies` (LULC_80)
- Class distribution: 14 class-0, 6 class-1
- Water bodies naturally have very low NDVI, confirming these are genuine values, not data errors

### 7.5 FlowAccumulation_z Extreme High Values

- **355 rows** with FlowAccumulation_z > 3
- Class distribution: **301 landslides (class 1), 54 non-landslide (class 0)**
- High flow accumulation = major river channels and valleys; these are known high-susceptibility zones

### 7.6 SPI in Master Features (highly skewed)

| Statistic | Value |
|-----------|-------|
| Min | 0.0175 |
| Median | 1.247 |
| Mean | 3,040 |
| Max | **495,757** |
| Rows with SPI > 10,000 | 161 |
| Rows with SPI > 100,000 | 71 |

> [!CAUTION]
> SPI is **extremely right-skewed** (mean 3,040 vs median 1.25). Log-transformation is mandatory before using SPI as a predictor. The z-score standardisation applied in the modelling dataset does NOT use SPI directly (it uses FlowAccumulation which is already log-transformed).

### 7.7 TWI Negative Values

- **160 rows** in the master file have TWI < 0 (range: −1.066 to 15.961)
- Negative TWI values can arise from specific DEM-derived algorithms but are physically unusual. These should be investigated before using TWI in modelling.

---

## 8. Coordinate Bounds Check

| Property | Dataset Value | Expected (HP Bbox) | Status |
|----------|-------------|-------------------|--------|
| Latitude min | 29.9697°N | 30.00°N | ⚠️ 25 points below |
| Latitude max | 33.0123°N | 32.99°N | ⚠️ Points above |
| Longitude min | 75.7086°E | 75.74°E | ⚠️ 16 points left |
| Longitude max | 79.8292°E | 79.80°E | ⚠️ Points right |

**25 rows** have latitude below 30.00°N (outside the expected HP boundary described in the README).  
**16 rows** have longitude outside the expected [75.74°E, 79.80°E] range.

This is likely due to the **5 km buffer** applied around the convex hull of landslide points when generating pseudo-absence samples. Points at the edge of the buffer may fall just outside the administrative boundary. These are probably not data errors but should be mapped and reviewed.

**Maximum elevation in dataset:** 6,430 m — consistent with Himachal Pradesh's high peaks; 90 rows exceed 5,500 m.

---

## 9. Suspicious Records

| Issue | Count | Class Split | Assessment |
|-------|-------|-------------|------------|
| `slope_deg = 0` (flat terrain) | 76 | 72 non-landslide, 4 landslide | 4 landslide on flat terrain: **suspicious** |
| `aspect_deg = 0` (true north or flat) | 232 | 189 non-landslide, 43 landslide | Possible flat-terrain artefact |
| Landslides above 4,000 m elevation | 10 | class 1 only | Plausible but rare; verify locations |
| Landslides on permanent water bodies (LULC_80) | **28** | class 1 | Suspicious — may indicate GEE pixel misalignment |
| Snow/ice (LULC_70) with landslide label | 0 | — | No issue — all 110 snow rows are class 0 |
| Points outside expected HP bbox | 41 | mixed | Likely buffer artefact from pseudo-absence sampling |

> [!WARNING]
> **28 landslide rows are classified as "Permanent water bodies"** by ESA WorldCover. Land-cover is derived from 2020 Sentinel-2 imagery while landslides span 1970–2023. A landslide site may now be a water body (e.g., dam lake created by a landslide), or pixel misalignment occurred. These 28 rows should be manually reviewed before training.

---

## 10. Redundant & Non-Predictor Columns

| Column | Type | Recommendation |
|--------|------|---------------|
| `sample_id` | Sequential integer | **Drop** — no predictive value |
| `gee_system_index` | String identifier | **Drop** — no predictive value |
| `latitude`, `longitude` | Coordinates | **Keep for spatial splitting only; do not use as ML features** |
| `nearest_inventory_sno` | Class-1-only link | **Drop** — data linkage column; NaN for 47 % of rows |
| `match_dist_m` | Class-1-only | **Drop** — NaN for 47 % of rows; not a geographic predictor |
| `n_samples_at_coord` | Post-hoc count | **Drop** — encodes label clustering; potential leakage |
| `aspect_deg` | Duplicate of `Aspect_z` raw | Keep as reference; **do not double-use** |
| `elevation_m` | Duplicate of `Elevation_z` raw | Keep as reference; **do not double-use** |
| `slope_deg` | Duplicate of `Slope_z` raw | Keep as reference; **do not double-use** |
| `lulc_code` | Duplicate of one-hot | **Drop** — perfectly redundant with LULC_* columns |
| `lulc_name` | Duplicate of one-hot | **Drop** — perfectly redundant |
| `LULC_90` | Only 1 non-zero row | **Drop** — quasi-constant; no modelling signal |

---

## 11. Cross-File Linkage Verification

| Check | Result | Status |
|-------|--------|--------|
| All 6,503 `master` SNOs present in `inventory` | 6,503 / 6,503 | ✅ |
| Master SNOs NOT in inventory | 0 | ✅ |
| All modelling `nearest_inventory_sno` values in inventory | 5,856 / 5,856 unique | ✅ |
| Modelling SNOs NOT in inventory | 0 | ✅ |
| 36 missing-feature SNOs in inventory | 36 / 36 | ✅ |
| Modelling dataset landslide rows | 6,531 | ✅ |
| Inventory records | 6,539 | — |
| Master features records | 6,503 | — |
| Unique nearest_inventory_sno in modelling | 5,856 | — |

> [!NOTE]
> There are **6,531 landslide rows** in the modelling dataset but **6,539 inventory records** and **6,503 master feature rows**. The modelling dataset uses a nearest-30 m pixel snap for each inventory record, so the counts differ due to: (a) 36 records with no GEE features, and (b) multiple inventory records snapping to the same 30 m pixel (some pixels represent multiple co-located events). The 5,856 unique `nearest_inventory_sno` values vs 6,531 landslide rows confirms ~675 rows are from pixel-level merging of co-located inventory records.

---

## 12. Supporting Files Audit

### `hp_landslide_inventory_clean.csv`

| Property | Value |
|----------|-------|
| Rows | 6,539 |
| Columns | 14 |
| All labels | 1 (landslide only) |
| Co-located records (n_records_at_coord > 1) | 992 |
| Max co-located at one point | 15 |
| Known year (non-null) | 483 (7.4 %) |
| Year range | 1970–2023 |

**District distribution:**

| District | Records |
|----------|---------|
| Shimla | 1,065 |
| Mandi | 1,000 |
| Solan | 993 |
| Sirmaur | 941 |
| Chamba | 907 |
| Kangra | 524 |
| Kullu | 445 |
| Bilaspur | 264 |
| Kinnaur | 218 |
| Lahaul and Spiti | 86 |
| Hamirpur | 56 |
| Una | 40 |

**Material distribution:** Rock (3,514), Debris (2,998), Soil (25), Glaciofluvial (1)  
**Movement distribution:** Slide (6,124), Fall (246), Subsidence (77), Topple (25), Flow (21), Creep (16), other/composite (56)

**Limitations noted:**
- Only 7.4 % of records have known occurrence years.
- 6,309 / 6,539 records (96.5 %) have no slide name.
- 992 records share coordinates — likely GPS precision artefacts or multiple events at the same location.

### `inventory_records_without_features_36.csv`

36 records covering Sirmaur, Solan, Shimla, Mandi, Bilaspur, Kullu, Lahaul and Spiti, Chamba districts. The records cluster geographically (many in Chamba at ~32.75°–32.87°N, 75.95°–76.18°E), suggesting a GEE masked-out or no-data region. Reason for extraction failure is not documented.

---

## 13. Scientific Assessment

### 13.1 How Positive (Landslide) Samples Were Generated

Landslide samples in `landslide_modelling_dataset_12409.csv` were created by:

1. Taking each of the 6,503–6,539 GSI inventory point coordinates.
2. Snapping each to the nearest **30 m SRTM pixel centre** (introducing ≤20.9 m displacement, confirmed by `match_dist_m` column).
3. Extracting environmental features using GEE `sampleRegions` at 30 m scale.
4. Z-score standardising all 7 predictors over the **combined 12,409-row dataset**.

Result: **6,531 landslide rows** (some inventory points collapsed into the same pixel).

### 13.2 How Pseudo-Absence (Non-Landslide) Samples Were Generated

Pseudo-absence samples were generated by:

1. Computing the **convex hull** of all landslide point locations.
2. Buffering the convex hull by **5 km**.
3. Generating **5,878 random points** within the buffered hull using `ee.FeatureCollection.randomPoints(seed=42)`.
4. **No exclusion buffer** was applied around known landslide locations.
5. Extracting the same environmental features at each random point.

### 13.3 Can Pseudo-Absences Be Treated as Confirmed Non-Landslide Locations?

**No.** Pseudo-absence samples cannot be treated as confirmed non-landslide ground truth for the following reasons:

1. **No field verification** was performed. Points are purely random within a geographic boundary.
2. **No exclusion buffer** around known landslides — some pseudo-absence points may be within or immediately adjacent to actual landslide zones.
3. The GSI inventory is **incomplete by design** — it records reported/documented events, not all events. Random points may coincide with undocumented historical landslides.
4. The **temporal mismatch** between inventory records (1970–2023) and satellite-derived features (2020 land cover, multi-year climate averages) means features may not reflect conditions at the time of each event.

**Practical implication:** Class 0 represents "sampled random locations within the study area" not "verified stable slopes." Model outputs should be interpreted as **relative susceptibility** (probability of landslide-like conditions), not absolute presence/absence probability.

### 13.4 Limitations of the Sampling Methodology

| # | Limitation | Severity | Impact on Model |
|---|-----------|----------|----------------|
| 1 | **Pseudo-absences are not verified stable ground** | High | May introduce label noise in class 0; reduces precision |
| 2 | **No exclusion buffer** around known landslides in absence sampling | High | Some class-0 points may be actual landslide zones |
| 3 | **Inventory is biased toward accessible/reported events** | Medium | Model may underestimate susceptibility in remote areas |
| 4 | **Only 7 of 40 features** are included in modelling dataset | Medium | Information loss; master features file has richer data |
| 5 | **Aspect z-score is linear** but aspect is circular | Medium | Aspect predictor may be distorted near 0°/360° |
| 6 | **Z-scores computed over full 12,409-row dataset** | High | **Preprocessing leakage** (see Section 14) |
| 7 | **1,081 co-located rows** (same pixel, multiple records) | High | Random splits will cause train/test pixel overlap |
| 8 | **No lithology, fault distance, or seismic layers** | Medium | Missing key geological drivers |
| 9 | **No temporal dimension** in predictors | Medium | Cannot model time-dependent susceptibility |
| 10 | **Convex hull + 5 km buffer** for absence area may be too broad | Low | Absence samples from low-susceptibility zones (e.g., plains) are geographically correct but ecologically dissimilar |

---

## 14. Preprocessing Leakage Analysis

### 14.1 Standardisation Leakage (Critical)

The 7 z-score predictor columns in the modelling dataset were computed over the **entire 12,409-row dataset** (mean ≈ 0.000, std ≈ 1.00004 confirmed).

**This means:** if a train/test split is applied to this dataset after the fact, the test rows have already been used to compute the standardisation parameters. This constitutes **preprocessing leakage** — the test set has "seen" its own statistics via the shared mean and standard deviation.

**Magnitude:** With 12,409 total rows and typical 80/20 splits (9,927 train / 2,482 test), the influence of test rows on the global mean/std is small (~20 %) but non-zero and statistically meaningful.

**Required fix:** When splitting for ML, use the **raw columns** (`aspect_deg`, `elevation_m`, `slope_deg`) and re-apply z-score standardisation using only training-set statistics. For the 4 predictors where raw values are not provided (`Dist_River_z`, `FlowAccumulation_z`, `NDVI_z`, `Rainfall_z`), the master features file contains the raw values that can be joined by `nearest_inventory_sno` for class-1 rows.

### 14.2 Other Leakage Risks

| Column | Leakage Risk | Recommendation |
|--------|-------------|----------------|
| `n_samples_at_coord` | Medium — encodes spatial clustering of labels | Drop from features |
| `nearest_inventory_sno` | High — directly encodes the event identity | Drop from features |
| `match_dist_m` | High — NaN for all class-0 implies the label | Drop from features |
| `lulc_code` / `lulc_name` | Low — same info as one-hot, no additional leakage | Drop as redundant |

---

## 15. Spatial Split Strategy

### 15.1 Why Random Splitting Is Inadequate

Standard random splitting (e.g., `train_test_split`) is inappropriate for this dataset because:

1. **Spatial autocorrelation:** Nearby geographic points have similar feature values. Models trained on spatially adjacent training points will appear to generalise well on nearby test points even if they would fail in a new region.
2. **Coordinate duplicates:** 1,081 rows share pixel coordinates. Random splitting will frequently assign the same pixel to both train and test, directly leaking spatial signal.
3. **Scientific validity:** Landslide susceptibility mapping aims to predict at new locations. Generalisation to unseen spatial areas is the true challenge.

### 15.2 Recommended Spatial Split Strategy

**Option A — Spatial Blocking (Latitude Bands)**

Divide the study area into geographic blocks using latitude:

| Zone | Latitude Range | Total Rows | Class 1 | Class 0 |
|------|---------------|-----------|---------|---------|
| South | ≤ 31.011°N | 4,132 | 1,728 | 2,404 |
| Mid | 31.011°–31.687°N | 4,145 | 2,430 | 1,715 |
| North | > 31.687°N | 4,132 | 2,373 | 1,759 |

For a 3-fold spatial cross-validation: train on 2 blocks, test on the held-out block. This ensures geographic separation between train and test.

> [!TIP]
> **Option B — Block Cross-Validation:** Divide the study area into a regular grid (e.g., 0.5° × 0.5° cells), assign each cell to a fold, and perform k-fold cross-validation where folds are geographically non-adjacent. This is the preferred approach for spatial ML, implemented via `scikit-learn`'s `GroupKFold` or the `spatial-cv` pattern.

**Option C — District-based Split**

Use district membership as the grouping variable. Hold out 2–3 districts as the test set, train on the remaining 9–10. This leverages the administrative geography and ensures full geographic separation.

### 15.3 Deduplication Before Splitting

Before splitting:
1. Identify all duplicate coordinate groups.
2. Keep only one row per unique pixel (or aggregate by majority vote for class).
3. After deduplication, apply spatial blocking.

---

## 16. Limitations Summary

| Priority | Issue | Affects |
|----------|-------|---------|
| 🔴 Critical | Z-scores computed over full dataset — preprocessing leakage | All models |
| 🔴 Critical | 1,081 rows share coordinates — random split causes pixel overlap | All evaluations |
| 🔴 Critical | Pseudo-absences are not confirmed negatives — label noise | All models |
| 🟠 High | Aspect_z uses linear z-score on a circular variable | Aspect feature |
| 🟠 High | Dist_River_z extremely right-skewed (skew=7.9, kurtosis=60.7) | Dist_River feature |
| 🟠 High | 28 landslide rows on water body LULC — possible misalignment | LULC features |
| 🟡 Medium | No exclusion buffer in absence sampling — some class-0 may be actual landslides | Label quality |
| 🟡 Medium | LULC_90 (wetland) has only 1 non-zero row — quasi-constant feature | Feature selection |
| 🟡 Medium | LULC_70 (snow/ice) perfectly separates — may cause tree overfitting | LULC_70 |
| 🟡 Medium | 41 points outside expected HP bbox (buffer artefact) | Spatial extent |
| 🟡 Medium | FlowAccumulation_z: 355 outliers (2.86 %) | Flow feature |
| 🟢 Low | 36 inventory records with no features (excluded from modelling dataset) | Coverage |
| 🟢 Low | Only 7.4 % of events have known year — no temporal analysis possible | Temporal |

---

## 17. Recommendations Before Model Training

**Must-do (before any modelling):**

1. **Re-standardise from raw values using training-set-only statistics.** Drop or ignore the pre-computed `*_z` columns if train/test splitting is needed. Use raw columns from the modelling dataset (`aspect_deg`, `elevation_m`, `slope_deg`) and join raw values for the other 4 predictors from the master features file.

2. **Apply spatial blocking, not random splitting.** Use latitude bands, district grouping, or regular grid blocks to ensure geographic separation between train and test.

3. **Deduplicate by coordinate** before splitting. For rows sharing the same pixel, keep one representative row (e.g., the first occurrence or one with the most complete data).

4. **Repair Aspect representation.** Replace `Aspect_z` with `sin(aspect_deg × π/180)` and `cos(aspect_deg × π/180)` before standardising.

**Recommended (improves model quality):**

5. **Drop truly redundant columns** from the feature matrix: `sample_id`, `gee_system_index`, `lulc_code`, `lulc_name`, `nearest_inventory_sno`, `match_dist_m`, `n_samples_at_coord`, `LULC_90`.

6. **For linear models only:** Drop one LULC column to avoid the dummy variable trap (perfect multicollinearity). `LULC_90` (quasi-constant) is the natural drop candidate.

7. **Apply log-transformation to Dist_River raw values** before standardisation (skew = 7.9 in z-scored form suggests the raw variable is extremely skewed). Similarly, `FlowAccumulation` raw data is already log-transformed per the data dictionary — verify this before applying additional log.

8. **Review and decide on the 28 landslide rows** assigned to water body LULC. Options: flag and exclude, or retain with LULC replaced by the most likely adjacent class.

9. **Monitor class imbalance** during evaluation. Use AUC-ROC, precision-recall AUC, and F1-score in addition to accuracy. With a 1.11:1 ratio, standard thresholds (0.5) should be acceptable but verify with class-weighted metrics.

10. **Consider using the full master features file** (40 features) instead of the 7-predictor modelling dataset for richer modelling. Join by `sno` / `nearest_inventory_sno`.

---

*Audit completed: 2026-11-16 | No model training performed | No dataset modification made*

