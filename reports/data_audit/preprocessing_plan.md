# RockWatch – Preprocessing Plan (Revision 3 — Final)

**Project:** RockWatch – Intelligent Landslide Susceptibility Prediction System  
**Document:** Phase 1 Preprocessing Plan — Revision 3  
**Date:** 2026-11-16  
**Status:** AWAITING APPROVAL — no implementation performed  
**Supersedes:** preprocessing_plan.md (Revision 2)

> [!IMPORTANT]
> This document revises Revision 2 in response to four specific scientific corrections. All statistics are sourced directly from the original dataset files using the investigation scripts. No data has been modified, no models trained, and no fabricated values appear anywhere in this report.

---

## Table of Contents

1. [Duplicate Coordinate Analysis — Fully Verified](#1-duplicate-coordinate-analysis--fully-verified)
2. [Data Standardisation — Mathematical Proof and Anomalies](#2-data-standardisation--mathematical-proof-and-anomalies)
3. [Raw Feature Availability](#3-raw-feature-availability)
4. [Feature Engineering Strategy](#4-feature-engineering-strategy)
5. [Spatial Splitting — Exact Counts from Deduplicated Data](#5-spatial-splitting--exact-counts-from-deduplicated-data)
6. [LULC Feature Strategy](#6-lulc-feature-strategy)
7. [Missing Value Strategy](#7-missing-value-strategy)
8. [Feature Selection and scaler_params.json](#8-feature-selection-and-scaler_paramsjson)
9. [Primary Preprocessing Strategy — Final Recommendation](#9-primary-preprocessing-strategy--final-recommendation)
10. [Optional Future Experiments](#10-optional-future-experiments)
11. [Limitations That Must Be Disclosed](#11-limitations-that-must-be-disclosed)
12. [Expected Processed Files — Exact Counts](#12-expected-processed-files--exact-counts)

---

## 1. Duplicate Coordinate Analysis — Fully Verified

### 1.1 Field-by-Field Conflict Audit (All 409 Groups)

A complete audit was performed across every field for all 409 duplicate-coordinate groups (1,081 total rows). Every group was checked for conflicts in every relevant field.

| Field category | Fields checked | Groups with conflicts |
|---------------|---------------|----------------------|
| Class label | `class` | **0 / 409** |
| Z-score predictors | All 7 `*_z` columns | **0 / 409** |
| LULC columns | All 10 `LULC_*` columns | **0 / 409** |
| Inventory SNO | `nearest_inventory_sno` | **0 / 409** |
| Match distance | `match_dist_m` | **0 / 409** |
| Co-location count | `n_samples_at_coord` | **0 / 409** |

**Result: No conflicts of any kind exist across all 409 duplicate-coordinate groups.** Every row sharing a coordinate with another row has identical values in every field that matters for modelling: class label, all seven z-score predictors, all LULC indicators, inventory link, and spatial metadata.

### 1.2 Nature of the Duplication — Confirmed

- All 409 groups have **identical `nearest_inventory_sno`** within the group
- All 409 groups have **different `gee_system_index`** values per row
- **Conclusion:** These are GEE pipeline artefacts — multiple export entries for the same pixel, generated when multiple inventory records (with the same or nearby coordinates) triggered separate `sampleRegions` calls on the same 30 m SRTM pixel centre. The pixel returns identical features each time because it represents a fixed raster cell.

### 1.3 What Deduplication Loses

| Category | Rows removed | Information lost |
|----------|------------|-----------------|
| Redundant pixel exports | **672** | None — all 672 removed rows are field-identical to their retained counterpart in every modelling-relevant column |
| Distinct multi-event signal | 0 | Nothing — no group has rows with different `nearest_inventory_sno` |

**Deduplication is scientifically justified.** Decision: retain the row with the minimum `sample_id` per unique (latitude, longitude). This is deterministic and reproducible. The number of inventory events associated with each retained pixel is documented in the original `n_samples_at_coord` column, which is preserved in the deduplicated file for reference.

### 1.4 Post-Deduplication Counts (Exact)

| Metric | Count |
|--------|-------|
| Rows before deduplication | **12,409** |
| Rows removed | **672** |
| Rows after deduplication | **11,737** |
| Class = 1 after deduplication | **5,859** |
| Class = 0 after deduplication | **5,878** |
| Class balance ratio (1 : 0) | **0.9968 ≈ 1 : 1** |
| Duplicate coordinates remaining | **0** |

---

## 2. Data Standardisation — Mathematical Proof and Anomalies

### 2.1 Global Z-Score Parameters (Confirmed from Data)

```
All 7 *_z columns:
  global_mean ≈ 0  (range: −7.3e-16 to +1.7e-16, i.e., floating-point zero)
  global_std  = 1.0000402958  (identical for all 7 columns)
```

The near-identical standard deviation (1.0000402958) across all 7 columns confirms they were standardised in one operation using the same denominator — consistent with `sklearn.preprocessing.StandardScaler(copy=True)` or equivalent, using `ddof=0` (population std). The slight deviation from 1.0 arises because `ddof=0` was used in computation but `ddof=1` (sample std) is what Python/pandas `std()` returns by default.

### 2.2 Verification: Was It Strictly Linear z-Score Standardisation?

Three predictors can be verified against raw counterparts available in the same modelling dataset file:

| Predictor | Test | Residual (max_abs) | Interpretation |
|-----------|------|--------------------|----------------|
| `Elevation_z` from `elevation_m` | `(elevation_m − mean) / std` | **1.48 × 10⁻⁴** | Near-zero; consistent with float32→float64 precision loss |
| `Slope_z` from `round(slope_deg, 0)` | `(int_slope − mean) / std` | **1.50 × 10⁻⁴** | Near-zero; Slope_z was computed from integer slope (see §2.4) |
| `Aspect_z` from `aspect_deg` | `(aspect_deg − mean) / std` | **5.19 × 10⁻³** | Near-zero; consistent with float32→float64 conversion |

The residuals are consistent with **floating-point precision artefacts only** (float32 GEE export → float64 CSV storage introduces rounding at the 5th–6th significant digit). There is no evidence of clipping, nonlinear transformation, or intentional rounding.

**Conclusion: The 7 `*_z` columns are confirmed to be ordinary linear z-score (mean subtraction, standard-deviation division) transformations of their source variables. No clipping or nonlinear steps are present.**

### 2.3 Mathematical Proof: Re-Standardisation Eliminates Global Constants

**Claim:** Re-standardising `z_global` using training-set statistics is mathematically identical to standardising the raw variable using training-set statistics. The original global constants `mu_global` and `sd_global` do not appear in the result.

**Proof:**

Let:
- $x$ = raw feature value  
- $\mu_G$, $\sigma_G$ = global mean and std (computed over all 12,409 rows)  
- $\mu_T$, $\sigma_T$ = training-set mean and std of $x$ (computed over training rows only)

The original z-score is:
$$z_G = \frac{x - \mu_G}{\sigma_G} \tag{1}$$

The target corrected z-score (using training statistics) is:
$$z_T = \frac{x - \mu_T}{\sigma_T} \tag{2}$$

From (1): $x = z_G \cdot \sigma_G + \mu_G \tag{3}$

Now compute the training-set statistics of $z_G$:
$$\mu_{T,z} = \text{mean}(z_G \text{ over training rows}) = \frac{\mu_T - \mu_G}{\sigma_G} \tag{4}$$

$$\sigma_{T,z} = \text{std}(z_G \text{ over training rows}) = \frac{\sigma_T}{\sigma_G} \tag{5}$$

Then:
$$\frac{z_G - \mu_{T,z}}{\sigma_{T,z}} = \frac{z_G - \frac{\mu_T - \mu_G}{\sigma_G}}{\frac{\sigma_T}{\sigma_G}} = \frac{z_G \cdot \sigma_G + \mu_G - \mu_T}{\sigma_T}$$

Substituting (3):
$$= \frac{x - \mu_T}{\sigma_T} = z_T \tag{6}$$

**QED. The result $z_T$ contains no reference to $\mu_G$ or $\sigma_G$. The original global constants are algebraically cancelled.**

**Practical implication:** Fitting a `StandardScaler` to the training rows of `z_global` and transforming all sets with it is **exactly equivalent** to computing the standardisation from scratch using raw values and training-set statistics — provided the z-score transformation was strictly linear (confirmed in §2.2). **No residual leakage from the original global standardisation remains after this correction.** The term "residual leakage" used in Revision 2 was therefore incorrect and is retracted.

### 2.4 Slope_z Quantisation Anomaly — Important Data Quality Finding

**Finding:** `Slope_z` has only **76 unique values** despite `slope_deg` having 11,370 unique values.

**Investigation confirms:** `Slope_z` was computed from integer-degree slope (i.e., `round(slope_deg, 0)`), not from the high-precision `slope_deg` stored in the file. When slope_deg is rounded to the nearest integer, it yields exactly 76 unique values (0°–75°, with one value at 76°), matching Slope_z exactly.

| Slope_z value | Rows | slope_deg range mapped to it |
|--------------|------|------------------------------|
| −1.991977 (z for 0°) | 76 | 0.0000° only |
| −1.917626 (z for 1°) | 323 | 0.9274°–1.4372° |
| −1.843274 (z for 2°) | 268 | 1.8543°–2.3870° |
| … (74 more integer-degree bins) | | |
| +3.733086 (z for 76°) | 1 | 76.8216° |

**Verification:** Recomputing $z$ from `round(slope_deg, 0)` gives residuals with `max_abs = 1.50 × 10⁻⁴` — identical to float32 precision noise. The match is confirmed.

**Consequence for modelling:**
- `Slope_z` is quantised to 76 bins, reducing effective precision to ±0.5° of slope
- For tree-based models, this is negligible — decision boundaries are insensitive to small within-bin variation
- For logistic regression, the quantised version is equivalent to treating slope as an ordered categorical variable — acceptable, with a slight loss of continuous-variable information
- The raw `slope_deg` column in the modelling file stores higher-precision values but is NOT what was used in Slope_z. The raw `slope_deg` from the **master features file** (`dtype: int64`, 74 unique values) confirms this: Slope_z was standardised from master-file integer slope, not modelling-dataset float slope.

**Decision:** Use `Slope_z` as-is in the primary pipeline (corrected to training-set reference via the mathematical equivalence proved in §2.3). The quantisation is a known property of the source data, not an error.

### 2.5 Clipping and Rounding — Confirmed Absent

| Check | Result |
|-------|--------|
| Any z-score column clipped at boundary | None — no pile-up at min/max values (beyond natural data floors) |
| `FlowAccumulation_z` min = −0.578 (5,137 rows) | Physical floor: log(flow accumulation) has a natural minimum at log(2) = 0.693 for single-cell basins. Not clipping. |
| `Aspect_z` min = −1.804 (232 rows) | Aspect = 0° for all 232 rows. Not clipping — aspect=0° is true north. |
| `Dist_River_z` min = −0.131 (179 rows) | Physical floor: dist_river = 0 m (points on the river). Not clipping. |
| z-score values match 6 decimal-place rounding | 0 / 12,409 (0.00 %) — no lossy rounding. Values stored at full float64 precision. |

### 2.6 Assumptions for the Mathematical Equivalence to Hold

The proof in §2.3 holds under the following assumptions, all of which are confirmed:

1. **Linear transformation only** ✅ — Verified: z-scores reconstruct from raw with residuals ≤ 5.2 × 10⁻³ (float32 noise)
2. **No clipping** ✅ — Confirmed: extreme values are physical data floors, not imposed boundaries
3. **No nonlinear transformation** ✅ — Confirmed: transformation is strictly `(x − μ) / σ`
4. **No lossy rounding of stored values** ✅ — Confirmed: z-score columns store full float64 precision
5. **Training set is a strict subset of the 12,409 rows** ✅ — The spatial split draws from the same dataset

---

## 3. Raw Feature Availability

### 3.1 Summary (Unchanged from Revision 2)

For 4 of the 7 predictors, raw values exist in the master features file for class=1 rows only (joined via `nearest_inventory_sno`). Class=0 rows have no raw counterpart available without a new GEE extraction.

**Primary pipeline decision (unchanged):** Use the corrected `*_z` columns for all 7 predictors for both classes. This is the only fully consistent representation available within this dataset.

### 3.2 Consequence Acknowledged

Re-standardising `*_z` using training-set statistics, as proved in §2.3, completely removes the influence of the original global constants. The fact that raw values are unavailable for class=0 is therefore **not a leakage concern for the primary pipeline** — it is a constraint on extended experiments only.

A fully leakage-free pipeline that also avoids the z-score quantisation artefact in Slope_z would require:
- Raw slope (integer or float) for all 12,409 rows — available from master for class=1
- GEE extraction of all 7 raw features for the 5,878 class=0 coordinates

This is documented as a future experiment (Section 10).

---

## 4. Feature Engineering Strategy

### 4.1 Aspect — Circular Transformation (Unchanged)

**Decision confirmed:**
- `sin_aspect = sin(aspect_deg × π / 180)` for all rows
- `cos_aspect = cos(aspect_deg × π / 180)` for all rows
- Rows where `slope_deg = 0` (76 rows): set `sin_aspect = 0`, `cos_aspect = 0` (flat terrain; aspect undefined)
- These two columns replace `Aspect_z` entirely; `Aspect_z` is not scaled and not passed to `scaler_params.json`

**Availability:** `aspect_deg` is present in the modelling dataset for all 12,409 rows — no join required.

### 4.2 Skew Transformations — Optional Extended Modelling Only

No skew transforms are applied to the primary pipeline's `*_z` columns. Skew transforms on master-file raw features (sqrt for dist_river, log1p for spi/pop_density/dist_agri/dist_urban) are reserved for Section 10 extended experiments.

---

## 5. Spatial Splitting — Exact Counts from Deduplicated Data

### 5.1 Split Methodology

Cuts are computed from the **deduplicated** (11,737-row) dataset, not the original 12,409 rows.

| Parameter | Value |
|-----------|-------|
| Latitude cut 1 (CUT1) — 70th pct of deduplicated data | **31.713044°N** |
| Latitude cut 2 (CUT2) — 85th pct of deduplicated data | **32.144235°N** |
| Boundary exclusion buffer | **500 m ≈ 0.004505°** applied symmetrically around each cut |
| CUT1 exclusion zone | [31.708539°N, 31.717548°N] |
| CUT2 exclusion zone | [32.139731°N, 32.148740°N] |

### 5.2 Exact Row Counts — Fully Reconciled

| Stage | Rows | Class = 1 | Class = 0 | Notes |
|-------|------|----------|----------|-------|
| Original dataset | **12,409** | 6,531 | 5,878 | Before any processing |
| Removed by deduplication | **−672** | −672 | 0 | GEE pixel artefacts |
| After deduplication | **11,737** | 5,859 | 5,878 | No duplicate coordinates |
| Removed by CUT1 exclusion zone | **−46** | −20 | −26 | lat ∈ [31.7085°, 31.7175°N] |
| Removed by CUT2 exclusion zone | **−22** | −5 | −17 | lat ∈ [32.1397°, 32.1487°N] |
| Total removed by exclusion zones | **−68** | −25 | −43 | |
| **TRAIN** (lat < 31.7085°N) | **8,188** | **4,024** (49.1 %) | **4,164** (50.9 %) | |
| **VAL** (31.7175°N – 32.1397°N) | **1,731** | **896** (51.8 %) | **835** (48.2 %) | |
| **TEST** (lat > 32.1487°N) | **1,750** | **914** (52.2 %) | **836** (47.8 %) | |
| **Reconciliation check** | | | | |
| TRAIN + VAL + TEST | 11,669 | | | |
| Deduped − excluded | 11,737 − 68 = 11,669 | | | ✅ Exact match |

**All counts reconcile exactly to zero residual.**

### 5.3 Latitude Ranges After Exclusion Zone

| Split | Min latitude | Max latitude | Gap to boundary |
|-------|-------------|-------------|----------------|
| TRAIN | 29.9697°N | 31.7082°N | ≥ 538 m from CUT1 |
| (Exclusion) | 31.7085°N | 31.7175°N | — |
| VAL | 31.7176°N | 32.1397°N | ≥ 509 m from both cuts |
| (Exclusion) | 32.1397°N | 32.1487°N | — |
| TEST | 32.1488°N | 33.0123°N | ≥ 509 m from CUT2 |

### 5.4 Longitude Distribution Across Splits

The latitude split also creates an **east–west gradient** that is a property of the data, not an artefact of the splitting method:

| Split | Lon median | Lon std | % rows west of 76.5°E | % rows east of 78°E |
|-------|-----------|--------|----------------------|---------------------|
| TRAIN | 77.40°E | 0.82° | 7.0 % | 23.9 % |
| VAL | 76.77°E | 0.72° | 33.1 % | 12.1 % |
| TEST | 76.26°E | 0.50° | 67.7 % | 1.1 % |

This occurs because landslide density in northern HP (Chamba district) is concentrated along the western corridor (~76°E). The model is therefore evaluated on a region that differs from training in both latitudinal (climatic/elevation) and longitudinal (geological) character.

### 5.5 Honest Spatial Independence Assessment

> [!IMPORTANT]
> The 500 m exclusion zone is a **precaution that removes the most geographically proximate cross-boundary pairs**, not evidence of complete spatial independence. It guarantees a minimum 509 m gap between any training and validation row, and between any validation and test row.

However:
- Spatial autocorrelation in regional environmental layers (rainfall, NDVI, temperature) extends to **5–50 km**
- The minimum cross-boundary distance before exclusion was **0.10 km**; after exclusion it is **0.51 km**
- **1,616 train-val pairs fall within 5 km of the boundary** (from sample-based estimate)
- Points within 5 km still share similar climate and terrain conditions

**What the split does achieve:**
- District-level separation: TEST is dominated by Chamba (903 of 914 class=1 test rows), a district with 0 training rows
- Elevation regime difference: northern HP is higher altitude with different vegetation
- Longitude shift: TEST is predominantly at 76°E; TRAIN is centred at 77.4°E

**What the split does not achieve:**
- Pixel-level independence at the boundary
- Full longitude coverage in TEST (virtually no eastern HP beyond 78°E in TEST zone)

All metrics should be reported with the disclosure that evaluation is on a geographically distinct but not fully spatially independent held-out region.

---

## 6. LULC Feature Strategy

### 6.1 LULC_90 Clarification

LULC_90 (Herbaceous wetland) has exactly **1 row** with value 1 (class=0, `sample_id=677`).

The term "zero variance" in Revision 1 was imprecise. The correct characterisation:

- Binary variance = $p \times (1-p) = \frac{1}{11737} \times \frac{11736}{11737} \approx 0.0000852$ (post-deduplication)
- This is non-zero but effectively zero for modelling purposes
- One positive example cannot produce a replicable or generalisable coefficient
- The column is also a perfect class-0 separator within the 1-row group

**Decision: DROP `LULC_90`.** Not because variance is literally zero, but because a single positive example yields no generalisable model signal and creates a degenerate split condition.

### 6.2 LULC_70 (Snow and Ice) — Retained with Caution

All 110 snow/ice rows in the original dataset → 110 rows in the deduplicated dataset (confirmed: no snow/ice rows were among the 672 removed duplicates since all duplicates were class=1 and all LULC_70=1 rows are class=0). Retain; monitor feature importance; apply regularisation for logistic regression.

### 6.3 LULC_80 Water Body Flag — Retained

The `flag_water_lulc` column (value = 1 for the 28 class=1 rows with LULC_80 = 1) is retained in the processed dataset. Its purpose:

1. **Sensitivity analysis:** Train with and without these rows to quantify their influence on AUC-ROC (expected to be < 0.005)
2. **Transparency for downstream users** of the processed files who may wish to investigate or exclude potential LULC misalignment cases
3. **Post-deduplication note:** Several of the 28 water-body/landslide rows share coordinates and will be reduced by deduplication. The flag column is computed after deduplication on the retained rows only.

---

## 7. Missing Value Strategy

### 7.1 Primary Pipeline — No Imputation Required

All 7 `*_z` predictor columns have **zero missing values** across all 12,409 original rows, and therefore also zero missing values in the 11,737-row deduplicated dataset, and in all three splits. No imputation is required.

### 7.2 The 37 Unmatched Class=1 Records

These 37 rows have `nearest_inventory_sno` values not present in the master features file. They are fully populated in the `*_z` columns. **In the primary pipeline, these rows require no special treatment.**

For optional extended modelling using raw master features, these 37 rows are **excluded** (not imputed). Training-set median imputation is explicitly not used to avoid applying training statistics to unjoined rows that could appear in validation or test.

### 7.3 Structural NaN Columns (Excluded from Feature Matrix)

| Column | NaN count | Action |
|--------|----------|--------|
| `nearest_inventory_sno` | 5,878 (all class=0) | Drop from feature matrix |
| `match_dist_m` | 5,878 (all class=0) | Drop from feature matrix |

---

## 8. Feature Selection and scaler_params.json

### 8.1 Final Feature Matrix

**For tree-based models (Random Forest, Gradient Boosting): 17 features**  
**For logistic regression: 16 features** (drop `LULC_100` as reference category)

| # | Feature | Source | Transform | Scaled? |
|---|---------|--------|-----------|---------|
| 1 | `sin_aspect` | `aspect_deg` | sin(deg × π/180); 0 if slope=0 | No — bounded [−1, +1] |
| 2 | `cos_aspect` | `aspect_deg` | cos(deg × π/180); 0 if slope=0 | No — bounded [−1, +1] |
| 3 | `Dist_River_z_corr` | `Dist_River_z` | Train-set re-standardisation | Yes |
| 4 | `Elevation_z_corr` | `Elevation_z` | Train-set re-standardisation | Yes |
| 5 | `FlowAccumulation_z_corr` | `FlowAccumulation_z` | Train-set re-standardisation | Yes |
| 6 | `NDVI_z_corr` | `NDVI_z` | Train-set re-standardisation | Yes |
| 7 | `Rainfall_z_corr` | `Rainfall_z` | Train-set re-standardisation | Yes |
| 8 | `Slope_z_corr` | `Slope_z` | Train-set re-standardisation | Yes |
| 9 | `LULC_10` | modelling dataset | None | No — binary |
| 10 | `LULC_20` | modelling dataset | None | No — binary |
| 11 | `LULC_30` | modelling dataset | None | No — binary |
| 12 | `LULC_40` | modelling dataset | None | No — binary |
| 13 | `LULC_50` | modelling dataset | None | No — binary |
| 14 | `LULC_60` | modelling dataset | None | No — binary |
| 15 | `LULC_70` | modelling dataset | None | No — binary |
| 16 | `LULC_80` | modelling dataset | None | No — binary |
| 17 | `LULC_100` | modelling dataset | None | No — binary; drop for LR |

### 8.2 scaler_params.json — Correct Contents

`Aspect_z` is **replaced** by `sin_aspect` and `cos_aspect`. Because sin/cos values are bounded to [−1, +1], they do not require standardisation.

`scaler_params.json` therefore contains training-set statistics for exactly **6 predictors** (not 7):

```json
{
  "fitted_on": "TRAIN split (after deduplication and exclusion zone)",
  "n_train_rows": 8188,
  "features": {
    "Dist_River_z": {"train_mean": <computed at implementation>, "train_std": <computed at implementation>},
    "Elevation_z":  {"train_mean": <computed at implementation>, "train_std": <computed at implementation>},
    "FlowAccumulation_z": {"train_mean": <computed at implementation>, "train_std": <computed at implementation>},
    "NDVI_z":       {"train_mean": <computed at implementation>, "train_std": <computed at implementation>},
    "Rainfall_z":   {"train_mean": <computed at implementation>, "train_std": <computed at implementation>},
    "Slope_z":      {"train_mean": <computed at implementation>, "train_std": <computed at implementation>}
  },
  "excluded": {
    "Aspect_z": "Replaced by sin_aspect and cos_aspect; not scaled"
  }
}
```

Values are populated during implementation, not before. Placeholder `<computed at implementation>` is explicit and intentional — these must not be pre-filled with global statistics.

### 8.3 Columns Excluded from Feature Matrix

```
EXCLUDED (non-predictor / redundant / diagnostic):
  sample_id, gee_system_index
  latitude, longitude         (spatial index only; not model inputs)
  Aspect_z                    (replaced by sin_aspect, cos_aspect)
  aspect_deg                  (raw source; replaced)
  elevation_m                 (raw; Elevation_z_corr is used instead)
  slope_deg                   (raw float; Slope_z_corr is used instead — see §2.4)
  lulc_code, lulc_name        (redundant with LULC_* one-hot columns)
  nearest_inventory_sno       (class-conditional linkage; NaN for 47 % of rows)
  match_dist_m                (class-conditional; NaN for 47 % of rows)
  n_samples_at_coord          (post-hoc spatial clustering count)
  LULC_90                     (quasi-constant: 1 / 11,737 rows positive)
  flag_water_lulc             (diagnostic flag; not a predictor)
  LULC_100                    (reference category; drop for logistic regression only)
```

---

## 9. Primary Preprocessing Strategy — Final Recommendation

A single, justified pipeline. Steps are ordered for reproducibility; no step uses information from downstream steps.

```
Step 1 — Load raw files (read-only, no modification)
   Input:  landslide_modelling_dataset_12409.csv
   Output: df (12,409 rows × 30 columns)

Step 2 — Add diagnostic flags
   • flag_water_lulc = 1 for rows where (LULC_80 == 1) AND (class == 1)
   • All other rows: flag_water_lulc = 0

Step 3 — Coordinate deduplication
   • Group by (latitude, longitude); retain row with min(sample_id)
   • Output: df_dedup (11,737 rows)
   • Verify: (df_dedup.duplicated(subset=['latitude','longitude']).sum() == 0)

Step 4 — Aspect engineering
   • sin_aspect = sin(aspect_deg × π / 180)
   • cos_aspect = cos(aspect_deg × π / 180)
   • Where slope_deg == 0: sin_aspect = 0, cos_aspect = 0

Step 5 — Drop LULC_90
   • df_dedup.drop(columns=['LULC_90'], inplace=True)

Step 6 — Compute latitude cut points from df_dedup
   • lat_sorted = np.sort(df_dedup['latitude'].values)
   • CUT1 = lat_sorted[int(len(lat_sorted) * 0.70) - 1]  → 31.713044°N
   • CUT2 = lat_sorted[int(len(lat_sorted) * 0.85) - 1]  → 32.144235°N
   • BUFFER = 500 / 111_000  → 0.004505°

Step 7 — Apply spatial split with boundary exclusion
   • excl_mask = rows within BUFFER of CUT1 or CUT2
   • df_work = df_dedup[~excl_mask]  (11,669 rows)
   • train = df_work[df_work['latitude'] < CUT1]              → 8,188 rows
   • val   = df_work[(df_work['latitude'] >= CUT1) &
                     (df_work['latitude'] < CUT2)]            → 1,731 rows
   • test  = df_work[df_work['latitude'] >= CUT2]             → 1,750 rows
   • Save: train_spatial.csv, val_spatial.csv, test_spatial.csv (all columns retained)

Step 8 — Fit StandardScaler on TRAIN rows only (6 predictors)
   • Columns: Dist_River_z, Elevation_z, FlowAccumulation_z, NDVI_z, Rainfall_z, Slope_z
   • Fit on training rows only; save to scaler_params.json
   • Apply transform to train, val, test → creates *_z_corr columns

Step 9 — Construct and save feature matrices
   • Select columns per §8.1
   • Save: features_train.csv, features_val.csv, features_test.csv
   • Save: scaler_params.json
   • Write: preprocessing_log.txt

Step 10 — Verification assertions
   • Assert: len(train) + len(val) + len(test) == 11_669
   • Assert: len(train) + len(val) + len(test) + 68 + 672 == 12_409
   • Assert: no duplicate (lat, lon) in any split
   • Assert: scaler was fitted on train rows only (check log)
   • Assert: all *_z_corr columns in train have mean ≈ 0 and std ≈ 1
```

### What This Pipeline Achieves

| Property | Status |
|----------|--------|
| Original files unmodified | ✅ |
| Duplicate pixel-exports removed (confirmed no information lost) | ✅ |
| Aspect correctly encoded as circular variable | ✅ |
| Standardisation constants from training rows only | ✅ |
| Original global standardisation constants algebraically eliminated | ✅ (proved in §2.3) |
| Consistent feature representation for class=0 and class=1 | ✅ Both use the same `*_z` columns |
| Exact row counts verified and reconciled | ✅ |
| Geographic separation with minimum 509 m boundary gap | ✅ |
| Complete spatial independence | ⚠️ Partial — 500 m is a precaution, not proof of independence |

---

## 10. Optional Future Experiments

**Not part of Phase 1 implementation.**

| Experiment | Description | Prerequisite |
|-----------|-------------|-------------|
| A — Extended 40-feature model | Join master features (class=1 only via SNO); exclude 37 unjoined rows; re-standardise from raw values | None beyond current data |
| B — Fully raw pipeline | Obtain GEE raw features for 5,878 class=0 coordinates; re-standardise everything from scratch | New GEE extraction |
| C — 1°×1° grid-block CV | GroupKFold with 1°×1° block IDs; compare to latitude-band split | None |
| D — Sensitivity analysis | Train/evaluate with and without flag_water_lulc=1 rows; with/without LULC_70 | None |
| E — Higher-resolution aspect | Test sin/cos aspect vs linear Aspect_z_corr; compare feature importance | None |

---

## 11. Limitations That Must Be Disclosed

| # | Limitation | Required disclosure |
|---|-----------|---------------------|
| 1 | **Slope_z quantisation** | "`Slope_z` was computed from integer-degree slope (0°–76°), not from the higher-precision float `slope_deg` stored in the same file. This introduces 76-bin quantisation of slope, equivalent to ±0.5° resolution." |
| 2 | **Pseudo-absence label uncertainty** | "Class=0 samples are randomly sampled pseudo-absences, not verified stable-ground locations. Some may coincide with unrecorded landslides." |
| 3 | **Spatial split partial independence** | "Train, validation, and test sets are separated by latitude bands with a 500 m exclusion zone. This reduces but does not eliminate spatial autocorrelation across boundaries. Spatial autocorrelation in regional climate and terrain layers extends 5–50 km." |
| 4 | **Longitude coverage imbalance** | "The test zone (Chamba/northern HP) is predominantly western (~76°E). Eastern Kinnaur (77–80°E at high latitude) is not represented in the test set." |
| 5 | **GEE pixel snapping** | "Landslide coordinates were snapped to the nearest 30 m SRTM pixel, introducing up to 20.9 m displacement." |
| 6 | **Limited predictor set** | "7 of 40 available environmental features are used. Lithology, fault distance, and seismic intensity are absent." |
| 7 | **Temporal mismatch** | "Environmental features are derived from 2020 land cover and multi-year climate averages. Inventory events span 1970–2023." |

---

## 12. Expected Processed Files — Exact Counts

| File | Location | Exact rows | Columns | Notes |
|------|----------|-----------|---------|-------|
| `modelling_deduped.csv` | `data/processed/` | **11,737** | 32 (original 30 + sin_aspect + cos_aspect + flag_water_lulc − LULC_90) | All original cols retained; diagnostic cols added |
| `train_spatial.csv` | `data/processed/` | **8,188** | 32 | TRAIN split; all columns |
| `val_spatial.csv` | `data/processed/` | **1,731** | 32 | VAL split; all columns |
| `test_spatial.csv` | `data/processed/` | **1,750** | 32 | TEST split; all columns |
| `features_train.csv` | `data/processed/` | **8,188** | 18 or 17 | 17 features + `class` (tree) or 16 features + `class` (LR) |
| `features_val.csv` | `data/processed/` | **1,731** | 18 or 17 | Same feature schema as train |
| `features_test.csv` | `data/processed/` | **1,750** | 18 or 17 | Same feature schema as train |
| `scaler_params.json` | `data/processed/` | — | — | 6 `*_z` column stats from TRAIN only |
| `preprocessing_log.txt` | `data/processed/` | — | — | Row counts, decisions, verification assertions |

**Total rows in train + val + test: 8,188 + 1,731 + 1,750 = 11,669**  
**Reconciliation: 12,409 (original) − 672 (dedup) − 68 (buffer) = 11,669 ✅**

---

*Revision 3 finalised: 2026-11-16 | No preprocessing implemented | No dataset modified | No models trained*  
*Awaiting user approval before implementation begins.*
