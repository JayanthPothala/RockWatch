"""
Verification A: Z-score standardisation type
- Confirm ordinary linear z-score (not clipped, not nonlinear)
- Verify back-calculation: if z = (x - mu) / sd, then x = z * sd + mu
- Demonstrate that re-standardising corrected_z = (z - train_mu_z) / train_sd_z
  is mathematically equivalent to: corrected_z = (x - train_mu_x) / train_sd_x
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df       = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")
df_master= pd.read_csv(f"{BASE}/landslide_features_master_6503.csv")

z_cols = ['Elevation_z','Slope_z','Rainfall_z']   # 3 for which we have raw counterparts

print("=== GLOBAL Z-SCORE PARAMETERS (from full 12409-row dataset) ===")
for col in ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']:
    mu = df[col].mean()
    sd = df[col].std(ddof=1)
    print(f"  {col}: global_mean={mu:.10e}, global_std={sd:.10f}")

print("\n=== LINEAR STANDARDISATION VERIFICATION ===")
# For the 3 predictors where raw values exist in BOTH files,
# verify: z_col == (raw_col - global_mean_raw) / global_std_raw
# Join class=1 rows with master on sno
class1 = df[df['class']==1].copy()
class1['sno_int'] = class1['nearest_inventory_sno'].astype(float).astype('Int64')
merged = class1.merge(
    df_master[['sno','elevation_m','slope_deg','rainfall_annual_mm',
               'dist_river_m','flow_accumulation_log','ndvi_median']],
    left_on='sno_int', right_on='sno', how='inner',
    suffixes=('_model','_master')
)
print(f"Merged rows for verification: {len(merged)}")

# Elevation
raw_name, z_name = 'elevation_m_master', 'Elevation_z'
mu_raw = merged[raw_name].mean()
sd_raw = merged[raw_name].std(ddof=1)
recomputed_z = (merged[raw_name] - mu_raw) / sd_raw
# Using GLOBAL (all 12409) mu/sd of Elevation_z
global_mu_z = df['Elevation_z'].mean()
global_sd_z = df['Elevation_z'].std(ddof=1)
residual = (merged[z_name] - recomputed_z)
print(f"\nElevation: global recomputed vs dataset z-scores")
print(f"  Residual (z_dataset - z_recomputed_from_master): mean={residual.mean():.8f}, std={residual.std():.8f}, max_abs={residual.abs().max():.8f}")

# Slope — use slope_deg from modelling dataset (available for all rows without join suffix)
# In modelling dataset, the column is slope_deg (no suffix)
# After merge with master, master has 'slope_deg' too — it may appear as slope_deg_model/_master
# Use the modelling dataset raw column instead
raw_name2, z_name2 = 'slope_deg_model', 'Slope_z'
if raw_name2 not in merged.columns:
    # Try without suffix — modelling dataset slope_deg was included in merge request but master also has it
    # Re-check: slope_deg is in BOTH modelling and master, so suffix was applied
    raw_name2 = 'slope_deg_master'
if raw_name2 not in merged.columns:
    raw_name2 = 'slope_deg'
mu2 = merged[raw_name2].mean()
sd2 = merged[raw_name2].std(ddof=1)
recomputed_z2 = (merged[raw_name2] - mu2) / sd2
residual2 = (merged[z_name2] - recomputed_z2)
print(f"\nSlope (using {raw_name2}): global recomputed vs dataset z-scores")
print(f"  Residual: mean={residual2.mean():.8f}, std={residual2.std():.8f}, max_abs={residual2.abs().max():.8f}")

# Dist_River
raw_name3, z_name3 = 'dist_river_m', 'Dist_River_z'
mu3 = merged[raw_name3].mean()
sd3 = merged[raw_name3].std(ddof=1)
recomputed_z3 = (merged[raw_name3] - mu3) / sd3
residual3 = (merged[z_name3] - recomputed_z3)
print(f"\nDist_River: global recomputed vs dataset z-scores")
print(f"  Residual: mean={residual3.mean():.8f}, std={residual3.std():.8f}, max_abs={residual3.abs().max():.8f}")
print("  NOTE: This compares z computed from class=1 raw only vs z from full 12409 rows")

print("\n=== CLIPPING CHECK (z-score predictors) ===")
for col in ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']:
    # If clipped, we'd expect a sharp pile-up at some boundary
    vals = df[col]
    # Check for repeated identical extreme values (sign of clipping)
    min_v, max_v = vals.min(), vals.max()
    n_at_min = (vals == min_v).sum()
    n_at_max = (vals == max_v).sum()
    # Look for unusual concentration at extremes
    pct99 = vals.quantile(0.99)
    pct01 = vals.quantile(0.01)
    n_above_pct99 = (vals > pct99).sum()
    print(f"  {col}: min_val={min_v:.5f} (n={n_at_min}), max_val={max_v:.5f} (n={n_at_max}), "
          f"above 99th pct ({pct99:.3f}): {n_above_pct99} rows")

print("\n=== ROUNDING CHECK (z-score predictors) ===")
for col in ['Elevation_z','Rainfall_z','Slope_z']:
    vals = df[col]
    # Count how many have more than 6 significant decimal places
    # Rounding artefacts would show repeated values or fewer decimals
    rounded_to_6 = vals.round(6)
    exact_match_6dp = (vals == rounded_to_6).sum()
    print(f"  {col}: values matching 6dp round: {exact_match_6dp}/{len(vals)} ({exact_match_6dp/len(vals)*100:.2f}%)")
    # Count unique values
    print(f"           unique values: {vals.nunique()} of {len(vals)}")

print("\n=== MATHEMATICAL PROOF OF RE-STANDARDISATION EQUIVALENCE ===")
print("""
PROOF:
Given:
  z_global = (x - mu_global) / sd_global          ... (1) original standardisation

We want corrected_z = (x - mu_train) / sd_train   ... (2) target (from raw)

From (1): x = z_global * sd_global + mu_global      ... (3)

Substituting (3) into (2):
  corrected_z = (z_global * sd_global + mu_global - mu_train) / sd_train
              = z_global * (sd_global / sd_train) + (mu_global - mu_train) / sd_train

This is a LINEAR transformation of z_global:
  corrected_z = A * z_global + B
  where A = sd_global / sd_train
        B = (mu_global - mu_train) / sd_train

Computing mu_train_z and sd_train_z from training rows of z_global:
  mu_train_z   = mean(z_global over training rows)
               = mean((x - mu_global)/sd_global over training rows)
               = (mu_train_x - mu_global) / sd_global

  sd_train_z   = std(z_global over training rows)
               = std((x - mu_global)/sd_global over training rows)
               = sd_train_x / sd_global

Then: corrected_z = (z_global - mu_train_z) / sd_train_z
                  = (z_global - (mu_train_x - mu_global)/sd_global) / (sd_train_x/sd_global)
                  = (z_global * sd_global - (mu_train_x - mu_global)) / sd_train_x
                  = (z_global * sd_global + mu_global - mu_train_x) / sd_train_x
                  = (x - mu_train_x) / sd_train_x    [substituting from (3)]

This is EXACTLY equation (2). QED.

CONCLUSION:
Re-standardising z_global using training-row mean and std of z_global
is mathematically identical to re-standardising raw x using training-row
mean and std of x. No residual leakage remains from the original global
constants, PROVIDED the transformation from x to z_global was strictly
linear (no clipping, no nonlinear transformation, no lossy rounding).
""")
