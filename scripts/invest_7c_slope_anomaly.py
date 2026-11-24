"""
Investigate why Slope_z has only 76 unique values despite slope_deg having 11370.
This is an important data quality finding.
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

print("=== SLOPE_Z UNIQUE VALUE ANOMALY ===")
print(f"Unique slope_deg: {df['slope_deg'].nunique()}")
print(f"Unique Slope_z:   {df['Slope_z'].nunique()}")

# What are the 76 unique Slope_z values?
slope_z_unique = sorted(df['Slope_z'].unique())
print(f"\nAll 76 unique Slope_z values:")
for v in slope_z_unique:
    count = (df['Slope_z'] == v).sum()
    # What slope_deg values map to this Slope_z?
    deg_vals = df[df['Slope_z'] == v]['slope_deg'].unique()
    print(f"  Slope_z={v:.6f}: {count:4d} rows, slope_deg range [{min(deg_vals):.4f}, {max(deg_vals):.4f}] ({len(deg_vals)} unique deg values)")

print("\n=== IMPLICATION: SLOPE_Z IS QUANTISED ===")
print("Slope_z has only 76 distinct values despite slope_deg having 11370 distinct values.")
print("This suggests Slope_z was computed from an INTEGER version of slope, NOT from slope_deg.")
print("\nTesting: Does slope_deg rounded to integer give 76 unique values?")
slope_int = df['slope_deg'].round(0).astype(int)
print(f"  Unique values of round(slope_deg): {slope_int.nunique()}")

# Try to verify: z-score from integer slope
mu_int = slope_int.mean()
sd_int = slope_int.std(ddof=1)
recomp_z_int = (slope_int - mu_int) / sd_int
residual_int = df['Slope_z'] - recomp_z_int
print(f"\n  Recomputed from round(slope_deg,0):")
print(f"  Residual: mean={residual_int.mean():.10f}, std={residual_int.std():.10f}, max_abs={residual_int.abs().max():.10f}")
confirmed_int = residual_int.abs().max() < 1e-6
print(f"  Matches Slope_z within 1e-6: {confirmed_int}")

# Try round(slope_deg, 1)
slope_1dp = df['slope_deg'].round(1)
print(f"\n  Unique values of round(slope_deg, 1): {slope_1dp.nunique()}")
mu_1 = slope_1dp.mean(); sd_1 = slope_1dp.std(ddof=1)
r1 = df['Slope_z'] - (slope_1dp - mu_1) / sd_1
print(f"  Residual: max_abs={r1.abs().max():.10f}")

# Check raw master slope
df_master = pd.read_csv(f"{BASE}/landslide_features_master_6503.csv")
print(f"\n  Master slope_deg dtype: {df_master['slope_deg'].dtype}")
print(f"  Master slope_deg unique: {df_master['slope_deg'].nunique()}")
print(f"  Master slope_deg sample: {sorted(df_master['slope_deg'].unique())[:10]}")
# Master has integer slope!
mu_m = df_master['slope_deg'].mean(); sd_m = df_master['slope_deg'].std(ddof=1)
print(f"  Master slope: mean={mu_m:.4f}, std={sd_m:.4f}")

print("\n=== ASPECT_Z RESIDUAL EXPLANATION ===")
print(f"Unique aspect_deg in modelling: {df['aspect_deg'].nunique()}")
print(f"Unique Aspect_z in modelling:   {df['Aspect_z'].nunique()}")
# Same test
mu_a = df['aspect_deg'].mean(); sd_a = df['aspect_deg'].std(ddof=1)
recomp_za = (df['aspect_deg'] - mu_a) / sd_a
res_a = df['Aspect_z'] - recomp_za
print(f"Residual Aspect_z: max_abs={res_a.abs().max():.8f}")
print(f"Note: aspect residual is 0.005 — consistent with floating-point rounding from float32 storage")

# Check if z-scores might have been computed from float32
print("\n=== FLOAT32 STORAGE HYPOTHESIS ===")
# If raw values were stored as float32 during GEE export, then converted back to float64
for col, raw in [('Slope_z', 'slope_deg'), ('Aspect_z', 'aspect_deg'), ('Elevation_z', 'elevation_m')]:
    raw_32 = df[raw].astype('float32').astype('float64')
    mu_32  = raw_32.mean(); sd_32 = raw_32.std(ddof=1)
    recomp = (raw_32 - mu_32) / sd_32
    res = df[col] - recomp
    print(f"  {col} from float32({raw}): max_abs_residual = {res.abs().max():.2e}")
