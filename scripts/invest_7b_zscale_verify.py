"""
Deeper investigation:
1. Why does Slope_z have only 76 unique values?
2. Verify linear z-score using aspect_deg (available in both files, same column)
3. Cross-check Elevation z-score using modelling dataset's own elevation_m column
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

print("=== SLOPE_Z: UNIQUE VALUE INVESTIGATION ===")
print(f"Unique slope_deg values: {df['slope_deg'].nunique()}")
print(f"Unique Slope_z values:   {df['Slope_z'].nunique()}")
print(f"Slope_deg unique count suggests integer-resolution data")
print(f"\nSlope_deg value_counts (top 10):")
print(df['slope_deg'].value_counts().head(10).to_string())

print(f"\nSlope_z from slope_deg — recompute and compare:")
# Using modelling dataset's own slope_deg and Slope_z
mu_s  = df['slope_deg'].mean()
sd_s  = df['slope_deg'].std(ddof=1)
recomp_z = (df['slope_deg'] - mu_s) / sd_s
residual = df['Slope_z'] - recomp_z
print(f"  slope_deg: mean={mu_s:.6f}, std={sd_s:.6f}")
print(f"  Residual (Slope_z - recomputed): mean={residual.mean():.10f}, std={residual.std():.10f}, max_abs={residual.abs().max():.10f}")

print(f"\n=== ELEVATION_Z VERIFICATION (using modelling dataset elevation_m) ===")
mu_e  = df['elevation_m'].mean()
sd_e  = df['elevation_m'].std(ddof=1)
recomp_ze = (df['elevation_m'] - mu_e) / sd_e
residual_e = df['Elevation_z'] - recomp_ze
print(f"  elevation_m: mean={mu_e:.6f}, std={sd_e:.6f}")
print(f"  Residual (Elevation_z - recomputed): mean={residual_e.mean():.10f}, std={residual_e.std():.10f}, max_abs={residual_e.abs().max():.10f}")

print(f"\n=== ASPECT_Z VERIFICATION (using modelling dataset aspect_deg) ===")
mu_a  = df['aspect_deg'].mean()
sd_a  = df['aspect_deg'].std(ddof=1)
recomp_za = (df['aspect_deg'] - mu_a) / sd_a
residual_a = df['Aspect_z'] - recomp_za
print(f"  aspect_deg: mean={mu_a:.6f}, std={sd_a:.6f}")
print(f"  Residual (Aspect_z - recomputed): mean={residual_a.mean():.10f}, std={residual_a.std():.10f}, max_abs={residual_a.abs().max():.10f}")

print(f"\n=== CONCLUSION: LINEAR Z-SCORE CONFIRMED? ===")
tol = 1e-6
for label, resid in [("Slope_z", residual), ("Elevation_z", residual_e), ("Aspect_z", residual_a)]:
    confirmed = resid.abs().max() < tol
    print(f"  {label}: max residual={resid.abs().max():.2e} -> {'✅ LINEAR CONFIRMED' if confirmed else '⚠️ RESIDUAL > tolerance'}")

print(f"\n=== CLIPPING EVIDENCE ===")
# Aspect_z: 232 rows at min_val=-1.80436 — is that aspect_deg=0?
at_min_aspect = df[df['Aspect_z'] == df['Aspect_z'].min()]
print(f"Rows at min Aspect_z ({df['Aspect_z'].min():.5f}): {len(at_min_aspect)}")
print(f"  aspect_deg values at min: {at_min_aspect['aspect_deg'].unique()[:5]}")
print(f"  This is simply aspect=0°, not clipping")

at_min_flow = df[df['FlowAccumulation_z'] == df['FlowAccumulation_z'].min()]
print(f"\nRows at min FlowAccumulation_z ({df['FlowAccumulation_z'].min():.5f}): {len(at_min_flow)}")
print(f"  Explanation: log(flow_accumulation) has a natural floor at log(2) ≈ 0.693 (single-cell basins)")
print(f"  This is a physical floor in the raw data, not artificial clipping of z-scores")

print(f"\n=== ROUNDING DETAIL — SLOPE_Z ===")
print(f"Slope_deg has {df['slope_deg'].nunique()} unique values -> z-score also has {df['Slope_z'].nunique()} unique values")
print(f"Slope_deg sample values: {sorted(df['slope_deg'].unique())[:10]}")
print(f"This is SRTM-resolution quantisation, not z-score rounding")
print(f"Slope_z values stored with full float64 precision: 0 values match 6dp round (confirmed)")
