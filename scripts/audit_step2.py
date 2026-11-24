import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"

df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

# ---- LULC distribution ----
print("=== LULC DISTRIBUTION ===")
lulc_dist = df.groupby(['lulc_code','lulc_name']).size().reset_index(name='count')
lulc_dist['pct'] = (lulc_dist['count'] / len(df) * 100).round(2)
print(lulc_dist.to_string())

# ---- LULC by class ----
print("\n=== LULC BY CLASS ===")
print(df.groupby(['class','lulc_name']).size().unstack(fill_value=0).to_string())

# ---- Constant or near-constant columns ----
print("\n=== COLUMN VARIANCE (z-score predictors) ===")
z_cols = ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']
print(df[z_cols].var().to_string())

# ---- Check if z-scores are truly standardised over all 12409 rows ----
print("\n=== Z-SCORE VALIDATION (mean~0, std~1 overall) ===")
for col in z_cols:
    print(f"{col}: mean={df[col].mean():.8f}, std={df[col].std(ddof=1):.8f}")

# ---- Check standardisation per class ----
print("\n=== Z-SCORE MEANS BY CLASS (data leakage check) ===")
print(df.groupby('class')[z_cols].mean().to_string())

# ---- Outlier detection (IQR method on z-score predictors) ----
print("\n=== OUTLIERS (beyond 3 SD from mean, using z-score columns) ===")
for col in z_cols:
    n_out = (df[col].abs() > 3).sum()
    pct = n_out / len(df) * 100
    print(f"{col}: {n_out} outliers ({pct:.2f}%); min={df[col].min():.4f}, max={df[col].max():.4f}")

# ---- NDVI_z extreme investigation ----
print("\n=== NDVI_z EXTREME VALUES ===")
print(df['NDVI_z'].describe())
ndvi_extreme = df[df['NDVI_z'] < -3]
print(f"Rows with NDVI_z < -3: {len(ndvi_extreme)}")
print(f"Class distribution of NDVI_z < -3: {ndvi_extreme['class'].value_counts().to_dict()}")
print(f"LULC names for NDVI_z < -3: {ndvi_extreme['lulc_name'].value_counts().to_dict()}")

# ---- FlowAccumulation_z extreme ----
print("\n=== FlowAccumulation_z EXTREME VALUES ===")
fa_extreme = df[df['FlowAccumulation_z'] > 3]
print(f"Rows with FlowAccumulation_z > 3: {len(fa_extreme)}")
print(f"Class dist: {fa_extreme['class'].value_counts().to_dict()}")

# ---- Dist_River_z skewness check ----
print("\n=== Dist_River_z DISTRIBUTION ===")
print(df['Dist_River_z'].describe())
print(f"Rows with Dist_River_z > 1: {(df['Dist_River_z'] > 1).sum()}")

# ---- Coordinate bounding box ----
print("\n=== COORDINATE BOUNDS ===")
print(f"Latitude:  {df['latitude'].min():.5f} to {df['latitude'].max():.5f}")
print(f"Longitude: {df['longitude'].min():.5f} to {df['longitude'].max():.5f}")
print(f"Expected (HP): 30.00-32.99N, 75.74-79.80E")
print(f"Points outside expected bbox:")
out_lat = ((df['latitude'] < 30.00) | (df['latitude'] > 32.99)).sum()
out_lon = ((df['longitude'] < 75.74) | (df['longitude'] > 79.80)).sum()
print(f"  Latitude outside: {out_lat}")
print(f"  Longitude outside: {out_lon}")

# ---- n_samples_at_coord ----
print("\n=== n_samples_at_coord DISTRIBUTION ===")
print(df['n_samples_at_coord'].value_counts().sort_index().to_string())

# ---- elevation check ----
print("\n=== ELEVATION RANGE CHECK ===")
print(f"Max elevation in data: {df['elevation_m'].max()} m")
print(f"HP max elevation (Reo Purgyil ~6816m): 6816 m")
print(f"Rows with elevation > 5500m: {(df['elevation_m'] > 5500).sum()}")
print(f"Rows with elevation < 300m: {(df['elevation_m'] < 300).sum()}")

