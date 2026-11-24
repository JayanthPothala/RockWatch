import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"

# ---- Load main modelling dataset ----
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

print("=== SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== COLUMNS ===")
for c in df.columns:
    print(c)

print("\n=== DTYPES ===")
print(df.dtypes.to_string())

print("\n=== HEAD (3 rows) ===")
print(df.head(3).to_string())

print("\n=== MISSING VALUES ===")
mv = df.isnull().sum()
print(mv[mv > 0].to_string())
print(f"Total missing cells: {df.isnull().sum().sum()}")

print("\n=== CLASS DISTRIBUTION ===")
print(df['class'].value_counts().to_string())
print(f"Class balance ratio: {df['class'].value_counts()[1] / df['class'].value_counts()[0]:.4f}")

print("\n=== DUPLICATE ROWS ===")
print(f"Exact duplicate rows: {df.duplicated().sum()}")

print("\n=== DUPLICATE COORDINATES ===")
coord_dupes = df.duplicated(subset=['latitude','longitude'], keep=False)
print(f"Rows sharing a lat/lon with another row: {coord_dupes.sum()}")
unique_duped_coords = df[coord_dupes].groupby(['latitude','longitude']).size()
print(f"Distinct duplicated coordinate pairs: {len(unique_duped_coords)}")
print(f"Max rows at one coordinate: {unique_duped_coords.max()}")

print("\n=== NUMERIC SUMMARY (z-score predictors) ===")
z_cols = ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']
print(df[z_cols].describe().to_string())

print("\n=== RAW TERRAIN COLUMNS ===")
raw_cols = ['aspect_deg','elevation_m','slope_deg']
print(df[raw_cols].describe().to_string())

