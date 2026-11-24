import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

z_cols = ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']

print("=== POINT-BISERIAL CORRELATION: z-score predictors vs class ===")
for col in z_cols:
    corr = df[col].corr(df['class'])
    print(f"{col}: r = {corr:.4f}")

print("\n=== INTER-PREDICTOR CORRELATION (z-score predictors) ===")
print(df[z_cols].corr().round(3).to_string())

print("\n=== ONE-HOT LULC COLUMN SUMS ===")
lulc_cols = ['LULC_10','LULC_20','LULC_30','LULC_40','LULC_50','LULC_60','LULC_70','LULC_80','LULC_90','LULC_100']
print(df[lulc_cols].sum().to_string())
print(f"\nRows where all LULC flags = 0: {(df[lulc_cols].sum(axis=1) == 0).sum()}")
print(f"Rows where more than 1 LULC flag = 1: {(df[lulc_cols].sum(axis=1) > 1).sum()}")
print(f"LULC column sum (should match lulc_code non-null count): each row sums to 1")

# Check LULC redundancy: lulc_code, lulc_name, and one-hot all same info
print("\n=== LULC REDUNDANCY: lulc_code maps to one-hot ===")
# lulc_code column is redundant with one-hot
for code in [10,20,30,40,50,60,70,80,90,100]:
    col = f"LULC_{code}"
    agree = (df[df['lulc_code'] == code][col] == 1).all()
    print(f"LULC_{code} matches lulc_code=={code}: {agree}")

print("\n=== DATA LEAKAGE ASSESSMENT ===")
print("Columns that should NOT be used as predictors:")
print("  - sample_id: row number, no predictive value")
print("  - gee_system_index: GEE identifier")
print("  - latitude, longitude: should only be used for spatial splitting")
print("  - nearest_inventory_sno: only for class=1, encodes identity of event")
print("  - match_dist_m: only for class=1, 5878 NaN for class=0")
print("  - n_samples_at_coord: post-hoc count, encodes label clustering pattern")
print("  - lulc_code / lulc_name: same info as one-hot (multicollinearity if used together)")

print("\n=== STANDARDISATION LEAKAGE CHECK ===")
print("Z-scores are computed over ALL 12,409 rows (class=0 and class=1 combined).")
print("If the test set is a subset of the same 12,409 rows, the test data has")
print("already influenced the standardisation parameters -> LEAKAGE.")
print("The z-score statistics must NOT be reused; re-standardise from scratch on training set only.")
print(f"\nMean of z-scored cols across full dataset:")
print(df[z_cols].mean().to_string())
print(f"\nStd of z-scored cols across full dataset:")
print(df[z_cols].std(ddof=1).to_string())

print("\n=== SKEWNESS OF RAW TERRAIN PREDICTORS ===")
raw_cols = ['aspect_deg','elevation_m','slope_deg']
for col in raw_cols:
    print(f"{col}: skew={df[col].skew():.4f}, kurt={df[col].kurt():.4f}")

print("\n=== SKEWNESS OF Z-SCORE PREDICTORS ===")
for col in z_cols:
    print(f"{col}: skew={df[col].skew():.4f}, kurt={df[col].kurt():.4f}")

print("\n=== SPATIAL SPLIT RECOMMENDATION ===")
print(f"Lat range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
print(f"Lon range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
# Approx spatial split by latitude thirds
q33 = df['latitude'].quantile(0.333)
q67 = df['latitude'].quantile(0.667)
print(f"Latitude 33rd pct: {q33:.4f}")
print(f"Latitude 67th pct: {q67:.4f}")
south_zone = df[df['latitude'] <= q33]
mid_zone = df[(df['latitude'] > q33) & (df['latitude'] <= q67)]
north_zone = df[df['latitude'] > q67]
print(f"\nSouth zone (<={q33:.4f}N): {len(south_zone)} rows, class 1: {(south_zone['class']==1).sum()}, class 0: {(south_zone['class']==0).sum()}")
print(f"Mid zone ({q33:.4f}-{q67:.4f}N): {len(mid_zone)} rows, class 1: {(mid_zone['class']==1).sum()}, class 0: {(mid_zone['class']==0).sum()}")
print(f"North zone (>{q67:.4f}N): {len(north_zone)} rows, class 1: {(north_zone['class']==1).sum()}, class 0: {(north_zone['class']==0).sum()}")

