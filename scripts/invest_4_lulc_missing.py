"""
Investigation 6 & 7: LULC columns + Missing value strategy
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")
df_master = pd.read_csv(f"{BASE}/landslide_features_master_6503.csv")

# ===================================================
# LULC INVESTIGATION
# ===================================================
print("=== LULC_90 DETAILED INVESTIGATION ===")
lulc90_rows = df[df['LULC_90'] == 1]
print(f"Rows with LULC_90=1: {len(lulc90_rows)}")
if len(lulc90_rows) > 0:
    print(lulc90_rows[['sample_id','latitude','longitude','class',
                         'Elevation_z','Slope_z','Rainfall_z','lulc_code','lulc_name',
                         'nearest_inventory_sno']].to_string())

print("\n=== ALL LULC COLUMN STATISTICS ===")
lulc_cols = [c for c in df.columns if c.startswith('LULC_')]
for col in lulc_cols:
    code = int(col.split('_')[1])
    subset = df[df[col]==1]
    n = len(subset)
    c1 = (subset['class']==1).sum()
    c0 = (subset['class']==0).sum()
    pct_c1 = c1/n*100 if n > 0 else 0
    print(f"{col} (n={n:5d}): class1={c1:5d} ({pct_c1:5.1f}%), class0={c0:5d} | lulc_name={subset['lulc_name'].iloc[0] if n>0 else 'N/A'}")

print("\n=== LULC_70 (SNOW/ICE) DETAILED ===")
snow = df[df['LULC_70']==1]
print(f"Rows: {len(snow)}")
print(f"Class dist: {snow['class'].value_counts().to_dict()}")
print(f"Elevation range: {snow['elevation_m'].min()} to {snow['elevation_m'].max()} m")
print(f"Lat range: {snow['latitude'].min():.4f} to {snow['latitude'].max():.4f}")
print(f"Slope range: {snow['slope_deg'].min():.2f} to {snow['slope_deg'].max():.2f}")

print("\n=== LULC_80 (WATER BODIES) WITH CLASS=1 ===")
water_ls = df[(df['LULC_80']==1) & (df['class']==1)]
print(f"Landslide rows on water bodies: {len(water_ls)}")
print(water_ls[['sample_id','latitude','longitude','elevation_m','slope_deg',
                 'Rainfall_z','nearest_inventory_sno','match_dist_m']].to_string())

print("\n=== LULC_50 (BUILT-UP) ANALYSIS ===")
buildup = df[df['LULC_50']==1]
c1 = (buildup['class']==1).sum()
c0 = (buildup['class']==0).sum()
print(f"Rows: {len(buildup)}, class1={c1} ({c1/len(buildup)*100:.1f}%), class0={c0}")

print("\n=== PERFECT SEPARATOR CHECK ===")
for col in lulc_cols:
    subset = df[df[col]==1]
    if len(subset) > 0 and subset['class'].nunique() == 1:
        only_class = subset['class'].iloc[0]
        print(f"  {col}: ALL {len(subset)} rows are class={only_class} (perfect separator!)")

print("\n=== ONE-HOT CONSISTENCY CHECK ===")
row_sums = df[lulc_cols].sum(axis=1)
print(f"All rows have exactly 1 LULC=1: {(row_sums == 1).all()}")
print(f"Rows with 0 LULC flags: {(row_sums == 0).sum()}")
print(f"Rows with >1 LULC flags: {(row_sums > 1).sum()}")

# ===================================================
# MISSING VALUE INVESTIGATION
# ===================================================
print("\n\n=== MISSING VALUES — MODELLING DATASET ===")
mv = df.isnull().sum()
print(mv[mv > 0].to_string())

print("\n=== MISSING VALUES — MASTER FEATURES ===")
mv_m = df_master.isnull().sum()
mv_m = mv_m[mv_m > 0]
if len(mv_m) == 0:
    print("No missing values in master features.")
else:
    print(mv_m.to_string())

print("\n=== STRUCTURAL MISSING VALUE CHECK ===")
# Confirm that nearest_inventory_sno NaN is exclusively class=0
na_sno = df[df['nearest_inventory_sno'].isnull()]
print(f"Rows where nearest_inventory_sno is NaN: {len(na_sno)}")
print(f"  Class distribution: {na_sno['class'].value_counts().to_dict()}")

na_match = df[df['match_dist_m'].isnull()]
print(f"Rows where match_dist_m is NaN: {len(na_match)}")
print(f"  Class distribution: {na_match['class'].value_counts().to_dict()}")

# Is there any row with class=1 but NaN nearest_inventory_sno?
class1_no_sno = df[(df['class']==1) & df['nearest_inventory_sno'].isnull()]
print(f"\nClass=1 rows without nearest_inventory_sno: {len(class1_no_sno)}")

print("\n=== MASTER FEATURES: TWI NEGATIVE VALUES ===")
print(f"TWI < 0: {(df_master['twi'] < 0).sum()} rows")
print(f"TWI stats: {df_master['twi'].describe().to_string()}")

print("\n=== MASTER FEATURES: CURVATURE VALUES ===")
for col in ['curvature_plan','curvature_profile']:
    print(f"{col}: min={df_master[col].min():.4f}, max={df_master[col].max():.4f}, skew={df_master[col].skew():.3f}")

