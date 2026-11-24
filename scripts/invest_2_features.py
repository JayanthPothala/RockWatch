"""
Investigation 2 & 3: Feature Availability
- Which raw values exist in modelling dataset vs master features
- Coordinate-based join feasibility between the two files
- Identifier (sno) join feasibility
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df_model  = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")
df_master = pd.read_csv(f"{BASE}/landslide_features_master_6503.csv")

print("=== MASTER FEATURES FILE COLUMNS ===")
for c in df_master.columns:
    print(f"  {c}: {df_master[c].dtype}")

print("\n=== RAW VALUES IN MODELLING DATASET ===")
print("Available raw (non-z-scored) columns in modelling dataset:")
raw_avail = ['aspect_deg','elevation_m','slope_deg']
for c in raw_avail:
    print(f"  {c}: min={df_model[c].min():.3f}, max={df_model[c].max():.3f}")

print("\n=== FEATURES ONLY IN MASTER (not in modelling dataset) ===")
master_raw_cols = [c for c in df_master.columns if c not in
                   ['sno','latitude','longitude','label','n_records_at_coord','duplicate_feature_vector']]
modelling_cols = set(df_model.columns)
master_only = [c for c in master_raw_cols if c not in modelling_cols]
print(f"Master-only feature columns: {len(master_only)}")
for c in master_only:
    print(f"  {c}")

print("\n=== IDENTIFIER JOIN: sno-based ===")
# class=1 rows have nearest_inventory_sno
class1 = df_model[df_model['class'] == 1].copy()
class1['sno_int'] = class1['nearest_inventory_sno'].astype(int)

# Check how many class=1 rows match a master sno
master_snos = set(df_master['sno'])
matched = class1['sno_int'].isin(master_snos).sum()
print(f"Class=1 rows: {len(class1)}")
print(f"Class=1 rows with sno in master: {matched} ({matched/len(class1)*100:.2f}%)")
print(f"Class=1 rows with sno NOT in master: {len(class1) - matched}")

# Unique snos
uniq_sno = class1['sno_int'].nunique()
master_sno_matched = sum(1 for s in class1['sno_int'].unique() if s in master_snos)
print(f"\nUnique nearest_inventory_sno values: {uniq_sno}")
print(f"Of these, in master sno set: {master_sno_matched}")
print(f"Of these, NOT in master sno set: {uniq_sno - master_sno_matched}")

print("\n=== COORDINATE JOIN: lat/lon-based ===")
# Round to 5 decimal places for fuzzy join
df_model['lat_r'] = df_model['latitude'].round(5)
df_model['lon_r'] = df_model['longitude'].round(5)
df_master['lat_r'] = df_master['latitude'].round(5)
df_master['lon_r'] = df_master['longitude'].round(5)

master_coords = set(zip(df_master['lat_r'], df_master['lon_r']))
model_coords = list(zip(df_model['lat_r'], df_model['lon_r']))
match_coord = sum(1 for c in model_coords if c in master_coords)
print(f"Model rows whose (lat,lon) matches master (±0.00001°): {match_coord} / {len(df_model)}")
print(f"  Class=1 matches: {sum(1 for i,c in enumerate(model_coords) if c in master_coords and df_model.iloc[i]['class']==1)}")
print(f"  Class=0 matches: {sum(1 for i,c in enumerate(model_coords) if c in master_coords and df_model.iloc[i]['class']==0)}")

print("\n=== COORDINATE MATCH AT DIFFERENT PRECISIONS ===")
for dp in [3, 4, 5, 6]:
    df_model['lat_r2'] = df_model['latitude'].round(dp)
    df_model['lon_r2'] = df_model['longitude'].round(dp)
    df_master['lat_r2'] = df_master['latitude'].round(dp)
    df_master['lon_r2'] = df_master['longitude'].round(dp)
    mc = set(zip(df_master['lat_r2'], df_master['lon_r2']))
    n = sum(1 for c in zip(df_model['lat_r2'], df_model['lon_r2']) if c in mc)
    print(f"  Rounded to {dp} dp: {n} / {len(df_model)} model rows matched ({n/len(df_model)*100:.2f}%)")

print("\n=== KEY FEATURES AVAILABLE IN MASTER (raw values for class=1) ===")
# The 4 z-score predictors without raw counterparts in modelling dataset
# Their raw names in master:
raw_map = {
    'Dist_River_z': 'dist_river_m',
    'FlowAccumulation_z': 'flow_accumulation_log',  # already log
    'NDVI_z': 'ndvi_median',
    'Rainfall_z': 'rainfall_annual_mm',
}
print("Z-score predictor -> raw name in master features:")
for z, raw in raw_map.items():
    if raw in df_master.columns:
        print(f"  {z} -> master['{raw}']: min={df_master[raw].min():.4f}, max={df_master[raw].max():.4f}, skew={df_master[raw].skew():.3f}")
    else:
        print(f"  {z} -> '{raw}' NOT FOUND in master")

print("\n=== SAMPLE MERGED ROW CHECK (sno join) ===")
# Merge class=1 model rows with master on sno
class1_m = class1.copy()
class1_m['sno'] = class1_m['sno_int']
merged = class1_m.merge(df_master[['sno','dist_river_m','flow_accumulation_log','ndvi_median',
                                    'rainfall_annual_mm','slope_deg','aspect_deg','elevation_m']],
                         on='sno', how='left', suffixes=('_model','_master'))
print(f"Merged rows: {len(merged)}")
print(f"dist_river_m null after merge: {merged['dist_river_m'].isnull().sum()}")
print(f"\nSample merge (first 3 rows):")
print(merged[['sample_id','class','Dist_River_z','dist_river_m',
              'Elevation_z','elevation_m_model','elevation_m_master',
              'Rainfall_z','rainfall_annual_mm']].head(3).to_string())

