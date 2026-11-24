"""
Investigation 4 & 5: Spatial Splitting + Feature Engineering
- District-based split feasibility
- Grid-based split
- Aspect circular transform validation
- Log1p for skewed features
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df_model  = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")
df_master = pd.read_csv(f"{BASE}/landslide_features_master_6503.csv")
df_inv    = pd.read_csv(f"{BASE}/hp_landslide_inventory_clean.csv")

z_cols = ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']

# =====================================================
# SPATIAL SPLIT INVESTIGATION
# =====================================================
print("=== GRID-BASED SPATIAL BLOCKS (0.5° x 0.5°) ===")
df_model['lat_block'] = (df_model['latitude'] // 0.5) * 0.5
df_model['lon_block'] = (df_model['longitude'] // 0.5) * 0.5
df_model['block_id'] = df_model['lat_block'].round(1).astype(str) + "_" + df_model['lon_block'].round(1).astype(str)

block_summary = df_model.groupby('block_id').agg(
    total=('class','count'),
    class1=('class','sum'),
    lat_min=('latitude','min'),
    lat_max=('latitude','max'),
    lon_min=('longitude','min'),
    lon_max=('longitude','max')
).reset_index()
block_summary['class0'] = block_summary['total'] - block_summary['class1']
block_summary['pct_class1'] = (block_summary['class1'] / block_summary['total'] * 100).round(1)
block_summary = block_summary.sort_values('lat_min')
print(f"Total 0.5°x0.5° blocks: {len(block_summary)}")
print(block_summary[['block_id','total','class1','class0','pct_class1','lat_min','lat_max','lon_min','lon_max']].to_string())

print("\n=== LATITUDE BAND SPLIT (70/15/15 target) ===")
# Sort by latitude and find split points
df_sorted = df_model.sort_values('latitude')
n = len(df_model)
train_target = int(n * 0.70)
val_target   = int(n * 0.15)
test_target  = n - train_target - val_target

# Find natural cut points by latitude
lat_sorted = df_sorted['latitude'].values
cut_train = lat_sorted[train_target - 1]
cut_val   = lat_sorted[train_target + val_target - 1]

print(f"Target split: train={train_target}, val={val_target}, test={test_target}")
print(f"Latitude cut points: train < {cut_train:.4f}°N, val {cut_train:.4f}°–{cut_val:.4f}°N, test > {cut_val:.4f}°N")

train_df = df_model[df_model['latitude'] < cut_train]
val_df   = df_model[(df_model['latitude'] >= cut_train) & (df_model['latitude'] < cut_val)]
test_df  = df_model[df_model['latitude'] >= cut_val]

for name, part in [('TRAIN', train_df), ('VAL', val_df), ('TEST', test_df)]:
    c1 = (part['class']==1).sum()
    c0 = (part['class']==0).sum()
    print(f"{name}: {len(part)} rows ({len(part)/n*100:.1f}%) | class1={c1} ({c1/len(part)*100:.1f}%) | class0={c0} ({c0/len(part)*100:.1f}%)")

print("\n=== DISTRICT-BASED SPLIT ===")
# Use inventory district info, link via nearest_inventory_sno
class1 = df_model[df_model['class']==1].copy()
class1['sno_int'] = class1['nearest_inventory_sno'].astype(int)
inv_sno_district = df_inv[['sno','district']].copy()
class1_w_district = class1.merge(inv_sno_district, left_on='sno_int', right_on='sno', how='left')
print("Class=1 rows with district info:")
print(class1_w_district['district'].value_counts().to_string())
print(f"\nClass=0 rows: no district assignment possible (pseudo-absences)")

# =====================================================
# FEATURE ENGINEERING
# =====================================================
print("\n=== ASPECT CIRCULAR TRANSFORMATION ===")
# Raw aspect_deg in modelling dataset
aspect_raw = df_model['aspect_deg']
print(f"aspect_deg: min={aspect_raw.min():.2f}, max={aspect_raw.max():.2f}, mean={aspect_raw.mean():.2f}")

sin_a = np.sin(np.radians(aspect_raw))
cos_a = np.cos(np.radians(aspect_raw))
print(f"sin_aspect: min={sin_a.min():.4f}, max={sin_a.max():.4f}, mean={sin_a.mean():.4f}")
print(f"cos_aspect: min={cos_a.min():.4f}, max={cos_a.max():.4f}, mean={cos_a.mean():.4f}")

# Correlation with class
print(f"\nCorrelation with class:")
print(f"  Aspect_z (linear z):  {df_model['Aspect_z'].corr(df_model['class']):.4f}")
print(f"  sin_aspect:           {sin_a.corr(df_model['class']):.4f}")
print(f"  cos_aspect:           {cos_a.corr(df_model['class']):.4f}")

# Check for flat areas (aspect_deg=0 means undefined on flat terrain)
print(f"\naspect_deg == 0 (possibly flat/undefined): {(aspect_raw == 0).sum()} rows")
print(f"slope_deg == 0 (flat, aspect undefined):     {(df_model['slope_deg'] == 0).sum()} rows")
flat_nonzero_aspect = ((df_model['slope_deg'] == 0) & (aspect_raw != 0)).sum()
print(f"slope=0 but aspect!=0 (artefact?):           {flat_nonzero_aspect} rows")

print("\n=== SKEWNESS OF RAW MASTER FEATURES (candidates for log1p) ===")
numeric_master = df_master.select_dtypes(include='number').drop(
    columns=['sno','label','n_records_at_coord'], errors='ignore')
numeric_master = numeric_master.drop(
    columns=['duplicate_feature_vector'] if 'duplicate_feature_vector' in numeric_master.columns else [], errors='ignore')

skew_df = pd.DataFrame({
    'col': numeric_master.columns,
    'skew': [numeric_master[c].skew() for c in numeric_master.columns],
    'min':  [numeric_master[c].min() for c in numeric_master.columns],
    'max':  [numeric_master[c].max() for c in numeric_master.columns],
    'mean': [numeric_master[c].mean() for c in numeric_master.columns],
})
skew_df['abs_skew'] = skew_df['skew'].abs()
skew_df = skew_df.sort_values('abs_skew', ascending=False)
print("Top 15 skewed master features (|skew| desc):")
print(skew_df.head(15)[['col','skew','min','max','mean']].to_string())

print("\n=== log1p EFFECT ON HIGHLY SKEWED FEATURES ===")
high_skew = ['spi','dist_river_m','dist_urban','dist_agri','pop_density','flow_accumulation_log']
for col in high_skew:
    if col in df_master.columns:
        raw = df_master[col]
        if (raw < 0).any():
            print(f"{col}: has negatives (min={raw.min():.4f}), log1p not directly applicable")
            continue
        log_vals = np.log1p(raw)
        print(f"{col}: raw skew={raw.skew():.3f} -> log1p skew={log_vals.skew():.3f} (min_raw={raw.min():.3f})")

print("\n=== DIST_RIVER_z BACK-CALCULATION CHECK ===")
# Try to verify what the raw dist_river_m distribution looks like
dr = df_master['dist_river_m']
print(f"dist_river_m: min={dr.min():.2f}, median={dr.median():.2f}, mean={dr.mean():.2f}, max={dr.max():.2f}, skew={dr.skew():.3f}")
log_dr = np.log1p(dr)
print(f"log1p(dist_river_m): skew={log_dr.skew():.3f}, min={log_dr.min():.4f}, max={log_dr.max():.4f}")
# Square root?
sqrt_dr = np.sqrt(dr)
print(f"sqrt(dist_river_m): skew={sqrt_dr.skew():.3f}")

