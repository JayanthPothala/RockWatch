"""
Investigation B: Spatial split longitude coverage and minimum distances
- Longitude distributions per latitude split
- Minimum distances between nearest points across split boundaries
- Geographic overlap / corridor analysis
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

# Latitude cuts from prior investigation (on full 12409 rows)
# After dedup we recompute, but use the same approximate splits here for analysis
CUT1 = 31.7796   # ~70th pct
CUT2 = 32.2340   # ~85th pct

train = df[df['latitude'] < CUT1]
val   = df[(df['latitude'] >= CUT1) & (df['latitude'] < CUT2)]
test  = df[df['latitude'] >= CUT2]

print("=== LONGITUDE DISTRIBUTION BY SPLIT ===")
for name, part in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
    print(f"\n{name} ({len(part)} rows):")
    print(f"  Lat:  {part['latitude'].min():.4f} to {part['latitude'].max():.4f}")
    print(f"  Lon:  {part['longitude'].min():.4f} to {part['longitude'].max():.4f}")
    print(f"  Lon quartiles: Q1={part['longitude'].quantile(0.25):.4f}, "
          f"Med={part['longitude'].median():.4f}, Q3={part['longitude'].quantile(0.75):.4f}")
    print(f"  Lon std: {part['longitude'].std():.4f}")
    q1,q2,q3 = (part['longitude'] < 76.5).sum(), \
                ((part['longitude']>=76.5)&(part['longitude']<78.0)).sum(), \
                (part['longitude']>=78.0).sum()
    n = len(part)
    print(f"  Lon bins: <76.5°E={q1}({q1/n*100:.1f}%), 76.5-78°E={q2}({q2/n*100:.1f}%), >=78°E={q3}({q3/n*100:.1f}%)")

print("\n=== LONGITUDE OVERLAP BETWEEN SPLITS ===")
train_lon = set(train['longitude'].round(2))
val_lon   = set(val['longitude'].round(2))
test_lon  = set(test['longitude'].round(2))
print(f"Train lon range: [{train['longitude'].min():.4f}, {train['longitude'].max():.4f}]")
print(f"Val   lon range: [{val['longitude'].min():.4f}, {val['longitude'].max():.4f}]")
print(f"Test  lon range: [{test['longitude'].min():.4f}, {test['longitude'].max():.4f}]")
print(f"Train-Val lon overlap at 2dp: {len(train_lon & val_lon)} / {len(val_lon)} val values")
print(f"Train-Test lon overlap at 2dp: {len(train_lon & test_lon)} / {len(test_lon)} test values")
print(f"Val-Test lon overlap at 2dp: {len(val_lon & test_lon)} / {len(test_lon)} test values")

print("\n=== MINIMUM DISTANCE ACROSS SPLIT BOUNDARIES ===")
# Near boundary: train rows within 0.05° of CUT1
train_near = train[train['latitude'] > CUT1 - 0.05]
val_near   = val[val['latitude'] < CUT1 + 0.05]

print(f"Train rows within 0.05° of CUT1 ({CUT1:.4f}°N): {len(train_near)}")
print(f"Val rows within 0.05° of CUT1:   {len(val_near)}")

# Approximate great-circle distance (degrees → km, rough)
def approx_min_dist_km(df1, df2, sample=200):
    """Sample-based minimum distance in km (Euclidean approx at ~31°N)."""
    if len(df1) > sample:
        df1 = df1.sample(sample, random_state=42)
    if len(df2) > sample:
        df2 = df2.sample(sample, random_state=42)
    lat1 = df1['latitude'].values[:, None]
    lon1 = df1['longitude'].values[:, None]
    lat2 = df2['latitude'].values[None, :]
    lon2 = df2['longitude'].values[None, :]
    # 1° lat ≈ 111 km; 1° lon at 31°N ≈ 111 * cos(31°) ≈ 95 km
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 95.0
    dist = np.sqrt(dlat**2 + dlon**2)
    return dist.min()

min_d_train_val = approx_min_dist_km(train_near, val_near)
print(f"\nMin distance (sample) between train/val border rows: {min_d_train_val:.2f} km")

val_near2 = val[val['latitude'] > CUT2 - 0.05]
test_near2 = test[test['latitude'] < CUT2 + 0.05]
print(f"\nVal rows within 0.05° of CUT2 ({CUT2:.4f}°N): {len(val_near2)}")
print(f"Test rows within 0.05° of CUT2: {len(test_near2)}")
min_d_val_test = approx_min_dist_km(val_near2, test_near2)
print(f"Min distance (sample) between val/test border rows: {min_d_val_test:.2f} km")

print("\n=== SPATIAL AUTOCORRELATION CONCERN: 30m PIXEL SCALE ===")
# How many train/val border pairs are within 1 km, 5 km, 10 km?
def count_within_km(df1, df2, km_thresh, sample=500):
    if len(df1) > sample:
        df1 = df1.sample(sample, random_state=42)
    if len(df2) > sample:
        df2 = df2.sample(sample, random_state=42)
    lat1 = df1['latitude'].values[:, None]
    lon1 = df1['longitude'].values[:, None]
    lat2 = df2['latitude'].values[None, :]
    lon2 = df2['longitude'].values[None, :]
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 95.0
    dist = np.sqrt(dlat**2 + dlon**2)
    # count pairs within threshold
    return (dist < km_thresh).sum()

for km in [1, 5, 10, 20]:
    n = count_within_km(train_near, val_near, km)
    print(f"  Train-Val border pairs within {km:2d} km (sample): {n}")

print("\n=== DISTRICT DISTRIBUTION BY SPLIT ===")
# Use inventory to get district for class=1 rows
df_inv = pd.read_csv(f"{BASE}/hp_landslide_inventory_clean.csv")
df_model = df.copy()
class1 = df_model[df_model['class']==1].copy()
class1['sno_int'] = class1['nearest_inventory_sno'].astype(float).astype('Int64')
class1_d = class1.merge(df_inv[['sno','district']], left_on='sno_int', right_on='sno', how='left')

for name, part in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
    ids = set(part['sample_id'])
    sub = class1_d[class1_d['sample_id'].isin(ids)]
    print(f"\n{name} class=1 district distribution:")
    print(sub['district'].value_counts().to_string())

print("\n=== ALTERNATIVE: GRID BLOCK SPLIT (1.0° x 1.0°) ===")
df['lat_blk1'] = (df['latitude'] // 1.0) * 1.0
df['lon_blk1'] = (df['longitude'] // 1.0) * 1.0
df['blk1_id']  = df['lat_blk1'].round(0).astype(str) + "_" + df['lon_blk1'].round(0).astype(str)

blk1 = df.groupby('blk1_id').agg(
    total=('class','count'),
    class1=('class','sum'),
    lat_min=('latitude','min'), lat_max=('latitude','max'),
    lon_min=('longitude','min'), lon_max=('longitude','max')
).reset_index()
blk1['class0'] = blk1['total'] - blk1['class1']
blk1['pct1'] = (blk1['class1']/blk1['total']*100).round(1)
blk1 = blk1.sort_values('total', ascending=False)
print(f"Total 1°×1° blocks: {len(blk1)}")
print(blk1.to_string())
