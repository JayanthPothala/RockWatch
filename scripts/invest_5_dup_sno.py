"""
Investigation A: Duplicate coordinate inconsistency
- Why do some groups have identical nearest_inventory_sno?
- Are these truly separate inventory records or repeated rows?
- What information is lost by deduplication?
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")
df_inv = pd.read_csv(f"{BASE}/hp_landslide_inventory_clean.csv")

dup_mask = df.duplicated(subset=['latitude','longitude'], keep=False)
df_dup = df[dup_mask].copy()
groups = df_dup.groupby(['latitude','longitude'])

print("=== NEAREST_INVENTORY_SNO UNIQUENESS WITHIN DUPLICATE GROUPS ===")
# For each coordinate group, how many distinct sno values?
sno_variety = []
for (lat, lon), grp in groups:
    n_total = len(grp)
    n_unique_sno = grp['nearest_inventory_sno'].nunique()
    n_null_sno   = grp['nearest_inventory_sno'].isnull().sum()
    sno_list = grp['nearest_inventory_sno'].dropna().unique().tolist()
    sno_variety.append({
        'lat': lat, 'lon': lon,
        'n_rows': n_total,
        'n_unique_sno': n_unique_sno,
        'n_null_sno': n_null_sno,
        'sno_values': sno_list
    })

sno_df = pd.DataFrame(sno_variety)

# Groups where all rows share the same single sno
same_sno = sno_df[sno_df['n_unique_sno'] == 1]
diff_sno = sno_df[sno_df['n_unique_sno'] > 1]

print(f"Total duplicate-coordinate groups: {len(sno_df)}")
print(f"Groups where ALL rows have the SAME nearest_inventory_sno: {len(same_sno)}")
print(f"Groups where rows have DIFFERENT nearest_inventory_sno values: {len(diff_sno)}")

if len(diff_sno) > 0:
    print(f"\n--- Groups with DIFFERENT SNOs (up to 10): ---")
    for _, row in diff_sno.head(10).iterrows():
        print(f"  ({row['lat']:.6f}, {row['lon']:.6f}): {row['n_rows']} rows, SNOs={row['sno_values']}")

print(f"\n--- Sample of SAME-SNO groups (3 examples): ---")
for _, row in same_sno.head(3).iterrows():
    subset = df[(df['latitude']==row['lat']) & (df['longitude']==row['lon'])]
    print(f"\n  ({row['lat']:.6f}, {row['lon']:.6f}): {row['n_rows']} rows, SNO={row['sno_values']}")
    print(subset[['sample_id','class','nearest_inventory_sno','match_dist_m',
                   'n_samples_at_coord','gee_system_index']].to_string())

print("\n=== INVENTORY RECORDS AT SAME COORDINATES ===")
# For same-SNO groups, look up the inventory
same_sno_examples = same_sno.head(5)
for _, row in same_sno_examples.iterrows():
    sno_val = int(row['sno_values'][0]) if row['sno_values'] else None
    if sno_val:
        inv_row = df_inv[df_inv['sno'] == sno_val]
        print(f"\nInventory record for SNO={sno_val}:")
        print(inv_row[['sno','latitude','longitude','district','material','movement','year','n_records_at_coord']].to_string())

print("\n=== INVENTORY n_records_at_coord FOR DUPLICATE PIXELS ===")
# For the same-sno groups, check if the inventory shows co-located records
all_snos_in_dups = sno_df['sno_values'].explode().dropna().astype(int).unique()
inv_colocated = df_inv[df_inv['sno'].isin(all_snos_in_dups)][['sno','n_records_at_coord','latitude','longitude']]
print(f"Inventory records linked to duplicate-pixel groups: {len(inv_colocated)}")
print(f"n_records_at_coord distribution for these:")
print(inv_colocated['n_records_at_coord'].value_counts().sort_index().to_string())

print("\n=== GEE_SYSTEM_INDEX UNIQUENESS WITHIN DUPLICATE GROUPS ===")
# Is the gee_system_index the same across duplicate rows?
gee_variety = []
for (lat, lon), grp in groups:
    n_unique_gee = grp['gee_system_index'].nunique()
    gee_variety.append(n_unique_gee)
gee_series = pd.Series(gee_variety)
print(f"Groups where GEE index is identical across all rows: {(gee_series == 1).sum()}")
print(f"Groups where GEE index varies: {(gee_series > 1).sum()}")

print("\n=== WHAT DEDUPLICATION LOSES ===")
# Count groups where multiple different SNOs exist (truly different events)
print(f"Groups with 2+ distinct SNOs (different events, same pixel): {len(diff_sno)}")
print(f"Groups with same SNO (identical event record repeated): {len(same_sno)}")
print(f"\nIf we keep 1 row per pixel, we lose knowledge of:")
print(f"  - {diff_sno['n_rows'].sum() - len(diff_sno)} extra rows from multi-event pixels")
print(f"  - {same_sno['n_rows'].sum() - len(same_sno)} exact-duplicate rows from single-event pixels")
print(f"Total rows removed by dedup: {len(df_dup) - len(sno_df)}")
