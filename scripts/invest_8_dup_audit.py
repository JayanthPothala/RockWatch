"""
Verification B: Complete duplicate-group field audit
- For every duplicate coordinate group: check class, all 7 z-scores, LULC, inventory fields
- Report any conflicts
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

z_cols   = ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']
lulc_cols= [c for c in df.columns if c.startswith('LULC_')]
inv_cols = ['nearest_inventory_sno','match_dist_m','n_samples_at_coord']

dup_mask = df.duplicated(subset=['latitude','longitude'], keep=False)
df_dup   = df[dup_mask].copy()
groups   = df_dup.groupby(['latitude','longitude'])

conflicts = {
    'class': 0, 'z_score': 0, 'lulc': 0,
    'inv_sno': 0, 'inv_dist': 0, 'n_samples': 0
}
conflict_examples = {}

for (lat, lon), grp in groups:
    # Class
    if grp['class'].nunique() > 1:
        conflicts['class'] += 1
        conflict_examples.setdefault('class', []).append((lat, lon, grp['class'].tolist()))

    # All 7 z-scores
    for col in z_cols:
        if grp[col].nunique() > 1:
            conflicts['z_score'] += 1
            conflict_examples.setdefault('z_score', []).append(
                (lat, lon, col, grp[col].tolist()))
            break  # one flag per group

    # LULC columns
    for col in lulc_cols:
        if grp[col].nunique() > 1:
            conflicts['lulc'] += 1
            conflict_examples.setdefault('lulc', []).append(
                (lat, lon, col, grp[col].tolist()))
            break

    # Inventory SNO
    if grp['nearest_inventory_sno'].nunique() > 1:
        conflicts['inv_sno'] += 1
        conflict_examples.setdefault('inv_sno', []).append(
            (lat, lon, grp['nearest_inventory_sno'].tolist()))

    # match_dist_m
    if grp['match_dist_m'].nunique() > 1:
        conflicts['inv_dist'] += 1

    # n_samples_at_coord
    if grp['n_samples_at_coord'].nunique() > 1:
        conflicts['n_samples'] += 1

total_groups = len(groups)
print("=== FULL DUPLICATE GROUP CONFLICT AUDIT ===")
print(f"Total duplicate-coordinate groups: {total_groups}")
print(f"Total rows in duplicate groups: {len(df_dup)}")
print(f"\nConflicts found:")
for k, v in conflicts.items():
    status = "✅ NONE" if v == 0 else f"❌ {v} groups"
    print(f"  {k:25s}: {status}")

if any(v > 0 for v in conflicts.values()):
    print("\n=== CONFLICT EXAMPLES ===")
    for k, examples in conflict_examples.items():
        print(f"\n--- {k} conflicts ---")
        for ex in examples[:3]:
            print(f"  {ex}")
else:
    print("\n✅ No conflicts found in any field across all 409 duplicate groups.")

print("\n=== DEDUPLICATION VERIFICATION ===")
# Simulate deduplication: keep min sample_id per (lat, lon)
df_dedup = df.loc[df.groupby(['latitude','longitude'])['sample_id'].idxmin()].copy()
print(f"Rows before dedup: {len(df)}")
print(f"Rows after dedup:  {len(df_dedup)}")
print(f"Rows removed:      {len(df) - len(df_dedup)}")
print(f"\nClass distribution after dedup:")
print(df_dedup['class'].value_counts().to_string())
print(f"Class balance ratio: {(df_dedup['class']==1).sum() / (df_dedup['class']==0).sum():.4f}")

# Verify no duplicate coords remain
remaining_dups = df_dedup.duplicated(subset=['latitude','longitude']).sum()
print(f"\nDuplicate coordinates remaining after dedup: {remaining_dups}")
