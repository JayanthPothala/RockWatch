"""
Investigation 1: Duplicate Coordinates
- Records sharing lat/lon in the modelling dataset
- Label conflicts within duplicate groups
- Feature variance within duplicate groups
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

z_cols = ['Aspect_z','Dist_River_z','Elevation_z','FlowAccumulation_z','NDVI_z','Rainfall_z','Slope_z']

# --- All groups sharing a coordinate ---
dup_mask = df.duplicated(subset=['latitude','longitude'], keep=False)
df_dup = df[dup_mask].copy()
groups = df_dup.groupby(['latitude','longitude'])

print(f"=== DUPLICATE COORDINATE OVERVIEW ===")
print(f"Total rows in duplicate groups : {len(df_dup)}")
print(f"Distinct duplicated coordinates: {df_dup.groupby(['latitude','longitude']).ngroups}")

# --- Group sizes ---
grp_sizes = groups.size()
print(f"\nGroup size distribution:")
print(grp_sizes.value_counts().sort_index().to_string())

# --- Label conflicts: same coordinate, different class ---
def has_conflict(g):
    return g['class'].nunique() > 1

conflict_groups = groups.filter(has_conflict)
n_conflict_coords = conflict_groups.groupby(['latitude','longitude']).ngroups
print(f"\n=== LABEL CONFLICTS ===")
print(f"Coordinates where class=0 and class=1 both appear: {n_conflict_coords}")
print(f"Rows involved in label conflicts: {len(conflict_groups)}")

# Show conflict examples
if n_conflict_coords > 0:
    print("\nFirst 5 conflict coordinate groups:")
    shown = 0
    for (lat, lon), grp in conflict_groups.groupby(['latitude','longitude']):
        print(f"\n  Coord ({lat:.6f}, {lon:.6f}): {len(grp)} rows")
        print(grp[['sample_id','class','Elevation_z','Slope_z','Rainfall_z','lulc_name']].to_string())
        shown += 1
        if shown >= 5:
            break

# --- Same class, same coordinate: feature variance ---
print("\n=== FEATURE VARIANCE WITHIN SAME-LABEL DUPLICATE GROUPS ===")
# Same-class groups (class=1 duplicates)
df_dup_1 = df_dup[df_dup['class'] == 1]
df_dup_0 = df_dup[df_dup['class'] == 0]

# Check if z-score features are identical within same-coord same-class groups
for cls, subset in [(1, df_dup_1), (0, df_dup_0)]:
    grp = subset.groupby(['latitude','longitude'])
    varies = {}
    for col in z_cols:
        varying_groups = (grp[col].nunique() > 1).sum()
        varies[col] = varying_groups
    total_groups = grp.ngroups
    print(f"\nClass {cls} duplicate groups: {total_groups} coordinate groups, {len(subset)} rows")
    for col, n in varies.items():
        print(f"  {col}: {n}/{total_groups} groups have varying values")

# --- Full identical feature vectors ---
print("\n=== FULLY IDENTICAL ROWS (all 7 z-cols identical at same coord) ===")
exact_dups = df_dup[df_dup.duplicated(subset=['latitude','longitude'] + z_cols, keep=False)]
print(f"Rows where coord AND all 7 z-scores identical: {len(exact_dups)}")

# --- Class=1 same-class group examples ---
print("\n=== SAMPLE CLASS=1 DUPLICATE GROUP (up to 5 rows at same coord) ===")
shown = 0
for (lat, lon), grp in df_dup_1.groupby(['latitude','longitude']):
    if len(grp) > 1:
        print(f"\nCoord ({lat:.6f},{lon:.6f}) — {len(grp)} rows:")
        print(grp[['sample_id','class','Aspect_z','Elevation_z','Slope_z','Rainfall_z',
                    'nearest_inventory_sno','match_dist_m','n_samples_at_coord']].to_string())
        shown += 1
        if shown >= 3:
            break

