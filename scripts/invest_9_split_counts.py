"""
Verification C: Exact split counts on the actual deduplicated dataset
- Dedup first
- Compute latitude cut points from deduplicated data
- Apply 500m (~0.0045 deg) exclusion zone
- Report exact counts, class distributions, reconciliation
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"
df = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")

# ---- Step 1: Deduplication ----
df_dedup = df.loc[df.groupby(['latitude','longitude'])['sample_id'].idxmin()].copy()
N_before = len(df)
N_after  = len(df_dedup)
N_removed = N_before - N_after
print(f"=== DEDUPLICATION ===")
print(f"Before dedup: {N_before}")
print(f"After dedup:  {N_after}")
print(f"Removed:      {N_removed}")
print(f"Class dist after dedup: class=1: {(df_dedup['class']==1).sum()}, class=0: {(df_dedup['class']==0).sum()}")

# ---- Step 2: Cut points from deduplicated data ----
lat_sorted = np.sort(df_dedup['latitude'].values)
n = len(lat_sorted)
idx_70 = int(np.floor(n * 0.70))
idx_85 = int(np.floor(n * 0.85))
CUT1 = lat_sorted[idx_70 - 1]
CUT2 = lat_sorted[idx_85 - 1]

print(f"\n=== LATITUDE CUT POINTS (from deduplicated dataset) ===")
print(f"Dataset size (deduplicated): {n}")
print(f"70th percentile index: {idx_70}, cut1 = {CUT1:.6f}°N")
print(f"85th percentile index: {idx_85}, cut2 = {CUT2:.6f}°N")

# ---- Step 3: Apply splits WITHOUT exclusion zone ----
raw_train = df_dedup[df_dedup['latitude'] < CUT1]
raw_val   = df_dedup[(df_dedup['latitude'] >= CUT1) & (df_dedup['latitude'] < CUT2)]
raw_test  = df_dedup[df_dedup['latitude'] >= CUT2]

print(f"\n=== PRE-EXCLUSION ZONE SPLIT SIZES ===")
for name, part in [('TRAIN', raw_train), ('VAL', raw_val), ('TEST', raw_test)]:
    c1 = (part['class']==1).sum()
    c0 = (part['class']==0).sum()
    print(f"{name}: {len(part):5d} rows | class1={c1:4d} ({c1/len(part)*100:.1f}%) | class0={c0:4d} ({c0/len(part)*100:.1f}%)")
print(f"Total: {len(raw_train)+len(raw_val)+len(raw_test)}")

# ---- Step 4: Apply 500m exclusion zone (500m ≈ 0.004504° at 31°N) ----
# 1° lat = 111.0 km; 500m = 0.004504°
BUFFER_DEG = 500 / 111_000  # exactly 500m in degrees
print(f"\n=== EXCLUSION ZONE ===")
print(f"Buffer size: 500 m = {BUFFER_DEG:.6f}°")
print(f"CUT1 exclusion zone: [{CUT1 - BUFFER_DEG:.6f}°N, {CUT1 + BUFFER_DEG:.6f}°N]")
print(f"CUT2 exclusion zone: [{CUT2 - BUFFER_DEG:.6f}°N, {CUT2 + BUFFER_DEG:.6f}°N]")

# Rows in exclusion zones
in_excl_cut1 = df_dedup[(df_dedup['latitude'] >= CUT1 - BUFFER_DEG) &
                          (df_dedup['latitude'] <= CUT1 + BUFFER_DEG)]
in_excl_cut2 = df_dedup[(df_dedup['latitude'] >= CUT2 - BUFFER_DEG) &
                          (df_dedup['latitude'] <= CUT2 + BUFFER_DEG)]

# Rows in BOTH exclusion zones (overlap check)
in_both = df_dedup[(df_dedup['latitude'] >= CUT1 - BUFFER_DEG) &
                    (df_dedup['latitude'] <= CUT1 + BUFFER_DEG) |
                   ((df_dedup['latitude'] >= CUT2 - BUFFER_DEG) &
                    (df_dedup['latitude'] <= CUT2 + BUFFER_DEG))]

print(f"\nRows in CUT1 exclusion zone: {len(in_excl_cut1)}")
print(f"  class=1: {(in_excl_cut1['class']==1).sum()}, class=0: {(in_excl_cut1['class']==0).sum()}")
print(f"Rows in CUT2 exclusion zone: {len(in_excl_cut2)}")
print(f"  class=1: {(in_excl_cut2['class']==1).sum()}, class=0: {(in_excl_cut2['class']==0).sum()}")

# Combined exclusion (union, no double-count)
excl_mask = (
    ((df_dedup['latitude'] >= CUT1 - BUFFER_DEG) & (df_dedup['latitude'] <= CUT1 + BUFFER_DEG)) |
    ((df_dedup['latitude'] >= CUT2 - BUFFER_DEG) & (df_dedup['latitude'] <= CUT2 + BUFFER_DEG))
)
N_excluded = excl_mask.sum()
print(f"\nTotal rows excluded (union of both zones): {N_excluded}")

# ---- Step 5: Final split counts after exclusion ----
df_work = df_dedup[~excl_mask].copy()

final_train = df_work[df_work['latitude'] < CUT1]
final_val   = df_work[(df_work['latitude'] >= CUT1) & (df_work['latitude'] < CUT2)]
final_test  = df_work[df_work['latitude'] >= CUT2]

print(f"\n=== FINAL SPLIT COUNTS (after dedup + exclusion zone) ===")
total_accounted = 0
for name, part in [('TRAIN', final_train), ('VAL', final_val), ('TEST', final_test)]:
    c1 = (part['class']==1).sum()
    c0 = (part['class']==0).sum()
    total_accounted += len(part)
    print(f"{name}: {len(part):5d} rows | class1={c1:4d} ({c1/len(part)*100:.1f}%) | "
          f"class0={c0:4d} ({c0/len(part)*100:.1f}%) | "
          f"lat: {part['latitude'].min():.4f}–{part['latitude'].max():.4f}°N")

print(f"\n=== RECONCILIATION ===")
print(f"Before dedup:                       {N_before:6d}")
print(f"Removed by dedup:                  -{N_removed:6d}")
print(f"After dedup:                        {N_after:6d}")
print(f"Removed by exclusion zones:        -{N_excluded:6d}")
print(f"Remaining (train+val+test):         {total_accounted:6d}")
print(f"Check (should be 0): {N_after - N_excluded - total_accounted}")

print(f"\n=== SPATIAL VALIDATION OF BUFFER ===")
# Min distance across boundary after exclusion
print(f"After exclusion, closest train row to CUT1: {CUT1 - final_train['latitude'].max():.6f}° = "
      f"{(CUT1 - final_train['latitude'].max())*111000:.0f} m")
print(f"After exclusion, closest val row to CUT1:   {final_val['latitude'].min() - CUT1:.6f}° = "
      f"{(final_val['latitude'].min() - CUT1)*111000:.0f} m")
print(f"After exclusion, closest val row to CUT2:   {CUT2 - final_val['latitude'].max():.6f}° = "
      f"{(CUT2 - final_val['latitude'].max())*111000:.0f} m")
print(f"After exclusion, closest test row to CUT2:  {final_test['latitude'].min() - CUT2:.6f}° = "
      f"{(final_test['latitude'].min() - CUT2)*111000:.0f} m")
print(f"\nNote: 500 m is a minimum boundary gap, not a proof of spatial independence.")
print(f"Spatial autocorrelation extends to 5-50 km for regional environmental layers.")
