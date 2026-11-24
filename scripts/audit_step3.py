import pandas as pd
import numpy as np

BASE = r"c:\Users\potha\OneDrive\Desktop\AI PROJECTS\RockWatch\data\raw\himachal_landslide"

# ---- Load all CSVs ----
df_model   = pd.read_csv(f"{BASE}/landslide_modelling_dataset_12409.csv")
df_inv     = pd.read_csv(f"{BASE}/hp_landslide_inventory_clean.csv")
df_master  = pd.read_csv(f"{BASE}/landslide_features_master_6503.csv")
df_missing = pd.read_csv(f"{BASE}/inventory_records_without_features_36.csv")

print("===== INVENTORY FILE =====")
print(f"Shape: {df_inv.shape}")
print(df_inv.dtypes.to_string())
print("\nMissing values:")
print(df_inv.isnull().sum()[df_inv.isnull().sum()>0].to_string())
print(f"\nLabel distribution: {df_inv['label'].value_counts().to_dict()}")
print(f"\nDistrict distribution:")
print(df_inv['district'].value_counts().to_string())
print(f"\nMaterial distribution:")
print(df_inv['material'].value_counts().to_string())
print(f"\nMovement distribution:")
print(df_inv['movement'].value_counts().to_string())
print(f"\nYear known/unknown:")
print(f"  Known year (non-null): {df_inv['year'].notna().sum()}")
print(f"  Unknown (blank): {df_inv['year'].isna().sum()}")
print(f"  Year range: {df_inv['year'].min()} to {df_inv['year'].max()}")
print(f"\nin_feature_table: {df_inv['in_feature_table'].value_counts().to_dict()}")
print(f"\nn_records_at_coord > 1 (co-located): {(df_inv['n_records_at_coord'] > 1).sum()}")
print(f"Max co-located records: {df_inv['n_records_at_coord'].max()}")

print("\n===== MASTER FEATURES FILE =====")
print(f"Shape: {df_master.shape}")
print("\nMissing values:")
mv = df_master.isnull().sum()
print(mv[mv > 0].to_string())
print(f"Total missing cells: {df_master.isnull().sum().sum()}")
print(f"\nduplicate_feature_vector count:")
print(df_master['duplicate_feature_vector'].value_counts().to_string())
print(f"\nn_records_at_coord distribution:")
print(df_master['n_records_at_coord'].value_counts().sort_index().head(10).to_string())
print(f"\nSPI stats (highly skewed):")
print(df_master['spi'].describe().to_string())
print(f"SPI > 10000: {(df_master['spi'] > 10000).sum()} rows")
print(f"SPI > 100000: {(df_master['spi'] > 100000).sum()} rows")
print(f"\nSlope_deg range: {df_master['slope_deg'].min()} to {df_master['slope_deg'].max()}")
print(f"\nTWI range: {df_master['twi'].min()} to {df_master['twi'].max()}")
print(f"TWI < 0: {(df_master['twi'] < 0).sum()} rows")

print("\n===== MISSING 36 RECORDS FILE =====")
print(f"Shape: {df_missing.shape}")
print(df_missing[['sno','latitude','longitude','district']].to_string())

print("\n===== CROSS-FILE LINKAGE CHECK =====")
# sno match between master and inventory
master_snos = set(df_master['sno'])
inv_snos = set(df_inv['sno'])
print(f"Inventory sno count: {len(inv_snos)}")
print(f"Master sno count: {len(master_snos)}")
print(f"Master snos in inventory: {len(master_snos & inv_snos)}")
print(f"Master snos NOT in inventory: {len(master_snos - inv_snos)}")

# modelling dataset nearest_inventory_sno
model_snos = set(df_model['nearest_inventory_sno'].dropna().astype(int))
print(f"\nModelling dataset unique nearest_inventory_sno (class=1): {len(model_snos)}")
print(f"Modelling sno in inventory: {len(model_snos & inv_snos)}")
print(f"Modelling sno NOT in inventory: {len(model_snos - inv_snos)}")

# How many landslide rows in modelling vs inventory
n_ls_model = (df_model['class'] == 1).sum()
print(f"\nLandslide rows in modelling dataset: {n_ls_model}")
print(f"Inventory records: {len(df_inv)}")
print(f"Master features records: {len(df_master)}")
print(f"Modelling dataset nearest_inventory_sno unique values: {df_model['nearest_inventory_sno'].nunique()}")

print("\n===== SUSPICIOUS RECORDS CHECK =====")
# Slope = 0 in modelling dataset
slope_zero = df_model[df_model['slope_deg'] == 0]
print(f"Rows with slope_deg = 0: {len(slope_zero)}, class dist: {slope_zero['class'].value_counts().to_dict()}")
# Aspect = 0 (true north - could be flat areas)
aspect_zero = df_model[df_model['aspect_deg'] == 0]
print(f"Rows with aspect_deg = 0: {len(aspect_zero)}, class dist: {aspect_zero['class'].value_counts().to_dict()}")
# High elevation landslides
high_elev_ls = df_model[(df_model['elevation_m'] > 4000) & (df_model['class'] == 1)]
print(f"\nLandslides above 4000m: {len(high_elev_ls)}")
# Snow cover class with landslide class
snow_ls = df_model[(df_model['LULC_70'] == 1) & (df_model['class'] == 1)]
print(f"Landslides on snow/ice LULC: {len(snow_ls)}")
# Water bodies with landslide
water_ls = df_model[(df_model['LULC_80'] == 1) & (df_model['class'] == 1)]
print(f"Landslides on water bodies LULC: {len(water_ls)}")

print("\n===== match_dist_m STATS (class=1) =====")
print(df_model[df_model['class']==1]['match_dist_m'].describe().to_string())

