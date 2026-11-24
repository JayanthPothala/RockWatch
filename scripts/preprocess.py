"""
RockWatch - Phase 1: Data Cleaning and Preprocessing Pipeline
============================================================
Implements the approved Preprocessing Plan (Revision 3):
1. Preserves original raw dataset files without modification.
2. Loads landslide_modelling_dataset_12409.csv.
3. Creates diagnostic water-body flag (flag_water_lulc = 1 for LULC_80==1 and class==1).
4. Deduplicates verified identical coordinates using min(sample_id) -> 11,737 rows.
5. Computes circular aspect features (sin_aspect, cos_aspect); sets 0 for slope_deg == 0.
6. Drops quasi-constant LULC_90 column.
7. Computes latitude cut points from deduplicated data (CUT1=31.713044°N, CUT2=32.144235°N).
8. Applies 500m (~0.004505°) exclusion buffer around boundaries (68 rows excluded).
9. Splits into TRAIN (8,188), VAL (1,731), TEST (1,750).
10. Fits StandardScaler strictly on TRAIN rows for 6 predictor columns.
11. Transforms train, val, and test splits with train-fitted scaler.
12. Generates feature matrices (17 features + class = 18 columns).
13. Saves all processed artifacts, scaler_params.json, and preprocessing_log.txt.
14. Executes and validates all verification assertions.
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_modelling_data
from src.features.engineer import add_water_body_flag, compute_aspect_trig
from src.data.preprocessor import (
    deduplicate_coordinates,
    re_standardise_features,
    build_feature_matrices,
    SCALED_PREDICTOR_COLS,
    FINAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.splitting.spatial_split import compute_latitude_cuts, apply_spatial_split


def run_pipeline():
    log_lines = []
    
    def log(msg: str):
        print(msg)
        log_lines.append(msg)
        
    log("=" * 80)
    log("ROCKWATCH - PHASE 1: DATA CLEANING & PREPROCESSING EXECUTION")
    log(f"Execution timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("Approved plan: Preprocessing Plan Revision 3")
    log("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: Load raw modelling dataset (Read-only)
    # ---------------------------------------------------------
    log("\n[Step 1] Loading original modelling dataset (read-only)...")
    df_raw = load_raw_modelling_data()
    n_raw = len(df_raw)
    n_raw_cols = df_raw.shape[1]
    log(f"  Loaded raw dataset: {n_raw:,} rows, {n_raw_cols} columns")
    assert n_raw == 12409, f"Expected 12,409 rows, got {n_raw}"
    assert n_raw_cols == 30, f"Expected 30 columns, got {n_raw_cols}"

    # ---------------------------------------------------------
    # STEP 2: Add diagnostic water-body flag
    # ---------------------------------------------------------
    log("\n[Step 2] Adding diagnostic flag_water_lulc...")
    df_step2 = add_water_body_flag(df_raw)
    n_flagged = df_step2["flag_water_lulc"].sum()
    log(f"  Diagnostic flag added. Rows with flag_water_lulc == 1: {n_flagged}")
    assert n_flagged == 28, f"Expected exactly 28 water-body landslide flags, got {n_flagged}"

    # ---------------------------------------------------------
    # STEP 3: Coordinate deduplication
    # ---------------------------------------------------------
    log("\n[Step 3] Performing coordinate deduplication (min sample_id per unique coordinate)...")
    df_dedup = deduplicate_coordinates(df_step2)
    n_dedup = len(df_dedup)
    n_removed = n_raw - n_dedup
    log(f"  Rows before deduplication: {n_raw:,}")
    log(f"  Duplicate pixel exports removed: {n_removed:,}")
    log(f"  Rows after deduplication: {n_dedup:,}")
    assert n_dedup == 11737, f"Expected 11,737 deduplicated rows, got {n_dedup}"
    assert n_removed == 672, f"Expected 672 removed duplicate rows, got {n_removed}"
    assert df_dedup.duplicated(subset=["latitude", "longitude"]).sum() == 0, "Duplicate coordinates remain!"

    # Class balance after deduplication
    c1_dedup = (df_dedup["class"] == 1).sum()
    c0_dedup = (df_dedup["class"] == 0).sum()
    log(f"  Class distribution after dedup: class=1: {c1_dedup:,} ({c1_dedup/n_dedup*100:.2f}%), "
        f"class=0: {c0_dedup:,} ({c0_dedup/n_dedup*100:.2f}%)")
    assert c1_dedup == 5859, f"Expected 5,859 class=1 rows, got {c1_dedup}"
    assert c0_dedup == 5878, f"Expected 5,878 class=0 rows, got {c0_dedup}"

    # ---------------------------------------------------------
    # STEP 4: Aspect circular engineering
    # ---------------------------------------------------------
    log("\n[Step 4] Engineering circular aspect features (sin_aspect, cos_aspect)...")
    df_eng = compute_aspect_trig(df_dedup)
    # Check flat terrain handling
    flat_mask = df_eng["slope_deg"] == 0
    n_flat = flat_mask.sum()
    log(f"  Flat terrain locations (slope_deg == 0): {n_flat} rows")
    if n_flat > 0:
        assert (df_eng.loc[flat_mask, "sin_aspect"] == 0.0).all(), "Flat terrain sin_aspect not 0.0"
        assert (df_eng.loc[flat_mask, "cos_aspect"] == 0.0).all(), "Flat terrain cos_aspect not 0.0"
    log(f"  sin_aspect range: [{df_eng['sin_aspect'].min():.4f}, {df_eng['sin_aspect'].max():.4f}]")
    log(f"  cos_aspect range: [{df_eng['cos_aspect'].min():.4f}, {df_eng['cos_aspect'].max():.4f}]")

    # ---------------------------------------------------------
    # STEP 5: Drop quasi-constant LULC_90
    # ---------------------------------------------------------
    log("\n[Step 5] Dropping quasi-constant LULC_90 column...")
    n_lulc90_positive = df_eng["LULC_90"].sum()
    log(f"  LULC_90 positive count before dropping: {n_lulc90_positive} / {len(df_eng)}")
    df_dedup_final = df_eng.drop(columns=["LULC_90"])
    log(f"  Shape of deduplicated working dataframe: {df_dedup_final.shape}")
    assert df_dedup_final.shape == (11737, 32), f"Expected (11737, 32), got {df_dedup_final.shape}"

    # ---------------------------------------------------------
    # STEP 6: Compute latitude cut points
    # ---------------------------------------------------------
    log("\n[Step 6] Computing deterministic latitude cut points from deduplicated data...")
    cut1, cut2 = compute_latitude_cuts(df_dedup_final, q1=0.70, q2=0.85)
    log(f"  CUT1 (70th percentile): {cut1:.6f}°N")
    log(f"  CUT2 (85th percentile): {cut2:.6f}°N")
    assert np.isclose(cut1, 31.713044, atol=1e-5), f"CUT1 {cut1} does not match expected 31.713044"
    assert np.isclose(cut2, 32.144235, atol=1e-5), f"CUT2 {cut2} does not match expected 32.144235"

    # ---------------------------------------------------------
    # STEP 7: Spatial split with 500m exclusion buffer
    # ---------------------------------------------------------
    log("\n[Step 7] Applying spatial split with 500m (~0.004505°) exclusion buffer...")
    train_df, val_df, test_df, excluded_df, split_meta = apply_spatial_split(
        df_dedup_final, cut1=cut1, cut2=cut2, buffer_m=500.0, km_per_deg_lat=111.0
    )
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_excl = len(excluded_df)
    n_total_splits = n_train + n_val + n_test

    log(f"  Buffer size: 500.0 m ({split_meta['buffer_deg']:.6f}°)")
    log(f"  Rows excluded around CUT1: {split_meta['n_excluded_cut1']}")
    log(f"  Rows excluded around CUT2: {split_meta['n_excluded_cut2']}")
    log(f"  Total rows excluded: {n_excl}")
    log(f"  TRAIN rows: {n_train:,} | class1: {(train_df['class']==1).sum():,} ({(train_df['class']==1).mean()*100:.2f}%), "
        f"class0: {(train_df['class']==0).sum():,} ({(train_df['class']==0).mean()*100:.2f}%)")
    log(f"  VAL   rows: {n_val:,} | class1: {(val_df['class']==1).sum():,} ({(val_df['class']==1).mean()*100:.2f}%), "
        f"class0: {(val_df['class']==0).sum():,} ({(val_df['class']==0).mean()*100:.2f}%)")
    log(f"  TEST  rows: {n_test:,} | class1: {(test_df['class']==1).sum():,} ({(test_df['class']==1).mean()*100:.2f}%), "
        f"class0: {(test_df['class']==0).sum():,} ({(test_df['class']==0).mean()*100:.2f}%)")

    # Assertions on split counts
    assert n_excl == 68, f"Expected 68 excluded rows, got {n_excl}"
    assert n_train == 8188, f"Expected 8,188 train rows, got {n_train}"
    assert n_val == 1731, f"Expected 1,731 val rows, got {n_val}"
    assert n_test == 1750, f"Expected 1,750 test rows, got {n_test}"
    assert n_total_splits == 11669, f"Expected 11,669 total split rows, got {n_total_splits}"

    # Class distribution checks
    assert (train_df["class"] == 1).sum() == 4024, "Train class=1 mismatch"
    assert (train_df["class"] == 0).sum() == 4164, "Train class=0 mismatch"
    assert (val_df["class"] == 1).sum() == 896, "Val class=1 mismatch"
    assert (val_df["class"] == 0).sum() == 835, "Val class=0 mismatch"
    assert (test_df["class"] == 1).sum() == 914, "Test class=1 mismatch"
    assert (test_df["class"] == 0).sum() == 836, "Test class=0 mismatch"

    # Full reconciliation assertion
    assert n_total_splits + n_excl + n_removed == n_raw, "Split reconciliation failed!"

    # ---------------------------------------------------------
    # STEP 8: Fit StandardScaler on TRAIN rows only (6 predictors)
    # ---------------------------------------------------------
    log("\n[Step 8] Fitting StandardScaler on TRAIN data only (6 z-score predictors)...")
    log(f"  Features scaled: {SCALED_PREDICTOR_COLS}")
    log("  Aspect_z is replaced by sin_aspect and cos_aspect; not scaled.")
    train_df_scaled, val_df_scaled, test_df_scaled, scaler_params = re_standardise_features(
        train_df, val_df, test_df, feature_cols=SCALED_PREDICTOR_COLS
    )

    for col in SCALED_PREDICTOR_COLS:
        p = scaler_params["features"][col]
        log(f"    {col:20s}: train_mean = {p['train_mean']:+.6f}, train_std = {p['train_std']:.6f}")

    # Verify scaling properties on TRAIN
    for col in SCALED_PREDICTOR_COLS:
        corr_col = f"{col}_corr"
        m = train_df_scaled[corr_col].mean()
        s = train_df_scaled[corr_col].std(ddof=0)
        assert abs(m) < 1e-6, f"Train {corr_col} mean not ~0 (got {m})"
        assert abs(s - 1.0) < 1e-6, f"Train {corr_col} std not ~1 (got {s})"
    log("  All *_z_corr features in TRAIN verified: mean == 0.000000, std == 1.000000")

    # ---------------------------------------------------------
    # STEP 9: Build feature matrices
    # ---------------------------------------------------------
    log("\n[Step 9] Constructing final feature matrices...")
    features_train, features_val, features_test = build_feature_matrices(
        train_df_scaled, val_df_scaled, test_df_scaled,
        feature_cols=FINAL_FEATURE_COLUMNS,
        target_col=TARGET_COLUMN,
    )
    log(f"  Features train shape: {features_train.shape}")
    log(f"  Features val   shape: {features_val.shape}")
    log(f"  Features test  shape: {features_test.shape}")
    assert features_train.shape == (8188, 18), f"Expected (8188, 18), got {features_train.shape}"
    assert features_val.shape == (1731, 18), f"Expected (1731, 18), got {features_val.shape}"
    assert features_test.shape == (1750, 18), f"Expected (1750, 18), got {features_test.shape}"

    # ---------------------------------------------------------
    # STEP 10: Save all artifacts
    # ---------------------------------------------------------
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n[Step 10] Saving processed files to {processed_dir}...")

    # 1. modelling_deduped.csv
    p_dedup = processed_dir / "modelling_deduped.csv"
    df_dedup_final.to_csv(p_dedup, index=False)
    log(f"  Saved: {p_dedup.name} ({len(df_dedup_final):,} rows, {df_dedup_final.shape[1]} cols)")

    # 2. train_spatial.csv, val_spatial.csv, test_spatial.csv (full metadata, 32 cols)
    p_train = processed_dir / "train_spatial.csv"
    p_val = processed_dir / "val_spatial.csv"
    p_test = processed_dir / "test_spatial.csv"
    train_df.to_csv(p_train, index=False)
    val_df.to_csv(p_val, index=False)
    test_df.to_csv(p_test, index=False)
    log(f"  Saved: {p_train.name} ({len(train_df):,} rows, {train_df.shape[1]} cols)")
    log(f"  Saved: {p_val.name} ({len(val_df):,} rows, {val_df.shape[1]} cols)")
    log(f"  Saved: {p_test.name} ({len(test_df):,} rows, {test_df.shape[1]} cols)")

    # 3. features_train.csv, features_val.csv, features_test.csv (17 features + class = 18 cols)
    p_feat_train = processed_dir / "features_train.csv"
    p_feat_val = processed_dir / "features_val.csv"
    p_feat_test = processed_dir / "features_test.csv"
    features_train.to_csv(p_feat_train, index=False)
    features_val.to_csv(p_feat_val, index=False)
    features_test.to_csv(p_feat_test, index=False)
    log(f"  Saved: {p_feat_train.name} ({len(features_train):,} rows, {features_train.shape[1]} cols)")
    log(f"  Saved: {p_feat_val.name} ({len(features_val):,} rows, {features_val.shape[1]} cols)")
    log(f"  Saved: {p_feat_test.name} ({len(features_test):,} rows, {features_test.shape[1]} cols)")

    # 4. scaler_params.json
    p_scaler = processed_dir / "scaler_params.json"
    with open(p_scaler, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)
    log(f"  Saved: {p_scaler.name}")

    # ---------------------------------------------------------
    # STEP 11: Document Limitations & Verification Summary
    # ---------------------------------------------------------
    log("\n" + "=" * 80)
    log("DOCUMENTED LIMITATIONS & SCIENTIFIC DISCLOSURES")
    log("=" * 80)
    log("1. Slope_z Quantisation:")
    log("   - Slope_z was computed by the original authors from integer-degree slope (0°-76°).")
    log("   - This produces exactly 76 discrete quantisation bins (~0.5° resolution).")
    log("2. Spatial Independence Limitation:")
    log("   - The 500 m boundary exclusion zone prevents immediate cross-boundary pixel leakage.")
    log("   - However, regional environmental variables (NDVI, rainfall, elevation) possess spatial")
    log("     autocorrelation lengths spanning 5-50 km; this boundary buffer is a precaution, not")
    log("     proof of complete spatial independence.")
    log("3. Longitude Distribution Imbalance:")
    log("   - The held-out test split is geographically located in northern HP (predominantly Chamba).")
    log("   - Eastern high-altitude regions (Kinnaur, Spiti at 77°-79°E) are not represented in test.")
    log("4. Pseudo-Absence Uncertainty:")
    log("   - Class=0 locations are randomly sampled pseudo-absences from GEE, not confirmed stable ground.")

    log("\n" + "=" * 80)
    log("RECONCILIATION & VERIFICATION SUMMARY")
    log("=" * 80)
    log(f"  Raw input rows:                   {n_raw:6d}")
    log(f"  - Duplicate coordinate removals: -{n_removed:6d}")
    log(f"  Deduplicated working set:         {n_dedup:6d}")
    log(f"  - 500m boundary exclusion rows:  -{n_excl:6d}")
    log(f"  Remaining active modelling rows:  {n_total_splits:6d}")
    log(f"    - TRAIN rows:                   {n_train:6d} ({n_train/n_total_splits*100:.2f}%)")
    log(f"    - VAL rows:                     {n_val:6d} ({n_val/n_total_splits*100:.2f}%)")
    log(f"    - TEST rows:                    {n_test:6d} ({n_test/n_total_splits*100:.2f}%)")
    log(f"  Total accounted:                  {n_total_splits + n_excl + n_removed:6d} (Reconciliation: EXACT)")
    log("=" * 80)
    log("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY.")

    # 5. preprocessing_log.txt
    p_log = processed_dir / "preprocessing_log.txt"
    with open(p_log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nExecution log written to: {p_log}")


if __name__ == "__main__":
    run_pipeline()
