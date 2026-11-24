"""
Unit and integration tests for RockWatch Phase 3:
Phase 3A: Local Model Explainability (SHAP TreeExplainer, Waterfall Plot, Narratives)
Phase 3B: GeoTIFF / Raster Upload and Map Visualization Engine
"""

import unittest
import io
import math
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import streamlit as st
from scipy.special import expit

from src.app.explainability import (
    compute_local_shap,
    create_shap_waterfall_plot,
    generate_shap_narrative,
)
from src.app.raster_engine import (
    ensure_sample_dem,
    extract_raster_metadata,
    generate_raster_overlay,
    query_raster_pixel_at_coord,
)
from src.app.views_predict import FEATURE_NAMES, render_predict_view
from src.app.views_map import create_scatter_map, render_map_view, load_map_data


class TestPhase3ExplainabilityAndRaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("models/trained/gradient_boosting_v1.joblib")
        cls.sample_tif = ensure_sample_dem()

    def setUp(self):
        for k in list(st.session_state.keys()):
            del st.session_state[k]

    def test_01_shap_tree_explainer_mathematical_exactness(self):
        """
        Verify that SHAP TreeExplainer satisfies exact additive attribution:
        f(x) = base_value + sum(shap_values) == decision_function(x)
        sigmoid(f(x)) == predict_proba(x)[1]
        across distinct environmental terrain configurations.
        """
        scenarios = [
            # 1. Steep monsoon slope
            {
                "slope_deg": 38.0, "elevation_m": 1650, "aspect_deg": 210.0,
                "dist_river_m": 180, "flow_accum_cells": 850, "rainfall_index": 1.6,
                "ndvi_val": 0.38, "selected_lulc": 60,
                "vector": [math.sin(math.radians(210)), math.cos(math.radians(210)), -0.1, -0.1, 0.2, -0.4, 1.6, 0.8] + [0, 0, 0, 0, 0, 1, 0, 0, 0]
            },
            # 2. Arid alpine valley
            {
                "slope_deg": 34.0, "elevation_m": 3600, "aspect_deg": 45.0,
                "dist_river_m": 2500, "flow_accum_cells": 80, "rainfall_index": -1.5,
                "ndvi_val": -0.05, "selected_lulc": 70,
                "vector": [math.sin(math.radians(45)), math.cos(math.radians(45)), 0.3, 1.4, -0.4, -2.8, -1.5, 0.5] + [0, 0, 0, 0, 0, 0, 1, 0, 0]
            },
            # 3. Valley floor plain
            {
                "slope_deg": 8.0, "elevation_m": 950, "aspect_deg": 90.0,
                "dist_river_m": 60, "flow_accum_cells": 4500, "rainfall_index": -0.2,
                "ndvi_val": 0.52, "selected_lulc": 40,
                "vector": [1.0, 0.0, -0.12, -0.65, 0.8, 0.4, -0.2, -1.4] + [0, 0, 0, 1, 0, 0, 0, 0, 0]
            },
        ]

        for sc in scenarios:
            feat_df = pd.DataFrame([sc["vector"]], columns=FEATURE_NAMES)
            shap_data, err = compute_local_shap(self.model, feat_df, sc)
            self.assertIsNone(err, f"SHAP error: {err}")
            self.assertIsNotNone(shap_data)

            # Mathematical exactness checks
            expected_decision = float(self.model.decision_function(feat_df.values)[0])
            expected_proba = float(self.model.predict_proba(feat_df.values)[0, 1])

            self.assertAlmostEqual(shap_data["fx_log_odds"], expected_decision, places=5)
            self.assertAlmostEqual(shap_data["derived_prob"], expected_proba, places=5)

            # Sum of SHAP values + base value
            total_sum = shap_data["base_value"] + shap_data["total_shap"]
            self.assertAlmostEqual(total_sum, expected_decision, places=5)

    def test_02_shap_waterfall_and_narrative_generation(self):
        """Test interactive horizontal waterfall chart and scientific narrative."""
        feat_df = pd.DataFrame([np.zeros(17)], columns=FEATURE_NAMES)
        raw_inputs = {"slope_deg": 35.0, "elevation_m": 2100, "dist_river_m": 250, "aspect_deg": 180, "selected_lulc": 50}
        shap_data, err = compute_local_shap(self.model, feat_df, raw_inputs)
        self.assertIsNone(err)

        fig = create_shap_waterfall_plot(shap_data, max_features=8)
        self.assertIsNotNone(fig)
        self.assertEqual(fig.data[0].type, "waterfall")
        self.assertEqual(fig.data[0].orientation, "h")

        narrative = generate_shap_narrative(shap_data)
        self.assertIn("regional baseline probability", narrative)
        self.assertIn("log-odds", narrative)
        self.assertTrue(len(narrative) > 50)

    def test_03_valid_geotiff_metadata_extraction(self):
        """Test extraction of CRS, dimensions, resolution, bounds, and pixel stats from GeoTIFF."""
        meta, err = extract_raster_metadata(self.sample_tif)
        self.assertIsNone(err, f"Metadata extraction failed: {err}")
        self.assertIsNotNone(meta)

        self.assertEqual(meta["width"], 150)
        self.assertEqual(meta["height"], 120)
        self.assertEqual(meta["bands"], 1)
        self.assertEqual(meta["epsg"], 4326)
        self.assertEqual(meta["nodata"], -9999.0)

        # Bounds check
        w84 = meta["wgs84_bounds"]
        self.assertAlmostEqual(w84["west"], 75.5, places=2)
        self.assertAlmostEqual(w84["east"], 79.2, places=2)
        self.assertAlmostEqual(w84["south"], 30.2, places=2)
        self.assertAlmostEqual(w84["north"], 33.3, places=2)
        self.assertTrue(meta["is_himachal_intersect"])

        # Statistics
        stats = meta["stats"]
        self.assertGreater(stats["min"], 400.0)
        self.assertLess(stats["max"], 6500.0)
        self.assertAlmostEqual(stats["valid_fraction"], 1.0)

    def test_04_invalid_and_corrupt_raster_handling(self):
        """Test that invalid bytes or non-raster files return helpful errors without crashing."""
        corrupt_bytes = b"This is a dummy text file, not a GeoTIFF raster!"
        meta, err = extract_raster_metadata(corrupt_bytes)
        self.assertIsNone(meta)
        self.assertIsNotNone(err)
        self.assertIn("Raster inspection failed", err)

        overlay, err = generate_raster_overlay(corrupt_bytes)
        self.assertIsNone(overlay)
        self.assertIsNotNone(err)

    def test_05_projected_crs_reprojection_utm(self):
        """
        Create a synthetic raster in UTM Zone 43N (EPSG:32643) and verify
        that metadata and overlay engine reproject bounds and imagery into WGS84.
        """
        ny, nx = 40, 50
        # UTM 43N bounds in Himachal Pradesh
        left, bottom, right, top = 650000, 3450000, 750000, 3550000
        elev = np.linspace(800, 3500, ny * nx, dtype=np.float32).reshape((ny, nx))
        transform_utm = from_bounds(left, bottom, right, top, nx, ny)

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=ny,
                width=nx,
                count=1,
                dtype="float32",
                crs="EPSG:32643",
                transform=transform_utm,
                nodata=-9999.0,
            ) as dst:
                dst.write(elev, 1)
            data_bytes = memfile.read()

        meta, err = extract_raster_metadata(data_bytes)
        self.assertIsNone(err)
        self.assertEqual(meta["epsg"], 32643)
        # Check that WGS84 bounds fall inside northern India bounds (~76-78°E, 31-32°N)
        w84 = meta["wgs84_bounds"]
        self.assertTrue(75.0 <= w84["west"] <= 78.0)
        self.assertTrue(76.5 <= w84["east"] <= 79.0)
        self.assertTrue(30.0 <= w84["south"] <= 33.0)

        overlay, err = generate_raster_overlay(data_bytes, colormap_name="viridis")
        self.assertIsNone(err)
        self.assertIsNotNone(overlay)
        self.assertEqual(len(overlay["coordinates"]), 4)

    def test_06_nodata_handling_and_pixel_masking(self):
        """Verify NoData pixels are masked out and return informative query messages."""
        ny, nx = 30, 30
        data = np.full((ny, nx), 2000.0, dtype=np.float32)
        # Set half the pixels to NoData
        data[:15, :] = -9999.0
        transform_wgs = from_bounds(76.0, 31.0, 78.0, 33.0, nx, ny)

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=ny,
                width=nx,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform_wgs,
                nodata=-9999.0,
            ) as dst:
                dst.write(data, 1)
            data_bytes = memfile.read()

        meta, err = extract_raster_metadata(data_bytes)
        self.assertIsNone(err)
        self.assertAlmostEqual(meta["stats"]["valid_fraction"], 0.5, places=2)

        overlay, err = generate_raster_overlay(data_bytes, colormap_name="terrain")
        self.assertIsNone(err)
        self.assertIsNotNone(overlay)

        # Query coordinate inside the NoData region (lat 32.5, lon 77.0)
        val, p_err = query_raster_pixel_at_coord(data_bytes, 32.5, 77.0)
        self.assertIsNone(val)
        self.assertIn("NoData pixel", p_err)

        # Query coordinate inside valid region (lat 31.5, lon 77.0)
        val, p_err = query_raster_pixel_at_coord(data_bytes, 31.5, 77.0)
        self.assertIsNone(p_err)
        self.assertAlmostEqual(val, 2000.0)

    def test_07_pixel_query_and_boundary_checks(self):
        """Test pixel querying at known coordinates and out-of-bounds rejection."""
        # Query Shimla coordinate on sample DEM
        val, err = query_raster_pixel_at_coord(self.sample_tif, 31.1048, 77.1734)
        self.assertIsNone(err)
        self.assertTrue(1000.0 <= val <= 5000.0, f"Expected realistic elevation, got {val}")

        # Query location far outside Himachal (e.g. Mumbai at 19.0760, 72.8777)
        val_oob, err_oob = query_raster_pixel_at_coord(self.sample_tif, 19.0760, 72.8777)
        self.assertIsNone(val_oob)
        self.assertIn("outside the raster boundary", err_oob)

    def test_08_large_raster_downsampling_safety(self):
        """Test safe downsampling of large rasters for fluid web rendering."""
        ny, nx = 500, 500
        data = np.linspace(500, 5000, ny * nx, dtype=np.float32).reshape((ny, nx))
        transform_wgs = from_bounds(75.5, 30.5, 78.5, 33.5, nx, ny)

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=ny,
                width=nx,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform_wgs,
                nodata=-9999.0,
            ) as dst:
                dst.write(data, 1)
            data_bytes = memfile.read()

        overlay, err = generate_raster_overlay(data_bytes, max_dim=250)
        self.assertIsNone(err)
        self.assertIsNotNone(overlay)
        self.assertTrue(overlay["source"].startswith("data:image/png;base64,"))

    def test_09_map_view_rendering_with_raster_layer(self):
        """Test that create_scatter_map and render_map_view handle raster overlays seamlessly."""
        df, _ = load_map_data()
        self.assertIsNotNone(df)
        sample_df = df.head(50)

        overlay, err = generate_raster_overlay(self.sample_tif, colormap_name="terrain")
        self.assertIsNone(err)

        fig = create_scatter_map(sample_df, raster_layer=overlay)
        self.assertIsNotNone(fig)
        # Check map layout has layers
        if hasattr(fig.layout, "map") and fig.layout.map:
            layers = fig.layout.map.layers
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0]["sourcetype"], "image")

        # Test view render execution
        try:
            render_map_view()
        except Exception as e:
            self.fail(f"render_map_view with raster components raised error: {e}")

    def test_10_predict_view_rendering_with_shap(self):
        """Test that render_predict_view renders when latest_shap is populated."""
        st.session_state["latest_prediction"] = {
            "timestamp": "2026-11-24 12:00:00 UTC",
            "score": 0.68,
            "tier_name": "High",
            "tier_class": "tier-high",
            "tier_color": "#f97316",
            "tier_desc": "High geomorphic conditioning.",
            "elevation_m": 1750,
            "slope_deg": 36.0,
            "aspect_deg": 200.0,
            "selected_lulc": 60,
            "rainfall_index": 1.4,
            "ndvi_val": 0.35,
            "dist_river_m": 190,
            "flow_accum_cells": 600,
        }

        feat_df = pd.DataFrame([np.zeros(17)], columns=FEATURE_NAMES)
        shap_data, _ = compute_local_shap(self.model, feat_df, st.session_state["latest_prediction"])
        st.session_state["latest_shap"] = shap_data

        try:
            render_predict_view()
        except Exception as e:
            self.fail(f"render_predict_view with SHAP components raised error: {e}")


if __name__ == "__main__":
    unittest.main()
