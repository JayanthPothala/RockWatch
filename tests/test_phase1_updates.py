"""
Unit and integration tests for RockWatch Phase 1: Scientific Alignment & UI Corrections.
"""

import unittest
import pandas as pd
import numpy as np
import streamlit as st
from src.app.views_home import load_home_statistics, render_home_view
from src.app.views_predict import (
    load_prediction_artifacts,
    apply_preset,
    render_predict_view,
    PRESETS,
    FEATURE_NAMES,
    GLOBAL_PARAMS,
)
from src.app.views_map import render_map_view
from src.app.views_performance import render_performance_view
from src.app.views_regional import render_regional_view
from src.app.views_about import render_about_view


class TestPhase1ScientificAlignment(unittest.TestCase):
    def setUp(self):
        # Reset session state for clean testing
        for k in list(st.session_state.keys()):
            del st.session_state[k]

    def test_01_dashboard_counts_against_data(self):
        """Verify dashboard statistics reconcile with actual data files."""
        stats = load_home_statistics()
        self.assertEqual(stats["n_total"], 11737, "Total rows mismatch")
        self.assertEqual(stats["n_events"], 5859, "Landslide events count mismatch")
        self.assertEqual(stats["n_pseudo"], 5878, "Pseudo-absence count mismatch")
        self.assertEqual(stats["n_events"] + stats["n_pseudo"], stats["n_total"])
        self.assertEqual(stats["min_elev"], 246, "Min elevation mismatch")
        self.assertEqual(stats["max_elev"], 6430, "Max elevation mismatch")
        self.assertEqual(stats["n_train"], 8188, "Train split mismatch")
        self.assertEqual(stats["n_val"], 1731, "Val split mismatch")
        self.assertEqual(stats["n_test"], 1750, "Test split mismatch")

    def test_02_all_scenario_presets_session_state(self):
        """Test session state binding across all 4 representative terrain presets."""
        for pkey, pdata in PRESETS.items():
            apply_preset(pkey)
            self.assertEqual(st.session_state["pred_elevation"], pdata["elevation"])
            self.assertEqual(st.session_state["pred_slope"], float(pdata["slope"]))
            self.assertEqual(st.session_state["pred_aspect"], float(pdata["aspect"]))
            self.assertEqual(st.session_state["pred_rainfall_index"], float(pdata["rainfall_index"]))
            self.assertEqual(st.session_state["pred_ndvi"], float(pdata["ndvi"]))
            self.assertEqual(st.session_state["pred_dist_river"], pdata["dist_river"])
            self.assertEqual(st.session_state["pred_flow_accum"], pdata["flow_accum"])
            self.assertEqual(st.session_state["pred_lulc"], pdata["lulc"])

    def test_03_trained_model_fresh_predictions(self):
        """Verify model produces fresh, responsive predictions for different terrain profiles."""
        model, scaler_params = load_prediction_artifacts()
        self.assertIsNotNone(model, "Model should be loaded")
        self.assertIsNotNone(scaler_params, "Scaler params should be loaded")

        def predict_point(elev, slope, aspect, rainfall, ndvi, dist_river, flow_accum, lulc):
            import math
            rad = math.radians(aspect) if slope > 0 else 0.0
            sin_asp = math.sin(rad) if slope > 0 else 0.0
            cos_asp = math.cos(rad) if slope > 0 else 0.0

            elev_z = (elev - GLOBAL_PARAMS["elevation_m"]["mean"]) / GLOBAL_PARAMS["elevation_m"]["std"]
            slope_z = (round(slope) - GLOBAL_PARAMS["slope_deg"]["mean"]) / GLOBAL_PARAMS["slope_deg"]["std"]
            dist_river_z = -0.1314 + (dist_river / 4389.0) * 0.8
            ndvi_z = (ndvi - 0.45) / 0.18
            rainfall_z = rainfall
            flow_accum_z = -0.5783 + (math.log1p(flow_accum) / 5.0) * 0.8

            sp = scaler_params["features"]
            elev_corr = (elev_z - sp["Elevation_z"]["train_mean"]) / sp["Elevation_z"]["train_std"]
            slope_corr = (slope_z - sp["Slope_z"]["train_mean"]) / sp["Slope_z"]["train_std"]
            dist_corr = (dist_river_z - sp["Dist_River_z"]["train_mean"]) / sp["Dist_River_z"]["train_std"]
            ndvi_corr = (ndvi_z - sp["NDVI_z"]["train_mean"]) / sp["NDVI_z"]["train_std"]
            rain_corr = (rainfall_z - sp["Rainfall_z"]["train_mean"]) / sp["Rainfall_z"]["train_std"]
            flow_corr = (flow_accum_z - sp["FlowAccumulation_z"]["train_mean"]) / sp["FlowAccumulation_z"]["train_std"]

            lulc_vec = [1 if lulc == c else 0 for c in [10, 20, 30, 40, 50, 60, 70, 80, 100]]
            x = [sin_asp, cos_asp, dist_corr, elev_corr, flow_corr, ndvi_corr, rain_corr, slope_corr] + lulc_vec
            df = pd.DataFrame([x], columns=FEATURE_NAMES)
            return float(model.predict_proba(df)[0, 1])

        # Kangra steep slope vs Kullu gentle valley floor
        p_steep = predict_point(1650, 38.0, 210.0, 1.6, 0.38, 180, 850, 60)
        p_flat = predict_point(950, 8.0, 90.0, -0.2, 0.52, 60, 4500, 40)

        self.assertGreater(p_steep, 0.50, f"Steep slope should exhibit high score (got {p_steep:.3f})")
        self.assertLess(p_flat, 0.35, f"Flat terrain should exhibit low score (got {p_flat:.3f})")
        self.assertGreater(p_steep, p_flat, "Steep terrain must have higher susceptibility than flat basin.")

    def test_04_all_navigation_views_load(self):
        """Verify all 6 pages render without raising exceptions."""
        try:
            render_home_view()
            render_predict_view()
            render_map_view()
            render_performance_view()
            render_regional_view()
            render_about_view()
        except Exception as e:
            self.fail(f"View rendering failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
