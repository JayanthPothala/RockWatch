"""
Unit and integration tests for RockWatch Phase 2: Assessment Report Generation Engine.
"""

import unittest
import json
import pandas as pd
import streamlit as st
from src.app.report_generator import (
    generate_markdown_report,
    generate_html_report,
    export_regional_summary_csv,
    export_regional_geojson,
)
from src.app.views_predict import render_predict_view
from src.app.views_regional import load_regional_data, render_regional_view


class TestPhase2Reports(unittest.TestCase):
    def setUp(self):
        # Clear session state
        for k in list(st.session_state.keys()):
            del st.session_state[k]

        self.sample_assessment = {
            "timestamp": "2026-11-24 12:00:00 UTC",
            "score": 0.6543,
            "tier_name": "High",
            "tier_color": "#f97316",
            "site_name": "Kangra Steep Monsoon Ridge",
            "inputs": {
                "elevation_m": 1650,
                "slope_deg": 38.0,
                "aspect_deg": 210.0,
                "selected_lulc": 60,
                "lulc_label": "Bare Ground / Sparse Rock (LULC 60)",
                "rainfall_index": 1.6,
                "ndvi_val": 0.38,
                "dist_river_m": 180,
                "flow_accum_cells": 850,
            },
        }

    def test_01_markdown_report_structure(self):
        """Verify Markdown report content, layout, metadata, and scientific disclaimers."""
        md = generate_markdown_report(self.sample_assessment)
        self.assertIn("# RockWatch: Landslide Susceptibility Assessment Report", md)
        self.assertIn("Kangra Steep Monsoon Ridge", md)
        self.assertIn("65.43%", md)
        self.assertIn("HIGH HAZARD", md)
        self.assertIn("Gradient Boosting Classifier", md)
        self.assertIn("Threshold $T = 0.37$", md)
        self.assertIn("Elevation", md)
        self.assertIn("1650 m", md)
        self.assertIn("38.0°", md)
        self.assertIn("Planning & Screening Advisories", md)
        self.assertIn("CRITICAL SCIENTIFIC & DISASTER MANAGEMENT NOTICE", md)
        # Verify strict negative requirements - disclaimer confirms it is NOT statutory authority
        self.assertNotIn("immediate evacuation order", md.lower())
        self.assertNotIn("mandatory construction ban", md.lower())
        self.assertIn("statutory authority", md.lower())

    def test_02_html_report_structure(self):
        """Verify HTML report tags, DOCTYPE, tables, and styling."""
        html = generate_html_report(self.sample_assessment)
        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIn("RockWatch: Landslide Susceptibility Assessment", html)
        self.assertIn("65.4%", html)
        self.assertIn("HIGH HAZARD", html)
        self.assertIn("<table", html)
        self.assertIn("1650 m", html)
        self.assertIn("SCIENTIFIC & DISASTER MANAGEMENT DISCLAIMER", html)
        self.assertIn("</html>", html)

    def test_03_csv_export(self):
        """Verify CSV export converts dataframes to valid CSV text with preserved headers."""
        sample_df = pd.DataFrame({
            "District": ["Shimla", "Mandi"],
            "Events": [1026, 932],
            "Mean_Score": [0.512, 0.485],
        })
        csv_text = export_regional_summary_csv(sample_df)
        lines = csv_text.strip().split("\n")
        self.assertEqual(lines[0].strip(), "District,Events,Mean_Score")
        self.assertIn("Shimla,1026,0.512", lines[1])

    def test_04_geojson_export_rfc7946(self):
        """Verify GeoJSON export adheres to RFC 7946 with valid Point geometries [lon, lat]."""
        df = load_regional_data().head(50)
        gj_str = export_regional_geojson(df)
        gj = json.loads(gj_str)

        self.assertEqual(gj.get("type"), "FeatureCollection")
        self.assertEqual(len(gj["features"]), 50)

        f0 = gj["features"][0]
        self.assertEqual(f0["type"], "Feature")
        self.assertEqual(f0["geometry"]["type"], "Point")
        coords = f0["geometry"]["coordinates"]
        self.assertEqual(len(coords), 2)
        # Longitude in [75.0, 81.0], Latitude in [28.5, 34.0]
        lon, lat = coords[0], coords[1]
        self.assertTrue(75.0 <= lon <= 81.0, f"Longitude {lon} out of bounds")
        self.assertTrue(28.5 <= lat <= 34.0, f"Latitude {lat} out of bounds")

        props = f0["properties"]
        self.assertIn("susceptibility_score", props)
        self.assertIn("susceptibility_class", props)
        self.assertIn("elevation_m", props)
        self.assertIn("slope_deg", props)

    def test_05_regional_filtering_and_exports(self):
        """Test corridor, district, and tier filtering for exports."""
        df = load_regional_data()

        # Test Shimla district filter
        shimla_df = df[df["district"] == "Shimla"]
        self.assertGreater(len(shimla_df), 0)
        gj_shimla = json.loads(export_regional_geojson(shimla_df))
        self.assertEqual(len(gj_shimla["features"]), len(shimla_df))
        for feat in gj_shimla["features"]:
            self.assertEqual(feat["properties"]["district"], "Shimla")

        # Test High / Very High tiers only
        high_df = df[df["susceptibility_class"].isin(["High", "Very High"])]
        gj_high = json.loads(export_regional_geojson(high_df.head(100)))
        for feat in gj_high["features"]:
            self.assertIn(feat["properties"]["susceptibility_class"], ["High", "Very High"])

    def test_06_views_rendering_with_reports(self):
        """Verify views_predict and views_regional render with report download buttons active."""
        # Set latest_prediction in session state
        st.session_state["latest_prediction"] = {
            "timestamp": "2026-11-24 12:00:00 UTC",
            "score": 0.72,
            "tier_name": "Very High",
            "tier_class": "tier-very-high",
            "tier_color": "#ef4444",
            "tier_desc": "Severe predisposition.",
            "elevation_m": 2100,
            "slope_deg": 35.0,
            "aspect_deg": 180.0,
            "selected_lulc": 10,
            "rainfall_index": 0.8,
            "ndvi_val": 0.42,
            "dist_river_m": 250,
            "flow_accum_cells": 300,
        }

        try:
            render_predict_view()
            render_regional_view()
        except Exception as e:
            self.fail(f"View rendering with export components failed: {e}")


if __name__ == "__main__":
    unittest.main()
