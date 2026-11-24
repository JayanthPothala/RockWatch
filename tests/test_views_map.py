"""
Tests for RockWatch Susceptibility Map View and Map Generation
"""

import unittest
import pandas as pd
import numpy as np
from src.app.views_map import load_map_data, create_scatter_map, TIER_COLOR_MAP, REQUIRED_COLUMNS


class TestSusceptibilityMapView(unittest.TestCase):
    def setUp(self):
        self.df, self.status = load_map_data()

    def test_01_data_loading(self):
        """Verify real geographical dataset loads cleanly without nulls or duplicates."""
        self.assertIsNotNone(self.df, "Map data frame should not be None.")
        self.assertEqual(len(self.df), 11737, "Dataset must contain exactly 11,737 records.")
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.df.columns, f"Required column {col} missing.")

        # Coordinate checks
        self.assertFalse(self.df["latitude"].isnull().any(), "Latitude has null values.")
        self.assertFalse(self.df["longitude"].isnull().any(), "Longitude has null values.")
        self.assertEqual(
            self.df.duplicated(subset=["latitude", "longitude"]).sum(),
            0,
            "Dataset should not contain duplicate coordinates."
        )

    def test_02_coordinate_bounds(self):
        """Verify all coordinates fall within the geographical domain of Himachal Pradesh."""
        lat_min, lat_max = self.df["latitude"].min(), self.df["latitude"].max()
        lon_min, lon_max = self.df["longitude"].min(), self.df["longitude"].max()

        self.assertGreaterEqual(lat_min, 29.5, f"Latitude {lat_min} below Himachal border.")
        self.assertLessEqual(lat_max, 33.5, f"Latitude {lat_max} above Himachal border.")
        self.assertGreaterEqual(lon_min, 75.0, f"Longitude {lon_min} west of Himachal border.")
        self.assertLessEqual(lon_max, 80.5, f"Longitude {lon_max} east of Himachal border.")

    def test_03_susceptibility_tiers(self):
        """Verify all 4 susceptibility tiers are present with valid probability scores [0, 1]."""
        self.assertTrue(self.df["susceptibility_score"].between(0.0, 1.0).all(), "Scores must be in [0, 1].")
        tier_counts = self.df["susceptibility_class"].value_counts()
        for tier in ["Low", "Moderate", "High", "Very High"]:
            self.assertIn(tier, tier_counts, f"Tier {tier} missing from dataset.")
            self.assertGreater(tier_counts[tier], 0, f"Tier {tier} has zero records.")

    def test_04_create_scatter_map_styles(self):
        """Verify modern Plotly map creation works with all supported basemaps."""
        sub_df = self.df.head(150)
        for style in ["open-street-map", "carto-positron", "carto-darkmatter"]:
            fig = create_scatter_map(sub_df, map_style=style)
            self.assertIsNotNone(fig, f"Figure was None for style {style}.")
            self.assertGreater(len(fig.data), 0, f"Figure had no data traces for style {style}.")

    def test_05_category_filters(self):
        """Verify each susceptibility category filter produces real plotted coordinates."""
        for tier in ["Low", "Moderate", "High", "Very High"]:
            sub_tier = self.df[self.df["susceptibility_class"] == tier]
            self.assertGreater(len(sub_tier), 0, f"Tier {tier} should have points.")
            fig = create_scatter_map(sub_tier.head(100), map_style="open-street-map")
            total_points = sum(len(tr.lat) for tr in fig.data if hasattr(tr, "lat") and tr.lat is not None)
            self.assertGreater(total_points, 0, f"No points mapped for category {tier}.")


if __name__ == "__main__":
    unittest.main()
