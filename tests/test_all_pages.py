"""
Test script to verify all 6 navigation views render without exceptions.
"""

import unittest
from unittest.mock import patch, MagicMock
import app
from src.app.views_home import render_home_view
from src.app.views_map import render_map_view, load_map_data
from src.app.views_predict import render_predict_view
from src.app.views_performance import render_performance_view
from src.app.views_regional import render_regional_view
from src.app.views_about import render_about_view


class TestAllPages(unittest.TestCase):
    def test_01_home_view(self):
        """Verify home view renders without error."""
        try:
            render_home_view()
        except Exception as e:
            self.fail(f"render_home_view raised exception: {e}")

    def test_02_map_view(self):
        """Verify map view renders without error."""
        try:
            render_map_view()
        except Exception as e:
            self.fail(f"render_map_view raised exception: {e}")

    def test_03_predict_view(self):
        """Verify predict view renders without error."""
        try:
            render_predict_view()
        except Exception as e:
            self.fail(f"render_predict_view raised exception: {e}")

    def test_04_performance_view(self):
        """Verify performance view renders without error."""
        try:
            render_performance_view()
        except Exception as e:
            self.fail(f"render_performance_view raised exception: {e}")

    def test_05_regional_view(self):
        """Verify regional view renders without error."""
        try:
            render_regional_view()
        except Exception as e:
            self.fail(f"render_regional_view raised exception: {e}")

    def test_06_about_view(self):
        """Verify about view renders without error."""
        try:
            render_about_view()
        except Exception as e:
            self.fail(f"render_about_view raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
