"""Unit tests for RockWatch preprocessor and feature engineering."""
import unittest
import numpy as np
import pandas as pd
from src.features.engineer import compute_aspect_trig, add_water_body_flag
from src.evaluation.metrics import evaluate_predictions, find_optimal_threshold


class TestRockWatchPipeline(unittest.TestCase):
    def test_aspect_trigonometry(self):
        df = pd.DataFrame({
            "aspect_deg": [0.0, 90.0, 180.0, 270.0, 45.0],
            "slope_deg": [0.0, 20.0, 25.0, 30.0, 0.0],
        })
        res = compute_aspect_trig(df)

        # Flat areas (slope=0) must be 0.0
        self.assertEqual(res.loc[0, "sin_aspect"], 0.0)
        self.assertEqual(res.loc[0, "cos_aspect"], 0.0)
        self.assertEqual(res.loc[4, "sin_aspect"], 0.0)
        self.assertEqual(res.loc[4, "cos_aspect"], 0.0)

        # 90 deg -> sin=1, cos=0
        self.assertAlmostEqual(res.loc[1, "sin_aspect"], 1.0, places=5)
        self.assertAlmostEqual(res.loc[1, "cos_aspect"], 0.0, places=5)

        # 180 deg -> sin=0, cos=-1
        self.assertAlmostEqual(res.loc[2, "sin_aspect"], 0.0, places=5)
        self.assertAlmostEqual(res.loc[2, "cos_aspect"], -1.0, places=5)

    def test_water_body_flag(self):
        df = pd.DataFrame({
            "LULC_80": [1, 1, 0, 0],
            "class": [1, 0, 1, 0],
        })
        res = add_water_body_flag(df)
        self.assertEqual(res["flag_water_lulc"].tolist(), [1, 0, 0, 0])

    def test_metrics_evaluation(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        res = evaluate_predictions(y_true, y_prob, threshold=0.50)

        self.assertEqual(res["auc_roc"], 1.0)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["f1"], 1.0)
        self.assertLess(res["brier_score"], 0.05)


if __name__ == "__main__":
    unittest.main()
