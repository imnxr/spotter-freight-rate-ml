"""Unit tests for src/features.py module."""

import unittest
import numpy as np
import pandas as pd

from src.data import load_december_chart_inputs, load_train_test, load_validation
from src.features import (
    DERIVED_FEATURE_COLUMNS,
    HISTORICAL_FEATURE_COLUMNS,
    calculate_rate_per_mile_target,
    compute_historical_pricing_features,
    engineer_features,
    reconstruct_rate_from_rpm,
)




class TestFeaturesModule(unittest.TestCase):
    """Test suite for feature engineering functionality."""

    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "load_id": ["TR-000001", "TR-000002"],
                "pickup": ["Richmond", "Dallas"],
                "delivery": ["Baltimore", "Nashville"],
                "pickup_lat": [38.09122, 31.83025],
                "pickup_lon": [-76.78906, -94.38343],
                "delivery_lat": [38.16908, 35.29479],
                "delivery_lon": [-72.74564, -88.08915],
                "distance": [274.3, 541.9],
                "equipment": ["Dry Van", "Reefer"],
                "weight": [30658.0, 35183.0],
                "date": ["2025-01-01", "2025-01-15"],
                "market_index": [0.95684, 0.98480],
                "quote_signal": [2.39595, 2.56300],
                "posted_rate": [645.41, 1380.28],
            }
        )

    def test_input_not_mutated(self):
        original_cols = list(self.sample_df.columns)
        original_shape = self.sample_df.shape

        out = engineer_features(self.sample_df)

        self.assertIsNot(out, self.sample_df)
        self.assertEqual(list(self.sample_df.columns), original_cols)
        self.assertEqual(self.sample_df.shape, original_shape)
        self.assertIn("load_id", self.sample_df.columns)

    def test_date_features_created(self):
        out = engineer_features(self.sample_df)

        expected_date_cols = [
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_year",
            "week_of_year",
            "days_since_start",
        ]
        for col in expected_date_cols:
            self.assertIn(col, out.columns)

        # Row 0: 2025-01-01 (Wednesday -> day_of_week=2, day=1, month=1, year=2025, days_since_start=0)
        self.assertEqual(out["year"].iloc[0], 2025)
        self.assertEqual(out["month"].iloc[0], 1)
        self.assertEqual(out["day"].iloc[0], 1)
        self.assertEqual(out["day_of_week"].iloc[0], 2)
        self.assertEqual(out["day_of_year"].iloc[0], 1)
        self.assertEqual(out["week_of_year"].iloc[0], 1)
        self.assertEqual(out["days_since_start"].iloc[0], 0)

        # Row 1: 2025-01-15 (14 days after start)
        self.assertEqual(out["days_since_start"].iloc[1], 14)

    def test_route_feature_correct(self):
        out = engineer_features(self.sample_df)
        self.assertIn("route", out.columns)
        self.assertEqual(out["route"].iloc[0], "Richmond__Baltimore")
        self.assertEqual(out["route"].iloc[1], "Dallas__Nashville")

    def test_coordinate_features_correct(self):
        out = engineer_features(self.sample_df)

        expected_lat_diff = abs(38.09122 - 38.16908)
        expected_lon_diff = abs(-76.78906 - (-72.74564))
        expected_mid_lat = (38.09122 + 38.16908) / 2.0
        expected_mid_lon = (-76.78906 + (-72.74564)) / 2.0

        np.testing.assert_almost_equal(out["abs_lat_diff"].iloc[0], expected_lat_diff)
        np.testing.assert_almost_equal(out["abs_lon_diff"].iloc[0], expected_lon_diff)
        np.testing.assert_almost_equal(out["midpoint_lat"].iloc[0], expected_mid_lat)
        np.testing.assert_almost_equal(out["midpoint_lon"].iloc[0], expected_mid_lon)

    def test_load_id_excluded_by_default(self):
        out = engineer_features(self.sample_df, drop_load_id=True)
        self.assertNotIn("load_id", out.columns)

    def test_load_id_preserved_when_flag_false(self):
        out = engineer_features(self.sample_df, drop_load_id=False)
        self.assertIn("load_id", out.columns)

    def test_existing_columns_preserved(self):
        out = engineer_features(self.sample_df)
        retained = ["pickup", "delivery", "equipment", "distance", "weight", "market_index", "quote_signal"]
        for col in retained:
            self.assertIn(col, out.columns)
            self.assertEqual(out[col].iloc[0], self.sample_df[col].iloc[0])

    def test_missing_values_do_not_crash(self):
        df_with_nans = pd.DataFrame(
            {
                "load_id": ["TR-001", "TR-002", "TR-003"],
                "pickup": ["Richmond", np.nan, "Dallas"],
                "delivery": [np.nan, "Baltimore", "Nashville"],
                "pickup_lat": [38.0, np.nan, 31.8],
                "pickup_lon": [-76.7, np.nan, -94.3],
                "delivery_lat": [np.nan, 38.1, 35.2],
                "delivery_lon": [np.nan, -72.7, -88.0],
                "distance": [274.3, np.nan, 541.9],
                "equipment": ["Dry Van", "Reefer", np.nan],
                "weight": [np.nan, 35000.0, np.nan],
                "date": ["2025-01-01", None, "2025-05-10"],
                "market_index": [np.nan, 0.98, np.nan],
                "quote_signal": [2.39, np.nan, 2.56],
            }
        )

        out = engineer_features(df_with_nans)
        self.assertEqual(len(out), 3)
        self.assertTrue(pd.isna(out["route"].iloc[0]))
        self.assertTrue(pd.isna(out["route"].iloc[1]))
        self.assertEqual(out["route"].iloc[2], "Dallas__Nashville")
        self.assertTrue(pd.isna(out["abs_lat_diff"].iloc[0]))
        self.assertTrue(pd.isna(out["year"].iloc[1]))
        self.assertEqual(out["year"].iloc[2], 2025)

    def test_real_datasets(self):
        train_df = load_train_test()
        feat_train = engineer_features(train_df)
        self.assertEqual(len(feat_train), len(train_df))
        self.assertNotIn("load_id", feat_train.columns)

        val_df = load_validation()
        feat_val = engineer_features(val_df)
        self.assertEqual(len(feat_val), len(val_df))

        dec_df = load_december_chart_inputs()
        feat_dec = engineer_features(dec_df)
        self.assertEqual(len(feat_dec), len(dec_df))

    def test_row_cannot_see_its_own_target(self):
        """Verify that a row cannot observe its own target in self-expanding history."""
        toy_df = pd.DataFrame({
            "load_id": ["TR-001"],
            "pickup": ["CityA"],
            "delivery": ["CityB"],
            "equipment": ["Dry Van"],
            "distance": [100.0],
            "date": ["2025-01-01"],
            "posted_rate": [9999.0],
        })
        out = compute_historical_pricing_features(toy_df)
        self.assertEqual(out["route_prev_count"].iloc[0], 0)
        self.assertTrue(pd.isna(out["route_hist_median_rate"].iloc[0]))
        self.assertTrue(pd.isna(out["route_hist_mean_rate"].iloc[0]))
        self.assertTrue(pd.isna(out["route_hist_median_rate_per_mile"].iloc[0]))

    def test_row_cannot_see_later_target(self):
        """Verify that earlier rows cannot be influenced by later target observations."""
        toy_df = pd.DataFrame({
            "load_id": ["TR-001", "TR-002", "TR-003"],
            "pickup": ["CityA", "CityA", "CityA"],
            "delivery": ["CityB", "CityB", "CityB"],
            "equipment": ["Dry Van", "Dry Van", "Dry Van"],
            "distance": [100.0, 100.0, 100.0],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "posted_rate": [100.0, 200.0, 50000.0],  # TR-003 has extreme future spike
        })
        out = compute_historical_pricing_features(toy_df)
        
        # Day 1: sees 0 past rows -> NaN
        self.assertEqual(out["route_prev_count"].iloc[0], 0)
        self.assertTrue(pd.isna(out["route_hist_median_rate"].iloc[0]))
        
        # Day 2: sees only Day 1 ($100.0)
        self.assertEqual(out["route_prev_count"].iloc[1], 1)
        self.assertEqual(out["route_hist_median_rate"].iloc[1], 100.0)
        
        # Day 3: sees Day 1 & Day 2 (median = 150.0), NOT itself ($50,000)
        self.assertEqual(out["route_prev_count"].iloc[2], 2)
        self.assertEqual(out["route_hist_median_rate"].iloc[2], 150.0)

    def test_unseen_routes_return_missing_historical_features(self):
        """Verify that routes not present in historical data return NaN and count 0."""
        hist_df = pd.DataFrame({
            "load_id": ["TR-001"],
            "pickup": ["CityA"],
            "delivery": ["CityB"],
            "equipment": ["Dry Van"],
            "distance": [100.0],
            "date": ["2025-01-01"],
            "posted_rate": [500.0],
        })
        val_df = pd.DataFrame({
            "load_id": ["TR-002"],
            "pickup": ["CityX"],
            "delivery": ["CityY"],  # Completely new unseen route
            "equipment": ["Dry Van"],
            "distance": [300.0],
            "date": ["2025-02-01"],
        })
        out = compute_historical_pricing_features(val_df, history_df=hist_df)
        self.assertEqual(out["route_prev_count"].iloc[0], 0)
        self.assertTrue(pd.isna(out["route_hist_median_rate"].iloc[0]))
        self.assertTrue(pd.isna(out["route_hist_mean_rate"].iloc[0]))
        self.assertTrue(pd.isna(out["route_hist_median_rate_per_mile"].iloc[0]))
        self.assertTrue(pd.isna(out["equipment_route_hist_median_rate"].iloc[0]))

    def test_validation_features_contain_no_validation_targets(self):
        """Verify that modifying validation targets has zero effect on computed historical features."""
        train_df = pd.DataFrame({
            "load_id": ["TR-001", "TR-002"],
            "pickup": ["CityA", "CityA"],
            "delivery": ["CityB", "CityB"],
            "equipment": ["Dry Van", "Dry Van"],
            "distance": [100.0, 100.0],
            "date": ["2025-01-01", "2025-01-02"],
            "posted_rate": [200.0, 400.0],
        })
        val_df_1 = pd.DataFrame({
            "load_id": ["TR-003"],
            "pickup": ["CityA"],
            "delivery": ["CityB"],
            "equipment": ["Dry Van"],
            "distance": [100.0],
            "date": ["2025-02-01"],
            "posted_rate": [300.0],
        })
        val_df_2 = pd.DataFrame({
            "load_id": ["TR-003"],
            "pickup": ["CityA"],
            "delivery": ["CityB"],
            "equipment": ["Dry Van"],
            "distance": [100.0],
            "date": ["2025-02-01"],
            "posted_rate": [999999.0],  # Arbitrary target
        })
        
        out1 = compute_historical_pricing_features(val_df_1, history_df=train_df)
        out2 = compute_historical_pricing_features(val_df_2, history_df=train_df)
        
        # Historical features must be identical regardless of validation target
        for col in HISTORICAL_FEATURE_COLUMNS:
            self.assertEqual(out1[col].iloc[0], out2[col].iloc[0])
        self.assertEqual(out1["route_hist_median_rate"].iloc[0], 300.0)

    def test_calculate_rate_per_mile_target(self):
        """Verify rate-per-mile target calculation and validation."""
        rates = pd.Series([500.0, 1000.0, 1500.0])
        distances = pd.Series([250.0, 500.0, 1000.0])
        rpm = calculate_rate_per_mile_target(rates, distances)
        
        np.testing.assert_allclose(rpm, [2.0, 2.0, 1.5])

        # Non-positive distance must raise ValueError
        invalid_distances = pd.Series([250.0, 0.0, 1000.0])
        with self.assertRaises(ValueError):
            calculate_rate_per_mile_target(rates, invalid_distances)

    def test_reconstruct_rate_from_rpm(self):
        """Verify absolute rate reconstruction from rate-per-mile."""
        pred_rpm = np.array([2.0, 2.5, 1.8])
        distances = np.array([250.0, 400.0, 1000.0])
        reconstructed = reconstruct_rate_from_rpm(pred_rpm, distances)
        
        np.testing.assert_allclose(reconstructed, [500.0, 1000.0, 1800.0])

    def test_rate_per_mile_roundtrip(self):
        """Verify exact roundtrip identity: rate == reconstruct(calc_rpm(rate, dist), dist)."""
        rng = np.random.RandomState(42)
        true_rates = rng.uniform(200.0, 8000.0, size=50)
        true_dists = rng.uniform(50.0, 3000.0, size=50)
        
        target_rpm = calculate_rate_per_mile_target(true_rates, true_dists)
        reconstructed = reconstruct_rate_from_rpm(target_rpm, true_dists)
        
        np.testing.assert_allclose(reconstructed, true_rates, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()

