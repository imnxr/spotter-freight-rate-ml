"""Unit tests for src/validation.py temporal splitting module."""

import unittest
import pandas as pd
import numpy as np

from src.data import load_train_test
from src.validation import (
    SplitSummary,
    get_split_summary,
    temporal_split,
    temporal_train_valid_split,
    validate_temporal_split,
)


class TestValidationModule(unittest.TestCase):
    """Test suite for temporal validation utilities."""

    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "load_id": [f"TR-{i:03d}" for i in range(1, 11)],
                "date": [
                    "2025-01-01",
                    "2025-01-05",
                    "2025-01-10",
                    "2025-01-15",
                    "2025-01-20",
                    "2025-02-01",
                    "2025-02-05",
                    "2025-02-10",
                    "2025-02-15",
                    "2025-02-20",
                ],
                "distance": [100.0 * i for i in range(1, 11)],
                "posted_rate": [500.0 + 50.0 * i for i in range(1, 11)],
            }
        )

    def test_temporal_split_basic(self):
        cutoff = "2025-02-01"
        train_df, valid_df = temporal_split(self.sample_df, cutoff_date=cutoff)

        self.assertEqual(len(train_df), 5)
        self.assertEqual(len(valid_df), 5)
        self.assertTrue((pd.to_datetime(train_df["date"]) < pd.Timestamp(cutoff)).all())
        self.assertTrue((pd.to_datetime(valid_df["date"]) >= pd.Timestamp(cutoff)).all())

    def test_get_split_summary(self):
        cutoff = "2025-02-01"
        train_df, valid_df = temporal_split(self.sample_df, cutoff_date=cutoff)
        summary = get_split_summary(train_df, valid_df)

        self.assertIsInstance(summary, SplitSummary)
        self.assertEqual(summary.train_rows, 5)
        self.assertEqual(summary.valid_rows, 5)
        self.assertEqual(summary.train_min_date, pd.Timestamp("2025-01-01"))
        self.assertEqual(summary.train_max_date, pd.Timestamp("2025-01-20"))
        self.assertEqual(summary.valid_min_date, pd.Timestamp("2025-02-01"))
        self.assertEqual(summary.valid_max_date, pd.Timestamp("2025-02-20"))

        summary_dict = summary.to_dict()
        self.assertEqual(summary_dict["train_rows"], 5)
        self.assertEqual(summary_dict["train_min_date"], "2025-01-01")
        self.assertIn("SplitSummary", str(summary))

    def test_temporal_train_valid_split_features_and_target(self):
        cutoff = "2025-02-01"
        X_train, y_train, X_valid, y_valid = temporal_train_valid_split(
            self.sample_df,
            cutoff_date=cutoff,
            target_column="posted_rate",
        )

        self.assertEqual(len(X_train), 5)
        self.assertEqual(len(y_train), 5)
        self.assertEqual(len(X_valid), 5)
        self.assertEqual(len(y_valid), 5)

        # Ensure load_id and target_column are NOT in feature matrices
        self.assertNotIn("load_id", X_train.columns)
        self.assertNotIn("load_id", X_valid.columns)
        self.assertNotIn("posted_rate", X_train.columns)
        self.assertNotIn("posted_rate", X_valid.columns)
        self.assertIn("distance", X_train.columns)
        self.assertIn("date", X_train.columns)

    def test_explicit_feature_columns_selection(self):
        cutoff = "2025-02-01"
        X_train, y_train, X_valid, y_valid = temporal_train_valid_split(
            self.sample_df,
            cutoff_date=cutoff,
            target_column="posted_rate",
            feature_columns=["distance"],
        )
        self.assertEqual(list(X_train.columns), ["distance"])
        self.assertEqual(list(X_valid.columns), ["distance"])

    def test_load_id_in_feature_columns_rejected(self):
        cutoff = "2025-02-01"
        with self.assertRaises(ValueError) as ctx:
            temporal_train_valid_split(
                self.sample_df,
                cutoff_date=cutoff,
                target_column="posted_rate",
                feature_columns=["load_id", "distance"],
            )
        self.assertIn("load_id must not be used as a model feature", str(ctx.exception))

    def test_missing_target_column(self):
        with self.assertRaises(KeyError):
            temporal_train_valid_split(
                self.sample_df,
                cutoff_date="2025-02-01",
                target_column="non_existent_target",
            )

    def test_empty_split_validation(self):
        # Cutoff earlier than any date -> train is empty
        with self.assertRaises(ValueError) as ctx_early:
            temporal_split(self.sample_df, cutoff_date="2024-01-01")
        self.assertIn("Training split is empty", str(ctx_early.exception))

        # Cutoff later than any date -> valid is empty
        with self.assertRaises(ValueError) as ctx_late:
            temporal_split(self.sample_df, cutoff_date="2026-01-01")
        self.assertIn("Validation split is empty", str(ctx_late.exception))

    def test_invalid_date_column(self):
        df_no_date = pd.DataFrame({"value": [1, 2, 3]})
        with self.assertRaises(KeyError):
            temporal_split(df_no_date, cutoff_date="2025-02-01")

    def test_validate_temporal_split_overlap_error(self):
        train_df = pd.DataFrame({"date": ["2025-01-01", "2025-02-05"]})
        valid_df = pd.DataFrame({"date": ["2025-02-01", "2025-02-10"]})
        with self.assertRaises(ValueError) as ctx:
            validate_temporal_split(train_df, valid_df)
        self.assertIn("Temporal ordering violation", str(ctx.exception))

    def test_real_dataset_temporal_split(self):
        train_test_df = load_train_test()
        cutoff = "2025-09-01"
        train_df, valid_df = temporal_split(train_test_df, cutoff_date=cutoff)

        summary = get_split_summary(train_df, valid_df)
        self.assertGreater(summary.train_rows, 0)
        self.assertGreater(summary.valid_rows, 0)
        self.assertEqual(summary.train_rows + summary.valid_rows, len(train_test_df))
        self.assertLess(summary.train_max_date, summary.valid_min_date)
        self.assertEqual(summary.valid_min_date, pd.Timestamp("2025-09-01"))


if __name__ == "__main__":
    unittest.main()
