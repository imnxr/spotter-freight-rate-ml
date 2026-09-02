"""Unit tests for src/data.py module."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.data import (
    TRAIN_TEST_COLUMNS,
    VALIDATION_COLUMNS,
    DECEMBER_COLUMNS,
    get_data_dir,
    load_train_test,
    load_validation,
    load_december_chart_inputs,
    load_december_inputs,
    parse_date_column,
    validate_columns,
    validate_train_test_data,
    validate_validation_data,
    validate_december_data,
)


class TestDataModule(unittest.TestCase):
    """Test suite for data loading, parsing, and validation."""

    def test_get_data_dir(self):
        data_dir = get_data_dir()
        self.assertIsInstance(data_dir, Path)
        self.assertTrue(data_dir.is_dir())
        self.assertTrue((data_dir / "train-test.csv").exists())

    def test_load_train_test_default(self):
        df = load_train_test()
        self.assertEqual(df.shape, (48000, 14))
        self.assertEqual(list(df.columns), list(TRAIN_TEST_COLUMNS))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertEqual(len(df), 48000)

    def test_load_train_test_no_parse_dates(self):
        df = load_train_test(parse_dates=False)
        self.assertEqual(df.shape, (48000, 14))
        self.assertEqual(df["date"].dtype, object)

    def test_load_validation_default(self):
        df = load_validation()
        self.assertEqual(df.shape, (12000, 13))
        self.assertEqual(list(df.columns), list(VALIDATION_COLUMNS))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertEqual(len(df), 12000)

    def test_load_december_chart_inputs_default(self):
        df = load_december_chart_inputs()
        self.assertEqual(df.shape, (31, 7))
        self.assertEqual(list(df.columns), list(DECEMBER_COLUMNS))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertEqual(len(df), 31)

    def test_load_december_alias(self):
        df = load_december_inputs()
        self.assertEqual(df.shape, (31, 7))

    def test_load_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_train_test("non_existent_file.csv")

    def test_parse_date_column(self):
        df = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "value": [10, 20]})
        result = parse_date_column(df, date_column="date")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))
        self.assertEqual(df["date"].dtype, object)  # Original DataFrame untouched

    def test_parse_date_column_missing_column(self):
        df = pd.DataFrame({"day": ["2025-01-01"]})
        with self.assertRaises(KeyError):
            parse_date_column(df, date_column="date")

    def test_validate_columns_valid(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        # Should not raise
        validate_columns(df, ["a", "b"])

    def test_validate_columns_missing(self):
        df = pd.DataFrame({"a": [1]})
        with self.assertRaises(ValueError) as ctx:
            validate_columns(df, ["a", "b"])
        self.assertIn("missing expected columns", str(ctx.exception))

    def test_validate_columns_extra(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with self.assertRaises(ValueError) as ctx:
            validate_columns(df, ["a", "b"], allow_extra=False)
        self.assertIn("unexpected extra columns", str(ctx.exception))

        # Allowed when allow_extra is True
        validate_columns(df, ["a", "b"], allow_extra=True)

    def test_validate_empty_dataframes(self):
        with self.assertRaises(ValueError):
            validate_train_test_data(pd.DataFrame(columns=list(TRAIN_TEST_COLUMNS)))
        with self.assertRaises(ValueError):
            validate_validation_data(pd.DataFrame(columns=list(VALIDATION_COLUMNS)))
        with self.assertRaises(ValueError):
            validate_december_data(pd.DataFrame(columns=list(DECEMBER_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
