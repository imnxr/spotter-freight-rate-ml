"""Data loading, date parsing, and schema validation utilities.

This module provides clean, reusable functions to load datasets for the
Spotter Freight Rate Prediction assessment:
- `load_train_test`: Loads `train-test.csv`
- `load_validation`: Loads `validation.csv`
- `load_december_chart_inputs`: Loads `december-chart-inputs.csv`
- `parse_date_column`: Parses date strings into pandas datetime objects
- `validate_columns`, `validate_train_test_data`, `validate_validation_data`,
  `validate_december_data`: Performs basic non-destructive input validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import pandas as pd

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

TRAIN_TEST_FILE = DEFAULT_DATA_DIR / "train-test.csv"
VALIDATION_FILE = DEFAULT_DATA_DIR / "validation.csv"
DECEMBER_CHART_FILE = DEFAULT_DATA_DIR / "december-chart-inputs.csv"

# Expected column schemas
TRAIN_TEST_COLUMNS: tuple[str, ...] = (
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate",
)

VALIDATION_COLUMNS: tuple[str, ...] = (
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
)

DECEMBER_COLUMNS: tuple[str, ...] = (
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
)


def get_data_dir() -> Path:
    """Return the default data directory path.

    Returns:
        Path: Path to the data directory.
    """
    return DEFAULT_DATA_DIR


def parse_date_column(
    df: pd.DataFrame,
    date_column: str = "date",
    errors: str = "raise",
) -> pd.DataFrame:
    """Parse a date column into pandas datetime objects.

    Returns a shallow copy of the DataFrame with the specified date column
    converted to datetime. Does not drop or modify other rows/columns.

    Args:
        df: Input DataFrame.
        date_column: Name of the column containing date values. Defaults to 'date'.
        errors: How to handle parsing errors ('raise', 'coerce', 'ignore'). Defaults to 'raise'.

    Returns:
        pd.DataFrame: DataFrame with the date column converted to datetime.

    Raises:
        KeyError: If date_column is not present in df.
        ValueError: If dates cannot be parsed and errors is 'raise'.
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame.")

    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors=errors)
    return df_copy


def validate_columns(
    df: pd.DataFrame,
    expected_columns: Sequence[str],
    allow_extra: bool = False,
    dataset_name: str = "Dataset",
) -> None:
    """Validate that a DataFrame contains the expected columns.

    Args:
        df: DataFrame to check.
        expected_columns: Sequence of expected column names.
        allow_extra: Whether to allow columns not in expected_columns. Defaults to False.
        dataset_name: Name of the dataset for error reporting. Defaults to 'Dataset'.

    Raises:
        ValueError: If required columns are missing or if unexpected columns are found.
    """
    current_cols = list(df.columns)
    missing = [col for col in expected_columns if col not in current_cols]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing expected columns: {missing}"
        )

    if not allow_extra:
        extra = [col for col in current_cols if col not in expected_columns]
        if extra:
            raise ValueError(
                f"{dataset_name} contains unexpected extra columns: {extra}"
            )


def validate_train_test_data(df: pd.DataFrame) -> None:
    """Perform basic non-destructive input validation on the train-test dataset.

    Checks column schema, non-emptiness, and basic target value validity.
    Does not drop or alter any rows.

    Args:
        df: train-test DataFrame.

    Raises:
        ValueError: If schema is invalid, DataFrame is empty, or posted_rate has invalid values.
    """
    if df.empty:
        raise ValueError("Train-test DataFrame is empty.")

    validate_columns(df, TRAIN_TEST_COLUMNS, allow_extra=False, dataset_name="Train-test data")

    if "posted_rate" in df.columns:
        numeric_rates = pd.to_numeric(df["posted_rate"], errors="coerce")
        if (numeric_rates <= 0).any():
            raise ValueError("Train-test data contains non-positive posted_rate values.")


def validate_validation_data(df: pd.DataFrame) -> None:
    """Perform basic non-destructive input validation on the validation dataset.

    Checks column schema and non-emptiness.
    Does not drop or alter any rows.

    Args:
        df: Validation DataFrame.

    Raises:
        ValueError: If schema is invalid or DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Validation DataFrame is empty.")

    validate_columns(df, VALIDATION_COLUMNS, allow_extra=False, dataset_name="Validation data")


def validate_december_data(df: pd.DataFrame) -> None:
    """Perform basic non-destructive input validation on the December chart inputs dataset.

    Checks column schema and row count (expected 31 rows for December).
    Does not drop or alter any rows.

    Args:
        df: December chart inputs DataFrame.

    Raises:
        ValueError: If schema is invalid or row count does not match 31 days.
    """
    if df.empty:
        raise ValueError("December inputs DataFrame is empty.")

    validate_columns(df, DECEMBER_COLUMNS, allow_extra=False, dataset_name="December inputs")

    if len(df) != 31:
        raise ValueError(f"December inputs expected 31 rows, found {len(df)}.")


def _resolve_path(path: Union[str, Path, None], default: Path) -> Path:
    """Helper to resolve a path argument to a Path object.

    Args:
        path: User-provided path string or Path, or None.
        default: Default Path fallback if path is None.

    Returns:
        Path: Resolved Path object.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    resolved = Path(path) if path is not None else default
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved


def load_train_test(
    path: Union[str, Path, None] = None,
    parse_dates: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the train-test dataset (train-test.csv).

    Args:
        path: Path to train-test.csv. If None, uses default path `data/train-test.csv`.
        parse_dates: Whether to parse the 'date' column as datetime. Defaults to True.
        validate: Whether to validate dataset schema. Defaults to True.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validate is True and validation fails.
    """
    file_path = _resolve_path(path, TRAIN_TEST_FILE)
    df = pd.read_csv(file_path)

    if parse_dates and "date" in df.columns:
        df = parse_date_column(df, date_column="date")

    if validate:
        validate_train_test_data(df)

    return df


def load_validation(
    path: Union[str, Path, None] = None,
    parse_dates: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the validation dataset (validation.csv).

    Args:
        path: Path to validation.csv. If None, uses default path `data/validation.csv`.
        parse_dates: Whether to parse the 'date' column as datetime. Defaults to True.
        validate: Whether to validate dataset schema. Defaults to True.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validate is True and validation fails.
    """
    file_path = _resolve_path(path, VALIDATION_FILE)
    df = pd.read_csv(file_path)

    if parse_dates and "date" in df.columns:
        df = parse_date_column(df, date_column="date")

    if validate:
        validate_validation_data(df)

    return df


def load_december_chart_inputs(
    path: Union[str, Path, None] = None,
    parse_dates: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the December chart inputs dataset (december-chart-inputs.csv).

    Args:
        path: Path to december-chart-inputs.csv. If None, uses default path `data/december-chart-inputs.csv`.
        parse_dates: Whether to parse the 'date' column as datetime. Defaults to True.
        validate: Whether to validate dataset schema. Defaults to True.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validate is True and validation fails.
    """
    file_path = _resolve_path(path, DECEMBER_CHART_FILE)
    df = pd.read_csv(file_path)

    if parse_dates and "date" in df.columns:
        df = parse_date_column(df, date_column="date")

    if validate:
        validate_december_data(df)

    return df


# Alias for convenience
load_december_inputs = load_december_chart_inputs
