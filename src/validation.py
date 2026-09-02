"""Temporal validation utilities for Spotter freight rate modeling.

This module provides non-destructive, strictly chronological splitting
utilities designed to mimic out-of-time production evaluation without data leakage:
- `temporal_split`: Splits a DataFrame into train and validation sets using a cutoff date.
- `temporal_train_valid_split`: Splits into (X_train, y_train, X_valid, y_valid),
  ensuring non-feature identifier columns like `load_id` are excluded.
- `get_split_summary`: Generates a structured summary of row counts and date ranges.
- `validate_temporal_split`: Enforces temporal integrity and non-empty partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence, Union

import pandas as pd

# Default columns to exclude from feature matrices
DEFAULT_EXCLUDE_COLUMNS: tuple[str, ...] = ("load_id",)


@dataclass(frozen=True)
class SplitSummary:
    """Summary metrics of a temporal train/validation partition.

    Attributes:
        train_rows: Number of rows in the training split.
        valid_rows: Number of rows in the validation split.
        train_min_date: Earliest date in the training split.
        train_max_date: Latest date in the training split.
        valid_min_date: Earliest date in the validation split.
        valid_max_date: Latest date in the validation split.
    """

    train_rows: int
    valid_rows: int
    train_min_date: pd.Timestamp
    train_max_date: pd.Timestamp
    valid_min_date: pd.Timestamp
    valid_max_date: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to a dictionary."""
        return {
            "train_rows": self.train_rows,
            "valid_rows": self.valid_rows,
            "train_min_date": self.train_min_date.strftime("%Y-%m-%d"),
            "train_max_date": self.train_max_date.strftime("%Y-%m-%d"),
            "valid_min_date": self.valid_min_date.strftime("%Y-%m-%d"),
            "valid_max_date": self.valid_max_date.strftime("%Y-%m-%d"),
        }

    def __str__(self) -> str:
        """Return a formatted string representation."""
        return (
            f"SplitSummary("
            f"Train: {self.train_rows:,} rows [{self.train_min_date.strftime('%Y-%m-%d')} to {self.train_max_date.strftime('%Y-%m-%d')}], "
            f"Valid: {self.valid_rows:,} rows [{self.valid_min_date.strftime('%Y-%m-%d')} to {self.valid_max_date.strftime('%Y-%m-%d')}]"
            f")"
        )


def validate_temporal_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    date_column: str = "date",
) -> None:
    """Validate that temporal partition constraints are satisfied.

    Verifies:
    1. Date column exists in both partitions.
    2. Dates can be coerced/parsed to datetime.
    3. Neither split is empty.
    4. Validation dates are strictly later than training dates (valid_min > train_max).

    Args:
        train_df: Training partition DataFrame.
        valid_df: Validation partition DataFrame.
        date_column: Column name containing dates. Defaults to 'date'.

    Raises:
        KeyError: If date_column is missing from either DataFrame.
        ValueError: If a split is empty, dates are invalid, or chronological order is violated.
    """
    if date_column not in train_df.columns:
        raise KeyError(f"Date column '{date_column}' not found in training DataFrame.")
    if date_column not in valid_df.columns:
        raise KeyError(f"Date column '{date_column}' not found in validation DataFrame.")

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if valid_df.empty:
        raise ValueError("Validation split is empty.")

    try:
        train_dates = pd.to_datetime(train_df[date_column])
        valid_dates = pd.to_datetime(valid_df[date_column])
    except Exception as exc:
        raise ValueError(f"Failed to parse dates in temporal split: {exc}") from exc

    if train_dates.isna().any() or valid_dates.isna().any():
        raise ValueError("Date column contains null or invalid datetime values.")

    train_max = train_dates.max()
    valid_min = valid_dates.min()

    if valid_min <= train_max:
        raise ValueError(
            f"Temporal ordering violation: validation minimum date ({valid_min.strftime('%Y-%m-%d')}) "
            f"is not strictly later than training maximum date ({train_max.strftime('%Y-%m-%d')})."
        )


def get_split_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    date_column: str = "date",
) -> SplitSummary:
    """Report row counts and date bounds for train and validation splits.

    Args:
        train_df: Training partition DataFrame.
        valid_df: Validation partition DataFrame.
        date_column: Column name containing dates. Defaults to 'date'.

    Returns:
        SplitSummary: Dataclass containing split counts and date ranges.

    Raises:
        KeyError: If date_column is missing from either DataFrame.
        ValueError: If either DataFrame is empty or contains unparsable dates.
    """
    if date_column not in train_df.columns:
        raise KeyError(f"Date column '{date_column}' not found in training DataFrame.")
    if date_column not in valid_df.columns:
        raise KeyError(f"Date column '{date_column}' not found in validation DataFrame.")

    if train_df.empty:
        raise ValueError("Cannot summarize empty training DataFrame.")
    if valid_df.empty:
        raise ValueError("Cannot summarize empty validation DataFrame.")

    train_dates = pd.to_datetime(train_df[date_column])
    valid_dates = pd.to_datetime(valid_df[date_column])

    return SplitSummary(
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        train_min_date=train_dates.min(),
        train_max_date=train_dates.max(),
        valid_min_date=valid_dates.min(),
        valid_max_date=valid_dates.max(),
    )


def temporal_split(
    df: pd.DataFrame,
    cutoff_date: Union[str, pd.Timestamp, date, datetime],
    date_column: str = "date",
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame temporally into training (< cutoff) and validation (>= cutoff).

    Args:
        df: Input DataFrame containing date_column.
        cutoff_date: Cutoff date string (e.g. '2025-09-01') or Timestamp.
        date_column: Column containing dates. Defaults to 'date'.
        validate: Whether to validate non-empty partitions and chronological integrity.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, valid_df) copies.

    Raises:
        KeyError: If date_column is not found in df.
        ValueError: If dates cannot be parsed or if validation checks fail.
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame.")

    try:
        cutoff_ts = pd.to_datetime(cutoff_date)
    except Exception as exc:
        raise ValueError(f"Invalid cutoff_date '{cutoff_date}': {exc}") from exc

    dates = pd.to_datetime(df[date_column])
    train_mask = dates < cutoff_ts
    valid_mask = dates >= cutoff_ts

    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    if validate:
        validate_temporal_split(train_df, valid_df, date_column=date_column)

    return train_df, valid_df


def temporal_train_valid_split(
    df: pd.DataFrame,
    cutoff_date: Union[str, pd.Timestamp, date, datetime],
    target_column: str = "posted_rate",
    feature_columns: Union[Sequence[str], None] = None,
    exclude_columns: Union[Sequence[str], None] = DEFAULT_EXCLUDE_COLUMNS,
    date_column: str = "date",
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Perform a temporal split returning (X_train, y_train, X_valid, y_valid).

    Guarantees non-feature identifier columns (such as `load_id`) are excluded
    from feature matrices by default.

    Args:
        df: Input DataFrame containing features, target, and date column.
        cutoff_date: Cutoff date string (e.g. '2025-09-01') or Timestamp.
        target_column: Target column name. Defaults to 'posted_rate'.
        feature_columns: Explicit list of feature column names to include.
            If None, all columns except target_column and exclude_columns are used.
        exclude_columns: Columns to exclude from features when feature_columns is None.
            Defaults to ('load_id',).
        date_column: Date column name used for temporal splitting. Defaults to 'date'.
        validate: Whether to run validation checks on the temporal split.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            (X_train, y_train, X_valid, y_valid)

    Raises:
        KeyError: If target_column, date_column, or specified feature_columns are missing.
        ValueError: If temporal split validation fails.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    train_df, valid_df = temporal_split(
        df=df,
        cutoff_date=cutoff_date,
        date_column=date_column,
        validate=validate,
    )

    if feature_columns is not None:
        missing_feats = [col for col in feature_columns if col not in df.columns]
        if missing_feats:
            raise KeyError(f"Feature columns not found in DataFrame: {missing_feats}")
        if "load_id" in feature_columns:
            raise ValueError("load_id must not be used as a model feature.")
        selected_features = list(feature_columns)
    else:
        excluded = set(exclude_columns or ()) | {target_column}
        selected_features = [col for col in df.columns if col not in excluded]

    X_train = train_df[selected_features].copy()
    y_train = train_df[target_column].copy()
    X_valid = valid_df[selected_features].copy()
    y_valid = valid_df[target_column].copy()

    return X_train, y_train, X_valid, y_valid
