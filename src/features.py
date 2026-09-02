"""Leakage-safe feature engineering utilities for freight rate prediction.

This module generates row-level structural, temporal, and spatial features
without data leakage:
- Date decomposition (year, month, day, day_of_week, day_of_year, week_of_year, days_since_start)
- Route identifiers (pickup + "__" + delivery)
- Geographic coordinate differences and midpoints
- Strict exclusion of non-feature identifiers (e.g. `load_id`)
- Zero target-derived features (no usage of `posted_rate` or aggregates)
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

# Default baseline reference date for chronological progression feature
DEFAULT_START_DATE: str = "2025-01-01"

# Engineered feature names created by this module
DERIVED_FEATURE_COLUMNS: tuple[str, ...] = (
    "route",
    "abs_lat_diff",
    "abs_lon_diff",
    "midpoint_lat",
    "midpoint_lon",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "days_since_start",
)


def engineer_features(
    df: pd.DataFrame,
    drop_load_id: bool = True,
    start_date: Union[str, pd.Timestamp] = DEFAULT_START_DATE,
    date_column: str = "date",
) -> pd.DataFrame:
    """Compute leakage-safe row-level features from input freight data.

    Returns a new DataFrame without mutating the input. Preserves all existing
    columns (except `load_id` if `drop_load_id=True`), and does not impute missing
    values or drop any rows.

    Engineered features:
    - `route`: string combination `{pickup}__{delivery}`
    - `abs_lat_diff`: `|pickup_lat - delivery_lat|` (or NaN if coords missing)
    - `abs_lon_diff`: `|pickup_lon - delivery_lon|` (or NaN if coords missing)
    - `midpoint_lat`: `(pickup_lat + delivery_lat) / 2` (or NaN if coords missing)
    - `midpoint_lon`: `(pickup_lon + delivery_lon) / 2` (or NaN if coords missing)
    - `year`: Integer year from `date_column`
    - `month`: Integer month from `date_column`
    - `day`: Integer day from `date_column`
    - `day_of_week`: Day of week (0=Monday, 6=Sunday)
    - `day_of_year`: Day of year (1-366)
    - `week_of_year`: ISO week of year (1-53)
    - `days_since_start`: Days elapsed since `start_date`

    Args:
        df: Input DataFrame containing freight records.
        drop_load_id: Whether to exclude `load_id` from the output features. Defaults to True.
        start_date: Baseline start date string or Timestamp for `days_since_start`.
            Defaults to '2025-01-01'.
        date_column: Name of the date column. Defaults to 'date'.

    Returns:
        pd.DataFrame: New DataFrame containing original and newly engineered features.

    Raises:
        KeyError: If date_column is missing from df.
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame.")

    out = df.copy()

    # 1. Route identifier
    if "pickup" in out.columns and "delivery" in out.columns:
        pickup_s = out["pickup"]
        delivery_s = out["delivery"]
        is_null_route = pickup_s.isna() | delivery_s.isna()
        out["route"] = np.where(
            is_null_route,
            np.nan,
            pickup_s.astype(str) + "__" + delivery_s.astype(str),
        )
    else:
        out["route"] = np.nan

    # 2. Geographic differences and midpoints
    has_lat = "pickup_lat" in out.columns and "delivery_lat" in out.columns
    if has_lat:
        p_lat = pd.to_numeric(out["pickup_lat"], errors="coerce")
        d_lat = pd.to_numeric(out["delivery_lat"], errors="coerce")
        out["abs_lat_diff"] = (p_lat - d_lat).abs()
        out["midpoint_lat"] = (p_lat + d_lat) / 2.0
    else:
        out["abs_lat_diff"] = np.nan
        out["midpoint_lat"] = np.nan

    has_lon = "pickup_lon" in out.columns and "delivery_lon" in out.columns
    if has_lon:
        p_lon = pd.to_numeric(out["pickup_lon"], errors="coerce")
        d_lon = pd.to_numeric(out["delivery_lon"], errors="coerce")
        out["abs_lon_diff"] = (p_lon - d_lon).abs()
        out["midpoint_lon"] = (p_lon + d_lon) / 2.0
    else:
        out["abs_lon_diff"] = np.nan
        out["midpoint_lon"] = np.nan

    # 3. Temporal decompositions
    dates = pd.to_datetime(out[date_column], errors="coerce")
    base_ts = pd.to_datetime(start_date)

    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["day"] = dates.dt.day
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_year"] = dates.dt.dayofyear
    out["week_of_year"] = dates.dt.isocalendar().week.astype("Int64")
    out["days_since_start"] = (dates - base_ts).dt.days

    # 4. Exclude load_id if requested
    if drop_load_id and "load_id" in out.columns:
        out = out.drop(columns=["load_id"])

    return out


HISTORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "route_prev_count",
    "route_hist_median_rate",
    "route_hist_mean_rate",
    "route_hist_median_rate_per_mile",
    "equipment_route_hist_median_rate",
    "origin_hist_median_rate",
    "destination_hist_median_rate",
)


def compute_historical_pricing_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    target_column: str = "posted_rate",
    date_column: str = "date",
    drop_load_id: bool = True,
) -> pd.DataFrame:
    """Compute leakage-safe historical pricing features for freight data.

    Calculates expanding historical aggregations using strictly earlier observations:
    - If `history_df` is None (training mode):
      Computes expanding history within `df`. Each row with date D only accesses
      rows in `df` where date < D (strict prior dates). A row never observes its own
      target or contemporaneous/future targets.
    - If `history_df` is provided (validation/inference mode):
      Computes historical aggregates from `history_df` using observations strictly
      prior to each validation row's date (or all of `history_df` if all training
      dates precede validation dates). Validation target values are never accessed.

    Engineered features:
    - `route_prev_count`: Count of strictly prior labeled observations for route
    - `route_hist_median_rate`: Median prior posted_rate for route
    - `route_hist_mean_rate`: Mean prior posted_rate for route
    - `route_hist_median_rate_per_mile`: Median prior (posted_rate / distance) for route
    - `equipment_route_hist_median_rate`: Median prior posted_rate for (pickup, delivery, equipment)
    - `origin_hist_median_rate`: Median prior posted_rate for pickup city
    - `destination_hist_median_rate`: Median prior posted_rate for delivery city

    Args:
        df: Input DataFrame to enrich with historical features.
        history_df: Optional reference history DataFrame with labeled targets.
        target_column: Name of target column in history. Defaults to 'posted_rate'.
        date_column: Name of date column. Defaults to 'date'.
        drop_load_id: Whether to drop `load_id` from output. Defaults to True.

    Returns:
        pd.DataFrame: New DataFrame containing original features plus historical pricing features.
    """
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found in DataFrame.")

    out = df.copy()

    # Helper routes
    if "pickup" in out.columns and "delivery" in out.columns:
        out["route"] = out["pickup"].astype(str) + "__" + out["delivery"].astype(str)
    else:
        out["route"] = np.nan

    if "equipment" in out.columns and "route" in out.columns:
        out["_equip_route"] = out["route"].astype(str) + "__" + out["equipment"].astype(str)
    else:
        out["_equip_route"] = np.nan

    # Initialize all historical columns with NaN
    for col in HISTORICAL_FEATURE_COLUMNS:
        out[col] = np.nan

    if history_df is None:
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' required for self-expanding historical features.")

        source_df = df.copy()
        if "route" not in source_df.columns:
            source_df["route"] = source_df["pickup"].astype(str) + "__" + source_df["delivery"].astype(str)
        source_df["_equip_route"] = source_df["route"].astype(str) + "__" + source_df["equipment"].astype(str)
        source_df["_rpm"] = source_df[target_column] / source_df["distance"]

        dates = pd.to_datetime(source_df[date_column]).sort_values().unique()

        for d in dates:
            mask_d = pd.to_datetime(out[date_column]) == d
            if not mask_d.any():
                continue

            past_mask = pd.to_datetime(source_df[date_column]) < d
            if not past_mask.any():
                out.loc[mask_d, "route_prev_count"] = 0
                continue

            past_df = source_df[past_mask]

            r_grp = past_df.groupby("route")
            r_count = r_grp[target_column].count()
            r_med = r_grp[target_column].median()
            r_mean = r_grp[target_column].mean()
            r_rpm_med = r_grp["_rpm"].median()

            er_med = past_df.groupby("_equip_route")[target_column].median()
            orig_med = past_df.groupby("pickup")[target_column].median()
            dest_med = past_df.groupby("delivery")[target_column].median()

            curr_routes = out.loc[mask_d, "route"]
            curr_er = out.loc[mask_d, "_equip_route"]
            curr_p = out.loc[mask_d, "pickup"]
            curr_d = out.loc[mask_d, "delivery"]

            out.loc[mask_d, "route_prev_count"] = curr_routes.map(r_count).fillna(0)
            out.loc[mask_d, "route_hist_median_rate"] = curr_routes.map(r_med)
            out.loc[mask_d, "route_hist_mean_rate"] = curr_routes.map(r_mean)
            out.loc[mask_d, "route_hist_median_rate_per_mile"] = curr_routes.map(r_rpm_med)
            out.loc[mask_d, "equipment_route_hist_median_rate"] = curr_er.map(er_med)
            out.loc[mask_d, "origin_hist_median_rate"] = curr_p.map(orig_med)
            out.loc[mask_d, "destination_hist_median_rate"] = curr_d.map(dest_med)

    else:
        if target_column not in history_df.columns:
            raise KeyError(f"Target column '{target_column}' required in history_df.")

        source_df = history_df.copy()
        if "route" not in source_df.columns:
            source_df["route"] = source_df["pickup"].astype(str) + "__" + source_df["delivery"].astype(str)
        source_df["_equip_route"] = source_df["route"].astype(str) + "__" + source_df["equipment"].astype(str)
        source_df["_rpm"] = source_df[target_column] / source_df["distance"]

        min_df_date = pd.to_datetime(out[date_column]).min()
        max_hist_date = pd.to_datetime(source_df[date_column]).max()

        if min_df_date > max_hist_date:
            r_grp = source_df.groupby("route")
            r_count = r_grp[target_column].count()
            r_med = r_grp[target_column].median()
            r_mean = r_grp[target_column].mean()
            r_rpm_med = r_grp["_rpm"].median()

            er_med = source_df.groupby("_equip_route")[target_column].median()
            orig_med = source_df.groupby("pickup")[target_column].median()
            dest_med = source_df.groupby("delivery")[target_column].median()

            out["route_prev_count"] = out["route"].map(r_count).fillna(0)
            out["route_hist_median_rate"] = out["route"].map(r_med)
            out["route_hist_mean_rate"] = out["route"].map(r_mean)
            out["route_hist_median_rate_per_mile"] = out["route"].map(r_rpm_med)
            out["equipment_route_hist_median_rate"] = out["_equip_route"].map(er_med)
            out["origin_hist_median_rate"] = out["pickup"].map(orig_med)
            out["destination_hist_median_rate"] = out["delivery"].map(dest_med)
        else:
            dates = pd.to_datetime(out[date_column]).unique()
            for d in dates:
                mask_d = pd.to_datetime(out[date_column]) == d
                past_df = source_df[pd.to_datetime(source_df[date_column]) < d]
                if not len(past_df):
                    out.loc[mask_d, "route_prev_count"] = 0
                    continue
                r_grp = past_df.groupby("route")
                out.loc[mask_d, "route_prev_count"] = out.loc[mask_d, "route"].map(r_grp[target_column].count()).fillna(0)
                out.loc[mask_d, "route_hist_median_rate"] = out.loc[mask_d, "route"].map(r_grp[target_column].median())
                out.loc[mask_d, "route_hist_mean_rate"] = out.loc[mask_d, "route"].map(r_grp[target_column].mean())
                out.loc[mask_d, "route_hist_median_rate_per_mile"] = out.loc[mask_d, "route"].map(r_grp["_rpm"].median())
                out.loc[mask_d, "equipment_route_hist_median_rate"] = out.loc[mask_d, "_equip_route"].map(past_df.groupby("_equip_route")[target_column].median())
                out.loc[mask_d, "origin_hist_median_rate"] = out.loc[mask_d, "pickup"].map(past_df.groupby("pickup")[target_column].median())
                out.loc[mask_d, "destination_hist_median_rate"] = out.loc[mask_d, "delivery"].map(past_df.groupby("delivery")[target_column].median())

    if "_equip_route" in out.columns:
        out = out.drop(columns=["_equip_route"])

    if drop_load_id and "load_id" in out.columns:
        out = out.drop(columns=["load_id"])

    return out


def calculate_rate_per_mile_target(
    posted_rate: Union[pd.Series, np.ndarray],
    distance: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
    """Calculate rate-per-mile target for transformed regression models.

    Computes: `target_rpm = posted_rate / distance`.
    Note: This is strictly a target transformation for training models,
    NEVER to be used as a contemporaneous input feature.

    Args:
        posted_rate: Target freight rate Series or array.
        distance: Haul distance Series or array.

    Returns:
        Series or array representing rate per mile.

    Raises:
        ValueError: If any distance values are non-positive (<= 0).
    """
    dist_arr = np.asarray(distance)
    if np.any(dist_arr <= 0):
        raise ValueError("Distance must be strictly positive (> 0) to calculate rate per mile.")
    return posted_rate / distance


def reconstruct_rate_from_rpm(
    predicted_rpm: Union[pd.Series, np.ndarray],
    distance: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
    """Reconstruct absolute dollar freight rate from predicted rate-per-mile.

    Computes: `predicted_rate = predicted_rpm * distance`.

    Args:
        predicted_rpm: Predicted rate per mile Series or array.
        distance: Haul distance Series or array.

    Returns:
        Series or array of reconstructed absolute dollar rates.
    """
    return predicted_rpm * distance


