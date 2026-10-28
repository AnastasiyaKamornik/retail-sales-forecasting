
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features from the Date column."""

    result = df.copy()
    dates = pd.to_datetime(result["Date"])

    # Calendar features
    result["Year"] = dates.dt.year
    result["Month"] = dates.dt.month
    result["Day"] = dates.dt.day

    # Monday = 1, Sunday = 7
    result["Weekday"] = dates.dt.dayofweek + 1

    result["WeekOfYear"] = dates.dt.isocalendar().week.astype("int16")
    result["Quarter"] = dates.dt.quarter
    result["DayOfYear"] = dates.dt.dayofyear

    # Cyclical encoding for month
    result["MonthSin"] = np.sin(
        2 * np.pi * result["Month"] / 12
    )
    result["MonthCos"] = np.cos(
        2 * np.pi * result["Month"] / 12
    )

    # Cyclical encoding for weekday
    result["WeekdaySin"] = np.sin(
        2 * np.pi * result["Weekday"] / 7
    )
    result["WeekdayCos"] = np.cos(
        2 * np.pi * result["Weekday"] / 7
    )

    return result


def add_segment_id(
    df: pd.DataFrame,
    store_col: str = "Store",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Assign a new segment ID after every gap in a store's daily history.
    """

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])

    result = result.sort_values(
        [store_col, date_col]
    ).copy()

    date_diff = (
        result.groupby(store_col, sort=False)[date_col]
        .diff()
        .dt.days
    )

    new_segment = date_diff.ne(1)

    result["segment_id"] = (
        new_segment.groupby(result[store_col], sort=False)
        .cumsum()
        .astype("int64")
    )

    return result


def add_sales_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 7, 14, 28),
    windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """
    Create sales lags and rolling statistics within continuous
    daily segments for each store.

    Rolling features use only observations preceding the current date.
    Zero sales are preserved. Missing dates are not filled artificially.
    """

    required_columns = {"Store", "Date", "Sales"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if any(lag < 1 for lag in lags):
        raise ValueError("All lags must be positive integers.")

    if any(window < 1 for window in windows):
        raise ValueError("All rolling windows must be positive integers.")

    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"])

    if result[["Store", "Date"]].isna().any().any():
        raise ValueError("Store and Date must not contain missing values.")

    if result.duplicated(["Store", "Date"]).any():
        raise ValueError("Duplicate Store-Date observations found.")

    # Assign segment IDs before calculating temporal features
    result = add_segment_id(result)

    group_cols = ["Store", "segment_id"]

    # Calculate features separately within each continuous segment
    grouped_sales = result.groupby(
        group_cols,
        sort=False,
    )["Sales"]

    for lag in lags:
        result[f"Sales_lag_{lag}"] = grouped_sales.shift(lag)

    for window in windows:
        # Exclude current-day sales to prevent target leakage
        past_sales = grouped_sales.shift(1)

        result[f"Sales_roll_mean_{window}"] = (
            past_sales
            .groupby(
                [result["Store"], result["segment_id"]],
                sort=False,
            )
            .rolling(
                window=window,
                min_periods=window,
            )
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        result[f"Sales_roll_std_{window}"] = (
            past_sales
            .groupby(
                [result["Store"], result["segment_id"]],
                sort=False,
            )
            .rolling(
                window=window,
                min_periods=window,
            )
            .std()
            .reset_index(level=[0, 1], drop=True)
        )

    return result.sort_index()
