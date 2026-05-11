from __future__ import annotations

import pandas as pd


def add_calendar_features(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add deterministic calendar features for hourly electricity data."""
    output = frame.copy()
    timestamp = pd.to_datetime(output[timestamp_col])

    output["hour"] = timestamp.dt.hour
    output["day_of_week"] = timestamp.dt.dayofweek
    output["month"] = timestamp.dt.month
    output["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype(int)
    return output


def add_lag_features(
    frame: pd.DataFrame,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged target values for baseline tabular models."""
    output = frame.copy()
    for lag in lags:
        output[f"{target_col}_lag_{lag}"] = output[target_col].shift(lag)
    return output

