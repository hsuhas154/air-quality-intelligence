from __future__ import annotations

import pandas as pd


def hourly_profile(df: pd.DataFrame, value_col: str = "aqi") -> pd.DataFrame:
    frame = df.copy()
    frame["hour"] = pd.to_datetime(frame["ts"]).dt.hour
    return frame.groupby("hour", as_index=False)[value_col].mean()


def weekday_profile(df: pd.DataFrame, value_col: str = "aqi") -> pd.DataFrame:
    frame = df.copy()
    frame["weekday"] = pd.to_datetime(frame["ts"]).dt.day_name()
    return frame.groupby("weekday", as_index=False)[value_col].mean()


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr(numeric_only=True)
