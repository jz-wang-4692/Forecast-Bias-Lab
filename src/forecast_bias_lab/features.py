from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from forecast_bias_lab.config import FEATURE_PANEL_METADATA_PATH, FEATURE_PANEL_PATH


LEAD_BUCKETS = [0, 7, 14, 28, 56, 98, 10_000]
LEAD_BUCKET_LABELS = ["0-7", "8-14", "15-28", "29-56", "57-98", "99+"]


def _stable_int(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _series_offsets(series_ids: pd.Series) -> pd.Series:
    unique = pd.Series(series_ids.unique(), index=series_ids.unique())
    offsets = unique.map(lambda value: _stable_int(str(value)) % 37)
    return series_ids.map(offsets).astype(float)


def add_feature_panel(weekly: pd.DataFrame, force_rebuild: bool = False) -> pd.DataFrame:
    metadata = {
        "weekly_rows": int(len(weekly)),
        "series": int(weekly["id"].nunique()),
        "min_week": str(pd.to_datetime(weekly["week_start"]).min().date()),
        "max_week": str(pd.to_datetime(weekly["week_start"]).max().date()),
    }
    cache_matches = False
    if FEATURE_PANEL_METADATA_PATH.exists():
        try:
            cache_matches = json.loads(FEATURE_PANEL_METADATA_PATH.read_text()) == metadata
        except json.JSONDecodeError:
            cache_matches = False
    if FEATURE_PANEL_PATH.exists() and cache_matches and not force_rebuild:
        return pd.read_parquet(FEATURE_PANEL_PATH)

    df = weekly.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])
    df["series_id"] = df["store_id"].astype(str) + "__" + df["item_id"].astype(str)
    df = df.sort_values(["series_id", "week_start"]).reset_index(drop=True)

    iso = df["week_start"].dt.isocalendar()
    df["week_of_year"] = iso.week.astype(int)
    df["month"] = df["week_start"].dt.month.astype(int)
    df["quarter"] = df["week_start"].dt.quarter.astype(int)
    df["is_event_week"] = df["event_days"].gt(0).astype(int)
    df["is_major_event_week"] = df["major_event_days"].gt(0).astype(int)
    df["is_snap_heavy_week"] = df["snap_days"].ge(3).astype(int)
    df["event_intensity"] = df["event_days"] / 7.0
    df["snap_intensity"] = df["snap_days"] / 7.0

    week_event = (
        df[["wm_yr_wk", "event_days", "major_event_days"]]
        .drop_duplicates("wm_yr_wk")
        .sort_values("wm_yr_wk")
    )
    week_event["event_days_next_2w"] = (
        week_event["event_days"].shift(-1).fillna(0)
        + week_event["event_days"].shift(-2).fillna(0)
    )
    week_event["major_event_next_2w"] = (
        week_event["major_event_days"].shift(-1).fillna(0)
        + week_event["major_event_days"].shift(-2).fillna(0)
    )
    df = df.merge(
        week_event[["wm_yr_wk", "event_days_next_2w", "major_event_next_2w"]],
        on="wm_yr_wk",
        how="left",
    )

    group = df.groupby("series_id", group_keys=False)
    for lag in [1, 2, 4, 8, 13, 52]:
        df[f"demand_lag_{lag}"] = group["demand"].shift(lag)
    for window in [4, 8, 13, 26]:
        df[f"rolling_mean_{window}"] = group["demand"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).mean()
        )
    df["rolling_std_8"] = group["demand"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).std()
    )
    df["rolling_max_13"] = group["demand"].transform(
        lambda s: s.shift(1).rolling(13, min_periods=2).max()
    )
    df["demand_trend_4_vs_13"] = df["rolling_mean_4"] / df["rolling_mean_13"].replace(0, np.nan)

    df["price_lag_1"] = group["sell_price"].shift(1)
    df["price_lag_4"] = group["sell_price"].shift(4)
    df["price_change_4w"] = (df["price_lag_1"] - df["price_lag_4"]) / df["price_lag_4"].replace(
        0, np.nan
    )
    df["price_vs_series_median"] = df["sell_price"] / group["sell_price"].transform("median")

    base_lead_by_cat = {"FOODS": 21.0, "HOBBIES": 44.0, "HOUSEHOLD": 64.0}
    state_adjustment = {"CA": 4.0, "TX": 9.0, "WI": 13.0}
    base_lead = df["cat_id"].map(base_lead_by_cat).astype(float)
    state_adj = df["state_id"].map(state_adjustment).astype(float)
    stable_offset = _series_offsets(df["series_id"])
    seasonal_delay = 7.0 * np.sin(2 * np.pi * df["week_of_year"] / 52.0)
    event_delay = 1.5 * df["event_days_next_2w"] + 0.75 * df["snap_days"]
    df["supplier_lead_time_days"] = (
        base_lead + state_adj + stable_offset + seasonal_delay + event_delay
    ).clip(3, 150)
    df["lead_bucket"] = pd.cut(
        df["supplier_lead_time_days"],
        bins=LEAD_BUCKETS,
        labels=LEAD_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    df["long_lead_eligible"] = df["supplier_lead_time_days"].ge(57).astype(int)

    df["expected_uncensored_demand"] = df["rolling_mean_8"]
    df["zero_sales_week"] = df["demand"].le(0).astype(int)
    df["stockout_censored_proxy"] = (
        df["zero_sales_week"].eq(1)
        & df["expected_uncensored_demand"].ge(3.0)
        & df["price_observed"].eq(1)
    ).astype(int)
    df["stockout_risk_lag_1"] = (
        group["stockout_censored_proxy"].shift(1).fillna(0).astype(int)
    )
    df["recent_zero_week_rate_8"] = group["zero_sales_week"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=2).mean()
    )
    df["availability_risk_score"] = (
        0.6 * df["stockout_risk_lag_1"]
        + 0.4 * df["recent_zero_week_rate_8"].fillna(0)
    )

    fill_columns = [
        "demand_lag_1",
        "demand_lag_2",
        "demand_lag_4",
        "demand_lag_8",
        "demand_lag_13",
        "demand_lag_52",
        "rolling_mean_4",
        "rolling_mean_8",
        "rolling_mean_13",
        "rolling_mean_26",
        "rolling_std_8",
        "rolling_max_13",
        "demand_trend_4_vs_13",
        "price_lag_1",
        "price_lag_4",
        "price_change_4w",
        "price_vs_series_median",
        "recent_zero_week_rate_8",
    ]
    for col in fill_columns:
        if col in df:
            series_median = df.groupby("series_id")[col].transform("median")
            global_median = df[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            df[col] = df[col].fillna(series_median).fillna(global_median).replace(
                [np.inf, -np.inf], global_median
            )

    FEATURE_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURE_PANEL_PATH, index=False)
    FEATURE_PANEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return df
