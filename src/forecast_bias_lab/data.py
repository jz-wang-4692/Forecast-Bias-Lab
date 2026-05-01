from __future__ import annotations

import shutil
import urllib.request
import zipfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_bias_lab.config import (
    M5_REQUIRED_FILES,
    M5_ZENODO_URL,
    M5_ZIP_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    WEEKLY_PANEL_METADATA_PATH,
    WEEKLY_PANEL_PATH,
)


ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(destination)


def ensure_m5_raw_data(force_download: bool = False) -> dict[str, Path]:
    """Download and extract the public M5 files needed for the analysis."""
    ensure_directories()
    required_paths = {name: RAW_DIR / name for name in M5_REQUIRED_FILES}
    if force_download or not all(path.exists() for path in required_paths.values()):
        if force_download or not M5_ZIP_PATH.exists():
            print(f"Downloading M5 dataset to {M5_ZIP_PATH} ...", flush=True)
            _download(M5_ZENODO_URL, M5_ZIP_PATH)

        print("Extracting required M5 CSV files ...", flush=True)
        with zipfile.ZipFile(M5_ZIP_PATH) as zf:
            members_by_base = {Path(member).name: member for member in zf.namelist()}
            missing = [name for name in M5_REQUIRED_FILES if name not in members_by_base]
            if missing:
                raise FileNotFoundError(f"Zip file is missing required files: {missing}")
            for name in M5_REQUIRED_FILES:
                target = RAW_DIR / name
                with zf.open(members_by_base[name]) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)

    return required_paths


def _day_columns(sales_path: Path, start_day: int, end_day: int) -> list[str]:
    columns = pd.read_csv(sales_path, nrows=0).columns.tolist()
    days = []
    for col in columns:
        if col.startswith("d_"):
            day = int(col.split("_", 1)[1])
            if start_day <= day <= end_day:
                days.append(col)
    if not days:
        raise ValueError(f"No day columns found between d_{start_day} and d_{end_day}")
    return days


def _load_calendar(calendar_path: Path, selected_days: list[str]) -> pd.DataFrame:
    calendar = pd.read_csv(calendar_path)
    calendar = calendar.loc[calendar["d"].isin(selected_days)].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    event_cols = ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    for col in event_cols:
        if col not in calendar:
            calendar[col] = np.nan
    calendar["event_count"] = (
        calendar[["event_name_1", "event_name_2"]].notna().sum(axis=1).astype(int)
    )
    calendar["major_event_day"] = calendar[["event_type_1", "event_type_2"]].isin(
        ["National", "Religious", "Sporting", "Cultural"]
    ).any(axis=1).astype(int)
    calendar["event_label"] = (
        calendar["event_name_1"].fillna(calendar["event_name_2"]).fillna("")
    )
    return calendar


def _state_snap_days(df: pd.DataFrame) -> np.ndarray:
    conditions = [
        df["state_id"].eq("CA"),
        df["state_id"].eq("TX"),
        df["state_id"].eq("WI"),
    ]
    choices = [df["snap_CA"], df["snap_TX"], df["snap_WI"]]
    return np.select(conditions, choices, default=0).astype(int)


def build_weekly_panel(
    raw_paths: dict[str, Path],
    sample_series_per_store_cat: int,
    start_day: int,
    end_day: int,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build a compact weekly item-store panel from M5 daily sales."""
    metadata = {
        "sample_series_per_store_cat": sample_series_per_store_cat,
        "start_day": start_day,
        "end_day": end_day,
    }
    cache_matches = False
    if WEEKLY_PANEL_METADATA_PATH.exists():
        try:
            cache_matches = json.loads(WEEKLY_PANEL_METADATA_PATH.read_text()) == metadata
        except json.JSONDecodeError:
            cache_matches = False
    if WEEKLY_PANEL_PATH.exists() and cache_matches and not force_rebuild:
        return pd.read_parquet(WEEKLY_PANEL_PATH)

    sales_path = raw_paths["sales_train_validation.csv"]
    selected_days = _day_columns(sales_path, start_day, end_day)
    calendar = _load_calendar(raw_paths["calendar.csv"], selected_days)

    usecols = ID_COLUMNS + selected_days
    print("Reading M5 sales matrix ...", flush=True)
    sales = pd.read_csv(sales_path, usecols=usecols)
    sales["_volume"] = sales[selected_days].sum(axis=1)
    sales = (
        sales.sort_values("_volume", ascending=False)
        .groupby(["store_id", "cat_id"], group_keys=False)
        .head(sample_series_per_store_cat)
        .drop(columns=["_volume"])
        .reset_index(drop=True)
    )

    sample_keys = sales[["store_id", "item_id"]].drop_duplicates()
    print(f"Selected {len(sales):,} item-store series for local analysis.", flush=True)

    print("Melting daily demand and aggregating to weekly product-node rows ...", flush=True)
    daily = sales.melt(
        id_vars=ID_COLUMNS,
        value_vars=selected_days,
        var_name="d",
        value_name="demand",
    )
    daily = daily.merge(
        calendar[
            [
                "d",
                "date",
                "wm_yr_wk",
                "event_count",
                "major_event_day",
                "event_label",
                "snap_CA",
                "snap_TX",
                "snap_WI",
            ]
        ],
        on="d",
        how="left",
    )
    daily["snap_day"] = _state_snap_days(daily)
    daily["has_event_label"] = daily["event_label"].ne("").astype(int)

    weekly = (
        daily.groupby(ID_COLUMNS + ["wm_yr_wk"], as_index=False)
        .agg(
            demand=("demand", "sum"),
            week_start=("date", "min"),
            week_end=("date", "max"),
            event_days=("event_count", lambda x: int((x > 0).sum())),
            major_event_days=("major_event_day", "sum"),
            snap_days=("snap_day", "sum"),
            event_label=("event_label", lambda x: "; ".join(sorted(set(v for v in x if v)))[:120]),
        )
        .sort_values(["id", "week_start"])
        .reset_index(drop=True)
    )

    print("Joining weekly price data ...", flush=True)
    prices = pd.read_csv(
        raw_paths["sell_prices.csv"],
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32"},
    )
    prices = prices.merge(sample_keys, on=["store_id", "item_id"], how="inner")
    prices = (
        prices.groupby(["store_id", "item_id", "wm_yr_wk"], as_index=False)
        .agg(sell_price=("sell_price", "mean"))
    )
    weekly = weekly.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    weekly["price_observed"] = weekly["sell_price"].notna().astype(int)
    weekly["sell_price"] = weekly.groupby("id")["sell_price"].transform(
        lambda s: s.ffill().bfill()
    )
    weekly["sell_price"] = weekly["sell_price"].fillna(weekly["sell_price"].median())

    WEEKLY_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(WEEKLY_PANEL_PATH, index=False)
    WEEKLY_PANEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return weekly
