"""
Modeling pipeline: XGBoost demand model, empirical-Bayes bias correction,
and pinball-optimal lift factors.

The pipeline produces four forecast variants for comparison:
1. baseline_forecast — rolling mean with event/trend multipliers
2. xgb_forecast — raw XGBoost point forecast (log-demand target)
3. xgb_bias_corrected — empirical-Bayes residual-ratio correction
4. lift_p50_forecast / lift_p90_forecast — pinball-optimal lift factors

The central question: which policy minimizes asymmetric inventory cost,
especially on long-lead items where replenishment flexibility is limited?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast_bias_lab.config import (
    DEFAULT_CALIBRATION_WEEKS,
    DEFAULT_TEST_WEEKS,
    RANDOM_SEED,
    REPORTS_DIR,
)
from forecast_bias_lab.metrics import metric_scorecard, newsvendor_cost
from forecast_bias_lab.lift_factors import train_lift_factors, apply_lift_factors


NUMERIC_FEATURES = [
    "week_of_year",
    "month",
    "quarter",
    "event_days",
    "major_event_days",
    "snap_days",
    "is_event_week",
    "is_major_event_week",
    "is_snap_heavy_week",
    "event_intensity",
    "snap_intensity",
    "event_days_next_2w",
    "major_event_next_2w",
    "sell_price",
    "price_observed",
    "price_lag_1",
    "price_change_4w",
    "price_vs_series_median",
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
    "supplier_lead_time_days",
    "long_lead_eligible",
    "stockout_risk_lag_1",
    "recent_zero_week_rate_8",
    "availability_risk_score",
]

CATEGORICAL_FEATURES = ["cat_id", "dept_id", "store_id", "state_id", "lead_bucket"]


@dataclass
class ModelingResult:
    scored_panel: pd.DataFrame
    scorecard: pd.DataFrame
    bias_correction_table: pd.DataFrame
    lift_factor_table: pd.DataFrame
    feature_importance: pd.DataFrame
    split_summary: pd.DataFrame


def _assign_splits(
    df: pd.DataFrame,
    test_weeks: int = DEFAULT_TEST_WEEKS,
    calibration_weeks: int = DEFAULT_CALIBRATION_WEEKS,
) -> pd.DataFrame:
    out = df.copy()
    weeks = np.array(sorted(out["week_start"].drop_duplicates()))
    if len(weeks) <= test_weeks + calibration_weeks + 8:
        raise ValueError("Not enough weeks to create train/calibration/test splits.")
    test_start = weeks[-test_weeks]
    calibration_start = weeks[-(test_weeks + calibration_weeks)]
    out["split"] = np.where(
        out["week_start"].ge(test_start),
        "test",
        np.where(out["week_start"].ge(calibration_start), "calibration", "train"),
    )
    return out


def _design_matrix(train: pd.DataFrame, apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    apply_x = apply[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    combined = pd.concat([train_x, apply_x], axis=0, keys=["train", "apply"])
    combined = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES, dtype=float)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    medians = combined.loc["train"].median(numeric_only=True)
    combined = combined.fillna(medians).fillna(0.0)
    return combined.loc["train"], combined.loc["apply"]


def _fit_xgb(
    train: pd.DataFrame, apply: pd.DataFrame,
) -> tuple[np.ndarray, XGBRegressor, list[str]]:
    x_train, x_apply = _design_matrix(train, apply)
    y = np.log1p(train["demand"].astype(float))
    model = XGBRegressor(
        n_estimators=320,
        max_depth=4,
        learning_rate=0.045,
        subsample=0.86,
        colsample_bytree=0.88,
        objective="reg:squarederror",
        reg_alpha=0.05,
        reg_lambda=1.5,
        min_child_weight=4,
        random_state=RANDOM_SEED,
        tree_method="hist",
    )
    model.fit(x_train, y)
    pred = np.expm1(model.predict(x_apply))
    pred = np.clip(pred, 0.0, None)
    return pred, model, x_train.columns.tolist()


def _baseline_forecast(df: pd.DataFrame) -> pd.Series:
    baseline = df["rolling_mean_8"].copy()
    event_multiplier = 1.0 + 0.08 * df["is_major_event_week"] + 0.04 * df["is_snap_heavy_week"]
    trend_multiplier = df["demand_trend_4_vs_13"].clip(0.75, 1.35)
    baseline = baseline * event_multiplier * trend_multiplier
    fallback = df[["demand_lag_1", "rolling_mean_4", "rolling_mean_13"]].median(axis=1)
    return baseline.fillna(fallback).fillna(df["demand"].median()).clip(lower=0)


def fit_bias_correction(
    calibration: pd.DataFrame,
    forecast_col: str = "xgb_forecast",
    group_cols: list[str] | None = None,
    shrinkage_k: float = 80.0,
) -> pd.DataFrame:
    """Empirical-Bayes residual-ratio correction by segment."""
    if group_cols is None:
        group_cols = ["cat_id", "store_id", "lead_bucket", "is_event_week"]
    global_ratio = (
        calibration["demand"].sum() + 1.0
    ) / (calibration[forecast_col].sum() + 1.0)
    table = (
        calibration.groupby(group_cols, as_index=False)
        .agg(
            rows=("demand", "size"),
            actual_sum=("demand", "sum"),
            forecast_sum=(forecast_col, "sum"),
            stockout_proxy_rate=("stockout_censored_proxy", "mean"),
        )
    )
    table["raw_ratio"] = (table["actual_sum"] + 1.0) / (table["forecast_sum"] + 1.0)
    table["shrinkage_weight"] = table["rows"] / (table["rows"] + shrinkage_k)
    table["global_ratio"] = global_ratio
    table["correction_factor"] = (
        table["shrinkage_weight"] * table["raw_ratio"]
        + (1.0 - table["shrinkage_weight"]) * global_ratio
    ).clip(0.55, 1.75)
    return table.sort_values("rows", ascending=False).reset_index(drop=True)


def apply_bias_correction(
    df: pd.DataFrame,
    correction_table: pd.DataFrame,
    forecast_col: str = "xgb_forecast",
    output_col: str = "xgb_bias_corrected",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["cat_id", "store_id", "lead_bucket", "is_event_week"]
    out = df.copy()
    global_ratio = float(correction_table["global_ratio"].iloc[0])
    out = out.merge(
        correction_table[group_cols + ["correction_factor"]],
        on=group_cols,
        how="left",
    )
    out["correction_factor"] = out["correction_factor"].fillna(global_ratio).clip(0.55, 1.75)
    out[output_col] = (out[forecast_col] * out["correction_factor"]).clip(lower=0)
    return out.drop(columns=["correction_factor"])


def _feature_importance(model: XGBRegressor, feature_names: list[str]) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    )

    def family(feature: str) -> str:
        categorical_prefixes = [
            "cat_id_", "dept_id_", "store_id_", "state_id_", "lead_bucket_",
        ]
        for prefix in categorical_prefixes:
            if feature.startswith(prefix):
                return prefix[:-1]
        if feature.startswith("demand_lag_"):
            return "demand_lag"
        if feature.startswith("rolling_mean_"):
            return "rolling_mean"
        if feature.startswith("rolling_"):
            return "rolling_distribution"
        if feature.startswith("price_"):
            return "price"
        if feature.startswith("event_") or feature.startswith("major_event"):
            return "calendar_event"
        if feature.startswith("snap_") or feature.startswith("is_snap"):
            return "snap"
        if feature.startswith("stockout_") or feature.startswith("availability_"):
            return "availability_risk"
        if feature.startswith("supplier_lead") or feature.startswith("long_lead"):
            return "lead_time"
        if feature.startswith("is_event"):
            return "calendar_event"
        return feature

    importance["feature_family"] = importance["feature"].map(family)
    family_df = (
        importance.groupby("feature_family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    top_raw = importance.sort_values("importance", ascending=False).head(30)
    family_df.to_csv(REPORTS_DIR / "feature_importance_grouped.csv", index=False)
    top_raw.to_csv(REPORTS_DIR / "feature_importance_raw_top30.csv", index=False)
    return family_df


def run_modeling(
    feature_panel: pd.DataFrame,
    test_weeks: int = DEFAULT_TEST_WEEKS,
    calibration_weeks: int = DEFAULT_CALIBRATION_WEEKS,
) -> ModelingResult:
    df = feature_panel.copy()
    df = _assign_splits(df, test_weeks=test_weeks, calibration_weeks=calibration_weeks)
    df["baseline_forecast"] = _baseline_forecast(df)

    train = df.loc[df["split"].eq("train")].copy()
    apply_df = df.loc[df["split"].isin(["calibration", "test"])].copy()
    print(f"Training XGBoost on {len(train):,} weekly rows; scoring {len(apply_df):,} rows.")
    pred, model, feature_names = _fit_xgb(train, apply_df)
    apply_df["xgb_forecast"] = pred
    df = df.merge(
        apply_df[["series_id", "week_start", "xgb_forecast"]],
        on=["series_id", "week_start"],
        how="left",
    )
    df["xgb_forecast"] = df["xgb_forecast"].fillna(df["baseline_forecast"])

    # --- Empirical-Bayes bias correction ---
    calibration = df.loc[df["split"].eq("calibration")].copy()
    correction_table = fit_bias_correction(calibration)
    scored = apply_bias_correction(df, correction_table)

    # --- Pinball-optimal lift factors ---
    print("Training pinball-optimal lift factors on calibration data ...")
    lift_table = train_lift_factors(calibration, forecast_col="xgb_forecast")
    scored = apply_lift_factors(
        scored, lift_table, forecast_col="xgb_forecast",
    )
    lift_table.to_csv(REPORTS_DIR / "lift_factor_table.csv", index=False)
    print(f"  Lift factors: {len(lift_table)} segments, "
          f"{lift_table['is_long_lead'].sum()} long-lead intervened.")

    # --- Cost columns ---
    scored["baseline_cost"] = newsvendor_cost(scored["demand"], scored["baseline_forecast"])
    scored["xgb_cost"] = newsvendor_cost(scored["demand"], scored["xgb_forecast"])
    scored["xgb_bias_corrected_cost"] = newsvendor_cost(
        scored["demand"], scored["xgb_bias_corrected"],
    )
    scored["lift_p50_cost"] = newsvendor_cost(scored["demand"], scored["lift_p50_forecast"])
    scored["lift_p90_cost"] = newsvendor_cost(scored["demand"], scored["lift_p90_forecast"])

    # --- Scorecard ---
    test = scored.loc[scored["split"].eq("test")].copy()
    scorecard = metric_scorecard(
        test,
        [
            "baseline_forecast",
            "xgb_forecast",
            "xgb_bias_corrected",
            "lift_p50_forecast",
            "lift_p90_forecast",
        ],
    )
    scorecard.to_csv(REPORTS_DIR / "model_scorecard.csv", index=False)
    correction_table.to_csv(REPORTS_DIR / "bias_correction_table.csv", index=False)

    split_summary = (
        scored.groupby("split", as_index=False)
        .agg(
            rows=("demand", "size"),
            series=("series_id", "nunique"),
            weeks=("week_start", "nunique"),
            demand=("demand", "sum"),
            stockout_proxy_rate=("stockout_censored_proxy", "mean"),
            long_lead_share=("long_lead_eligible", "mean"),
        )
    )
    feature_importance = _feature_importance(model, feature_names)
    return ModelingResult(
        scored, scorecard, correction_table, lift_table, feature_importance, split_summary,
    )
