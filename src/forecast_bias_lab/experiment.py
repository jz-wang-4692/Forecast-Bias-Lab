"""
Clustered A/B-style experiment readout for long-lead forecast policies.

This module simulates a clustered policy readout:
series are hash-assigned to treatment/control, the treatment receives the
adjusted forecast policy, and the control receives the raw ML forecast.

The estimator uses:
- Pre-period cost adjustment (calibration-window baseline cost)
- Lagged demand, event/SNAP flags, category/state/lead-bucket controls
- Cluster-robust standard errors by item-store series

Two readouts are produced:
1. Bias-corrected vs raw ML forecast
2. Lift-factor p90 vs raw ML forecast

The paired counterfactual delta is also reported because this offline backtest
observes both potential outcomes for every row.  In a live experiment, only the
assigned outcome would be observed.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _stable_assignment(series_id: str) -> int:
    digest = hashlib.md5(str(series_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def _run_single_readout(
    test: pd.DataFrame,
    treatment_cost_col: str,
    control_cost_col: str,
    treatment_label: str,
    control_label: str,
) -> pd.DataFrame:
    """Run a single treatment-vs-control readout with cluster-robust SEs."""
    df = test.copy()
    df["assigned_treatment"] = df["series_id"].map(_stable_assignment).astype(int)
    df["assigned_cost"] = np.where(
        df["assigned_treatment"].eq(1),
        df[treatment_cost_col],
        df[control_cost_col],
    )
    df["paired_cost_delta"] = df[treatment_cost_col] - df[control_cost_col]

    formula = (
        "assigned_cost ~ assigned_treatment + pre_cost_per_unit + log_demand_lag_1 "
        "+ event_or_snap_week + C(cat_id) + C(state_id) + C(lead_bucket)"
    )
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["series_id"]},
    )

    coef = float(model.params["assigned_treatment"])
    se = float(model.bse["assigned_treatment"])
    ci_low, ci_high = model.conf_int().loc["assigned_treatment"].astype(float).tolist()
    control_mean = float(df.loc[df["assigned_treatment"].eq(0), "assigned_cost"].mean())
    treatment_mean = float(df.loc[df["assigned_treatment"].eq(1), "assigned_cost"].mean())
    paired_delta = float(df["paired_cost_delta"].mean())
    paired_relative = paired_delta / float(df[control_cost_col].mean())

    return pd.DataFrame([{
        "estimand": "ATE on weekly long-lead inventory cost",
        "treatment": treatment_label,
        "control": control_label,
        "rows": len(df),
        "clusters": df["series_id"].nunique(),
        "treatment_clusters": df.loc[df["assigned_treatment"].eq(1), "series_id"].nunique(),
        "control_clusters": df.loc[df["assigned_treatment"].eq(0), "series_id"].nunique(),
        "control_mean_assigned_cost": control_mean,
        "treatment_mean_assigned_cost": treatment_mean,
        "regression_ate": coef,
        "cluster_robust_se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(model.pvalues["assigned_treatment"]),
        "relative_ate_vs_control_mean": coef / control_mean if control_mean else np.nan,
        "paired_counterfactual_delta": paired_delta,
        "paired_counterfactual_relative_delta": paired_relative,
        "r_squared": float(model.rsquared),
    }])


def run_experiment_readout(
    scored_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run multi-arm experiment readout on the holdout period.

    Returns (experiment_panel, readout_table) where readout_table has one row
    per treatment arm compared against the raw ML control.
    """
    calibration = scored_panel.loc[scored_panel["split"].eq("calibration")].copy()
    test = scored_panel.loc[
        scored_panel["split"].eq("test") & scored_panel["long_lead_eligible"].eq(1)
    ].copy()
    if test.empty:
        raise ValueError("No long-lead rows available for experiment readout.")

    # Pre-period cost adjustment
    pre = (
        calibration.groupby("series_id", as_index=False)
        .agg(
            pre_baseline_cost_per_demand=("baseline_cost", lambda x: float(x.sum())),
            pre_demand=("demand", "sum"),
        )
    )
    pre["pre_cost_per_unit"] = pre["pre_baseline_cost_per_demand"] / pre["pre_demand"].clip(
        lower=1,
    )
    test = test.merge(pre[["series_id", "pre_cost_per_unit"]], on="series_id", how="left")
    test["pre_cost_per_unit"] = test["pre_cost_per_unit"].fillna(
        test["pre_cost_per_unit"].median(),
    )
    test["log_demand_lag_1"] = np.log1p(test["demand_lag_1"].clip(lower=0))
    test["event_or_snap_week"] = (
        test["is_event_week"].eq(1) | test["is_snap_heavy_week"].eq(1)
    ).astype(int)

    # Readout 1: bias-corrected vs raw ML
    readout_bias = _run_single_readout(
        test,
        treatment_cost_col="xgb_bias_corrected_cost",
        control_cost_col="xgb_cost",
        treatment_label="xgb_bias_corrected",
        control_label="xgb_forecast",
    )

    # Readout 2: lift-factor p90 vs raw ML
    readout_lift = _run_single_readout(
        test,
        treatment_cost_col="lift_p90_cost",
        control_cost_col="xgb_cost",
        treatment_label="lift_p90_forecast",
        control_label="xgb_forecast",
    )

    readout = pd.concat([readout_bias, readout_lift], ignore_index=True)
    return test, readout
