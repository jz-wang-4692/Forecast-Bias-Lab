"""
Forecast evaluation metrics with emphasis on asymmetric loss and pinball scoring.

The key insight this module encodes: WAPE and bias are useful diagnostics, but the
decision-relevant metric for inventory planning is the asymmetric newsvendor cost.
Under-forecasting is more expensive than over-forecasting when stockouts carry
higher penalties than excess inventory — and this asymmetry is sharper for long-lead
items where replenishment flexibility is limited.

Pinball loss at τ=0.5 measures median accuracy; at τ=0.9 it measures the quality
of a safety-stock-level forecast.  The τ=0.9 pinball penalizes under-forecast 9×
more than over-forecast, mirroring the cost structure of high-service-level planning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_bias_lab.config import (
    PINBALL_TAU_P50,
    PINBALL_TAU_P90,
    UNDERAGE_COST,
    OVERAGE_COST,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def wape(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> float:
    """Weighted Absolute Percentage Error = Σ|actual - forecast| / Σ|actual|."""
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(actual_arr))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual_arr - forecast_arr)) / denom)


def forecast_bias(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> float:
    """Signed bias = Σ(forecast - actual) / Σ(actual).  Positive = over-forecast."""
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    denom = np.sum(actual_arr)
    if denom == 0:
        return np.nan
    return float(np.sum(forecast_arr - actual_arr) / denom)


def underforecast_rate(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> float:
    """Fraction of rows where forecast < actual."""
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    return float(np.mean(forecast_arr < actual_arr))


def pinball_loss(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
    tau: float,
) -> float:
    """Quantile (pinball) loss at level τ.

    τ=0.5 → symmetric absolute error (median loss).
    τ=0.9 → under-forecast penalized 9× more than over-forecast.
    """
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    diff = actual_arr - forecast_arr
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def weighted_pinball(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
    tau: float,
) -> float:
    """Demand-weighted pinball = Σ pinball_loss / Σ demand.

    This normalizes
    by total demand so that high-volume items don't dominate purely by scale.
    """
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    diff = actual_arr - forecast_arr
    penalty = np.maximum(tau * diff, (tau - 1.0) * diff)
    denom = np.sum(actual_arr)
    if denom == 0:
        return np.nan
    return float(np.sum(penalty) / denom)


def newsvendor_cost(
    actual: pd.Series | np.ndarray,
    forecast: pd.Series | np.ndarray,
    underage_cost: float = UNDERAGE_COST,
    overage_cost: float = OVERAGE_COST,
    service_buffer: float = 0.0,
) -> np.ndarray:
    """Row-level asymmetric inventory cost (newsvendor model).

    order_qty = forecast × (1 + service_buffer)
    cost = underage_cost × max(actual - order, 0) + overage_cost × max(order - actual, 0)
    """
    actual_arr = np.asarray(actual, dtype=float)
    order_qty = np.asarray(forecast, dtype=float) * (1.0 + service_buffer)
    shortage = np.maximum(actual_arr - order_qty, 0.0)
    excess = np.maximum(order_qty - actual_arr, 0.0)
    return underage_cost * shortage + overage_cost * excess


# ---------------------------------------------------------------------------
# Scorecard builder
# ---------------------------------------------------------------------------

def metric_scorecard(df: pd.DataFrame, forecast_cols: list[str]) -> pd.DataFrame:
    """Build a comparison scorecard across multiple forecast columns."""
    rows = []
    actual = df["demand"]
    demand_sum = float(actual.sum())
    for col in forecast_cols:
        cost = newsvendor_cost(actual, df[col])
        rows.append(
            {
                "forecast": col,
                "wape": wape(actual, df[col]),
                "bias": forecast_bias(actual, df[col]),
                "underforecast_rate": underforecast_rate(actual, df[col]),
                "pinball_p50": pinball_loss(actual, df[col], tau=PINBALL_TAU_P50),
                "pinball_p90": pinball_loss(actual, df[col], tau=PINBALL_TAU_P90),
                "weighted_pinball_p50": weighted_pinball(actual, df[col], tau=PINBALL_TAU_P50),
                "weighted_pinball_p90": weighted_pinball(actual, df[col], tau=PINBALL_TAU_P90),
                "inventory_cost": float(cost.sum()),
                "inventory_cost_per_unit_demand": float(cost.sum() / demand_sum)
                if demand_sum
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    best_cost = out["inventory_cost"].min()
    out["cost_lift_vs_best"] = out["inventory_cost"] / best_cost - 1.0
    return out.sort_values("inventory_cost").reset_index(drop=True)
