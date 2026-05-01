"""
Pinball-optimal multiplicative lift factors by lead-bucket × event context.

This module implements a grid-search approach to finding the multiplicative
correction factor that minimizes demand-weighted pinball loss within each
segment.  The design mirrors how real-world forecast calibration works for
promotional / event-driven demand:

1. Segment the calibration data by (lead_bucket × is_event_week).
2. For each segment, grid-search a scalar factor f ∈ [0.5, 3.0] that
   minimizes weighted_pinball(demand, forecast × f, τ).
3. Apply shrinkage: small groups are blended toward 1.0 to avoid overfitting.
4. Enforce the constraint: factor_p50 ≤ factor_p90 at every segment.
5. Only long-lead buckets (57-98, 99+) receive intervention; short-lead
   buckets keep factor = 1.0 because the cost asymmetry is less severe
   and the base forecast is already adequate.

The key insight: optimizing for WAPE or MSE produces a median/mean-targeting
forecast, but inventory planning needs a quantile-targeting forecast.  The
lift factor shifts the forecast toward the service-level quantile without
retraining the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_bias_lab.config import (
    LEAD_BUCKET_LABELS,
    LIFT_FACTOR_MAX,
    LIFT_FACTOR_MIN,
    LIFT_FACTOR_STEP,
    LONG_LEAD_BUCKETS,
    PINBALL_TAU_P50,
    PINBALL_TAU_P90,
    SHRINKAGE_FULL_OBS,
    SHRINKAGE_MIN_OBS,
)


# ---------------------------------------------------------------------------
# Pinball penalty functions (vectorized)
# ---------------------------------------------------------------------------

def _penalty_p50(demand: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Pinball loss at τ=0.5 (symmetric absolute error)."""
    return 0.5 * np.abs(demand - forecast)


def _penalty_p90(demand: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Pinball loss at τ=0.9 (under-forecast penalized 9× more)."""
    diff = np.abs(demand - forecast)
    under = np.where(demand > forecast, 0.9 * diff, 0.0)
    over = np.where(demand < forecast, 0.1 * diff, 0.0)
    return under + over


def _weighted_error(
    demand: np.ndarray,
    forecast: np.ndarray,
    penalty_fn,
) -> float:
    """Demand-weighted error = Σ penalty / Σ demand."""
    return float(penalty_fn(demand, forecast).sum() / max(demand.sum(), 1e-8))


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _grid_search_factor(
    demand: np.ndarray,
    forecast: np.ndarray,
    penalty_fn,
) -> tuple[float, float]:
    """Find the multiplicative factor that minimizes weighted penalty."""
    best_factor = 1.0
    best_we = float("inf")
    for f in np.arange(LIFT_FACTOR_MIN, LIFT_FACTOR_MAX + LIFT_FACTOR_STEP, LIFT_FACTOR_STEP):
        we = penalty_fn(demand, forecast * f).sum() / max(demand.sum(), 1e-8)
        if we < best_we:
            best_we = we
            best_factor = float(f)
    return round(best_factor, 2), best_we


# ---------------------------------------------------------------------------
# Train lift factors
# ---------------------------------------------------------------------------

def train_lift_factors(
    calibration: pd.DataFrame,
    forecast_col: str = "xgb_forecast",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Train pinball-optimal lift factors on calibration data.

    Returns a DataFrame with one row per (lead_bucket × is_event_week) segment,
    containing the optimal p50 and p90 factors plus diagnostics.

    Only long-lead segments get intervention; short-lead segments get factor=1.0.
    """
    if group_cols is None:
        group_cols = ["lead_bucket", "is_event_week"]

    results = []
    for keys, group in calibration.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        lead_bucket = key_dict.get("lead_bucket", "")

        demand = group["demand"].values.astype(float)
        forecast = group[forecast_col].values.astype(float)
        n = len(group)

        is_long = lead_bucket in LONG_LEAD_BUCKETS

        if is_long:
            f50_raw, _ = _grid_search_factor(demand, forecast, _penalty_p50)
            f90_raw, _ = _grid_search_factor(demand, forecast, _penalty_p90)

            # Shrinkage toward 1.0 for small groups
            alpha = np.clip(
                (n - SHRINKAGE_MIN_OBS) / max(SHRINKAGE_FULL_OBS - SHRINKAGE_MIN_OBS, 1),
                0, 1,
            )
            f50 = round(alpha * f50_raw + (1 - alpha) * 1.0, 2)
            f90 = round(alpha * f90_raw + (1 - alpha) * 1.0, 2)

            # Constraint: f_p50 ≤ f_p90
            if f50 > f90:
                f50 = f90

            intervention = "INTERVENE"
        else:
            f50_raw = 1.0
            f90_raw = 1.0
            f50 = 1.0
            f90 = 1.0
            intervention = "ORIGINAL"

        # Compute before/after weighted errors for diagnostics
        we_p50_before = _weighted_error(demand, forecast, _penalty_p50)
        we_p90_before = _weighted_error(demand, forecast, _penalty_p90)
        we_p50_after = _weighted_error(demand, forecast * f50, _penalty_p50)
        we_p90_after = _weighted_error(demand, forecast * f90, _penalty_p90)

        row = {
            **key_dict,
            "n_obs": n,
            "is_long_lead": is_long,
            "intervention": intervention,
            "f_p50_raw": f50_raw,
            "f_p90_raw": f90_raw,
            "f_p50": f50,
            "f_p90": f90,
            "we_p50_before": round(we_p50_before, 6),
            "we_p50_after": round(we_p50_after, 6),
            "we_p90_before": round(we_p90_before, 6),
            "we_p90_after": round(we_p90_after, 6),
            "delta_p50_pct": round(
                (we_p50_before - we_p50_after) / max(we_p50_before, 1e-8) * 100, 2
            ),
            "delta_p90_pct": round(
                (we_p90_before - we_p90_after) / max(we_p90_before, 1e-8) * 100, 2
            ),
        }
        results.append(row)

    return pd.DataFrame(results).sort_values("n_obs", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Apply lift factors
# ---------------------------------------------------------------------------

def apply_lift_factors(
    df: pd.DataFrame,
    factor_table: pd.DataFrame,
    forecast_col: str = "xgb_forecast",
    output_col_p50: str = "lift_p50_forecast",
    output_col_p90: str = "lift_p90_forecast",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply trained lift factors to produce p50 and p90 adjusted forecasts.

    The p50 forecast targets the median; the p90 forecast targets the 90th
    percentile of demand, suitable for safety-stock planning.

    Row-level constraint: adj_p50 ≤ adj_p90 is enforced.
    """
    if group_cols is None:
        group_cols = ["lead_bucket", "is_event_week"]

    merge_cols = group_cols + ["f_p50", "f_p90"]
    out = df.copy()
    out = out.merge(
        factor_table[merge_cols],
        on=group_cols,
        how="left",
    )
    out["f_p50"] = out["f_p50"].fillna(1.0)
    out["f_p90"] = out["f_p90"].fillna(1.0)

    out[output_col_p50] = (out[forecast_col] * out["f_p50"]).clip(lower=0)
    out[output_col_p90] = (out[forecast_col] * out["f_p90"]).clip(lower=0)

    # Row-level constraint: p50 ≤ p90
    out[output_col_p90] = out[[output_col_p50, output_col_p90]].max(axis=1)

    return out


# ---------------------------------------------------------------------------
# Factor stability analysis (year-over-year comparison)
# ---------------------------------------------------------------------------

def compare_factor_stability(
    factors_period_a: pd.DataFrame,
    factors_period_b: pd.DataFrame,
    label_a: str = "period_a",
    label_b: str = "period_b",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compare lift factors trained on two different calibration periods.

    This answers: "How stable are the optimal factors across time?"
    Large drift means the factors are overfitting to the calibration window;
    small drift means they generalize and can be deployed with confidence.
    """
    if group_cols is None:
        group_cols = ["lead_bucket", "is_event_week"]

    a = factors_period_a[group_cols + ["f_p50", "f_p90", "n_obs"]].copy()
    a.columns = group_cols + [f"f50_{label_a}", f"f90_{label_a}", f"n_{label_a}"]

    b = factors_period_b[group_cols + ["f_p50", "f_p90", "n_obs"]].copy()
    b.columns = group_cols + [f"f50_{label_b}", f"f90_{label_b}", f"n_{label_b}"]

    merged = a.merge(b, on=group_cols, how="outer")
    merged["f50_drift"] = merged[f"f50_{label_b}"] - merged[f"f50_{label_a}"]
    merged["f90_drift"] = merged[f"f90_{label_b}"] - merged[f"f90_{label_a}"]

    return merged
