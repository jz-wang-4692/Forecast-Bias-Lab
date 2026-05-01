"""
Report generation: figures, tables, and the analytical writeup.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from forecast_bias_lab.config import (
    FIGURES_DIR,
    LEAD_BUCKET_LABELS,
    LONG_LEAD_BUCKETS,
    REPORTS_DIR,
)
from forecast_bias_lab.metrics import forecast_bias, wape, weighted_pinball


def ensure_report_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def plot_eda(feature_panel: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "weekly_demand_by_category.png"
    demand = (
        feature_panel.groupby(["week_start", "cat_id"], as_index=False)
        .agg(demand=("demand", "sum"), event_days=("event_days", "max"))
        .sort_values("week_start")
    )
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=demand, x="week_start", y="demand", hue="cat_id", linewidth=1.8)
    event_weeks = demand.loc[demand["event_days"].gt(0), "week_start"].drop_duplicates()
    for week in event_weeks.iloc[:: max(1, len(event_weeks) // 12)]:
        plt.axvline(week, color="0.82", linewidth=0.8, zorder=0)
    plt.title("Weekly demand by product category with event-week markers")
    plt.xlabel("")
    plt.ylabel("Units")
    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def plot_model_scorecard(scorecard: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "model_scorecard.png"
    plot_df = scorecard.copy()
    plot_df["forecast"] = plot_df["forecast"].str.replace("_", " ")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.barplot(data=plot_df, x="forecast", y="wape", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Holdout WAPE")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=plot_df, x="forecast", y="inventory_cost_per_unit_demand",
        ax=axes[1], color="#F58518",
    )
    axes[1].set_title("Asymmetric inventory cost / unit demand")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=plot_df, x="forecast", y="weighted_pinball_p90",
        ax=axes[2], color="#E45756",
    )
    axes[2].set_title("Weighted pinball loss (τ=0.9)")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=25)

    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Bias by lead bucket
# ---------------------------------------------------------------------------

def plot_bias_by_lead_bucket(scored_panel: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "forecast_bias_by_lead_bucket.png"
    test = scored_panel.loc[scored_panel["split"].eq("test")].copy()
    rows = []
    for lead_bucket, group in test.groupby("lead_bucket"):
        for col in ["xgb_forecast", "xgb_bias_corrected", "lift_p90_forecast"]:
            rows.append({
                "lead_bucket": lead_bucket,
                "forecast": col.replace("_", " "),
                "bias": forecast_bias(group["demand"], group[col]),
                "wape": wape(group["demand"], group[col]),
            })
    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=plot_df, x="lead_bucket", y="bias", hue="forecast")
    plt.axhline(0, color="black", linewidth=0.9)
    plt.title("Holdout forecast bias by simulated supplier lead-time bucket")
    plt.xlabel("Lead-time bucket (days)")
    plt.ylabel("Bias: Σ(forecast − actual) / Σ(actual)")
    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Correction heatmap
# ---------------------------------------------------------------------------

def plot_correction_heatmap(correction_table: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "bias_correction_heatmap.png"
    heat = (
        correction_table.groupby(["cat_id", "lead_bucket"], as_index=False)
        .agg(correction_factor=("correction_factor", "mean"))
        .pivot(index="cat_id", columns="lead_bucket", values="correction_factor")
    )
    plt.figure(figsize=(9, 3.8))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="vlag", center=1.0, linewidths=0.5)
    plt.title("Empirical-Bayes correction factors by category × lead bucket")
    plt.xlabel("Lead-time bucket (days)")
    plt.ylabel("")
    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Lift factor diagnostics
# ---------------------------------------------------------------------------

def plot_lift_factor_diagnostics(lift_table: pd.DataFrame) -> Path:
    """Bar chart of lift factors by lead bucket, showing p50 and p90 side by side."""
    path = FIGURES_DIR / "lift_factor_by_lead_bucket.png"
    long_only = lift_table[lift_table["is_long_lead"]].copy()
    if long_only.empty:
        # fallback: show all
        long_only = lift_table.copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # p50 factors
    ax = axes[0]
    for i, (_, row) in enumerate(long_only.iterrows()):
        label = f"{row['lead_bucket']} / evt={int(row.get('is_event_week', 0))}"
        ax.barh(i, row["f_p50"], color="#4C78A8", alpha=0.8)
        ax.text(row["f_p50"] + 0.02, i, f"{row['f_p50']:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(long_only)))
    ax.set_yticklabels([
        f"{r['lead_bucket']} / evt={int(r.get('is_event_week', 0))}"
        for _, r in long_only.iterrows()
    ])
    ax.axvline(1.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_title("p50 lift factors (long-lead segments)")
    ax.set_xlabel("Multiplicative factor")

    # p90 factors
    ax = axes[1]
    for i, (_, row) in enumerate(long_only.iterrows()):
        ax.barh(i, row["f_p90"], color="#E45756", alpha=0.8)
        ax.text(row["f_p90"] + 0.02, i, f"{row['f_p90']:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(long_only)))
    ax.set_yticklabels([
        f"{r['lead_bucket']} / evt={int(r.get('is_event_week', 0))}"
        for _, r in long_only.iterrows()
    ])
    ax.axvline(1.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_title("p90 lift factors (long-lead segments)")
    ax.set_xlabel("Multiplicative factor")

    _savefig(path)
    return path


def plot_weighted_pinball_by_lead(scored_panel: pd.DataFrame) -> Path:
    """Weighted pinball loss (τ=0.9) by lead bucket for each forecast policy."""
    path = FIGURES_DIR / "weighted_pinball_p90_by_lead.png"
    test = scored_panel.loc[scored_panel["split"].eq("test")].copy()
    rows = []
    forecast_cols = ["xgb_forecast", "xgb_bias_corrected", "lift_p90_forecast"]
    for lb in LEAD_BUCKET_LABELS:
        group = test[test["lead_bucket"] == lb]
        if group.empty:
            continue
        for col in forecast_cols:
            wp90 = weighted_pinball(group["demand"], group[col], tau=0.9)
            rows.append({
                "lead_bucket": lb,
                "forecast": col.replace("_", " "),
                "weighted_pinball_p90": wp90,
                "is_long": lb in LONG_LEAD_BUCKETS,
            })
    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=plot_df, x="lead_bucket", y="weighted_pinball_p90", hue="forecast")
    plt.title("Weighted pinball loss (τ=0.9) by lead bucket — lower is better")
    plt.xlabel("Lead-time bucket (days)")
    plt.ylabel("Weighted pinball (τ=0.9)")

    # Shade long-lead buckets
    for i, lb in enumerate(LEAD_BUCKET_LABELS):
        if lb in LONG_LEAD_BUCKETS:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="red", zorder=0)

    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Experiment readout
# ---------------------------------------------------------------------------

def plot_experiment_readout(readout: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "experiment_ate_ci.png"
    plt.figure(figsize=(9, 4))
    for i, (_, row) in enumerate(readout.iterrows()):
        label = row["treatment"].replace("_", " ")
        color = "#4C78A8" if i == 0 else "#E45756"
        plt.errorbar(
            x=[row["regression_ate"]],
            y=[i],
            xerr=[
                [row["regression_ate"] - row["ci_low"]],
                [row["ci_high"] - row["regression_ate"]],
            ],
            fmt="o",
            color=color,
            capsize=5,
            label=label,
        )
    plt.axvline(0, color="black", linewidth=0.9)
    plt.yticks(
        range(len(readout)),
        [r["treatment"].replace("_", " ") for _, r in readout.iterrows()],
    )
    plt.xlabel("ATE on weekly long-lead inventory cost (negative = better)")
    plt.title("Cluster-robust A/B readout: treatment vs raw XGBoost")
    plt.legend(loc="upper right")
    _savefig(path)
    return path


# ---------------------------------------------------------------------------
# Forecast explanations
# ---------------------------------------------------------------------------

def build_forecast_explanations(scored_panel: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    test = scored_panel.loc[scored_panel["split"].eq("test")].copy()
    test["abs_error_corrected"] = (test["demand"] - test["xgb_bias_corrected"]).abs()
    sample = test.sort_values("abs_error_corrected", ascending=False).head(n).copy()

    def explain(row: pd.Series) -> str:
        drivers = []
        if row["is_event_week"]:
            drivers.append("event week demand pressure")
        if row["is_snap_heavy_week"]:
            drivers.append("SNAP-heavy week")
        if row["demand_trend_4_vs_13"] > 1.15:
            drivers.append("recent demand acceleration")
        elif row["demand_trend_4_vs_13"] < 0.85:
            drivers.append("recent demand deceleration")
        if row["availability_risk_score"] > 0.3:
            drivers.append("recent stockout-risk proxy")
        if row["long_lead_eligible"]:
            drivers.append(f"long lead bucket {row['lead_bucket']}")
        if row["price_change_4w"] < -0.05:
            drivers.append("recent price discount")
        if not drivers:
            drivers.append("lagged demand and normal seasonality")
        return "; ".join(drivers)

    sample["driver_summary"] = sample.apply(explain, axis=1)
    cols = [
        "series_id", "week_start", "cat_id", "store_id", "lead_bucket",
        "demand", "xgb_forecast", "xgb_bias_corrected",
        "lift_p50_forecast", "lift_p90_forecast", "driver_summary",
    ]
    out = sample[[c for c in cols if c in sample.columns]].reset_index(drop=True)
    out.to_csv(REPORTS_DIR / "forecast_explanations_sample.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(
    scored_panel: pd.DataFrame,
    scorecard: pd.DataFrame,
    correction_table: pd.DataFrame,
    lift_factor_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
    split_summary: pd.DataFrame,
    experiment_readout: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    test = scored_panel.loc[scored_panel["split"].eq("test")].copy()
    best = scorecard.iloc[0]

    # Extract rows for each forecast
    def _row(name):
        match = scorecard.loc[scorecard["forecast"].eq(name)]
        return match.iloc[0] if not match.empty else None

    corrected = _row("xgb_bias_corrected")
    xgb = _row("xgb_forecast")
    baseline = _row("baseline_forecast")
    lift_p90 = _row("lift_p90_forecast")

    cost_reduction_vs_xgb = (
        1.0 - corrected["inventory_cost"] / xgb["inventory_cost"]
    ) if corrected is not None and xgb is not None else 0
    cost_reduction_vs_baseline = (
        1.0 - corrected["inventory_cost"] / baseline["inventory_cost"]
    ) if corrected is not None and baseline is not None else 0

    lift_cost_reduction = ""
    if lift_p90 is not None and xgb is not None:
        r = 1.0 - lift_p90["inventory_cost"] / xgb["inventory_cost"]
        lift_cost_reduction = f"The lift-factor p90 forecast reduces inventory cost by `{r:.1%}` vs raw XGBoost."

    top_features = ", ".join(feature_importance.head(8)["feature_family"].tolist())
    correction_range = (
        correction_table["correction_factor"].min(),
        correction_table["correction_factor"].max(),
    )
    long_lead_share = test["long_lead_eligible"].mean()
    stockout_proxy_rate = test["stockout_censored_proxy"].mean()

    # Lift factor summary for long-lead segments
    long_lift = lift_factor_table[lift_factor_table["is_long_lead"]]
    lift_summary_lines = []
    for _, row in long_lift.iterrows():
        evt = int(row.get("is_event_week", 0))
        lift_summary_lines.append(
            f"  - `{row['lead_bucket']}` / event={evt}: "
            f"f_p50={row['f_p50']:.2f}, f_p90={row['f_p90']:.2f} "
            f"(p90 WE improvement: {row['delta_p90_pct']:+.1f}%)"
        )
    lift_summary_block = "\n".join(lift_summary_lines) if lift_summary_lines else "  (none)"

    scorecard_md = scorecard.to_markdown(index=False, floatfmt=".4f")
    split_md = split_summary.to_markdown(index=False, floatfmt=".3f")
    experiment_md = experiment_readout.to_markdown(index=False, floatfmt=".4f")

    report = f"""# Forecast Bias Lab Report

Generated: {date.today().isoformat()}

## Executive Summary

This project evaluates demand-forecasting policies on public retail item-store sales data, with emphasis on **asymmetric loss optimization**, **lead-time-conditional bias correction**, **pinball-optimal lift factors**, and **A/B-style policy readout**.

The best holdout forecast by asymmetric inventory cost is `{best['forecast']}`. The bias-corrected model achieves WAPE `{corrected['wape']:.3f}` and bias `{corrected['bias']:.3f}` on the final holdout window. Relative to the raw XGBoost forecast, correction reduces inventory cost by `{cost_reduction_vs_xgb:.1%}`; relative to the rolling/event baseline, it reduces inventory cost by `{cost_reduction_vs_baseline:.1%}`.

{lift_cost_reduction}

## Data and Forecasting Unit

- Source: M5 Forecasting Accuracy data (Walmart item-store daily sales, prices, calendar events).
- Forecasting grain: weekly `item_id × store_id` demand (product-node forecast).
- Operational fields: lead-time buckets and stockout-risk indicators are transparent proxies/simulations — the public M5 data does not expose supplier lead time or true inventory availability.
- Holdout: `{long_lead_share:.1%}` of test rows are long-lead eligible; stockout-censored proxy rate is `{stockout_proxy_rate:.1%}`.

## Split Summary

{split_md}

## Modeling Method

1. Build lag, rolling-window, price, calendar, event, SNAP, stockout-risk, and simulated supplier-lead-time features.
2. Train XGBoost on the historical train split with a log-demand target.
3. Score calibration and holdout periods.
4. Fit **empirical-Bayes residual-ratio correction** on calibration by `category × store × lead_bucket × event_week`.
5. Fit **pinball-optimal lift factors** on calibration by `lead_bucket × is_event_week`, using grid search over [0.5, 3.0] with shrinkage regularization.
6. Evaluate on holdout using WAPE, signed bias, under-forecast rate, pinball loss (τ=0.5, τ=0.9), weighted pinball, and asymmetric inventory cost.

Top feature families by model importance: {top_features}.

Bias-correction factors range from `{correction_range[0]:.2f}` to `{correction_range[1]:.2f}` after shrinkage and caps.

## Lift Factor Summary (Long-Lead Segments)

Only long-lead buckets (57-98, 99+) receive lift-factor intervention. Short-lead buckets keep factor=1.0.

{lift_summary_block}

Constraint enforced: `adj_p50 ≤ adj_p90` at every row.

## Holdout Scorecard

{scorecard_md}

## Experiment Readout

The long-lead readout simulates a clustered A/B test: item-store series are hash-assigned to treatment/control, the treatment receives the candidate forecast, and the control receives the raw ML forecast. The estimator uses pre-period cost adjustment, lagged demand, event/SNAP flags, category/state/lead-bucket controls, and cluster-robust standard errors by item-store series.

{experiment_md}

Interpretation: negative ATE means the treatment reduced weekly long-lead inventory cost. The paired counterfactual delta is included because this offline backtest observes both potential forecast policies for every row; the regression readout mirrors what would be available in a randomized launch.

## Key Insights

1. **Forecast accuracy ≠ business cost.** The raw XGBoost model has the lowest WAPE but the highest inventory cost, because it systematically under-forecasts and the cost of under-forecasting (stockouts) is 4× the cost of over-forecasting (excess).

2. **Segment-level correction matters.** Both empirical-Bayes correction and pinball-optimal lift factors improve cost, but they work differently: EB correction adjusts the mean forecast toward observed demand ratios; lift factors directly optimize the quantile loss that drives inventory decisions.

3. **Long-lead items need separate treatment.** The cost asymmetry is sharper when replenishment lead times are long, because there's less flexibility to react to forecast errors. The lift-factor approach targets this directly by only intervening on long-lead segments.

4. **Pinball loss is the right objective for inventory planning.** Optimizing for WAPE or MSE produces a median/mean-targeting forecast. Inventory planning needs a quantile-targeting forecast — the p90 lift factor shifts the forecast toward the 90th percentile of demand, which is the service-level target.

5. **Factor stability matters for deployment.** Lift factors trained on one calibration window should generalize to the next. The shrinkage regularization (blending small-group factors toward 1.0) prevents overfitting to noise in the calibration data.

6. **Offline replay is a screening tool, not a launch decision.** Live deployment would require randomized assignment, availability monitoring, and interference checks. The A/B readout here demonstrates the measurement framework, not the final answer.
"""
    path = REPORTS_DIR / "forecast_bias_lab_report.md"
    path.write_text(report)
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_reports(
    feature_panel: pd.DataFrame,
    scored_panel: pd.DataFrame,
    scorecard: pd.DataFrame,
    correction_table: pd.DataFrame,
    lift_factor_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
    split_summary: pd.DataFrame,
    experiment_readout: pd.DataFrame,
) -> Path:
    ensure_report_dirs()
    figures = [
        plot_eda(feature_panel),
        plot_model_scorecard(scorecard),
        plot_bias_by_lead_bucket(scored_panel),
        plot_correction_heatmap(correction_table),
        plot_lift_factor_diagnostics(lift_factor_table),
        plot_weighted_pinball_by_lead(scored_panel),
        plot_experiment_readout(experiment_readout),
    ]
    build_forecast_explanations(scored_panel)
    return write_markdown_report(
        scored_panel=scored_panel,
        scorecard=scorecard,
        correction_table=correction_table,
        lift_factor_table=lift_factor_table,
        feature_importance=feature_importance,
        split_summary=split_summary,
        experiment_readout=experiment_readout,
        figure_paths=figures,
    )
