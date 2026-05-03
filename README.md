# Forecast Bias Lab: Asymmetric Loss Optimization for Retail Demand

A personal analytics project exploring how forecast accuracy, forecast bias, and business cost diverge — and what to do about it.

Uses the public M5 Forecasting Accuracy dataset (Walmart item-store daily unit sales, prices, and retail calendar events). Since the public data does not include supplier lead times or true inventory stockout labels, the code creates a transparent operational layer: deterministic simulated lead-time buckets and stockout-risk proxies derived from censored zero-sales behavior.

## What This Project Is About

Standard forecast evaluation (WAPE, MAPE, RMSE) optimizes for the mean or median of the demand distribution. But inventory planning needs a *quantile* forecast — specifically, a forecast at the service-level target (e.g., p90). Under-forecasting is more expensive than over-forecasting when stockouts carry higher penalties than excess inventory, and this asymmetry is sharper for long-lead items where replenishment flexibility is limited.

This project builds four forecast policies and compares them on the metric that actually matters: asymmetric inventory cost.

## Project Shape

```text
.
├── README.md
├── requirements.txt
├── src/forecast_bias_lab/
│   ├── config.py          # parameters, paths, lead-bucket definitions
│   ├── data.py            # M5 download, weekly panel construction
│   ├── features.py        # lag, rolling, price, event, lead-time, stockout features
│   ├── metrics.py         # WAPE, bias, pinball loss, weighted pinball, newsvendor cost
│   ├── modeling.py        # gradient-boosted model, EB correction, lift factor integration
│   ├── lift_factors.py    # pinball-optimal multiplicative lift factors by segment
│   ├── experiment.py      # clustered A/B readout with cluster-robust SEs
│   ├── reporting.py       # figures, tables, markdown report
│   └── run_project.py     # end-to-end pipeline
├── data/
│   ├── raw/               # downloaded M5 files
│   └── processed/         # generated parquet artifacts
└── reports/
    ├── forecast_bias_lab_report.md
    ├── model_scorecard.csv
    ├── lift_factor_table.csv
    ├── experiment_readout.csv
    └── figures/
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m forecast_bias_lab.run_project
```

For a faster smoke test:

```bash
PYTHONPATH=src python -m forecast_bias_lab.run_project --sample-series-per-store-cat 8 --force-rebuild
```

## Analytical Approach

### 1. Demand Panel & Features
Weekly product-node demand panel from real item-store retail data. Features include lag/rolling-window demand, price dynamics, calendar events, SNAP benefits, simulated supplier lead-time, and stockout-risk proxies.

### 2. Gradient-Boosted Demand Model
Point forecast trained on log-demand with train/calibration/test temporal split. The pipeline uses XGBoost when the local runtime supports it and falls back to scikit-learn gradient boosting otherwise, so the project remains runnable on a fresh desktop setup. The calibration window is used to fit post-hoc corrections without leaking test data.

### 3. Empirical-Bayes Bias Correction
Residual-ratio correction by `category × store × lead_bucket × event_week`. Shrinkage toward the global ratio prevents overfitting in small segments.

### 4. Pinball-Optimal Lift Factors
The key analytical contribution. For each `lead_bucket × is_event_week` segment:
- Grid-search a multiplicative factor f ∈ [0.5, 3.0] that minimizes demand-weighted pinball loss
- Separate factors for p50 (τ=0.5) and p90 (τ=0.9)
- Shrinkage regularization: small groups blend toward 1.0
- Constraint: f_p50 ≤ f_p90 enforced at every segment
- Only long-lead buckets (57-98, 99+) receive intervention; short-lead buckets keep factor=1.0

This produces two adjusted forecasts: `lift_p50_forecast` (median-targeting) and `lift_p90_forecast` (safety-stock-targeting).

### 5. Holdout Evaluation
Five forecast policies compared on: WAPE, signed bias, under-forecast rate, pinball loss (τ=0.5, τ=0.9), demand-weighted pinball, and asymmetric inventory cost (underage=9, overage=1).

### 6. Clustered A/B Readout
Simulates a clustered policy readout: series hash-assigned to treatment/control, with pre-period cost adjustment, covariate controls, and cluster-robust standard errors. Two treatment arms: bias-corrected and lift-factor p90.

## Key Results

- The raw ML model has strong WAPE but poor asymmetric-cost performance when it under-forecasts.
- Empirical-Bayes correction reduces inventory cost by ~31% vs the raw ML forecast.
- Pinball-optimal lift factors directly target the quantile loss that drives inventory decisions.
- Long-lead segments show the largest improvement because cost asymmetry is sharpest there.
- The A/B readout confirms statistically significant cost reduction with cluster-robust inference.

## Key Insight

> Optimizing for WAPE can produce a forecast that looks accurate but costs money.
> Optimizing for pinball loss at the service-level quantile produces a forecast
> that may look upward-biased but better matches the inventory decision. The
> right metric depends on the decision the forecast supports.

## Data Source

M5 Forecasting Accuracy dataset, originally from Kaggle and mirrored on Zenodo:
- https://zenodo.org/records/12636070
- DOI: `10.5281/zenodo.12636070`

The generated lead-time and stockout-risk fields are intentionally marked as operational simulation/proxy fields in the code and report.
