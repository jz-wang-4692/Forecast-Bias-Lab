# Forecast Bias Lab Report

Generated: 2026-04-30

## Executive Summary

This project evaluates demand-forecasting policies on public retail item-store sales data, with emphasis on **asymmetric loss optimization**, **lead-time-conditional bias correction**, **pinball-optimal lift factors**, and **A/B-style policy readout**.

The best holdout forecast by asymmetric inventory cost is `baseline_forecast`. The bias-corrected model achieves WAPE `0.319` and bias `0.050` on the final holdout window. Relative to the raw XGBoost forecast, correction reduces inventory cost by `21.8%`; relative to the rolling/event baseline, it reduces inventory cost by `-0.3%`.

The lift-factor p90 forecast reduces inventory cost by `11.1%` vs raw XGBoost.

## Data and Forecasting Unit

- Source: M5 Forecasting Accuracy data (Walmart item-store daily sales, prices, calendar events).
- Forecasting grain: weekly `item_id × store_id` demand (product-node forecast).
- Operational fields: lead-time buckets and stockout-risk indicators are transparent proxies/simulations — the public M5 data does not expose supplier lead time or true inventory availability.
- Holdout: `80.4%` of test rows are long-lead eligible; stockout-censored proxy rate is `2.4%`.

## Split Summary

| split       |   rows |   series |   weeks |   demand |   stockout_proxy_rate |   long_lead_share |
|:------------|-------:|---------:|--------:|---------:|----------------------:|------------------:|
| calibration |  12600 |     1050 |      12 |   790078 |                 0.043 |             0.702 |
| test        |  16800 |     1050 |      16 |  1093665 |                 0.024 |             0.804 |
| train       |  78750 |     1050 |      75 |  5180888 |                 0.032 |             0.729 |

## Modeling Method

1. Build lag, rolling-window, price, calendar, event, SNAP, stockout-risk, and simulated supplier-lead-time features.
2. Train XGBoost on the historical train split with a log-demand target.
3. Score calibration and holdout periods.
4. Fit **empirical-Bayes residual-ratio correction** on calibration by `category × store × lead_bucket × event_week`.
5. Fit **pinball-optimal lift factors** on calibration by `lead_bucket × is_event_week`, using grid search over [0.5, 3.0] with shrinkage regularization.
6. Evaluate on holdout using WAPE, signed bias, under-forecast rate, pinball loss (τ=0.5, τ=0.9), weighted pinball, and asymmetric inventory cost.

Top feature families by model importance: demand_lag, rolling_mean, calendar_event, store_id, availability_risk, price, cat_id, dept_id.

Bias-correction factors range from `1.06` to `1.31` after shrinkage and caps.

## Lift Factor Summary (Long-Lead Segments)

Only long-lead buckets (57-98, 99+) receive lift-factor intervention. Short-lead buckets keep factor=1.0.

  - `57-98` / event=1: f_p50=1.08, f_p90=1.68 (p90 WE improvement: +51.1%)
  - `57-98` / event=0: f_p50=1.11, f_p90=1.70 (p90 WE improvement: +55.0%)
  - `99+` / event=1: f_p50=1.09, f_p90=1.68 (p90 WE improvement: +60.7%)
  - `99+` / event=0: f_p50=1.12, f_p90=1.67 (p90 WE improvement: +63.5%)

Constraint enforced: `adj_p50 ≤ adj_p90` at every row.

## Holdout Scorecard

| forecast           |   wape |    bias |   underforecast_rate |   pinball_p50 |   pinball_p90 |   weighted_pinball_p50 |   weighted_pinball_p90 |   inventory_cost |   inventory_cost_per_unit_demand |   cost_lift_vs_best |
|:-------------------|-------:|--------:|---------------------:|--------------:|--------------:|-----------------------:|-----------------------:|-----------------:|---------------------------------:|--------------------:|
| baseline_forecast  | 0.3457 |  0.0957 |               0.3912 |       11.2519 |        8.7589 |                 0.1728 |                 0.1345 |      788098.6477 |                           0.7206 |              0.0000 |
| xgb_bias_corrected | 0.3190 |  0.0499 |               0.3859 |       10.3819 |        9.0825 |                 0.1595 |                 0.1395 |      790215.9450 |                           0.7225 |              0.0027 |
| lift_p90_forecast  | 0.5192 |  0.3178 |               0.1927 |       16.8995 |        8.6234 |                 0.2596 |                 0.1325 |      898162.6202 |                           0.8212 |              0.1397 |
| lift_p50_forecast  | 0.3103 | -0.0362 |               0.4695 |       10.1014 |       11.0437 |                 0.1552 |                 0.1696 |      907880.0091 |                           0.8301 |              0.1520 |
| xgb_forecast       | 0.3113 | -0.0970 |               0.5552 |       10.1325 |       12.6594 |                 0.1556 |                 0.1945 |     1010326.6593 |                           0.9238 |              0.2820 |

## Experiment Readout

The long-lead readout simulates a clustered A/B test: item-store series are hash-assigned to treatment/control, the treatment receives the candidate forecast, and the control receives the raw ML forecast. The estimator uses pre-period cost adjustment, lagged demand, event/SNAP flags, category/state/lead-bucket controls, and cluster-robust standard errors by item-store series.

| estimand                               | treatment          | control      |   rows |   clusters |   treatment_clusters |   control_clusters |   control_mean_assigned_cost |   treatment_mean_assigned_cost |   regression_ate |   cluster_robust_se |   ci_low |   ci_high |   p_value |   relative_ate_vs_control_mean |   paired_counterfactual_delta |   paired_counterfactual_relative_delta |   r_squared |
|:---------------------------------------|:-------------------|:-------------|-------:|-----------:|---------------------:|-------------------:|-----------------------------:|-------------------------------:|-----------------:|--------------------:|---------:|----------:|----------:|-------------------------------:|------------------------------:|---------------------------------------:|------------:|
| ATE on weekly long-lead inventory cost | xgb_bias_corrected | xgb_forecast |  13509 |        911 |                  485 |                426 |                      48.9667 |                        36.9261 |         -12.5246 |              3.3723 | -19.1342 |   -5.9150 |    0.0002 |                        -0.2558 |                      -10.8713 |                                -0.2255 |      0.1102 |
| ATE on weekly long-lead inventory cost | lift_p90_forecast  | xgb_forecast |  13509 |        911 |                  485 |                426 |                      48.9667 |                        40.8682 |          -8.6972 |              3.2253 | -15.0187 |   -2.3756 |    0.0070 |                        -0.1776 |                       -7.8612 |                                -0.1630 |      0.1420 |

Interpretation: negative ATE means the treatment reduced weekly long-lead inventory cost. The paired counterfactual delta is included because this offline backtest observes both potential forecast policies for every row; the regression readout mirrors what would be available in a randomized launch.

## Key Insights

1. **Forecast accuracy ≠ business cost.** The raw XGBoost model has the lowest WAPE but the highest inventory cost, because it systematically under-forecasts and the cost of under-forecasting (stockouts) is 4× the cost of over-forecasting (excess).

2. **Segment-level correction matters.** Both empirical-Bayes correction and pinball-optimal lift factors improve cost, but they work differently: EB correction adjusts the mean forecast toward observed demand ratios; lift factors directly optimize the quantile loss that drives inventory decisions.

3. **Long-lead items need separate treatment.** The cost asymmetry is sharper when replenishment lead times are long, because there's less flexibility to react to forecast errors. The lift-factor approach targets this directly by only intervening on long-lead segments.

4. **Pinball loss is the right objective for inventory planning.** Optimizing for WAPE or MSE produces a median/mean-targeting forecast. Inventory planning needs a quantile-targeting forecast — the p90 lift factor shifts the forecast toward the 90th percentile of demand, which is the service-level target.

5. **Factor stability matters for deployment.** Lift factors trained on one calibration window should generalize to the next. The shrinkage regularization (blending small-group factors toward 1.0) prevents overfitting to noise in the calibration data.

6. **Offline replay is a screening tool, not a launch decision.** Live deployment would require randomized assignment, availability monitoring, and interference checks. The A/B readout here demonstrates the measurement framework, not the final answer.
