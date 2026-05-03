# Forecast Bias Lab Report

Generated: 2026-05-03

## Executive Summary

This project evaluates demand-forecasting policies on public retail item-store sales data, with emphasis on **asymmetric loss optimization**, **lead-time-conditional bias correction**, **pinball-optimal lift factors**, and **A/B-style policy readout**.

The best holdout forecast by asymmetric inventory cost is `xgb_bias_corrected`. The bias-corrected model achieves WAPE `0.320` and bias `0.071` on the final holdout window. Relative to the raw ML forecast, correction reduces inventory cost by `31.0%`; relative to the rolling/event baseline, it reduces inventory cost by `2.2%`.

The lift-factor p90 forecast reduces inventory cost by `30.9%` vs raw ML forecast.

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
2. Train a gradient-boosted model on the historical train split with a log-demand target.
3. Score calibration and holdout periods.
4. Fit **empirical-Bayes residual-ratio correction** on calibration by `category × store × lead_bucket × event_week`.
5. Fit **pinball-optimal lift factors** on calibration by `lead_bucket × is_event_week`, using grid search over [0.5, 3.0] with shrinkage regularization.
6. Evaluate on holdout using WAPE, signed bias, under-forecast rate, pinball loss (τ=0.5, τ=0.9), weighted pinball, and asymmetric inventory cost.

Top feature families by model importance: demand_lag, rolling_mean, dept_id, availability_risk, rolling_distribution, cat_id, lead_bucket, store_id.

Bias-correction factors range from `1.07` to `1.36` after shrinkage and caps.

## Lift Factor Summary (Long-Lead Segments)

Only long-lead buckets (57-98, 99+) receive lift-factor intervention. Short-lead buckets keep factor=1.0.

  - `57-98` / event=1: f_p50=1.08, f_p90=1.71 (p90 WE improvement: +50.0%)
  - `57-98` / event=0: f_p50=1.11, f_p90=1.75 (p90 WE improvement: +54.6%)
  - `99+` / event=1: f_p50=1.10, f_p90=1.78 (p90 WE improvement: +57.5%)
  - `99+` / event=0: f_p50=1.16, f_p90=1.80 (p90 WE improvement: +60.9%)

Constraint enforced: `adj_p50 ≤ adj_p90` at every row.

## Holdout Scorecard

| forecast           |   wape |    bias |   underforecast_rate |   pinball_p50 |   pinball_p90 |   weighted_pinball_p50 |   weighted_pinball_p90 |   inventory_cost |   inventory_cost_per_unit_demand |   cost_lift_vs_best |
|:-------------------|-------:|--------:|---------------------:|--------------:|--------------:|-----------------------:|-----------------------:|-----------------:|---------------------------------:|--------------------:|
| xgb_bias_corrected | 0.3198 |  0.0709 |               0.3613 |       10.4099 |        8.5638 |                 0.1599 |                 0.1316 |     1438718.0350 |                           1.3155 |              0.0000 |
| lift_p90_forecast  | 0.5510 |  0.3595 |               0.1824 |       17.9364 |        8.5754 |                 0.2755 |                 0.1317 |     1440661.1558 |                           1.3173 |              0.0014 |
| baseline_forecast  | 0.3457 |  0.0957 |               0.3912 |       11.2519 |        8.7589 |                 0.1728 |                 0.1345 |     1471488.0480 |                           1.3455 |              0.0228 |
| lift_p50_forecast  | 0.3074 | -0.0284 |               0.4590 |       10.0045 |       10.7435 |                 0.1537 |                 0.1650 |     1804899.8003 |                           1.6503 |              0.2545 |
| xgb_forecast       | 0.3067 | -0.0932 |               0.5523 |        9.9841 |       12.4106 |                 0.1534 |                 0.1906 |     2084978.4437 |                           1.9064 |              0.4492 |

## Experiment Readout

The long-lead readout simulates a clustered A/B test: item-store series are hash-assigned to treatment/control, the treatment receives the adjusted forecast, and the control receives the raw ML forecast. The estimator uses pre-period cost adjustment, lagged demand, event/SNAP flags, category/state/lead-bucket controls, and cluster-robust standard errors by item-store series.

| estimand                               | treatment          | control      |   rows |   clusters |   treatment_clusters |   control_clusters |   control_mean_assigned_cost |   treatment_mean_assigned_cost |   regression_ate |   cluster_robust_se |   ci_low |   ci_high |   p_value |   relative_ate_vs_control_mean |   paired_counterfactual_delta |   paired_counterfactual_relative_delta |   r_squared |
|:---------------------------------------|:-------------------|:-------------|-------:|-----------:|---------------------:|-------------------:|-----------------------------:|-------------------------------:|-----------------:|--------------------:|---------:|----------:|----------:|-------------------------------:|------------------------------:|---------------------------------------:|------------:|
| ATE on weekly long-lead inventory cost | xgb_bias_corrected | xgb_forecast |  13509 |        911 |                  485 |                426 |                     100.6981 |                        65.6680 |         -35.8322 |              7.1483 | -49.8426 |  -21.8217 |    0.0000 |                        -0.3558 |                      -31.8658 |                                -0.3228 |      0.1003 |
| ATE on weekly long-lead inventory cost | lift_p90_forecast  | xgb_forecast |  13509 |        911 |                  485 |                426 |                     100.6981 |                        52.9024 |         -48.3356 |              6.8244 | -61.7111 |  -34.9601 |    0.0000 |                        -0.4800 |                      -46.1805 |                                -0.4677 |      0.1167 |

Interpretation: negative ATE means the treatment reduced weekly long-lead inventory cost. The paired counterfactual delta is included because this offline backtest observes both potential forecast policies for every row; the regression readout mirrors what would be available in a randomized launch.

## Key Insights

1. **Forecast accuracy ≠ business cost.** The raw ML model can look strong on WAPE while performing poorly on inventory cost, because it systematically under-forecasts and the configured cost of under-forecasting is 9× the cost of over-forecasting.

2. **Segment-level correction matters.** Both empirical-Bayes correction and pinball-optimal lift factors improve cost, but they work differently: EB correction adjusts the mean forecast toward observed demand ratios; lift factors directly optimize the quantile loss that drives inventory decisions.

3. **Long-lead items need separate treatment.** The cost asymmetry is sharper when replenishment lead times are long, because there's less flexibility to react to forecast errors. The lift-factor approach targets this directly by only intervening on long-lead segments.

4. **Pinball loss is the right objective for inventory planning.** Optimizing for WAPE or MSE produces a median/mean-targeting forecast. Inventory planning needs a quantile-targeting forecast — the p90 lift factor shifts the forecast toward the 90th percentile of demand, which is the service-level target.

5. **Factor stability matters for deployment.** Lift factors trained on one calibration window should generalize to the next. The shrinkage regularization (blending small-group factors toward 1.0) prevents overfitting to noise in the calibration data.

6. **Offline replay is a screening tool, not a launch decision.** Live deployment would require randomized assignment, availability monitoring, and interference checks. The A/B readout here demonstrates the measurement framework, not the final answer.
