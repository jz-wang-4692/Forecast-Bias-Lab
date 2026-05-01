from __future__ import annotations

import argparse

from forecast_bias_lab.config import (
    DEFAULT_CALIBRATION_WEEKS,
    DEFAULT_END_DAY,
    DEFAULT_SAMPLE_SERIES_PER_STORE_CAT,
    DEFAULT_START_DAY,
    DEFAULT_TEST_WEEKS,
    REPORTS_DIR,
)
from forecast_bias_lab.data import build_weekly_panel, ensure_m5_raw_data
from forecast_bias_lab.experiment import run_experiment_readout
from forecast_bias_lab.features import add_feature_panel
from forecast_bias_lab.modeling import run_modeling
from forecast_bias_lab.reporting import ensure_report_dirs, generate_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Forecast Bias Lab analysis.")
    parser.add_argument(
        "--sample-series-per-store-cat",
        type=int,
        default=DEFAULT_SAMPLE_SERIES_PER_STORE_CAT,
        help="Top item-store series retained per store/category.",
    )
    parser.add_argument("--start-day", type=int, default=DEFAULT_START_DAY)
    parser.add_argument("--end-day", type=int, default=DEFAULT_END_DAY)
    parser.add_argument("--test-weeks", type=int, default=DEFAULT_TEST_WEEKS)
    parser.add_argument("--calibration-weeks", type=int, default=DEFAULT_CALIBRATION_WEEKS)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_report_dirs()
    raw_paths = ensure_m5_raw_data(force_download=args.force_download)
    weekly = build_weekly_panel(
        raw_paths=raw_paths,
        sample_series_per_store_cat=args.sample_series_per_store_cat,
        start_day=args.start_day,
        end_day=args.end_day,
        force_rebuild=args.force_rebuild,
    )
    print(f"Weekly panel: {weekly.shape[0]:,} rows, {weekly['id'].nunique():,} series.")

    features = add_feature_panel(weekly, force_rebuild=args.force_rebuild)
    print(f"Feature panel: {features.shape[0]:,} rows, {features.shape[1]:,} columns.")

    model_result = run_modeling(
        features,
        test_weeks=args.test_weeks,
        calibration_weeks=args.calibration_weeks,
    )
    experiment_panel, experiment_readout = run_experiment_readout(model_result.scored_panel)
    experiment_panel.to_csv(REPORTS_DIR / "experiment_panel.csv", index=False)
    experiment_readout.to_csv(REPORTS_DIR / "experiment_readout.csv", index=False)

    report_path = generate_reports(
        feature_panel=features,
        scored_panel=model_result.scored_panel,
        scorecard=model_result.scorecard,
        correction_table=model_result.bias_correction_table,
        lift_factor_table=model_result.lift_factor_table,
        feature_importance=model_result.feature_importance,
        split_summary=model_result.split_summary,
        experiment_readout=experiment_readout,
    )
    model_result.scored_panel.to_parquet(REPORTS_DIR / "scored_panel.parquet", index=False)

    print("\n" + "=" * 70)
    print("Model scorecard:")
    print("=" * 70)
    print(model_result.scorecard.to_string(index=False))
    print("\n" + "=" * 70)
    print("Lift factor table (long-lead segments):")
    print("=" * 70)
    long_lift = model_result.lift_factor_table[model_result.lift_factor_table["is_long_lead"]]
    if not long_lift.empty:
        print(long_lift[["lead_bucket", "is_event_week", "n_obs", "f_p50", "f_p90",
                          "delta_p50_pct", "delta_p90_pct"]].to_string(index=False))
    print("\n" + "=" * 70)
    print("Experiment readout:")
    print("=" * 70)
    print(experiment_readout.to_string(index=False))
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
