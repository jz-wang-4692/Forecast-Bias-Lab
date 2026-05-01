from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

M5_ZENODO_URL = (
    "https://zenodo.org/records/12636070/files/"
    "m5-forecasting-accuracy.zip?download=1"
)
M5_ZIP_PATH = RAW_DIR / "m5-forecasting-accuracy.zip"
M5_REQUIRED_FILES = [
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
]

DEFAULT_START_DAY = 1200
DEFAULT_END_DAY = 1913
DEFAULT_SAMPLE_SERIES_PER_STORE_CAT = 35
DEFAULT_TEST_WEEKS = 16
DEFAULT_CALIBRATION_WEEKS = 12
RANDOM_SEED = 20260430

WEEKLY_PANEL_PATH = PROCESSED_DIR / "weekly_m5_retail_panel.parquet"
WEEKLY_PANEL_METADATA_PATH = PROCESSED_DIR / "weekly_m5_retail_panel.metadata.json"
FEATURE_PANEL_PATH = PROCESSED_DIR / "weekly_feature_panel.parquet"
FEATURE_PANEL_METADATA_PATH = PROCESSED_DIR / "weekly_feature_panel.metadata.json"

# --- Pinball / asymmetric cost parameters ---
PINBALL_TAU_P50 = 0.5
PINBALL_TAU_P90 = 0.9
UNDERAGE_COST = 4.0   # cost per unit of under-forecast (lost sale / stockout)
OVERAGE_COST = 1.0    # cost per unit of over-forecast (excess inventory)

# --- Lift factor grid search ---
LIFT_FACTOR_MIN = 0.50
LIFT_FACTOR_MAX = 3.00
LIFT_FACTOR_STEP = 0.01
SHRINKAGE_MIN_OBS = 50     # below this → full shrinkage toward 1.0
SHRINKAGE_FULL_OBS = 400   # above this → use raw grid-searched factor

# --- Lead bucket definitions ---
LEAD_BUCKET_EDGES = [0, 7, 14, 28, 56, 98, 10_000]
LEAD_BUCKET_LABELS = ["0-7", "8-14", "15-28", "29-56", "57-98", "99+"]
LONG_LEAD_BUCKETS = {"57-98", "99+"}
