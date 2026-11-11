"""Project-wide constants: paths, random seed and column groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "bank-additional-full.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "bank_marketing_cleaned.csv"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "y"

# `duration` is only known after the call ends, so it leaks the outcome
# and must not be used by a model that decides whom to call.
LEAKAGE_COLS = ["duration"]

NOMINAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous", "emp_var_rate",
    "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
]
MACRO_COLS = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]
