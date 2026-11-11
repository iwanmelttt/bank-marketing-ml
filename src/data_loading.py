"""Load the raw UCI Bank Marketing file and tidy column names."""
import pandas as pd

from .config import RAW_PATH, TARGET


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Read the semicolon-separated raw file and snake_case the column names."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    return df


def split_xy(df: pd.DataFrame):
    """Return features and a 0/1 target."""
    y = (df[TARGET] == "yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y
