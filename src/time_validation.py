"""Extension 1: time-based (chronological) validation."""
from .data_loading import split_xy


def chronological_split(clean, test_frac=0.20):
    """Time-ordered split: the first 80% of rows train the model, the last 20% test it.

    The raw file is in date order (May 2008 to Nov 2010), so this simulates
    training on the past and predicting the future.
    """
    n_test = int(round(len(clean) * test_frac))
    X_tr, y_tr = split_xy(clean.iloc[:-n_test])
    X_te, y_te = split_xy(clean.iloc[-n_test:])
    return X_tr, X_te, y_tr, y_te