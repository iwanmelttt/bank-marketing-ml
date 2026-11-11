"""Cleaning, feature engineering and the leakage-safe preprocessing pipeline (Part D).

Two kinds of step live here:

* ``clean_and_engineer`` is *stateless* (it learns nothing from the data), so it
  is safe to apply to the whole dataset before the train/test split. Its output
  is the published cleaned dataset.
* ``build_preprocessor`` returns *fitted* steps (imputation, scaling, encoding,
  correlation filter, PCA, SMOTE). These are wrapped in a pipeline and fitted on
  training folds only, so no information leaks from validation or test data.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LEAKAGE_COLS, RANDOM_STATE, TARGET

# Columns produced by clean_and_engineer
ENG_NOMINAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
ENG_NUMERIC = ["age", "campaign_log", "previous", "prev_contacted", "emp_var_rate",
               "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]


def clean_and_engineer(df: pd.DataFrame, drop_leakage: bool = True) -> pd.DataFrame:
    """Deterministic cleaning and feature engineering.

    1. Remove exact duplicate rows (12 in the raw file).
    2. Drop `duration` (target leakage: only known after the call).
    3. Strip whitespace from text values.
    4. `default`: only 3 'yes' values, so merge them into 'unknown'. The
       remaining split ('no' vs 'unknown/yes') is what carries signal.
    5. `pdays`: 999 is a sentinel for "never contacted" (96% of rows). Replace
       with a binary `prev_contacted` flag and drop the sentinel-coded column.
    6. `campaign`: heavily right-skewed count (skew about 4.8). Use log1p.
    7. 'unknown' values in categorical columns are KEPT as a level here; the
       choice of imputing them instead is made inside the fitted pipeline.
    """
    out = df.drop_duplicates().reset_index(drop=True).copy()
    if drop_leakage:
        out = out.drop(columns=[c for c in LEAKAGE_COLS if c in out.columns])
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].str.strip()
    out["default"] = out["default"].replace({"yes": "unknown"})
    out["prev_contacted"] = (out["pdays"] != 999).astype(int)
    out = out.drop(columns=["pdays"])
    out["campaign_log"] = np.log1p(out["campaign"])
    out = out.drop(columns=["campaign"])
    cols = ENG_NOMINAL + ENG_NUMERIC + (["duration"] if "duration" in out.columns else []) + [TARGET]
    return out[cols]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from every pair whose |Spearman rho| exceeds `threshold`.

    Fitted on training data only. Greedy: scans columns in order and drops a
    column if it is highly correlated with one already kept.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr(method="spearman").abs()
        keep = []
        for c in corr.columns:
            if all(corr.loc[c, k] <= self.threshold for k in keep):
                keep.append(c)
        self.keep_ = keep
        self.dropped_ = [c for c in X.columns if c not in keep]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_, dtype=object)


def build_preprocessor(unknown_strategy: str = "category", corr_threshold=0.90,
                       scale: bool = True, numeric=None, nominal=None) -> ColumnTransformer:
    """ColumnTransformer for numeric and nominal columns.

    unknown_strategy:
        'category' -> keep 'unknown' as its own level (treats missingness as information)
        'mode'     -> replace 'unknown' with the most frequent level of the training fold
    """
    numeric = numeric or ENG_NUMERIC
    nominal = nominal or ENG_NOMINAL
    num_steps = []
    if corr_threshold is not None:
        num_steps.append(("corr_filter", CorrelationFilter(corr_threshold)))
    num_steps.append(("scale", StandardScaler() if scale else "passthrough"))

    cat_steps = []
    if unknown_strategy == "mode":
        cat_steps.append(("impute", SimpleImputer(missing_values="unknown", strategy="most_frequent")))
    cat_steps.append(("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                              min_frequency=30, sparse_output=False)))
    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), nominal)],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre


def make_pipeline(model, unknown_strategy="category", sampler=None, pca_components=None,
                  corr_threshold=0.90, scale=True, numeric=None, nominal=None) -> ImbPipeline:
    """Full leakage-safe pipeline: preprocess -> [PCA] -> [resampler] -> model.

    sampler: None, 'smote', or 'undersample'. Resampling sits inside the
    imblearn pipeline so it is applied to training folds only, never to the
    validation or test data.
    """
    steps = [("prep", build_preprocessor(unknown_strategy, corr_threshold, scale, numeric, nominal))]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))
    if sampler == "smote":
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE)))
    elif sampler == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        steps.append(("sampler", RandomUnderSampler(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return ImbPipeline(steps)
