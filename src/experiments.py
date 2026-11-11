"""Controlled experiments that justify preprocessing choices (Parts B and D).

Each experiment changes ONE preprocessing decision and holds everything else
fixed, then compares 5-fold stratified cross-validation scores on the training
set only. The test set is never touched here.
"""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import CV_FOLDS, RANDOM_STATE
from .preprocessing import ENG_NUMERIC, build_preprocessor, make_pipeline

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1",
           "recall": "recall", "precision": "precision"}


def _cv():
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _lr(cw="balanced"):
    return LogisticRegression(max_iter=3000, class_weight=cw, random_state=RANDOM_STATE)


def _hgb(cw="balanced"):
    return HistGradientBoostingClassifier(class_weight=cw, random_state=RANDOM_STATE)


def cv_score(pipe, X, y):
    t = time.perf_counter()
    r = cross_validate(pipe, X, y, cv=_cv(), scoring=SCORING, n_jobs=-1)
    out = {k: float(np.mean(r[f"test_{k}"])) for k in SCORING}
    out["pr_auc_std"] = float(np.std(r["test_pr_auc"]))
    out["fit_seconds"] = round(time.perf_counter() - t, 1)
    return out


def preprocessing_experiments(X, y) -> pd.DataFrame:
    """One-factor-at-a-time comparison of preprocessing options."""
    configs = {
        # ---- unknown handling
        ("Unknown handling", "LR", "keep 'unknown' as category"): make_pipeline(_lr()),
        ("Unknown handling", "LR", "mode imputation"): make_pipeline(_lr(), unknown_strategy="mode"),
        ("Unknown handling", "HGB", "keep 'unknown' as category"): make_pipeline(_hgb()),
        ("Unknown handling", "HGB", "mode imputation"): make_pipeline(_hgb(), unknown_strategy="mode"),
        # ---- class imbalance
        ("Imbalance", "LR", "none"): make_pipeline(_lr(None)),
        ("Imbalance", "LR", "class weights"): make_pipeline(_lr()),
        ("Imbalance", "LR", "SMOTE"): make_pipeline(_lr(None), sampler="smote"),
        ("Imbalance", "LR", "random undersampling"): make_pipeline(_lr(None), sampler="undersample"),
        ("Imbalance", "HGB", "none"): make_pipeline(_hgb(None)),
        ("Imbalance", "HGB", "class weights"): make_pipeline(_hgb()),
        ("Imbalance", "HGB", "SMOTE"): make_pipeline(_hgb(None), sampler="smote"),
        ("Imbalance", "HGB", "random undersampling"): make_pipeline(_hgb(None), sampler="undersample"),
        # ---- scaling (matters for the linear model only)
        ("Scaling", "LR", "standard scaling"): make_pipeline(_lr()),
        ("Scaling", "LR", "no scaling"): make_pipeline(_lr(), scale=False),
        # ---- correlation-based feature selection
        ("Feature selection", "LR", "correlation filter |rho|>0.90"): make_pipeline(_lr()),
        ("Feature selection", "LR", "all features"): make_pipeline(_lr(), corr_threshold=None),
        ("Feature selection", "HGB", "correlation filter |rho|>0.90"): make_pipeline(_hgb()),
        ("Feature selection", "HGB", "all features"): make_pipeline(_hgb(), corr_threshold=None),
        # ---- PCA
        ("PCA", "LR", "no PCA"): make_pipeline(_lr()),
        ("PCA", "LR", "PCA (95% variance)"): make_pipeline(_lr(), pca_components=0.95),
        ("PCA", "HGB", "no PCA"): make_pipeline(_hgb()),
        ("PCA", "HGB", "PCA (95% variance)"): make_pipeline(_hgb(), pca_components=0.95),
    }
    rows = []
    for (exp, model, variant), pipe in configs.items():
        r = cv_score(pipe, X, y)
        rows.append({"experiment": exp, "model": model, "variant": variant, **r})
    return pd.DataFrame(rows).set_index(["experiment", "model", "variant"]).round(4)


def leakage_benchmark(X_with_duration, y) -> pd.DataFrame:
    """Quantify how much `duration` inflates performance (why it must be dropped)."""
    rows = {}
    rows["HGB without duration (deployable)"] = cv_score(make_pipeline(_hgb()), X_with_duration.drop(columns="duration"), y)
    rows["HGB with duration (leaky)"] = cv_score(
        make_pipeline(_hgb(), numeric=ENG_NUMERIC + ["duration"]), X_with_duration, y)
    return pd.DataFrame(rows).T.round(4)


def pca_variance(X):
    from sklearn.decomposition import PCA
    Z = build_preprocessor().fit_transform(X)
    pca = PCA(random_state=RANDOM_STATE).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return Z.shape[1], cum


def mutual_information(X, y) -> pd.Series:
    """Mutual information between each (encoded) feature and the target."""
    pre = build_preprocessor(corr_threshold=None)
    Z = pre.fit_transform(X)
    mi = mutual_info_classif(Z, y, random_state=RANDOM_STATE)
    return pd.Series(mi, index=Z.columns, name="mutual_info").sort_values(ascending=False)


def unsupervised_probe(X, y, ks=range(2, 9), sample=10000) -> pd.DataFrame:
    """Part B evidence: do natural clusters line up with the subscription label?"""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Z = build_preprocessor().fit_transform(X.iloc[idx])
    yy = np.asarray(y)[idx]
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        rates = pd.Series(yy).groupby(km.labels_).mean()
        rows.append({"k": k, "silhouette": silhouette_score(Z, km.labels_, sample_size=5000, random_state=RANDOM_STATE),
                     "ARI_vs_label": adjusted_rand_score(yy, km.labels_),
                     "max_cluster_sub_rate": rates.max(), "min_cluster_sub_rate": rates.min()})
    return pd.DataFrame(rows).set_index("k").round(4)
