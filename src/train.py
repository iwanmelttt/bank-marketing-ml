"""End-to-end training and evaluation pipeline.

Run from the project root:
    python -m src.train
It writes the cleaned dataset, figures, metrics tables and fitted models.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import evaluation as ev
from .config import CLEAN_PATH, CV_FOLDS, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from .data_loading import load_raw, split_xy
from .models import MODEL_LABELS, base_models, stacking_ensemble, tune, voting_ensemble
from .preprocessing import clean_and_engineer


def prepare_data():
    raw = load_raw()
    clean = clean_and_engineer(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)
    X, y = split_xy(clean)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
    return raw, clean, X_tr, X_te, y_tr, y_te


def tune_base_models(X_tr, y_tr, imbalance="class_weight"):
    tuned, info = {}, {}
    for name, pipe in base_models(imbalance).items():
        t = time.perf_counter()
        s = tune(name, pipe, X_tr, y_tr)
        tuned[name] = s.best_estimator_
        info[name] = {"best_cv_pr_auc_3fold": float(s.best_score_),
                      "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                      for k, v in s.best_params_.items()},
                      "search_seconds": round(time.perf_counter() - t, 1)}
        print(f"tuned {name}: PR-AUC={s.best_score_:.4f} in {info[name]['search_seconds']}s")
    return tuned, info


def oof_probas(estimators: dict, X_tr, y_tr):
    """5-fold out-of-fold probabilities: an unbiased training-set estimate per model."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for name, est in estimators.items():
        t = time.perf_counter()
        out[name] = cross_val_predict(clone(est), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        print(f"oof {name}: {time.perf_counter() - t:.0f}s")
    return out


def rank_models(oof: dict, y_tr) -> pd.DataFrame:
    rows = {n: ev.metrics(y_tr, p, ev.best_f1_threshold(y_tr, p)) for n, p in oof.items()}
    return ev.results_table(rows).sort_values("pr_auc", ascending=False)


def build_ensembles(tuned: dict, top2: list):
    members = {n: clone(tuned[n]) for n in top2}
    return {"voting": voting_ensemble(members), "stacking": stacking_ensemble(members)}


def fit_and_test(estimators: dict, X_tr, y_tr, X_te, y_te, thresholds: dict):
    fitted, probas, rows_default, rows_tuned, cis = {}, {}, {}, {}, {}
    for name, est in estimators.items():
        t = time.perf_counter()
        m = clone(est).fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        p = m.predict_proba(X_te)[:, 1]
        pred_s = time.perf_counter() - t
        fitted[name], probas[name] = m, p
        rows_default[name] = ev.metrics(y_te, p, 0.5)
        rows_tuned[name] = {**ev.metrics(y_te, p, thresholds[name]),
                            "fit_seconds": fit_s, "predict_seconds": pred_s}
        cis[name] = ev.bootstrap_ci(y_te.values, p)
    return fitted, probas, ev.results_table(rows_default), ev.results_table(rows_tuned), cis


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    raw, clean, X_tr, X_te, y_tr, y_te = prepare_data()
    print("clean shape", clean.shape, "train", X_tr.shape, "test", X_te.shape)

    tuned, tune_info = tune_base_models(X_tr, y_tr)
    oof = oof_probas(tuned, X_tr, y_tr)
    ranking = rank_models(oof, y_tr)
    top2 = list(ranking.index[:2])
    print("OOF ranking\n", ranking[["roc_auc", "pr_auc", "f1"]], "\nTop two:", top2)

    ens = build_ensembles(tuned, top2)
    oof.update(oof_probas(ens, X_tr, y_tr))
    oof_table = rank_models(oof, y_tr)
    thresholds = {n: ev.best_f1_threshold(y_tr, p) for n, p in oof.items()}

    all_est = {**tuned, **ens}
    fitted, probas, test_default, test_tuned, cis = fit_and_test(all_est, X_tr, y_tr, X_te, y_te, thresholds)
    print("TEST (tuned thresholds)\n", test_tuned[["roc_auc", "pr_auc", "f1", "recall", "precision"]])

    best_single = top2[0]
    diffs = {e: {"pr_auc": ev.paired_bootstrap_diff(y_te.values, probas[e], probas[best_single]),
                 "roc_auc": ev.paired_bootstrap_diff(y_te.values, probas[e], probas[best_single],
                                                     metric=ev.roc_auc_score)}
             for e in ("voting", "stacking")}

    labels = {**MODEL_LABELS, "voting": "Soft Voting", "stacking": "Stacking"}
    named = {labels[k]: v for k, v in probas.items()}
    ev.plot_curves(y_te, named)
    ev.plot_confusion_matrices(y_te, named, {labels[k]: v for k, v in thresholds.items()})
    ev.plot_gain_chart(y_te, named)

    oof_table.to_csv(RESULTS_DIR / "oof_cv_results.csv")
    test_default.to_csv(RESULTS_DIR / "test_results_threshold_0.5.csv")
    test_tuned.to_csv(RESULTS_DIR / "test_results_tuned_threshold.csv")
    json.dump({"tuning": tune_info, "top2": top2, "thresholds": thresholds, "bootstrap_ci": cis,
               "ensemble_vs_best_single": diffs}, open(RESULTS_DIR / "summary.json", "w"), indent=2)
    joblib.dump(fitted, RESULTS_DIR / "fitted_models.joblib", compress=3)
    return locals()


if __name__ == "__main__":
    main()
