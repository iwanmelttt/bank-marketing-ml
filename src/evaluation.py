"""Metrics, threshold selection, bootstrap confidence intervals and plots (Part H)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from .config import FIG_DIR, RANDOM_STATE


def lift_at(y_true, proba, frac=0.20):
    """Share of all subscribers captured when calling the top `frac` of clients by score."""
    y_true = np.asarray(y_true)
    n = int(np.ceil(frac * len(y_true)))
    top = np.argsort(-proba)[:n]
    captured = y_true[top].sum() / y_true.sum()
    return captured, captured / frac


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def metrics(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    cap20, lift20 = lift_at(y_true, proba, 0.20)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "capture_top20": cap20,
        "lift_top20": lift20,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ci(y_true, proba, n_boot=1000, seed=RANDOM_STATE):
    """Percentile 95% CIs for ROC-AUC and PR-AUC on the test set."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
        aps.append(average_precision_score(y_true[idx], proba[idx]))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"roc_auc_ci": q(aucs), "pr_auc_ci": q(aps)}


def paired_bootstrap_diff(y_true, proba_a, proba_b, n_boot=1000, seed=RANDOM_STATE,
                          metric=average_precision_score):
    """Paired bootstrap of metric(a) - metric(b). Returns mean diff, 95% CI and P(diff <= 0)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        d.append(metric(y_true[idx], proba_a[idx]) - metric(y_true[idx], proba_b[idx]))
    d = np.array(d)
    return {"mean_diff": float(d.mean()), "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
            "p_le_zero": float((d <= 0).mean())}


def plot_curves(y_true, probas: dict, fname="10_roc_pr_curves.png"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_true, p)
        a2.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    a2.axhline(np.mean(y_true), color="k", ls="--", lw=1, label=f"baseline ({np.mean(y_true):.3f})")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    a1.legend(loc="lower right", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(y_true, probas: dict, thresholds: dict, fname="11_confusion_matrices.png"):
    n = len(probas)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    for ax, (name, p) in zip(np.atleast_1d(axes), probas.items()):
        cm = confusion_matrix(y_true, (p >= thresholds[name]).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["no", "yes"])
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\n(threshold={thresholds[name]:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_gain_chart(y_true, probas: dict, fname="12_cumulative_gains.png"):
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.linspace(0, 1, 101)
    for name, p in probas.items():
        order = np.argsort(-p)
        cum = np.cumsum(y_true[order]) / y_true.sum()
        frac = np.arange(1, len(y_true) + 1) / len(y_true)
        ax.plot(frac, cum, label=name)
    ax.plot(x, x, "k--", lw=1, label="random calling")
    ax.axvline(0.2, color="grey", ls=":", lw=1)
    ax.set(xlabel="Share of clients called (ranked by score)", ylabel="Share of subscribers reached",
           title="Cumulative gains (test set)")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)


def results_table(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).T.round(4)
