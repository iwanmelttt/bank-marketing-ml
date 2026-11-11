"""Exploratory data analysis and data-quality checks (Part C)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIG_DIR, NOMINAL_COLS, NUMERIC_COLS, TARGET

sns.set_theme(style="whitegrid", palette="deep")


def _is_text(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics plus skewness and kurtosis for numeric columns."""
    num = df.select_dtypes("number")
    out = num.describe().T
    out["skew"] = num.skew()
    out["kurtosis"] = num.kurtosis()
    return out.round(3)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing (NaN), 'unknown' placeholders, cardinality and dtype."""
    rows = []
    for c in df.columns:
        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "n_unique": df[c].nunique(),
            "nan": int(df[c].isna().sum()),
            "unknown": int((df[c] == "unknown").sum()) if _is_text(df[c]) else 0,
        })
    rep = pd.DataFrame(rows).set_index("column")
    rep["unknown_pct"] = (100 * rep["unknown"] / len(df)).round(2)
    return rep


def duplicate_report(df: pd.DataFrame) -> dict:
    dup = df.duplicated(keep=False)
    feature_dup = df.drop(columns=[TARGET]).duplicated(keep=False)
    conflicting = df[feature_dup].groupby(
        list(df.drop(columns=[TARGET]).columns), dropna=False)[TARGET].nunique()
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "rows_involved_in_duplicates": int(dup.sum()),
        "feature_duplicates_with_conflicting_label": int((conflicting > 1).sum()),
    }


def iqr_outliers(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Count values beyond 1.5 x IQR from the quartiles (Tukey fences)."""
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"column": c, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": n, "pct": round(100 * n / len(df), 2)})
    return pd.DataFrame(rows).set_index("column")


def inconsistency_checks(df: pd.DataFrame) -> dict:
    """Logical consistency rules that should hold in this data."""
    return {
        # pdays = 999 means "never contacted before", so previous should be 0
        "pdays999_but_previous>0": int(((df.pdays == 999) & (df.previous > 0)).sum()),
        # poutcome 'nonexistent' should mean previous == 0
        "poutcome_nonexistent_but_previous>0": int(((df.poutcome == "nonexistent") & (df.previous > 0)).sum()),
        "previous>0_but_poutcome_nonexistent": int(((df.previous > 0) & (df.poutcome == "nonexistent")).sum()),
        # housing and loan are 'unknown' on exactly the same rows (one system gap)
        "housing_unknown_rows": int((df.housing == "unknown").sum()),
        "housing_and_loan_unknown_same_rows": bool(((df.housing == "unknown") == (df.loan == "unknown")).all()),
        "default_yes_count": int((df["default"] == "yes").sum()),
        "education_illiterate_count": int((df.education == "illiterate").sum()),
        "duration_zero_calls": int((df.duration == 0).sum()),
        "pdays_999_pct": round(100 * (df.pdays == 999).mean(), 2),
    }


def class_balance(df):
    vc = df[TARGET].value_counts()
    return pd.DataFrame({"count": vc, "pct": (100 * vc / vc.sum()).round(2)})


def subscription_rate_by(df, col):
    t = df.groupby(col)[TARGET].agg(n="size", rate=lambda s: (s == "yes").mean())
    return t.sort_values("rate", ascending=False)


# ----------------------------------------------------------------- figures
def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vc = df[TARGET].value_counts()
    ax.bar(vc.index, vc.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n({100*v/vc.sum():.1f}%)", ha="center", va="bottom")
    ax.set_title("Target class balance (y)")
    ax.set_ylabel("Clients")
    ax.set_ylim(0, vc.max() * 1.2)
    _save(fig, "01_class_balance.png")


def plot_numeric_distributions(df):
    cols = ["age", "duration", "campaign", "pdays", "previous"] + [
        "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]
    fig, axes = plt.subplots(2, 5, figsize=(21, 8.5))
    for i, (ax, c) in enumerate(zip(axes.ravel(), cols)):
        sns.histplot(data=df, x=c, hue=TARGET, ax=ax, bins=40, stat="density",
                     common_norm=False, element="step", legend=(i == 0))
        ax.set_title(f"{c} (skew={df[c].skew():.2f})", fontsize=11)
        ax.set_xlabel("")
    fig.suptitle("Numeric feature distributions by outcome (density, each class normalised)")
    fig.tight_layout()
    _save(fig, "02_numeric_distributions.png")


def plot_boxplots(df):
    cols = ["age", "duration", "campaign", "previous"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, c in zip(axes, cols):
        sns.boxplot(data=df, x=TARGET, y=c, ax=ax)
        ax.set_title(c)
    fig.suptitle("Outlier inspection (box plots by outcome)", y=1.03)
    _save(fig, "03_boxplots.png")


def plot_correlation(df):
    num = df.select_dtypes("number").copy()
    num["y"] = (df[TARGET] == "yes").astype(int)
    corr = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Spearman rho"})
    ax.set_title("Spearman correlation matrix (numeric features and target)")
    _save(fig, "04_correlation_heatmap.png")
    return corr


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def categorical_association(df):
    return pd.Series({c: cramers_v(df[c], df[TARGET]) for c in NOMINAL_COLS},
                     name="cramers_v").sort_values(ascending=False).round(3)


def plot_categorical_rates(df):
    cols = ["job", "education", "month", "poutcome", "contact", "default"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    base = (df[TARGET] == "yes").mean()
    for ax, c in zip(axes.ravel(), cols):
        t = subscription_rate_by(df, c)
        ax.barh(t.index.astype(str), t["rate"], color="#4C72B0")
        ax.axvline(base, color="#C44E52", ls="--", label=f"overall {base:.1%}")
        ax.set_title(f"Subscription rate by {c}")
        ax.invert_yaxis()
        ax.legend(loc="lower right")
    _save(fig, "05_categorical_rates.png")


def plot_unknowns(df):
    rep = quality_report(df)
    rep = rep[rep["unknown"] > 0].sort_values("unknown_pct")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(rep.index, rep["unknown_pct"], color="#8172B3")
    for i, v in enumerate(rep["unknown_pct"]):
        ax.text(v, i, f" {v:.2f}%", va="center")
    ax.set_xlabel("% of rows coded 'unknown'")
    ax.set_title("Disguised missing values")
    _save(fig, "06_unknown_values.png")


def plot_macro_over_time(df):
    fig, ax1 = plt.subplots(figsize=(11, 4))
    idx = np.arange(len(df))
    roll = (df[TARGET] == "yes").astype(int).rolling(1000).mean()
    ax1.plot(idx, df["euribor3m"], color="#4C72B0", label="euribor3m")
    ax1.set_ylabel("euribor3m")
    ax2 = ax1.twinx()
    ax2.plot(idx, roll, color="#DD8452", label="subscription rate (rolling 1,000)")
    ax2.set_ylabel("rolling subscription rate")
    ax1.set_xlabel("Row order (chronological, May 2008 to Nov 2010)")
    fig.legend(loc="upper center", ncol=2)
    ax1.set_title("Macro context drifts over time and drives the outcome", pad=25)
    _save(fig, "07_temporal_drift.png")


def run_all_figures(df):
    plot_class_balance(df)
    plot_numeric_distributions(df)
    plot_boxplots(df)
    corr = plot_correlation(df)
    plot_categorical_rates(df)
    plot_unknowns(df)
    plot_macro_over_time(df)
    return corr
