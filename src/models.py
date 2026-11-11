"""Base models, hyperparameter search spaces and ensembles (Parts E and F)."""
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import RANDOM_STATE
from .preprocessing import make_pipeline

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "hgb": "Histogram Gradient Boosting",
}


def base_models(imbalance: str = "class_weight") -> dict:
    """Three structurally different classifiers.

    All share the same cleaning, encoding, scaling and imbalance handling. The
    correlation filter is applied to Logistic Regression only: collinear inputs
    make linear coefficients unstable, whereas tree ensembles are unaffected by
    collinearity and the Part D experiment showed the filter costs HGB about
    0.006 PR-AUC and 0.009 ROC-AUC.

    imbalance: 'class_weight' (cost-sensitive learning) or 'smote'.
    """
    cw = "balanced" if imbalance == "class_weight" else None
    sampler = "smote" if imbalance == "smote" else None
    return {
        "logreg": make_pipeline(
            LogisticRegression(max_iter=3000, class_weight=cw, random_state=RANDOM_STATE),
            sampler=sampler),
        "rf": make_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight=("balanced_subsample" if cw else None),
                                   n_jobs=-1, random_state=RANDOM_STATE),
            sampler=sampler, corr_threshold=None),
        "hgb": make_pipeline(
            HistGradientBoostingClassifier(class_weight=cw, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=20,
                                           random_state=RANDOM_STATE),
            sampler=sampler, corr_threshold=None),
    }


SEARCH_SPACES = {
    "logreg": {
        "model__C": loguniform(1e-3, 1e2),
        "model__penalty": ["l2"],
    },
    "rf": {
        "model__max_depth": [6, 8, 10, 12, 16, None],
        "model__min_samples_leaf": randint(5, 60),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "hgb": {
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__max_leaf_nodes": randint(8, 48),
        "model__min_samples_leaf": randint(20, 200),
        "model__l2_regularization": loguniform(1e-3, 10),
        "model__max_iter": [200, 400, 600],
    },
}

N_ITER = {"logreg": 12, "rf": 12, "hgb": 20}


def tune(name, pipe, X, y, n_iter=None, folds=3, scoring="average_precision"):
    """Randomised search with stratified CV, optimising PR-AUC (average precision)."""
    search = RandomizedSearchCV(
        pipe, SEARCH_SPACES[name], n_iter=n_iter or N_ITER[name], scoring=scoring,
        cv=StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=(1 if name == "rf" else -1), refit=True,
    )
    search.fit(X, y)
    return search


def voting_ensemble(estimators: dict, weights=None):
    """Soft voting: averages the predicted probabilities of the members."""
    return VotingClassifier(list(estimators.items()), voting="soft", weights=weights, n_jobs=1)


def stacking_ensemble(estimators: dict, cv=5):
    """Stacking: a logistic-regression meta-learner trained on out-of-fold probabilities."""
    return StackingClassifier(
        list(estimators.items()),
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1, passthrough=False,
    )
