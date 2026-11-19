"""The model zoo: four learners plus two ensembles over them.

Why these and not others is argued in the report; the short version is
that this is a small, five-column, tabular problem with a non-linear but
not exotic decision surface, which is squarely gradient-boosting
territory. Logistic regression is kept as the honest baseline, because a
boosted tree that cannot beat a linear model is not worth deploying.

The stacking ensemble is deliberately hand-rolled rather than taken from
``sklearn.ensemble.StackingClassifier``: the stock version builds its
meta-features with a random internal CV, which would mix pixels from one
flood event across folds and reintroduce exactly the leakage the
event-based split exists to prevent. Here the meta-features come from
``GroupKFold`` over event ids.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import Config


def make_logistic(cfg: Config) -> Pipeline:
    p = cfg.section("models")["logistic_regression"]
    # Scaling matters here and nowhere else: elevation is ~10^3 and slope
    # ~10^1, so an unscaled linear model is dominated by units, not signal.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=p["C"], max_iter=p["max_iter"],
                                       random_state=cfg.seed)),
        ]
    )


def make_random_forest(cfg: Config) -> RandomForestClassifier:
    p = cfg.section("models")["random_forest"]
    return RandomForestClassifier(
        n_estimators=p["n_estimators"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        random_state=cfg.seed,
        n_jobs=-1,
    )


def make_xgboost(cfg: Config) -> XGBClassifier:
    p = cfg.section("models")["xgboost"]
    return XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        eval_metric=p["eval_metric"],
        random_state=cfg.seed,
        n_jobs=-1,
        tree_method="hist",
    )


class AverageEnsemble(ClassifierMixin, BaseEstimator):
    """Unweighted mean of the member models' flood probabilities.

    The simplest ensemble worth reporting: no meta-model, nothing to tune,
    and it either beats its members or it does not. Implemented against the
    scikit-learn estimator API so that tooling built on it — permutation
    importance, scorers, cross-validation — works unchanged.
    """

    def __init__(self, members: dict[str, object] | None = None):
        self.members = members or {}

    def fit(self, X, y, groups=None):  # noqa: D102 - sklearn-style
        self.classes_ = np.unique(y)
        self.members_ = {name: clone(m) for name, m in self.members.items()}
        for model in self.members_.values():
            model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        probs = [m.predict_proba(X)[:, 1] for m in self.members_.values()]
        p1 = np.mean(probs, axis=0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class EventStackingEnsemble(ClassifierMixin, BaseEstimator):
    """Stacking whose meta-features respect the event boundary.

    Base models produce out-of-fold probabilities under ``GroupKFold`` on
    event id; a logistic meta-model learns how to combine them. Because no
    base prediction is ever made on an event the base model trained on, the
    meta-model is fitted on honest inputs.
    """

    def __init__(self, members: dict[str, object] | None = None, seed: int = 42,
                 n_splits: int = 5):
        self.members = members or {}
        self.seed = seed
        self.n_splits = n_splits

    def fit(self, X, y, groups=None):  # noqa: D102
        if groups is None:
            raise ValueError("EventStackingEnsemble needs event ids as `groups`.")
        self.classes_ = np.unique(y)
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=min(self.n_splits, n_groups))

        meta_features = []
        for model in self.members.values():
            oof = cross_val_predict(
                clone(model), X, y, groups=groups, cv=cv,
                method="predict_proba", n_jobs=1,
            )[:, 1]
            meta_features.append(oof)

        self.meta_ = LogisticRegression(max_iter=2000, random_state=self.seed)
        self.meta_.fit(np.column_stack(meta_features), y)

        # Refit the bases on everything for inference.
        self.members_ = {name: clone(m) for name, m in self.members.items()}
        for model in self.members_.values():
            model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        base = np.column_stack([m.predict_proba(X)[:, 1] for m in self.members_.values()])
        return self.meta_.predict_proba(base)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models(cfg: Config) -> dict[str, object]:
    """All candidates, in the order they are reported."""
    logistic = make_logistic(cfg)
    rf = make_random_forest(cfg)
    xgb = make_xgboost(cfg)
    return {
        "logistic_regression": logistic,
        "random_forest": rf,
        "xgboost": xgb,
        "average_ensemble": AverageEnsemble(
            {"lr": make_logistic(cfg), "rf": make_random_forest(cfg), "xgb": make_xgboost(cfg)}
        ),
        "stacking_ensemble": EventStackingEnsemble(
            {"rf": make_random_forest(cfg), "xgb": make_xgboost(cfg)}, seed=cfg.seed
        ),
    }
