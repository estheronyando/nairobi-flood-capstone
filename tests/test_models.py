"""Tests for the model zoo, the selection rule, and the serialized artefact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from floodrisk.models import AverageEnsemble, EventStackingEnsemble, build_models
from floodrisk.train import select_model


@pytest.fixture(scope="module")
def toy():
    """A small, separable, grouped dataset - enough to exercise the API."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 5)).astype("float32")
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    groups = np.array([f"E{i % 6:02d}" for i in range(n)])
    return X, y, groups


class TestModelZoo:
    def test_all_candidates_present(self, cfg):
        models = build_models(cfg)
        assert set(models) == {
            "logistic_regression", "random_forest", "xgboost",
            "average_ensemble", "stacking_ensemble",
        }

    def test_ensembles_are_sklearn_classifiers(self, cfg):
        """Permutation importance and scorers need the estimator protocol."""
        models = build_models(cfg)
        for name in ("average_ensemble", "stacking_ensemble"):
            assert is_classifier(models[name]), f"{name} is not a sklearn classifier"

    def test_ensembles_fit_and_predict(self, cfg, toy):
        X, y, groups = toy
        for name in ("average_ensemble", "stacking_ensemble"):
            model = build_models(cfg)[name]
            model.fit(X, y, groups=groups)
            proba = model.predict_proba(X)
            assert proba.shape == (len(y), 2)
            assert np.allclose(proba.sum(axis=1), 1.0)
            assert ((proba >= 0) & (proba <= 1)).all()

    def test_stacking_requires_groups(self, cfg, toy):
        """Without event ids the meta-features would be built on a random CV,
        which is the leakage the event split exists to avoid."""
        X, y, _ = toy
        with pytest.raises(ValueError, match="groups"):
            build_models(cfg)["stacking_ensemble"].fit(X, y)


class TestSelectionRule:
    def _frame(self, scores: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame({"roc_auc": scores})

    def test_prefers_simpler_model_within_tolerance(self, cfg):
        chosen, note = select_model(
            cfg,
            self._frame({"stacking_ensemble": 0.9068, "xgboost": 0.9058,
                         "random_forest": 0.8999, "logistic_regression": 0.79}),
        )
        assert chosen == "xgboost"
        assert "tolerance" in note

    def test_takes_the_best_when_nothing_simpler_ties(self, cfg):
        chosen, _ = select_model(
            cfg,
            self._frame({"stacking_ensemble": 0.95, "xgboost": 0.80,
                         "logistic_regression": 0.70}),
        )
        assert chosen == "stacking_ensemble"

    def test_prefers_the_linear_baseline_when_it_ties(self, cfg):
        """If logistic regression is as good, deploy logistic regression."""
        chosen, _ = select_model(
            cfg,
            self._frame({"xgboost": 0.9000, "logistic_regression": 0.8990}),
        )
        assert chosen == "logistic_regression"


class TestSerializedArtefact:
    def test_carries_everything_needed_to_serve(self, model_artefact):
        for key in ("model", "model_name", "features", "threshold",
                    "train_events", "test_events", "metrics", "seed"):
            assert key in model_artefact, f"artefact missing '{key}'"

    def test_feature_order_matches_config(self, cfg, model_artefact):
        """Inference builds its matrix from this list; a reorder silently
        feeds slope into the elevation column."""
        assert model_artefact["features"] == cfg.feature_columns

    def test_train_and_test_events_disjoint(self, model_artefact):
        assert not set(model_artefact["train_events"]) & set(model_artefact["test_events"])

    def test_predicts_probabilities(self, cfg, model_artefact):
        X = np.array([[1600.0, 2.0, 1.0, 50.0, 120.0]], dtype="float32")
        p = model_artefact["model"].predict_proba(X)
        assert p.shape == (1, 2)
        assert 0.0 <= float(p[0, 1]) <= 1.0

    def test_reported_scores_are_plausible(self, model_artefact):
        """A sanity band, not a target: below 0.5 is worse than chance and
        above 0.99 on held-out events would mean leakage, not skill."""
        auc = float(model_artefact["metrics"]["roc_auc"])
        assert 0.5 < auc < 0.99, f"suspicious held-out ROC AUC: {auc}"
