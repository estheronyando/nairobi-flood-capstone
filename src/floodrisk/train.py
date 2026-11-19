"""Step 3: train every candidate and evaluate on unseen flood events.

Metrics are computed once, on the held-out events, and written to
``outputs/tables/model_comparison.csv``. Probability quality (Brier score,
reliability curve) is reported next to discrimination because the output of
this system is a probability surface people will threshold, not a label.
"""

from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Config
from .models import AverageEnsemble, EventStackingEnsemble, build_models

log = logging.getLogger(__name__)


def _metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_model(cfg: Config, comparison: pd.DataFrame) -> tuple[str, str]:
    """Choose the model to deploy: simplest among those statistically tied for best.

    Taking the argmax of a single metric on 1,600 test rows rewards noise.
    A difference of 0.001 ROC AUC here is far inside run-to-run variation,
    so the rule is: find the top score, keep everything within `tolerance`
    of it, and deploy the simplest of those. Complexity costs fit time,
    explainability and operational risk, and has to be paid for in
    measurable accuracy.
    """
    sel = cfg.section("selection")
    metric, tol = sel["metric"], float(sel["tolerance"])
    order = list(sel["simplicity_order"])

    top_name = str(comparison[metric].idxmax())
    top_score = float(comparison.loc[top_name, metric])
    tied = [m for m in comparison.index if top_score - float(comparison.loc[m, metric]) <= tol]
    ranked = sorted(tied, key=lambda m: order.index(m) if m in order else len(order))
    chosen = ranked[0]

    if chosen == top_name:
        note = f"Selected {chosen}: best {metric} ({top_score:.4f}) and nothing simpler ties it."
    else:
        note = (
            f"Selected {chosen} ({metric}={float(comparison.loc[chosen, metric]):.4f}) over "
            f"{top_name} ({top_score:.4f}): within the {tol} tolerance and simpler. "
            f"Tied set: {tied}."
        )
    return chosen, note


def train_and_evaluate(cfg: Config, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit every model on the training events, score on the held-out ones."""
    features = cfg.feature_columns
    threshold = float(cfg.section("evaluation")["threshold"])

    X_train = train[features].to_numpy(dtype="float32")
    y_train = train["label"].to_numpy(dtype=int)
    groups = train["event"].to_numpy()
    X_test = test[features].to_numpy(dtype="float32")
    y_test = test["label"].to_numpy(dtype=int)

    rows, fitted, probabilities = [], {}, {}
    for name, model in build_models(cfg).items():
        t0 = time.time()
        if isinstance(model, (AverageEnsemble, EventStackingEnsemble)):
            model.fit(X_train, y_train, groups=groups)
        else:
            model.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        proba = model.predict_proba(X_test)[:, 1]
        m = _metrics(y_test, proba, threshold)
        m.update({"model": name, "fit_seconds": round(fit_seconds, 2)})
        rows.append(m)
        fitted[name] = model
        probabilities[name] = proba
        log.info(
            "  %-20s ROC AUC %.4f | F1 %.4f | Brier %.4f | %.1fs",
            name, m["roc_auc"], m["f1"], m["brier"], fit_seconds,
        )

    comparison = (
        pd.DataFrame(rows)
        .set_index("model")
        .loc[:, ["accuracy", "precision", "recall", "f1", "roc_auc", "brier",
                 "tp", "fp", "fn", "tn", "fit_seconds"]]
        .sort_values("roc_auc", ascending=False)
        .round(4)
    )
    comparison.to_csv(cfg.tables_dir / "model_comparison.csv")
    log.info("Model comparison -> %s", cfg.tables_dir / "model_comparison.csv")

    best_name, selection_note = select_model(cfg, comparison)
    log.info("%s", selection_note)

    artefact = {
        "model": fitted[best_name],
        "model_name": best_name,
        "features": features,
        "threshold": threshold,
        "train_events": sorted(train["event"].unique().tolist()),
        "test_events": sorted(test["event"].unique().tolist()),
        "metrics": comparison.loc[best_name].to_dict(),
        "selection_note": selection_note,
        "seed": cfg.seed,
    }
    model_path = cfg.models_dir / "flood_model.joblib"
    joblib.dump(artefact, model_path)
    log.info("Serialized best model -> %s", model_path)

    # A model card a reviewer can read without loading the pickle.
    card = {k: v for k, v in artefact.items() if k != "model"}
    card["metrics"] = {k: float(v) for k, v in card["metrics"].items()}
    card["all_models"] = json.loads(comparison.to_json(orient="index"))
    with open(cfg.models_dir / "model_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)

    return {
        "comparison": comparison,
        "fitted": fitted,
        "probabilities": probabilities,
        "best_name": best_name,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
    }


def run(cfg: Config, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return train_and_evaluate(cfg, train, test)
