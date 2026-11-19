"""Quantify what the event-based split is protecting against.

The repository reports scores on four flood events the model never saw. The
obvious alternative - pool every sampled pixel and take a random 80/20 split -
is what most tutorials do, and it is wrong here: pixels from the same storm are
spatially autocorrelated and share one rainfall field, so a random split leaves
near-duplicate rows on both sides of the partition and the test score measures
memorisation as much as generalisation.

This script trains the deployed model configuration twice on the same sample,
once under each split, and writes the gap to outputs/tables/split_comparison.csv.
It is a diagnostic: nothing here changes the deployed model or the reported
scores. If the gap is large, any score published from a random split should be
treated as inflated.

    python scripts/check_split_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floodrisk.config import load_config          # noqa: E402
from floodrisk.models import build_models          # noqa: E402
from floodrisk.sampling import build_training_table, split_by_event  # noqa: E402

MODEL = "xgboost"   # the deployed configuration


def main() -> int:
    cfg = load_config()
    table = build_training_table(cfg)
    feats = cfg.section("features")["columns"]

    rows = []

    # A. The design this project uses: whole events held out.
    train, test = split_by_event(cfg, table)
    assert not (set(train["event"]) & set(test["event"])), "event leak"
    model = build_models(cfg)[MODEL]
    model.fit(train[feats], train["label"])
    p = model.predict_proba(test[feats])[:, 1]
    rows.append({
        "split": "event_holdout",
        "description": "4 flood events never seen in training",
        "test_rows": len(test),
        "roc_auc": roc_auc_score(test["label"], p),
        "f1": f1_score(test["label"], (p >= 0.5).astype(int)),
    })

    # B. The naive alternative: random pixels, event id ignored.
    tr2, te2 = train_test_split(
        table, test_size=0.2, random_state=cfg.seed, stratify=table["label"]
    )
    shared = set(tr2["event"]) & set(te2["event"])
    model2 = build_models(cfg)[MODEL]
    model2.fit(tr2[feats], tr2["label"])
    p2 = model2.predict_proba(te2[feats])[:, 1]
    rows.append({
        "split": "random_pixels",
        "description": f"random 80/20; {len(shared)} events appear on both sides",
        "test_rows": len(te2),
        "roc_auc": roc_auc_score(te2["label"], p2),
        "f1": f1_score(te2["label"], (p2 >= 0.5).astype(int)),
    })

    out = pd.DataFrame(rows)
    d_auc = out.loc[1, "roc_auc"] - out.loc[0, "roc_auc"]
    d_f1 = out.loc[1, "f1"] - out.loc[0, "f1"]

    path = cfg.tables_dir / "split_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nRandom-split inflation: {d_auc:+.4f} ROC AUC, {d_f1:+.4f} F1")
    print("The event holdout is the honest number; the random split is not.")
    print(f"Written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
