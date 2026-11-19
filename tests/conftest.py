"""Shared fixtures.

The model and raster stack are expensive to load, so they are built once
per session. Tests that need a trained model are skipped rather than
failed when the pipeline has not been run yet, so a fresh clone can run
`pytest` before `run_pipeline.py` and get a sensible result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floodrisk.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def model_artefact(cfg):
    import joblib

    path = cfg.models_dir / "flood_model.joblib"
    if not path.exists():
        pytest.skip("No trained model; run `python scripts/run_pipeline.py` first.")
    return joblib.load(path)


@pytest.fixture(scope="session")
def training_table(cfg):
    import pandas as pd

    path = cfg.tables_dir / "training_table.csv"
    if not path.exists():
        pytest.skip("No training table; run `python scripts/run_pipeline.py` first.")
    return pd.read_csv(path)
