"""Step 2: turn 50 rasters into one modelling table.

For each flood event we draw an equal number of flood and non-flood pixels
and read the five predictors at those exact cells. The result is a tidy
table carrying its own provenance (event id, row, col) so any row can be
traced back to a pixel on the map.

Two choices here matter more than they look:

* **Negatives exclude permanent water.** Rivers and dams are wet in every
  scene; leaving them in the non-flood class would let the model separate
  the classes on "is this a river", which is not the question.
* **Sampling is balanced per event, not globally.** Flood pixels are about
  0.08% of the grid. Balancing per event stops the two or three largest
  events from dominating, but it does distort prevalence, which is why the
  operating threshold has to be revisited before the map is read
  operationally (see the report's limitations).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .rasters import read_band

log = logging.getLogger(__name__)


def _static_stack(cfg: Config) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Static predictors and the mask of cells where all of them are valid."""
    arrays: dict[str, np.ndarray] = {}
    valid = None
    for name, path in cfg.static_rasters.items():
        arr, ok = read_band(path)
        arrays[name] = arr
        valid = ok if valid is None else (valid & ok)
    assert valid is not None
    return arrays, valid


def build_training_table(cfg: Config) -> pd.DataFrame:
    """Draw the event-balanced sample and return the modelling table."""
    sampling = cfg.section("sampling")
    n_pos = int(sampling["per_event_positive"])
    n_neg = int(sampling["per_event_negative"])
    rng = np.random.default_rng(int(sampling["seed"]))

    static, static_valid = _static_stack(cfg)

    if sampling["exclude_permanent_water"]:
        perm, _ = read_band(cfg.auxiliary_rasters["perm_water"])
        perm_mask = perm > 0
    else:
        perm_mask = np.zeros_like(static_valid, dtype=bool)

    frames = []
    for event in cfg.discover_events():
        label, label_valid = read_band(cfg.event_raster("label", event))
        rain7, rain7_valid = read_band(cfg.event_raster("rain7", event))
        rain_sum, rain_sum_valid = read_band(cfg.event_raster("rain_sum", event))

        usable = static_valid & label_valid & rain7_valid & rain_sum_valid & ~perm_mask
        flood = (label > 0) & usable
        dry = (label == 0) & usable

        pos_idx = _choose(np.flatnonzero(flood.ravel()), n_pos, rng, event, "flood")
        neg_idx = _choose(np.flatnonzero(dry.ravel()), n_neg, rng, event, "non-flood")
        flat = np.concatenate([pos_idx, neg_idx])
        rows, cols = np.unravel_index(flat, label.shape)

        frames.append(
            pd.DataFrame(
                {
                    "event": event,
                    "row": rows,
                    "col": cols,
                    "dem": static["dem"].ravel()[flat],
                    "slope": static["slope"].ravel()[flat],
                    "builtup": static["builtup"].ravel()[flat],
                    "rain7": rain7.ravel()[flat],
                    "rain_sum": rain_sum.ravel()[flat],
                    "label": np.concatenate(
                        [np.ones(len(pos_idx), int), np.zeros(len(neg_idx), int)]
                    ),
                }
            )
        )
        log.info("  %s: %d flood + %d non-flood", event, len(pos_idx), len(neg_idx))

    table = pd.concat(frames, ignore_index=True)

    before = len(table)
    feature_cols = cfg.feature_columns
    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    if len(table) != before:
        log.warning("Dropped %d rows with non-finite predictors", before - len(table))

    out = cfg.tables_dir / "training_table.csv"
    table.to_csv(out, index=False)
    log.info("Training table: %d rows, %d events -> %s",
             len(table), table["event"].nunique(), out)
    return table


def _choose(candidates: np.ndarray, n: int, rng: np.random.Generator,
            event: str, kind: str) -> np.ndarray:
    if candidates.size < n:
        raise ValueError(
            f"Event {event} has only {candidates.size} usable {kind} pixels, need {n}."
        )
    return rng.choice(candidates, size=n, replace=False)


def split_by_event(cfg: Config, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on event id, never on pixels.

    Pixels inside one flood footprint are strongly autocorrelated. A random
    pixel split puts a cell and its neighbour on opposite sides of the
    boundary, so the test set is effectively memorised and every metric is
    optimistic. Holding out whole events asks the honest question: does this
    model generalise to a flood it has never seen?
    """
    test_events = set(cfg.test_events)
    present = set(table["event"].unique())
    missing = test_events - present
    if missing:
        raise ValueError(f"Configured test events not in the data: {sorted(missing)}")

    test = table[table["event"].isin(test_events)].copy()
    train = table[~table["event"].isin(test_events)].copy()
    if train.empty or test.empty:
        raise ValueError("Event split produced an empty side; check split.test_events.")

    assert not (set(train["event"]) & set(test["event"])), "event leakage across the split"
    log.info(
        "Split: %d train rows (%d events) / %d test rows (%d events)",
        len(train), train["event"].nunique(), len(test), test["event"].nunique(),
    )
    return train, test


def run(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = build_training_table(cfg)
    train, test = split_by_event(cfg, table)
    return table, train, test
