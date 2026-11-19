"""Step 1 of the pipeline: audit the raw rasters before anything is modelled.

Writes ``outputs/tables/data_audit.csv`` (one row per raster) and
``outputs/tables/event_audit.csv`` (one row per flood event). The audit is
a gate, not decoration: it fails the run if layers are not co-registered
or if an event carries too few flood pixels to sample from.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .rasters import assert_same_grid, read_band

log = logging.getLogger(__name__)


def audit_rasters(cfg: Config) -> pd.DataFrame:
    """Per-raster inventory: grid, dtype, nodata share, value range."""
    events = cfg.discover_events()

    layers: dict[str, "object"] = {}
    layers.update(cfg.static_rasters)
    layers.update(cfg.auxiliary_rasters)
    for ev in events:
        for kind in ("label", "rain7", "rain_sum"):
            layers[f"{kind}:{ev}"] = cfg.event_raster(kind, ev)

    grid = assert_same_grid(cfg.reference_raster, layers)
    log.info("All %d layers co-registered on %s", len(layers), grid.describe())

    rows = []
    for name, path in layers.items():
        arr, valid = read_band(path)
        vals = arr[valid]
        rows.append(
            {
                "layer": name,
                "file": path.name,
                "width": grid.width,
                "height": grid.height,
                "crs": grid.crs,
                "cells": arr.size,
                "invalid_pct": round(100.0 * float((~valid).mean()), 4),
                "min": round(float(vals.min()), 4) if vals.size else np.nan,
                "max": round(float(vals.max()), 4) if vals.size else np.nan,
                "mean": round(float(vals.mean()), 4) if vals.size else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    out = cfg.tables_dir / "data_audit.csv"
    df.to_csv(out, index=False)
    log.info("Raster audit -> %s", out)
    return df


def audit_events(cfg: Config) -> pd.DataFrame:
    """Per-event flood-pixel counts, after removing permanent water.

    Permanent water is excluded so the positive class is transient flooding
    rather than rivers and dams that are wet in every image.
    """
    perm_path = cfg.auxiliary_rasters["perm_water"]
    perm, _ = read_band(perm_path)
    perm_mask = perm > 0

    sampling = cfg.section("sampling")
    need = int(sampling["per_event_positive"])

    rows = []
    for ev in cfg.discover_events():
        label, label_valid = read_band(cfg.event_raster("label", ev))
        flood = (label > 0) & label_valid
        flood_excl = flood & ~perm_mask
        rows.append(
            {
                "event": ev,
                "flood_pixels": int(flood.sum()),
                "flood_in_permanent_water": int((flood & perm_mask).sum()),
                "flood_pixels_used": int(flood_excl.sum()),
                "prevalence_pct": round(100.0 * float(flood_excl.mean()), 4),
                "sufficient_for_sampling": bool(flood_excl.sum() >= need),
            }
        )

    df = pd.DataFrame(rows)
    out = cfg.tables_dir / "event_audit.csv"
    df.to_csv(out, index=False)

    short = df.loc[~df["sufficient_for_sampling"], "event"].tolist()
    if short:
        raise ValueError(
            f"Events with fewer than {need} usable flood pixels: {short}. "
            "Lower sampling.per_event_positive in config.yaml or drop these events."
        )
    log.info("Event audit -> %s (%d events, all with >= %d flood pixels)", out, len(df), need)
    return df


def audit_rainfall_consistency(cfg: Config) -> pd.DataFrame:
    """Flag events whose two rainfall layers disagree implausibly.

    ``rain7`` is a seven-day antecedent accumulation and ``rain_sum`` the
    event total. The two are not required to sit in a fixed relationship —
    a short, intense storm after a wet week legitimately gives
    ``rain7 > rain_sum`` — but an order-of-magnitude gap suggests the
    layers were built over mismatched windows, or transposed. This does
    not stop the run; it records the finding so the report can state it
    rather than discover it later.
    """
    rows = []
    for event in cfg.discover_events():
        r7, r7_ok = read_band(cfg.event_raster("rain7", event))
        rs, rs_ok = read_band(cfg.event_raster("rain_sum", event))
        med7 = float(np.median(r7[r7_ok])) if r7_ok.any() else np.nan
        medsum = float(np.median(rs[rs_ok])) if rs_ok.any() else np.nan
        ratio = med7 / medsum if medsum else np.inf
        rows.append(
            {
                "event": event,
                "median_rain7_mm": round(med7, 2),
                "median_rain_sum_mm": round(medsum, 2),
                "ratio_rain7_over_rain_sum": round(ratio, 2),
                "implausible": bool(ratio > 5 or ratio < 0.05),
            }
        )

    df = pd.DataFrame(rows)
    out = cfg.tables_dir / "rainfall_consistency.csv"
    df.to_csv(out, index=False)

    flagged = df.loc[df["implausible"], "event"].tolist()
    if flagged:
        log.warning(
            "Rainfall layers look inconsistent for %s - see %s. Kept in the data; "
            "noted in the report.", flagged, out.name,
        )
    return df


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    return {
        "rasters": audit_rasters(cfg),
        "events": audit_events(cfg),
        "rainfall": audit_rainfall_consistency(cfg),
    }
