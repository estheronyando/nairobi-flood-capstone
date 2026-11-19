#!/usr/bin/env python3
"""Run the whole pipeline: audit -> sample -> train -> evaluate -> predict.

    python scripts/run_pipeline.py                 # everything
    python scripts/run_pipeline.py --skip-predict  # stop before the raster sweep
    python scripts/run_pipeline.py --scenario E14_2025_04

This is the single entry point a reviewer needs. It is deterministic: one
seed drives sampling and every model, so a rerun reproduces the tables in
`outputs/` exactly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib  # noqa: E402

from floodrisk import audit, evaluate, predict, sampling, train  # noqa: E402
from floodrisk.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--scenario", default=None,
                        help="rainfall scenario for the map: median, max, or an event id")
    parser.add_argument("--skip-predict", action="store_true",
                        help="skip the full-raster sweep (the slow step)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pipeline")
    started = time.time()

    cfg = load_config(args.config)
    log.info("Repository root: %s", cfg.root)
    log.info("Raw data:        %s", cfg.raw_data)

    log.info("\n[1/5] Auditing rasters")
    audit.run(cfg)

    log.info("\n[2/5] Building the event-balanced sample")
    table, train_df, test_df = sampling.run(cfg)

    log.info("\n[3/5] Training candidates and scoring on held-out events")
    result = train.run(cfg, train_df, test_df)

    log.info("\n[4/5] Diagnostics")
    evaluate.run(cfg, table, test_df, result)

    if args.skip_predict:
        log.info("\n[5/5] Skipped (--skip-predict)")
    else:
        log.info("\n[5/5] Scoring the full raster")
        artefact = joblib.load(cfg.models_dir / "flood_model.joblib")
        raster = predict.predict_surface(cfg, artefact, scenario=args.scenario)
        evaluate.plot_susceptibility_map(cfg, raster)

    log.info("\nDone in %.1fs. Artifacts in %s", time.time() - started, cfg.outputs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
