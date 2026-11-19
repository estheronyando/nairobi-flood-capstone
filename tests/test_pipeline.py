"""Tests for the data contract and the split that keeps the evaluation honest.

The split test is the important one. An event-based holdout is the single
assumption the reported scores rest on; if pixels from one flood event ever
appear on both sides, every metric in the report becomes optimistic and
nothing downstream would reveal it. So it is asserted, not trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from floodrisk import sampling
from floodrisk.rasters import assert_same_grid, read_band


class TestDataContract:
    def test_fifteen_events_discovered(self, cfg):
        events = cfg.discover_events()
        assert len(events) == 15, f"expected 15 flood events, found {len(events)}"
        assert len(set(events)) == len(events), "duplicate event ids"

    def test_all_layers_share_the_reference_grid(self, cfg):
        layers = {**cfg.static_rasters, **cfg.auxiliary_rasters}
        for event in cfg.discover_events():
            for kind in ("label", "rain7", "rain_sum"):
                layers[f"{kind}:{event}"] = cfg.event_raster(kind, event)
        grid = assert_same_grid(cfg.reference_raster, layers)
        assert (grid.width, grid.height) == (1858, 1108)
        assert grid.crs == "EPSG:32737"

    def test_labels_are_binary(self, cfg):
        for event in cfg.discover_events()[:3]:
            arr, valid = read_band(cfg.event_raster("label", event))
            assert set(np.unique(arr[valid])) <= {0.0, 1.0}

    def test_configured_test_events_exist(self, cfg):
        assert set(cfg.test_events).issubset(set(cfg.discover_events()))


class TestSampling:
    def test_sample_is_balanced_per_event(self, cfg, training_table):
        counts = training_table.groupby(["event", "label"]).size().unstack(fill_value=0)
        assert (counts[0] == counts[1]).all(), "classes not balanced within every event"

    def test_no_missing_predictors(self, cfg, training_table):
        X = training_table[cfg.feature_columns]
        assert X.notna().all().all(), "NaNs survived into the modelling table"
        assert np.isfinite(X.to_numpy()).all(), "non-finite predictors in the modelling table"

    def test_no_duplicate_pixels_within_an_event(self, training_table):
        dupes = training_table.duplicated(subset=["event", "row", "col"]).sum()
        assert dupes == 0, f"{dupes} pixels sampled twice in the same event"

    def test_sampling_is_reproducible(self, cfg, training_table):
        """Re-drawing with the same seed must reproduce the same pixels."""
        again = sampling.build_training_table(cfg)
        key = ["event", "row", "col", "label"]
        a = training_table[key].sort_values(key).reset_index(drop=True)
        b = again[key].sort_values(key).reset_index(drop=True)
        assert a.equals(b), "sampling is not deterministic under a fixed seed"


class TestEventSplit:
    def test_train_and_test_events_are_disjoint(self, cfg, training_table):
        train, test = sampling.split_by_event(cfg, training_table)
        assert not set(train["event"]) & set(test["event"])

    def test_no_pixel_appears_on_both_sides(self, cfg, training_table):
        """The leakage check with teeth: no (event,row,col) crosses the split."""
        train, test = sampling.split_by_event(cfg, training_table)
        tr = set(map(tuple, train[["event", "row", "col"]].to_numpy()))
        te = set(map(tuple, test[["event", "row", "col"]].to_numpy()))
        assert not (tr & te)

    def test_split_sizes_match_the_configuration(self, cfg, training_table):
        train, test = sampling.split_by_event(cfg, training_table)
        assert test["event"].nunique() == len(cfg.test_events)
        assert train["event"].nunique() == 15 - len(cfg.test_events)

    def test_rejects_an_unknown_test_event(self, cfg, training_table):
        bad = type(cfg)(raw={**cfg.raw, "split": {"test_events": ["E99_1999_99"]}},
                        root=cfg.root)
        with pytest.raises(ValueError, match="not in the data"):
            sampling.split_by_event(bad, training_table)
