"""Step 5: apply the model to every cell and write the susceptibility raster.

The map answers a conditional question — *given this much rain, how
susceptible is this pixel* — so the rainfall scenario is an explicit input
rather than a hidden constant. ``median`` uses the per-pixel median across
the fifteen observed events (a typical storm); naming an event reruns that
event's actual rainfall.

Permanent water is written as nodata. The model was trained to find
transient flooding with rivers and dams excluded, so colouring them
"high susceptibility" would be reporting a fact about the map, not a
prediction.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

from .config import Config
from .rasters import read_band, write_geotiff

log = logging.getLogger(__name__)

NODATA = -9999.0


def build_scenario(cfg: Config, scenario: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rain7, rain_sum)`` grids for the requested scenario."""
    events = cfg.discover_events()
    if scenario in events:
        rain7, _ = read_band(cfg.event_raster("rain7", scenario))
        rain_sum, _ = read_band(cfg.event_raster("rain_sum", scenario))
        return rain7, rain_sum

    if scenario not in {"median", "max"}:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Use 'median', 'max', or one of: {events}"
        )

    reducer = np.nanmedian if scenario == "median" else np.nanmax
    r7 = np.stack([read_band(cfg.event_raster("rain7", e))[0] for e in events])
    rs = np.stack([read_band(cfg.event_raster("rain_sum", e))[0] for e in events])
    # Cells with no rainfall observation in any event reduce to NaN; that is
    # expected at the grid edge and those cells are dropped by the valid mask.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return reducer(r7, axis=0), reducer(rs, axis=0)


def predict_surface(cfg: Config, model_artefact: dict, scenario: str | None = None) -> Path:
    """Score every valid cell and write a single-band probability GeoTIFF."""
    pred_cfg = cfg.section("prediction")
    scenario = scenario or pred_cfg["scenario"]
    chunk = int(pred_cfg["chunk_rows"])
    model = model_artefact["model"]
    features = model_artefact["features"]

    dem, dem_ok = read_band(cfg.static_rasters["dem"])
    slope, slope_ok = read_band(cfg.static_rasters["slope"])
    builtup, built_ok = read_band(cfg.static_rasters["builtup"])
    rain7, rain_sum = build_scenario(cfg, scenario)
    perm, _ = read_band(cfg.auxiliary_rasters["perm_water"])

    bands = {"dem": dem, "slope": slope, "builtup": builtup,
             "rain7": rain7, "rain_sum": rain_sum}

    valid = (
        dem_ok & slope_ok & built_ok
        & np.isfinite(rain7) & np.isfinite(rain_sum)
        & (perm == 0)
    )
    height, width = dem.shape
    out = np.full((height, width), NODATA, dtype="float32")

    log.info("Scoring %s cells (scenario=%s)", f"{int(valid.sum()):,}", scenario)
    for start in range(0, height, chunk):
        stop = min(start + chunk, height)
        m = valid[start:stop]
        if not m.any():
            continue
        X = np.column_stack([bands[f][start:stop][m] for f in features]).astype("float32")
        out[start:stop][m] = model.predict_proba(X)[:, 1].astype("float32")

    target = cfg.susceptibility_raster
    if scenario != pred_cfg["scenario"]:
        target = target.with_name(f"{target.stem}_{scenario}{target.suffix}")
    write_geotiff(target, out, cfg.reference_raster, nodata=NODATA)

    scored = out[out != NODATA]
    log.info(
        "Susceptibility raster -> %s  (mean %.3f, >=0.5 on %.2f%% of scored cells)",
        target, float(scored.mean()), 100.0 * float((scored >= 0.5).mean()),
    )
    return target


class PointScorer:
    """Reads predictors at arbitrary coordinates. Used by the API.

    Kept separate from :func:`predict_surface` because the service needs
    single-pixel lookups with the rasters held open, not a full sweep.
    """

    def __init__(self, cfg: Config, model_artefact: dict, scenario: str = "median"):
        self.cfg = cfg
        self.model = model_artefact["model"]
        self.features = model_artefact["features"]
        self.scenario = scenario

        self.dem, self.dem_ok = read_band(cfg.static_rasters["dem"])
        self.slope, self.slope_ok = read_band(cfg.static_rasters["slope"])
        self.builtup, self.built_ok = read_band(cfg.static_rasters["builtup"])
        self.rain7, self.rain_sum = build_scenario(cfg, scenario)
        self.perm, _ = read_band(cfg.auxiliary_rasters["perm_water"])

        with rasterio.open(cfg.reference_raster) as ds:
            self.transform = ds.transform
            self.crs = ds.crs
            self.height, self.width = ds.height, ds.width
            self.bounds = ds.bounds
        self._to_raster = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    # -- geometry -------------------------------------------------------
    def rowcol(self, lat: float, lon: float) -> tuple[int, int]:
        x, y = self._to_raster.transform(lon, lat)
        row, col = rasterio.transform.rowcol(self.transform, x, y)
        return int(row), int(col)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def extent_wgs84(self) -> dict[str, float]:
        west, south = self._to_wgs84.transform(self.bounds.left, self.bounds.bottom)
        east, north = self._to_wgs84.transform(self.bounds.right, self.bounds.top)
        return {"min_lat": south, "max_lat": north, "min_lon": west, "max_lon": east}

    # -- scoring --------------------------------------------------------
    def features_at(self, row: int, col: int,
                    rain7: float | None = None,
                    rain_sum: float | None = None) -> dict[str, float] | None:
        """Predictor values at a cell, or ``None`` where the cell is unusable."""
        if not self.in_bounds(row, col):
            return None
        if not (self.dem_ok[row, col] and self.slope_ok[row, col] and self.built_ok[row, col]):
            return None
        r7 = self.rain7[row, col] if rain7 is None else float(rain7)
        rs = self.rain_sum[row, col] if rain_sum is None else float(rain_sum)
        if not (np.isfinite(r7) and np.isfinite(rs)):
            return None
        return {
            "dem": float(self.dem[row, col]),
            "slope": float(self.slope[row, col]),
            "builtup": float(self.builtup[row, col]),
            "rain7": float(r7),
            "rain_sum": float(rs),
        }

    def score(self, lat: float, lon: float,
              rain7: float | None = None,
              rain_sum: float | None = None) -> dict:
        row, col = self.rowcol(lat, lon)
        if not self.in_bounds(row, col):
            return {"status": "outside_extent", "row": row, "col": col}
        feats = self.features_at(row, col, rain7, rain_sum)
        if feats is None:
            return {"status": "no_data_at_location", "row": row, "col": col}
        X = np.array([[feats[f] for f in self.features]], dtype="float32")
        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            "status": "ok",
            "row": row,
            "col": col,
            "probability": prob,
            "permanent_water": bool(self.perm[row, col] > 0),
            "features": feats,
        }


def run(cfg: Config, model_artefact: dict) -> Path:
    return predict_surface(cfg, model_artefact)
