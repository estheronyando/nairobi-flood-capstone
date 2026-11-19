"""The prediction service.

Three things a client can do: ask whether the service is healthy, ask what
the model is and how it scored, and ask for flood probabilities — either at
specific coordinates or as the whole 30 m GeoTIFF.

The rainfall scenario is a request parameter: callers may name an observed
event, or take the median or wettest observed storm. They may *not* supply
arbitrary rainfall values. The deployed model's response to rainfall is not
monotonic (see outputs/tables/rainfall_response.csv), so a free-form what-if
would return answers that look authoritative and point the wrong way. Every
response carries that caveat rather than leaving the caller to discover it.

Run locally:   uvicorn floodrisk.api:app --reload
In Docker:     docker compose up --build   (see README)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import load_config
from .predict import PointScorer

log = logging.getLogger(__name__)

STATE: dict[str, object] = {}

# Measured, not asserted: outputs/tables/rainfall_response.csv is regenerated
# by the pipeline and is what this sentence is based on.
RAINFALL_CAVEAT = (
    "Terrain-driven susceptibility, not a rainfall forecast. The model's response to "
    "rainfall is not monotonic, so differences between scenarios reflect learned "
    "event associations rather than a causal effect of rain. Do not read this as "
    "'how likely is flooding if it rains X mm'."
)

LIMITATIONS = [
    "Labels are Sentinel-1 water masks, not verified ground truth; SAR under-detects "
    "flooding beneath dense urban fabric.",
    "Training is balanced 50/50 while real prevalence is about 0.08% per event, so 0.5 "
    "is a presentational threshold, not a calibrated decision boundary.",
    RAINFALL_CAVEAT,
    "Output is a hazard indicator. It is not a flood-risk map until exposure and "
    "vulnerability layers are joined to it.",
]


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class Point(BaseModel):
    lat: float = Field(..., description="Latitude, WGS84 decimal degrees", examples=[-1.2921])
    lon: float = Field(..., description="Longitude, WGS84 decimal degrees", examples=[36.8219])

    @field_validator("lat")
    @classmethod
    def _lat_range(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError("lat must be between -90 and 90")
        return v

    @field_validator("lon")
    @classmethod
    def _lon_range(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError("lon must be between -180 and 180")
        return v


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}

    points: list[Point] = Field(..., min_length=1, max_length=1000)
    scenario: str = Field(
        "median",
        description="'median' (typical storm), 'max' (wettest observed), or an event id "
                    "such as E14_2025_04. Each uses rainfall fields actually observed "
                    "over Nairobi; arbitrary rainfall values are deliberately not "
                    "accepted - see the note on /model.",
    )


class PointPrediction(BaseModel):
    lat: float
    lon: float
    status: Literal["ok", "outside_extent", "no_data_at_location"]
    probability: float | None = None
    risk_band: str | None = None
    permanent_water: bool | None = None
    features: dict[str, float] | None = None


class PredictResponse(BaseModel):
    scenario: str
    model_name: str
    threshold: float
    predictions: list[PointPrediction]
    caveat: str = Field(
        default=RAINFALL_CAVEAT,
        description="Travels with every response so a probability is never read "
                    "without the limitation that governs it.",
    )


def risk_band(p: float) -> str:
    """Bands for reading the surface. These are presentational, not calibrated
    decision thresholds — see the report's limitations on prevalence."""
    if p >= 0.70:
        return "high"
    if p >= 0.40:
        return "moderate"
    return "low"


# --------------------------------------------------------------------------
# Lifespan: load the model and rasters once
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = load_config()
    model_path = cfg.models_dir / "flood_model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"No trained model at {model_path}. Run `python scripts/run_pipeline.py` first."
        )
    artefact = joblib.load(model_path)
    log.info("Loaded %s (trained on %d events)", artefact["model_name"],
             len(artefact["train_events"]))

    STATE["cfg"] = cfg
    STATE["artefact"] = artefact
    STATE["scorers"] = {}
    STATE["card"] = json.loads((cfg.models_dir / "model_card.json").read_text(encoding="utf-8"))
    # Warm the default scenario so the first request is not the slow one.
    STATE["scorers"]["median"] = PointScorer(cfg, artefact, scenario="median")

    resp = cfg.tables_dir / "rainfall_response.csv"
    if resp.exists():
        import csv

        with open(resp, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        means = [float(r["mean_probability"]) for r in rows]
        rising = sum(1 for a, b in zip(means, means[1:]) if b > a)
        STATE["rainfall_response"] = {
            "monotone_steps": rising,
            "total_steps": len(means) - 1,
            "monotonic": rising == len(means) - 1,
            "source": "outputs/tables/rainfall_response.csv",
        }
    log.info("Service ready")
    yield
    STATE.clear()


app = FastAPI(
    title="Nairobi Flood Susceptibility API",
    description=__doc__,
    version=__version__,
    lifespan=lifespan,
)


def get_scorer(scenario: str) -> PointScorer:
    """Scorers are cached per scenario; building one reads several rasters."""
    scorers: dict[str, PointScorer] = STATE["scorers"]  # type: ignore[assignment]
    if scenario not in scorers:
        cfg = STATE["cfg"]
        try:
            scorers[scenario] = PointScorer(cfg, STATE["artefact"], scenario=scenario)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return scorers[scenario]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health", summary="Liveness and readiness")
def health() -> dict:
    ready = "artefact" in STATE
    return {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "model_name": STATE.get("artefact", {}).get("model_name") if ready else None,
        "api_version": __version__,
    }


@app.get("/model", summary="Model card: features, training events, held-out scores")
def model_card() -> dict:
    if "card" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded")
    card = dict(STATE["card"])  # type: ignore[arg-type]
    card["limitations"] = LIMITATIONS
    card["rainfall_response"] = STATE.get("rainfall_response")
    return card


@app.get("/extent", summary="Geographic extent the model covers")
def extent() -> dict:
    scorer = get_scorer("median")
    return {
        "crs": str(scorer.crs),
        "wgs84": scorer.extent_wgs84(),
        "grid": {"width": scorer.width, "height": scorer.height, "pixel_m": 30},
    }


@app.get("/scenarios", summary="Rainfall scenarios available")
def scenarios() -> dict:
    cfg = STATE["cfg"]
    return {
        "named": ["median", "max"],
        "events": cfg.discover_events(),  # type: ignore[union-attr]
        "note": ("Each scenario uses rainfall fields actually observed over Nairobi. "
                 "Arbitrary rainfall values are not accepted: the model's rainfall "
                 "response is not monotonic, so a what-if would be misleading."),
    }


@app.post("/predict", response_model=PredictResponse, summary="Flood probability at points")
def predict(req: PredictRequest) -> PredictResponse:
    scorer = get_scorer(req.scenario)
    artefact = STATE["artefact"]

    predictions = []
    for pt in req.points:
        out = scorer.score(pt.lat, pt.lon)
        prob = out.get("probability")
        predictions.append(
            PointPrediction(
                lat=pt.lat,
                lon=pt.lon,
                status=out["status"],
                probability=round(prob, 4) if prob is not None else None,
                risk_band=risk_band(prob) if prob is not None else None,
                permanent_water=out.get("permanent_water"),
                features={k: round(v, 3) for k, v in out["features"].items()}
                if out.get("features") else None,
            )
        )

    return PredictResponse(
        scenario=req.scenario,
        model_name=artefact["model_name"],  # type: ignore[index]
        threshold=artefact["threshold"],    # type: ignore[index]
        predictions=predictions,
    )


@app.get("/map", summary="Download the susceptibility GeoTIFF")
def get_map(
    scenario: str = Query("median", description="Scenario the raster was produced for"),
) -> FileResponse:
    cfg = STATE["cfg"]
    base: Path = cfg.susceptibility_raster  # type: ignore[union-attr]
    path = base if scenario == cfg.section("prediction")["scenario"] else base.with_name(  # type: ignore[union-attr]
        f"{base.stem}_{scenario}{base.suffix}"
    )
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No raster for scenario '{scenario}'. Generate it with "
                f"`python scripts/run_pipeline.py --scenario {scenario}`."
            ),
        )
    return FileResponse(path, media_type="image/tiff", filename=path.name)
