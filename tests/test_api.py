"""Tests for the prediction service.

These exercise the API against the real model and rasters: a mocked model
would pass while the thing that actually breaks in deployment - the
coordinate transform, the feature order, a missing artefact - goes
unchecked.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Inside Nairobi City County, on the reference grid.
NAIROBI_CBD = {"lat": -1.2921, "lon": 36.8219}
MATHARE = {"lat": -1.2630, "lon": 36.8600}
# Comfortably outside the raster extent.
FAR_AWAY = {"lat": -4.0435, "lon": 39.6682}   # Mombasa


@pytest.fixture(scope="module")
def client(cfg):
    if not (cfg.models_dir / "flood_model.joblib").exists():
        pytest.skip("No trained model; run `python scripts/run_pipeline.py` first.")
    from floodrisk.api import app

    with TestClient(app) as c:
        yield c


class TestHealthAndMetadata:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_model_card(self, client):
        body = client.get("/model").json()
        assert body["features"] == ["dem", "slope", "builtup", "rain7", "rain_sum"]
        assert len(body["train_events"]) == 11
        assert len(body["test_events"]) == 4
        assert 0.5 < body["metrics"]["roc_auc"] < 0.99

    def test_extent_covers_nairobi(self, client):
        w = client.get("/extent").json()["wgs84"]
        assert w["min_lat"] < NAIROBI_CBD["lat"] < w["max_lat"]
        assert w["min_lon"] < NAIROBI_CBD["lon"] < w["max_lon"]

    def test_scenarios_lists_every_event(self, client):
        body = client.get("/scenarios").json()
        assert len(body["events"]) == 15
        assert "median" in body["named"]


class TestPredict:
    def test_returns_a_probability_inside_the_extent(self, client):
        r = client.post("/predict", json={"points": [NAIROBI_CBD]})
        assert r.status_code == 200
        pred = r.json()["predictions"][0]
        assert pred["status"] == "ok"
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["risk_band"] in {"low", "moderate", "high"}
        assert set(pred["features"]) == {"dem", "slope", "builtup", "rain7", "rain_sum"}

    def test_batch(self, client):
        r = client.post("/predict", json={"points": [NAIROBI_CBD, MATHARE]})
        assert len(r.json()["predictions"]) == 2

    def test_point_outside_the_grid_is_reported_not_guessed(self, client):
        """Mombasa is not in the training domain; the service must say so
        rather than return a number that looks like a prediction."""
        r = client.post("/predict", json={"points": [FAR_AWAY]})
        pred = r.json()["predictions"][0]
        assert pred["status"] == "outside_extent"
        assert pred["probability"] is None

    def test_arbitrary_rainfall_is_not_accepted(self, client):
        """The model's rainfall response is not monotonic, so a free-form
        what-if would return confident answers pointing the wrong way. The
        contract must reject the attempt rather than quietly ignore it."""
        r = client.post("/predict", json={
            "points": [MATHARE], "rain7": 300.0, "rain_sum": 500.0,
        })
        assert r.status_code == 422

    def test_every_response_carries_the_caveat(self, client):
        body = client.post("/predict", json={"points": [MATHARE]}).json()
        assert "not monotonic" in body["caveat"]

    def test_model_card_publishes_measured_limitations(self, client):
        card = client.get("/model").json()
        assert any("prevalence" in L for L in card["limitations"])
        rr = card["rainfall_response"]
        assert rr is not None and rr["monotonic"] is False

    def test_named_event_scenario(self, client):
        r = client.post("/predict", json={"points": [MATHARE], "scenario": "E14_2025_04"})
        assert r.status_code == 200
        assert r.json()["scenario"] == "E14_2025_04"

    def test_unknown_scenario_is_rejected(self, client):
        r = client.post("/predict", json={"points": [MATHARE], "scenario": "not_a_scenario"})
        assert r.status_code == 422

    @pytest.mark.parametrize("bad", [{"lat": 200, "lon": 36.8}, {"lat": -1.3, "lon": 400}])
    def test_invalid_coordinates_rejected(self, client, bad):
        assert client.post("/predict", json={"points": [bad]}).status_code == 422

    def test_empty_request_rejected(self, client):
        assert client.post("/predict", json={"points": []}).status_code == 422


class TestMap:
    def test_serves_the_geotiff(self, client):
        r = client.get("/map")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/tiff"
        assert len(r.content) > 100_000
        assert r.content[:4] in (b"II*\x00", b"MM\x00*"), "not a TIFF"

    def test_missing_scenario_is_a_404_with_a_hint(self, client):
        r = client.get("/map", params={"scenario": "E99_1999_01"})
        assert r.status_code == 404
        assert "run_pipeline" in r.json()["detail"]
