"""A calm, user-focused view of the Nairobi flood-susceptibility API.

This app deliberately contains no model or fallback calculation. Every assessment
comes from a ``/predict`` call to the trained service; if that service cannot be
reached, the interface says so plainly instead of inventing a result.

    export FLOOD_API_URL=http://localhost:8000
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from html import escape
import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("FLOOD_API_URL", "http://localhost:8000")
TIMEOUT = 60.0

PLACES = {
    "Nairobi CBD": (-1.2921, 36.8219),
    "Mathare valley": (-1.2630, 36.8600),
    "Kibera": (-1.3130, 36.7800),
    "Embakasi / Nairobi River": (-1.3190, 36.9250),
    "Kasarani": (-1.2270, 36.8870),
    "Karen": (-1.3350, 36.7100),
}

FEATURE_LABELS = {
    "dem": ("Elevation", "m"),
    "slope": ("Ground slope", "degrees"),
    "builtup": ("Built-up area", ""),
    "rain7": ("Recent rainfall (7 days)", "mm"),
    "rain_sum": ("Total event rainfall", "mm"),
}

st.set_page_config(page_title="Nairobi Flood Susceptibility", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str) -> dict | None:
    """Read public API metadata without surfacing transport details to users."""
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def api_predict(points: list[dict], scenario: str) -> tuple[dict | None, str | None]:
    """Request an assessment and translate failures into useful UI copy."""
    try:
        response = httpx.post(
            f"{API_URL}/predict",
            json={"points": points, "scenario": scenario},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 422:
            return None, "This assessment could not be completed. Check the location and rainfall scenario, then try again."
        return None, "The prediction service could not complete the assessment. Please try again shortly."
    except httpx.RequestError:
        return None, "The prediction service could not be reached. Confirm that it is running, then try again."
    except ValueError:
        return None, "The prediction service returned an unreadable response. Please try again shortly."


def initialise_state() -> None:
    st.session_state.setdefault("selected_place", "Enter coordinates")
    st.session_state.setdefault("assessment_lat", PLACES["Nairobi CBD"][0])
    st.session_state.setdefault("assessment_lon", PLACES["Nairobi CBD"][1])
    st.session_state.setdefault("assessment", None)
    st.session_state.setdefault("reference_assessments", {})


def apply_preset_location() -> None:
    """Keep coordinate entry and the selected reference location in sync."""
    place = st.session_state["selected_place"]
    if place in PLACES:
        st.session_state["assessment_lat"], st.session_state["assessment_lon"] = PLACES[place]


def mark_custom_location() -> None:
    st.session_state["selected_place"] = "Enter coordinates"


def location_name(place: str, latitude: float, longitude: float) -> str:
    if place in PLACES:
        return place
    return f"Coordinates: {latitude:.4f}, {longitude:.4f}"


def scenario_label(scenario: str) -> str:
    return {
        "median": "Typical observed storm",
        "max": "Wettest observed storm",
    }.get(scenario, scenario)


def feature_rows(features: dict[str, float]) -> pd.DataFrame:
    """Make returned predictors readable while retaining their source names."""
    rows = []
    for technical_name, value in features.items():
        label, unit = FEATURE_LABELS.get(technical_name, (technical_name, ""))
        if technical_name == "builtup" and value in {0, 1, 0.0, 1.0}:
            display_value = "Yes" if value else "No"
        else:
            display_value = f"{value:,.3f}" + (f" {unit}" if unit else "")
        rows.append(
            {
                "Environmental factor": label,
                "Value": display_value,
                "Technical name": technical_name,
                "Raw value": f"{value:.3f}",
            }
        )
    return pd.DataFrame(rows)


def render_assessment(assessment: dict) -> None:
    """Render a stored API response without recomputing a prediction."""
    prediction = assessment["prediction"]
    status = prediction.get("status")
    place = assessment["location"]

    if status == "outside_extent":
        st.warning(
            f"**{place} is outside the assessed area.** Choose a location inside Nairobi City County "
            "to receive a susceptibility assessment."
        )
        return
    if status != "ok":
        st.warning(
            f"**An assessment is not available for {place}.** The service has no usable data at this point. "
            "Try a nearby location or one of the reference locations."
        )
        return

    band = str(prediction.get("risk_band", "unknown")).lower()
    band_label = band.capitalize()
    score = prediction.get("probability")
    score_text = f"{score:.1%}" if isinstance(score, (float, int)) else "Not available"
    safe_place = escape(place)
    safe_scenario = escape(scenario_label(assessment["scenario"]))

    st.markdown(
        f"""
        <section class="assessment-card">
          <p class="card-kicker">LOCATION ASSESSMENT</p>
          <h2>{safe_place}</h2>
          <p class="scenario-label">Rainfall scenario: <strong>{safe_scenario}</strong></p>
          <div class="susceptibility-block">
            <span>Susceptibility</span>
            <strong class="risk-badge risk-{escape(band)}">{escape(band_label)}</strong>
          </div>
          <div class="score-block">
            <span>Model score</span>
            <strong>{score_text}</strong>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The model score is a susceptibility indicator for this 30 m location, not a flood forecast "
        "or an individual likelihood of flood damage."
    )

    caveat = assessment.get("caveat")
    if caveat:
        st.info(caveat)

    features = prediction.get("features") or {}
    if features:
        with st.expander("Location details", expanded=False):
            st.caption("These are the environmental inputs returned by the model service for this location.")
            st.dataframe(feature_rows(features), hide_index=True, width="stretch")


def render_reference_locations(scenario: str) -> None:
    """Keep useful reference assessments available without competing with results."""
    with st.expander("Reference locations", expanded=False):
        st.caption("Compare a small set of well-known Nairobi locations under the selected rainfall scenario.")
        reference_cache: dict[str, dict] = st.session_state["reference_assessments"]
        if scenario not in reference_cache:
            if st.button("Show reference assessments", key=f"reference-{scenario}"):
                with st.spinner("Loading reference assessments..."):
                    response, error = api_predict(
                        [{"lat": lat, "lon": lon} for lat, lon in PLACES.values()], scenario
                    )
                if error:
                    st.error(error)
                elif response:
                    reference_cache[scenario] = response
                    st.session_state["reference_assessments"] = reference_cache
                    st.rerun()

        response = reference_cache.get(scenario)
        if response:
            rows = []
            for name, prediction in zip(PLACES, response.get("predictions", [])):
                probability = prediction.get("probability")
                elevation = (prediction.get("features") or {}).get("dem")
                rows.append(
                    {
                        "Location": name,
                        "Susceptibility": str(prediction.get("risk_band") or "Not available").capitalize(),
                        "Model score": f"{probability:.1%}" if isinstance(probability, (float, int)) else "Not available",
                        "Elevation (m)": f"{elevation:,.1f}" if isinstance(elevation, (float, int)) else "Not available",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_model_information(card: dict, extent: dict) -> None:
    """Provide scientific transparency as optional secondary content."""
    with st.expander("About the model", expanded=False):
        metrics = card.get("metrics") or {}
        test_events = card.get("test_events") or []
        first, second, third = st.columns(3)
        first.metric("Model", card.get("model_name", "Not available"))
        second.metric("Held-out ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
        third.metric("Held-out F1", f"{metrics.get('f1', 0):.4f}")
        if test_events:
            st.caption(
                f"Performance was measured on {len(test_events)} flood events not used for model training."
            )

        limitations = card.get("limitations") or []
        if limitations:
            st.markdown("**Limitations**")
            for limitation in limitations:
                st.markdown(f"- {limitation}")

        coverage = extent.get("wgs84") or {}
        if coverage:
            st.markdown("**Coverage**")
            st.caption(
                "Assessments are available only within the trained raster area "
                f"(latitude {coverage['min_lat']:.3f} to {coverage['max_lat']:.3f}; "
                f"longitude {coverage['min_lon']:.3f} to {coverage['max_lon']:.3f})."
            )


def inject_styles() -> None:
    """Small, stable style additions that support hierarchy and accessibility."""
    st.markdown(
        """
        <style>
          .block-container { max-width: 1160px; padding-top: 2.4rem; padding-bottom: 3rem; }
          [data-testid="stSidebar"] { border-right: 1px solid #dce5e2; }
          [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
          .page-kicker, .card-kicker { color: #4b635b; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 0.25rem; }
          .page-subtitle { color: #47615a; font-size: 1.05rem; margin: -0.25rem 0 2rem; }
          .sidebar-title { color: #173b31; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.15rem; }
          .assessment-card { background: linear-gradient(135deg, #f4f9f7 0%, #ffffff 100%); border: 1px solid #d5e4de; border-radius: 14px; padding: 1.5rem 1.6rem; margin: 0.7rem 0 0.6rem; }
          .assessment-card h2 { color: #173b31; font-size: 1.65rem; margin: 0 0 0.25rem; }
          .scenario-label { color: #536a63; margin: 0 0 1.3rem; }
          .susceptibility-block, .score-block { display: flex; align-items: center; justify-content: space-between; gap: 1rem; border-top: 1px solid #dbe7e2; padding-top: 0.9rem; margin-top: 0.9rem; color: #3e554e; }
          .susceptibility-block > span, .score-block > span { font-size: 0.92rem; }
          .score-block strong { color: #173b31; font-size: 1.45rem; }
          .risk-badge { border-radius: 999px; display: inline-block; font-size: 0.9rem; font-weight: 750; letter-spacing: 0.04em; padding: 0.32rem 0.8rem; text-transform: uppercase; }
          .risk-low { background: #d8eee0; color: #174c2f; }
          .risk-moderate { background: #ffebbd; color: #6a4300; }
          .risk-high { background: #f9d8d4; color: #8d2119; }
          .risk-unknown { background: #e8eceb; color: #34413d; }
        </style>
        """,
        unsafe_allow_html=True,
    )


initialise_state()
inject_styles()

st.markdown('<p class="page-kicker">GEOSPATIAL DECISION SUPPORT</p>', unsafe_allow_html=True)
st.title("Nairobi Urban Flood Susceptibility")
st.markdown(
    '<p class="page-subtitle">Explore flood susceptibility across Nairobi using geospatial and machine-learning analysis.</p>',
    unsafe_allow_html=True,
)

health = api_get("/health")
if not health or health.get("status") != "ok":
    st.error(
        "The prediction service is currently unavailable. Start the API service, then refresh this page to assess a location."
    )
    st.stop()

card = api_get("/model") or {}
extent = api_get("/extent") or {}
scenario_data = api_get("/scenarios") or {"named": ["median"], "events": []}
scenarios = scenario_data.get("named", ["median"]) + scenario_data.get("events", [])
if not scenarios:
    scenarios = ["median"]

with st.sidebar:
    st.markdown('<p class="sidebar-title">Assess a location</p>', unsafe_allow_html=True)
    scenario = st.selectbox(
        "Rainfall scenario",
        scenarios,
        help="Each option uses rainfall observed over Nairobi. The tool does not support arbitrary rainfall inputs.",
    )
    st.divider()
    st.selectbox(
        "Reference location",
        ["Enter coordinates"] + list(PLACES),
        key="selected_place",
        on_change=apply_preset_location,
    )
    st.caption("Choose a reference location or enter a point in WGS84 decimal degrees.")
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        format="%.4f",
        key="assessment_lat",
        on_change=mark_custom_location,
    )
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        format="%.4f",
        key="assessment_lon",
        on_change=mark_custom_location,
    )
    assess = st.button("Assess Susceptibility", type="primary", width="stretch")

if assess:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        st.error("Enter a valid latitude (-90 to 90) and longitude (-180 to 180).")
    else:
        with st.spinner("Assessing location..."):
            response, error = api_predict([{"lat": latitude, "lon": longitude}], scenario)
        if error:
            st.error(error)
        elif response and response.get("predictions"):
            st.session_state["assessment"] = {
                "prediction": response["predictions"][0],
                "scenario": response.get("scenario", scenario),
                "location": location_name(st.session_state["selected_place"], latitude, longitude),
                "caveat": response.get("caveat"),
            }

assessment = st.session_state.get("assessment")
if assessment:
    render_assessment(assessment)
else:
    st.info(
        "Select a rainfall scenario and a Nairobi location in the sidebar, then choose **Assess Susceptibility**. "
        "Your result will appear here."
    )

render_reference_locations(scenario)
render_model_information(card, extent)
