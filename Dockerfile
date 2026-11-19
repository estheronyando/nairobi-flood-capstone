# Nairobi flood-susceptibility prediction service.
#
# Python 3.13 is required, not incidental: rasterio, pyproj, xgboost, shap
# and numpy at the pinned versions publish no cp311/cp312 wheels, so an
# older base image would fall back to building them from source.
#
# The image ships the trained model, the 30 m rasters the service reads
# predictors from, and the precomputed susceptibility surface, so
# `docker run` gives a working API with no volumes to wire up. Mount your
# own data over /app/data if you retrain.

FROM python:3.13-slim

# rasterio and pyproj ship manylinux wheels with GDAL/PROJ bundled, so no
# system GDAL is needed. Rasterio's GDAL extension does dynamically require
# libexpat from Debian; curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libexpat1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first so edits to source do not invalidate the wheel layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/
COPY outputs/ ./outputs/

# Run as a non-root user.
RUN useradd --create-home --uid 1000 flood && chown -R flood:flood /app
USER flood

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Respect Render's PORT at deployment time while retaining port 8000 for
# `docker compose up` locally.
CMD ["sh", "-c", "exec uvicorn floodrisk.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
