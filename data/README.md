# Data placement

The raw GeoTIFF rasters are intentionally excluded from version control. They are required for full pipeline runs and live point predictions, but are too large and subject to their source terms for this repository.

Place the prepared raster stack in `data/nairobi_flood_data/` before running the full pipeline or starting the API locally. The repository still includes the trained model, model card, diagnostics, figures, and model-contract tests so reviewers can inspect the implementation without downloading the raw data.
