"""Nairobi urban flood-susceptibility modelling.

Pipeline stages, in order:

1. :mod:`floodrisk.audit`     - verify the rasters before trusting them
2. :mod:`floodrisk.sampling`  - event-balanced pixel sample + event split
3. :mod:`floodrisk.train`     - fit the candidates, score on unseen events
4. :mod:`floodrisk.evaluate`  - curves, calibration, importance, SHAP
5. :mod:`floodrisk.predict`   - score every cell, write the GeoTIFF
"""

__version__ = "1.0.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
