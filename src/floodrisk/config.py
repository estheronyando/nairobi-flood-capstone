"""Configuration loading and path resolution.

One source of truth for paths and parameters: ``config.yaml`` at the
repository root. Every module takes a :class:`Config` rather than
reading the YAML itself, so a reviewer can point the pipeline at a
different data directory by editing one file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# Event ids look like E01_2016_04 or E02_2016_04b.
EVENT_RE = re.compile(r"^label_event_water_(?P<event>E\d{2}_\d{4}_\d{2}[a-z]?)\.tif$")


@dataclass
class Config:
    """Resolved configuration. Paths are absolute by the time they land here."""

    raw: dict[str, Any]
    root: Path = REPO_ROOT

    # -- paths -------------------------------------------------------
    @property
    def raw_data(self) -> Path:
        return self._path(self.raw["paths"]["raw_data"])

    @property
    def models_dir(self) -> Path:
        return self._path(self.raw["paths"]["models"])

    @property
    def outputs_dir(self) -> Path:
        return self._path(self.raw["paths"]["outputs"])

    @property
    def figures_dir(self) -> Path:
        return self._path(self.raw["paths"]["figures"])

    @property
    def tables_dir(self) -> Path:
        return self._path(self.raw["paths"]["tables"])

    @property
    def susceptibility_raster(self) -> Path:
        return self._path(self.raw["prediction"]["output_raster"])

    # -- raster layout -----------------------------------------------
    @property
    def reference_raster(self) -> Path:
        return self.raw_data / self.raw["raster"]["reference"]

    @property
    def static_rasters(self) -> dict[str, Path]:
        return {k: self.raw_data / v for k, v in self.raw["raster"]["static"].items()}

    @property
    def auxiliary_rasters(self) -> dict[str, Path]:
        return {k: self.raw_data / v for k, v in self.raw["raster"]["auxiliary"].items()}

    def event_raster(self, kind: str, event: str) -> Path:
        """Path to an event-specific layer (``label``, ``rain7``, ``rain_sum``)."""
        pattern = self.raw["raster"]["event_patterns"][kind]
        return self.raw_data / pattern.format(event=event)

    def discover_events(self) -> list[str]:
        """Event ids present on disk, sorted. Driven by the label files."""
        events = []
        for p in sorted(self.raw_data.glob("label_event_water_*.tif")):
            m = EVENT_RE.match(p.name)
            if m:
                events.append(m.group("event"))
        if not events:
            raise FileNotFoundError(
                f"No label rasters found in {self.raw_data}. Expected files named "
                "label_event_water_<EVENT>.tif - see README, 'Getting the data'."
            )
        return events

    # -- split --------------------------------------------------------
    @property
    def test_events(self) -> list[str]:
        return list(self.raw["split"]["test_events"])

    def train_events(self) -> list[str]:
        return [e for e in self.discover_events() if e not in set(self.test_events)]

    # -- parameters ----------------------------------------------------
    @property
    def feature_columns(self) -> list[str]:
        return list(self.raw["features"]["columns"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw[name])

    # -- helpers --------------------------------------------------------
    def _path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.outputs_dir, self.figures_dir, self.tables_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load ``config.yaml`` (or an override path) into a :class:`Config`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw=raw, root=cfg_path.resolve().parent)
    cfg.ensure_dirs()
    return cfg
