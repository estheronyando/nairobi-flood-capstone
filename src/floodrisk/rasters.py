"""Raster I/O and grid checks.

Every layer in this project is expected to sit on one reference grid (the
DEM). Rather than resample silently, the pipeline asserts the grids match
and stops if they do not, because a half-pixel offset between a predictor
and a label is invisible in the outputs but corrupts every sample drawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


@dataclass(frozen=True)
class Grid:
    """The spatial definition every layer must share."""

    width: int
    height: int
    crs: str
    transform: tuple[float, ...]

    @classmethod
    def from_dataset(cls, ds: rasterio.DatasetReader) -> "Grid":
        return cls(
            width=ds.width,
            height=ds.height,
            crs=str(ds.crs),
            transform=tuple(round(v, 6) for v in ds.transform[:6]),
        )

    def describe(self) -> str:
        return f"{self.width}x{self.height} @ {self.crs}, pixel {abs(self.transform[0]):.1f} m"


def read_grid(path: Path) -> Grid:
    with rasterio.open(path) as ds:
        return Grid.from_dataset(ds)


def read_band(path: Path, dtype: str = "float32") -> tuple[np.ndarray, np.ndarray]:
    """Read band 1 as ``(values, valid_mask)``.

    Validity means finite and not equal to the file's nodata value. These
    rasters carry no nodata tag, encoding gaps as NaN in the float layers,
    so both conditions are checked.
    """
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(dtype)
        nodata = ds.nodata
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return arr, valid


def read_profile(path: Path) -> dict:
    with rasterio.open(path) as ds:
        return ds.profile.copy()


def assert_same_grid(reference: Path, others: dict[str, Path]) -> Grid:
    """Raise if any layer differs from the reference grid."""
    ref = read_grid(reference)
    problems = []
    for name, path in others.items():
        if not path.exists():
            problems.append(f"  {name}: missing file {path.name}")
            continue
        g = read_grid(path)
        if g != ref:
            problems.append(f"  {name}: {g.describe()} != reference {ref.describe()}")
    if problems:
        raise ValueError(
            "Layers are not co-registered with "
            f"{reference.name} ({ref.describe()}):\n" + "\n".join(problems)
        )
    return ref


def write_geotiff(path: Path, array: np.ndarray, reference: Path, nodata: float = -9999.0,
                  dtype: str = "float32") -> Path:
    """Write ``array`` on the reference grid, compressed, with nodata set."""
    profile = read_profile(reference)
    profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    return path
