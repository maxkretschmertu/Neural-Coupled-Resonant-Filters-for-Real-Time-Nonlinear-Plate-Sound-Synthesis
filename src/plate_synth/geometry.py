from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryDescription:
    """Normalized geometry passed to a modal backend.

    The geometry contains no material parameters and no absolute physical size.
    Phase 1 only provides rectangles; Phase 2 will extend the same contract to
    rectangle/circle/triangle morph geometries.
    """

    sdf: np.ndarray
    aspect_ratio: float
    kind: str = "rectangle"

    @property
    def grid_size(self) -> int:
        return int(self.sdf.shape[0])


def rectangle_sdf(aspect_ratio: float, grid_size: int = 64) -> np.ndarray:
    """Return a true signed-distance field for a centered normalized rectangle.

    Negative values are inside. The longest rectangle dimension spans [-0.9,
    0.9] in the canonical square; the other dimension follows the aspect ratio.
    """
    if aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be > 0")
    if grid_size < 8:
        raise ValueError("grid_size must be >= 8")

    if aspect_ratio >= 1.0:
        half_x = 0.9
        half_y = 0.9 / aspect_ratio
    else:
        half_x = 0.9 * aspect_ratio
        half_y = 0.9

    axis = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    qx = np.abs(x) - half_x
    qy = np.abs(y) - half_y

    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return np.ascontiguousarray(outside + inside, dtype=np.float64)


def make_rectangle_geometry(
    length_x_m: float,
    length_y_m: float,
    grid_size: int = 64,
) -> GeometryDescription:
    """Create the normalized Phase-1 geometry from physical dimensions."""
    if length_x_m <= 0.0 or length_y_m <= 0.0:
        raise ValueError("rectangle lengths must be > 0")
    aspect = float(length_x_m / length_y_m)
    return GeometryDescription(
        sdf=rectangle_sdf(aspect, grid_size),
        aspect_ratio=aspect,
        kind="rectangle",
    )
