from __future__ import annotations

import numpy as np

from .material_coordinates import clamp_material_point


def bilinear_sample_mode_maps(
    mode_shapes: np.ndarray,
    u: float,
    v: float,
) -> np.ndarray:
    """Sample mode maps stored on a canonical [-1,1]^2 material grid.

    Mode-map pixels outside the canonical material disk are zero. Under the
    current project contract (`simply_supported`) displacement is also zero at
    the plate boundary, so bilinear interpolation toward those exterior zeros
    is consistent with the active boundary condition. A future free-edge model
    must revisit this sampling rule instead of reusing it silently.
    """
    shapes = np.asarray(mode_shapes, dtype=np.float64)
    if shapes.ndim != 3:
        raise ValueError("mode_shapes must have shape (N, H, W)")

    _, height, width = shapes.shape
    if height < 2 or width < 2:
        raise ValueError("mode shape maps must be at least 2x2")

    u, v = clamp_material_point(float(u), float(v))
    gx = (u + 1.0) * 0.5 * (width - 1)
    gy = (v + 1.0) * 0.5 * (height - 1)
    x0, y0 = int(np.floor(gx)), int(np.floor(gy))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    tx, ty = gx - x0, gy - y0

    values = (
        (1.0 - tx) * (1.0 - ty) * shapes[:, y0, x0]
        + tx * (1.0 - ty) * shapes[:, y0, x1]
        + (1.0 - tx) * ty * shapes[:, y1, x0]
        + tx * ty * shapes[:, y1, x1]
    )
    return np.ascontiguousarray(values, dtype=np.float64)
