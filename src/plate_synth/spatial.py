from __future__ import annotations

import numpy as np


def clamp_unit(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def analytical_rectangle_weights(
    mode_indices: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    """Evaluate simply-supported rectangle shapes at normalized coordinates.

    This is the exact spatial weighting used by v12:
        phi_lm(x, y) = sin(l*pi*x) * sin(m*pi*y)
    """
    xy = np.asarray(mode_indices, dtype=np.int64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("mode_indices must have shape (N, 2)")

    x = clamp_unit(x)
    y = clamp_unit(y)
    l = xy[:, 0]
    m = xy[:, 1]
    weights = np.sin(l * np.pi * x) * np.sin(m * np.pi * y)
    return np.ascontiguousarray(weights, dtype=np.float64)


def bilinear_sample_mode_maps(
    mode_shapes: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    """Sample N mode-shape maps at one normalized (x, y) location.

    `mode_shapes` has shape (N, H, W). x=0 is the left edge and y=0 the
    bottom edge. Array row 0 is treated as the bottom row of the canonical
    modal grid. The future neural backend will use this function directly.
    """
    shapes = np.asarray(mode_shapes, dtype=np.float64)
    if shapes.ndim != 3:
        raise ValueError("mode_shapes must have shape (N, H, W)")

    _, height, width = shapes.shape
    if height < 2 or width < 2:
        raise ValueError("mode shape maps must be at least 2 x 2")

    x = clamp_unit(x)
    y = clamp_unit(y)

    gx = x * (width - 1)
    gy = y * (height - 1)

    x0 = int(np.floor(gx))
    y0 = int(np.floor(gy))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    tx = gx - x0
    ty = gy - y0

    v00 = shapes[:, y0, x0]
    v10 = shapes[:, y0, x1]
    v01 = shapes[:, y1, x0]
    v11 = shapes[:, y1, x1]

    values = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return np.ascontiguousarray(values, dtype=np.float64)
