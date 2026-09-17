from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Protocol

import numpy as np

from .config import SynthParameters
from .geometry import GeometryDescription
from .spatial import analytical_rectangle_weights, bilinear_sample_mode_maps


@dataclass(frozen=True)
class ModalBasis:
    """Geometry-dependent modal data produced by a modal backend.

    Crucially, this object contains no material-scaled frequencies. A future
    neural backend therefore only has to predict geometry-dependent modal
    factors and mode shapes. Physical size, material, damping and tuning are
    resolved afterwards by ordinary code.
    """

    modal_factors: np.ndarray
    mode_shapes: np.ndarray
    geometry_sdf: np.ndarray
    integration_weights: np.ndarray
    legacy_mode_indices: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        return int(self.modal_factors.shape[0])

    def sample_weights(self, x: float, y: float) -> np.ndarray:
        if self.legacy_mode_indices is not None:
            return analytical_rectangle_weights(self.legacy_mode_indices, x, y)
        return bilinear_sample_mode_maps(self.mode_shapes, x, y)


class ModalBackend(Protocol):
    """Geometry-only modal predictor contract used by analytical and NN backends."""

    def predict(self, geometry: GeometryDescription, n_modes: int) -> ModalBasis:
        ...


class LegacyRectangleBackend:
    """Rectangle reference backend preserving the historical v12 ordering.

    For n_modes=90 the generated pairs are exactly the old order=10 list.
    Other counts use the smallest legacy rectangular enumeration large enough
    to supply the requested number of modes, then truncate it.
    """

    def __init__(self, shape_grid_size: int = 64) -> None:
        if shape_grid_size < 8:
            raise ValueError("shape_grid_size must be >= 8")
        self.shape_grid_size = int(shape_grid_size)

    @staticmethod
    def legacy_mode_indices(n_modes: int) -> np.ndarray:
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        order = max(2, int(ceil((1.0 + sqrt(1.0 + 4.0 * n_modes)) / 2.0)))
        pairs = [(m, n) for m in range(1, order + 1) for n in range(1, order)]
        return np.asarray(pairs[:n_modes], dtype=np.int64).reshape(-1, 2)

    def _mode_shape_maps(
        self,
        mode_indices: np.ndarray,
        geometry: GeometryDescription,
    ) -> np.ndarray:
        """Evaluate rectangle modes on the same canonical grid as the SDF."""
        n = geometry.grid_size
        aspect = geometry.aspect_ratio
        if aspect >= 1.0:
            half_x = 0.9
            half_y = 0.9 / aspect
        else:
            half_x = 0.9 * aspect
            half_y = 0.9

        axis = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        x, y = np.meshgrid(axis, axis, indexing="xy")
        u = (x / half_x + 1.0) * 0.5
        v = (y / half_y + 1.0) * 0.5
        inside = geometry.sdf <= 0.0

        shapes = np.zeros((mode_indices.shape[0], n, n), dtype=np.float64)
        for k, (l, m) in enumerate(mode_indices):
            values = np.sin(l * np.pi * u) * np.sin(m * np.pi * v)
            shapes[k, inside] = values[inside]
        return np.ascontiguousarray(shapes)

    def predict(self, geometry: GeometryDescription, n_modes: int) -> ModalBasis:
        if geometry.kind != "rectangle":
            raise ValueError("LegacyRectangleBackend only supports rectangles")

        indices = self.legacy_mode_indices(n_modes)
        l = indices[:, 0].astype(np.float64)
        m = indices[:, 1].astype(np.float64)

        # Exact geometry-dependent factor used by v12. Material and absolute
        # scale are deliberately not applied here.
        factors = l**2 + geometry.aspect_ratio * m**2
        shapes = self._mode_shape_maps(indices, geometry)

        inside = (geometry.sdf <= 0.0).astype(np.float64)
        total = float(np.sum(inside))
        weights = inside / total if total > 0.0 else inside

        return ModalBasis(
            modal_factors=np.ascontiguousarray(factors, dtype=np.float64),
            mode_shapes=shapes,
            geometry_sdf=np.ascontiguousarray(geometry.sdf, dtype=np.float64),
            integration_weights=np.ascontiguousarray(weights, dtype=np.float64),
            legacy_mode_indices=indices,
        )


def resolve_modal_frequencies_hz(
    basis: ModalBasis,
    params: SynthParameters,
) -> np.ndarray:
    """Apply physical size/material scaling and musical tuning outside the backend.

    This preserves the v12 rectangle expression:
        omega = pi^2/(Lx*Ly) * sqrt(D/(rho*H)) * modal_factor
    while keeping D, rho, H and frequency_scale out of the modal predictor.
    """
    if min(
        params.length_x_m,
        params.length_y_m,
        params.flexural_rigidity,
        params.density,
        params.thickness_m,
        params.frequency_scale,
    ) <= 0.0:
        raise ValueError("physical dimensions/material/frequency_scale must be > 0")

    scale = (
        np.pi**2
        / (params.length_x_m * params.length_y_m)
        * np.sqrt(params.flexural_rigidity / (params.density * params.thickness_m))
    )
    omega = scale * np.asarray(basis.modal_factors, dtype=np.float64)
    frequencies = omega / (2.0 * np.pi)
    frequencies *= float(params.frequency_scale)
    return np.ascontiguousarray(frequencies, dtype=np.float64)
