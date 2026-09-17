from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import SynthParameters
from .spatial import analytical_rectangle_weights, bilinear_sample_mode_maps


@dataclass(frozen=True)
class ModalBasis:
    """Complete linear modal description consumed by the resonator synth.

    `modal_factors` is intentionally generic. In the future neural backend it
    will contain geometry-dependent dimensionless modal information. For the
    legacy backend it stores the unscaled v12 modal factor.

    `mode_shapes` is always provided on a canonical regular grid so the same
    object can later be used for pickup sampling and modal-basis projection.
    `legacy_mode_indices` is optional and exists only to reproduce the exact
    analytical v12 strike/pickup weighting during regression.
    """

    modal_factors: np.ndarray
    frequencies_hz: np.ndarray
    mode_shapes: np.ndarray
    geometry_sdf: np.ndarray
    integration_weights: np.ndarray
    legacy_mode_indices: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        return int(self.frequencies_hz.shape[0])

    def sample_weights(self, x: float, y: float) -> np.ndarray:
        """Return phi_k(x, y) for every mode."""
        if self.legacy_mode_indices is not None:
            # Preserve exact v12 weighting for the legacy reference backend.
            return analytical_rectangle_weights(self.legacy_mode_indices, x, y)
        return bilinear_sample_mode_maps(self.mode_shapes, x, y)

    def with_frequencies(self, frequencies_hz: np.ndarray) -> "ModalBasis":
        freqs = np.ascontiguousarray(frequencies_hz, dtype=np.float64)
        if freqs.shape != self.frequencies_hz.shape:
            raise ValueError("frequency array shape must match the modal basis")
        return ModalBasis(
            modal_factors=self.modal_factors,
            frequencies_hz=freqs,
            mode_shapes=self.mode_shapes,
            geometry_sdf=self.geometry_sdf,
            integration_weights=self.integration_weights,
            legacy_mode_indices=self.legacy_mode_indices,
        )


class ModalBackend(Protocol):
    """Common interface for analytical and future neural modal models."""

    def predict(self, params: SynthParameters) -> ModalBasis:
        ...


class LegacyRectangleBackend:
    """Analytical rectangle backend matching v12 as closely as possible.

    This backend is a regression/reference implementation. The historical v12
    mode enumeration and frequency expression are intentionally preserved here
    rather than silently changed during the refactor.
    """

    def __init__(self, shape_grid_size: int = 64) -> None:
        if shape_grid_size < 8:
            raise ValueError("shape_grid_size must be >= 8")
        self.shape_grid_size = int(shape_grid_size)

    @staticmethod
    def legacy_mode_indices(order: int) -> np.ndarray:
        """Return the exact x * (x - 1) enumeration used in v12."""
        if order < 1:
            raise ValueError("legacy_mode_order must be >= 1")
        pairs = [(m, n) for m in range(1, order + 1) for n in range(1, order)]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    @staticmethod
    def legacy_frequencies_hz(
        mode_indices: np.ndarray,
        length_x_m: float,
        length_y_m: float,
        flexural_rigidity: float,
        density: float,
        thickness_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reproduce the v12 `modes_to_freqs()` expression exactly."""
        if min(length_x_m, length_y_m, flexural_rigidity, density, thickness_m) <= 0:
            raise ValueError("plate dimensions and material parameters must be > 0")

        v = length_x_m / length_y_m
        scale = (
            np.pi**2
            / (length_x_m * length_y_m)
            * np.sqrt(flexural_rigidity / (density * thickness_m))
        )

        l = mode_indices[:, 0].astype(np.float64)
        m = mode_indices[:, 1].astype(np.float64)
        factors = l**2 + v * m**2
        omega = scale * factors
        frequencies = omega / (2.0 * np.pi)

        return (
            np.ascontiguousarray(factors, dtype=np.float64),
            np.ascontiguousarray(frequencies, dtype=np.float64),
        )

    def _mode_shape_maps(self, mode_indices: np.ndarray) -> np.ndarray:
        n = self.shape_grid_size
        axis = np.linspace(0.0, 1.0, n, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")

        shapes = np.empty((mode_indices.shape[0], n, n), dtype=np.float64)
        for k, (l, m) in enumerate(mode_indices):
            shapes[k] = np.sin(l * np.pi * x_grid) * np.sin(m * np.pi * y_grid)
        return np.ascontiguousarray(shapes)

    def predict(self, params: SynthParameters) -> ModalBasis:
        mode_indices = self.legacy_mode_indices(params.legacy_mode_order)
        factors, frequencies = self.legacy_frequencies_hz(
            mode_indices=mode_indices,
            length_x_m=params.length_x_m,
            length_y_m=params.length_y_m,
            flexural_rigidity=params.flexural_rigidity,
            density=params.density,
            thickness_m=params.thickness_m,
        )

        mode_shapes = self._mode_shape_maps(mode_indices)
        grid_size = self.shape_grid_size

        # The Phase-1 legacy geometry is a full canonical rectangle.
        # Negative-inside SDF convention is reserved for the future geometry
        # generator; for now this simple field is sufficient as metadata.
        geometry_sdf = -np.ones((grid_size, grid_size), dtype=np.float64)
        integration_weights = np.full(
            (grid_size, grid_size),
            1.0 / (grid_size * grid_size),
            dtype=np.float64,
        )

        return ModalBasis(
            modal_factors=factors,
            frequencies_hz=frequencies,
            mode_shapes=mode_shapes,
            geometry_sdf=np.ascontiguousarray(geometry_sdf),
            integration_weights=np.ascontiguousarray(integration_weights),
            legacy_mode_indices=mode_indices,
        )
