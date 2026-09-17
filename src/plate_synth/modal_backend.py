from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from .config import SynthParameters
from .geometry import GeometryDescription
from .material_coordinates import make_material_grid
from .spatial import bilinear_sample_mode_maps


class UnsupportedGeometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModalBasis:
    """Geometry-dependent, material-independent modal basis.

    modal_factors follow
        omega_k = modal_factor_k * sqrt(D/(rho*H)) / size_m^2
    before the user frequency-scale control.
    """

    modal_factors: np.ndarray
    mode_shapes: np.ndarray
    integration_weights: np.ndarray
    geometry: GeometryDescription

    @property
    def n_modes(self) -> int:
        return int(self.modal_factors.shape[0])

    def sample_weights(self, u: float, v: float) -> np.ndarray:
        return bilinear_sample_mode_maps(self.mode_shapes, u, v)


class ModalBackend(Protocol):
    def predict(self, geometry: GeometryDescription, n_modes: int) -> ModalBasis:
        ...


class AnalyticRectangleBackend:
    """Simply-supported rectangle backend used until the NN is trained.

    It is not a compatibility layer. Modes are sorted by increasing physical
    modal factor and use the rectangular Kirchhoff-plate expression on the
    unit-area canonical domain. Non-rectangle geometries are rejected rather
    than approximated with fake modes.
    """

    def __init__(self, mode_grid_size: int = 64) -> None:
        if mode_grid_size < 16:
            raise ValueError("mode_grid_size must be >= 16")
        self.mode_grid_size = int(mode_grid_size)

    @staticmethod
    def _is_rectangle_endpoint(geometry: GeometryDescription) -> bool:
        return abs(float(geometry.morph)) <= 1e-12

    @staticmethod
    def _aspect(shape_mod: float, max_stretch: float = 4.0) -> float:
        return float(np.exp(np.log(max_stretch) * np.clip(shape_mod, 0.0, 1.0)))

    def predict(self, geometry: GeometryDescription, n_modes: int) -> ModalBasis:
        if not self._is_rectangle_endpoint(geometry):
            raise UnsupportedGeometryError(
                "No trained neural modal backend is available for morphed/circle/triangle geometries yet."
            )
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")

        aspect = self._aspect(geometry.shape_mod)
        lx = np.sqrt(aspect)
        ly = 1.0 / np.sqrt(aspect)

        order = max(4, int(np.ceil(np.sqrt(n_modes))) + 3)
        while order * order < n_modes:
            order += 1
        candidates: list[tuple[float, int, int]] = []
        for m in range(1, order + 1):
            for n in range(1, order + 1):
                factor = np.pi**2 * (m * m / (lx * lx) + n * n / (ly * ly))
                candidates.append((float(factor), m, n))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        selected = candidates[:n_modes]
        factors = np.asarray([item[0] for item in selected], dtype=np.float64)

        grid = make_material_grid(geometry, self.mode_grid_size)
        x = grid.physical_x
        y = grid.physical_y
        xi = x / lx + 0.5
        eta = y / ly + 0.5
        shapes = np.zeros((n_modes, self.mode_grid_size, self.mode_grid_size), dtype=np.float64)
        for k, (_, m, n) in enumerate(selected):
            values = np.sin(m * np.pi * xi) * np.sin(n * np.pi * eta)
            values = np.where(grid.mask, values, 0.0)
            norm = np.sqrt(np.sum(grid.integration_weights * values * values))
            if norm > 1e-12:
                values /= norm
            shapes[k] = values

        return ModalBasis(
            modal_factors=np.ascontiguousarray(factors),
            mode_shapes=np.ascontiguousarray(shapes),
            integration_weights=np.ascontiguousarray(grid.integration_weights),
            geometry=geometry,
        )


def resolve_modal_frequencies_hz(basis: ModalBasis, params: SynthParameters) -> np.ndarray:
    if min(params.size_m, params.flexural_rigidity, params.density, params.thickness_m, params.frequency_scale) <= 0.0:
        raise ValueError("size/material/frequency scale must be positive")
    omega = (
        np.asarray(basis.modal_factors, dtype=np.float64)
        * np.sqrt(params.flexural_rigidity / (params.density * params.thickness_m))
        / (params.size_m * params.size_m)
    )
    return np.ascontiguousarray(omega * float(params.frequency_scale) / (2.0 * np.pi), dtype=np.float64)
