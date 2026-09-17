from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from .config import SynthParameters, SUPPORTED_BOUNDARY_CONDITIONS
from .geometry import GeometryDescription
from .material_coordinates import make_material_grid
from .spatial import bilinear_sample_mode_maps


class UnsupportedGeometryError(RuntimeError):
    pass


class UnsupportedBoundaryConditionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModalBasis:
    """Geometry-dependent, material-independent modal basis.

    `modal_factors` follow
        omega_k = modal_factor_k * sqrt(D/(rho*H)) / size_m^2
    before the user frequency-scale control.

    `area_weights` are geometric quadrature weights on the unit-area
    normalized plate. `inner_product_weights` are normalized to sum to one and
    are used for mode normalization / projection. Neither includes rho*H.
    """

    modal_factors: np.ndarray
    mode_shapes: np.ndarray
    area_weights: np.ndarray
    inner_product_weights: np.ndarray
    geometry: GeometryDescription
    boundary_condition: str

    @property
    def n_modes(self) -> int:
        return int(self.modal_factors.shape[0])

    def sample_weights(self, u: float, v: float) -> np.ndarray:
        return bilinear_sample_mode_maps(self.mode_shapes, u, v)


class ModalBackend(Protocol):
    def predict(
        self,
        geometry: GeometryDescription,
        n_modes: int,
        boundary_condition: str,
    ) -> ModalBasis:
        ...


class AnalyticRectangleBackend:
    """Simply-supported analytical Kirchhoff rectangle backend.

    This is the exact analytical endpoint model used before the neural backend
    exists. Candidate enumeration is expanded adaptively until the requested
    modes are provably the globally lowest N positive (m,n) rectangle modes.
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
        return float(
            np.exp(np.log(max_stretch) * np.clip(shape_mod, 0.0, 1.0))
        )

    @staticmethod
    def _factor(m: int, n: int, lx: float, ly: float) -> float:
        return float(
            np.pi**2 * (m * m / (lx * lx) + n * n / (ly * ly))
        )

    @classmethod
    def _lowest_mode_candidates(
        cls,
        n_modes: int,
        lx: float,
        ly: float,
    ) -> list[tuple[float, int, int]]:
        """Return the globally lowest N rectangle modes.

        Because the factor is monotone in both positive indices, once the
        current N-th factor lies below the minimum possible omitted mode
        f(m_max+1, 1) / f(1, n_max+1), no unenumerated mode can enter the set.
        """
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")

        m_max = max(2, int(np.ceil(np.sqrt(n_modes))))
        n_max = m_max

        for _ in range(32):
            candidates = [
                (cls._factor(m, n, lx, ly), m, n)
                for m in range(1, m_max + 1)
                for n in range(1, n_max + 1)
            ]
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))

            if len(candidates) < n_modes:
                m_max *= 2
                n_max *= 2
                continue

            threshold = candidates[n_modes - 1][0]
            next_m = cls._factor(m_max + 1, 1, lx, ly)
            next_n = cls._factor(1, n_max + 1, lx, ly)

            if threshold <= min(next_m, next_n):
                return candidates[:n_modes]

            if next_m < threshold:
                m_max *= 2
            if next_n < threshold:
                n_max *= 2

        raise RuntimeError("failed to bound rectangle modal candidate search")

    def predict(
        self,
        geometry: GeometryDescription,
        n_modes: int,
        boundary_condition: str,
    ) -> ModalBasis:
        if not self._is_rectangle_endpoint(geometry):
            raise UnsupportedGeometryError(
                "No trained neural modal backend is available for "
                "morphed/circle/triangle geometries yet."
            )
        if boundary_condition not in SUPPORTED_BOUNDARY_CONDITIONS:
            raise UnsupportedBoundaryConditionError(
                f"unsupported boundary condition: {boundary_condition}"
            )
        if boundary_condition != "simply_supported":
            raise UnsupportedBoundaryConditionError(
                "AnalyticRectangleBackend currently supports simply_supported only"
            )
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")

        aspect = self._aspect(geometry.shape_mod)
        lx = np.sqrt(aspect)
        ly = 1.0 / np.sqrt(aspect)
        selected = self._lowest_mode_candidates(n_modes, lx, ly)
        factors = np.asarray([item[0] for item in selected], dtype=np.float64)

        grid = make_material_grid(geometry, self.mode_grid_size)
        x = grid.physical_x
        y = grid.physical_y
        xi = x / lx + 0.5
        eta = y / ly + 0.5

        shapes = np.zeros(
            (n_modes, self.mode_grid_size, self.mode_grid_size),
            dtype=np.float64,
        )
        for k, (_, m, n) in enumerate(selected):
            values = np.sin(m * np.pi * xi) * np.sin(n * np.pi * eta)
            values = np.where(grid.mask, values, 0.0)
            norm = np.sqrt(
                np.sum(grid.inner_product_weights * values * values)
            )
            if norm > 1e-12:
                values /= norm
            shapes[k] = values

        return ModalBasis(
            modal_factors=np.ascontiguousarray(factors),
            mode_shapes=np.ascontiguousarray(shapes),
            area_weights=np.ascontiguousarray(grid.area_weights),
            inner_product_weights=np.ascontiguousarray(
                grid.inner_product_weights
            ),
            geometry=geometry,
            boundary_condition=boundary_condition,
        )


def resolve_modal_frequencies_hz(
    basis: ModalBasis,
    params: SynthParameters,
) -> np.ndarray:
    if basis.boundary_condition != params.boundary_condition:
        raise ValueError("modal basis boundary condition does not match parameters")
    if min(
        params.size_m,
        params.flexural_rigidity,
        params.density,
        params.thickness_m,
        params.frequency_scale,
    ) <= 0.0:
        raise ValueError("size/material/frequency scale must be positive")

    omega = (
        np.asarray(basis.modal_factors, dtype=np.float64)
        * np.sqrt(
            params.flexural_rigidity
            / (params.density * params.thickness_m)
        )
        / (params.size_m * params.size_m)
    )
    return np.ascontiguousarray(
        omega * float(params.frequency_scale) / (2.0 * np.pi),
        dtype=np.float64,
    )


def physical_mass_weights(
    basis: ModalBasis,
    params: SynthParameters,
) -> np.ndarray:
    """Return physical lumped mass weights rho * H * dA.

    The normalized geometry has area one and is scaled uniformly by size_m,
    therefore physical dA = size_m^2 * dA_normalized.
    """
    scale = params.density * params.thickness_m * params.size_m**2
    return np.ascontiguousarray(
        np.asarray(basis.area_weights, dtype=np.float64) * scale,
        dtype=np.float64,
    )
