from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import GeometryDescription
from ..material_coordinates import make_material_grid
from .plate_solver import PlateEigenSolution


@dataclass(frozen=True)
class SampledModes:
    """FEM modes sampled onto the canonical Phase-2 material grid."""

    modal_factors: np.ndarray
    eigenvalues: np.ndarray
    mode_shapes: np.ndarray
    area_weights: np.ndarray
    inner_product_weights: np.ndarray
    degenerate_group_id: np.ndarray
    residuals: np.ndarray

    @property
    def n_modes(self) -> int:
        return int(self.modal_factors.shape[0])


def canonicalize_mode_sign(mode_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Choose a deterministic representative of the phi <-> -phi ambiguity."""
    values = np.asarray(mode_map, dtype=np.float64).copy()
    valid = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    if valid.size == 0:
        raise ValueError("material mask is empty")
    flat = values.ravel()
    idx = valid[int(np.argmax(np.abs(flat[valid])))]
    if flat[idx] < 0.0:
        values *= -1.0
    return values


def detect_degenerate_groups(
    eigenvalues: np.ndarray,
    relative_gap: float = 5e-4,
) -> np.ndarray:
    """Group consecutive numerically near-degenerate eigenvalues.

    Group identifiers are stable integers.  A singleton still receives its own
    group id; Phase 4 can inspect group sizes to choose vector or subspace loss.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    if lam.ndim != 1 or lam.size == 0:
        raise ValueError("eigenvalues must be a non-empty 1D array")
    if relative_gap < 0.0:
        raise ValueError("relative_gap must be non-negative")

    groups = np.zeros(lam.size, dtype=np.int16)
    group = 0
    for i in range(1, lam.size):
        gap = abs(lam[i] - lam[i - 1]) / max(abs(lam[i]), abs(lam[i - 1]), 1e-30)
        if gap > relative_gap:
            group += 1
        groups[i] = group
    return groups


def sample_modes_on_material_grid(
    solution: PlateEigenSolution,
    geometry: GeometryDescription,
    *,
    n_modes: int = 32,
    grid_size: int = 64,
    degeneracy_relative_gap: float = 5e-4,
    boundary_margin: float = 1e-9,
) -> SampledModes:
    """Evaluate FEM displacement modes on the fixed canonical material grid."""
    if n_modes < 1 or n_modes > solution.n_solved:
        raise ValueError("n_modes must be within the solved eigenpair count")

    # n_solve > n_modes is intentional: the extra eigenpairs let us verify
    # that the fixed stored cutoff does not split a repeated eigenspace.
    if solution.n_solved > n_modes:
        lam_lo = float(solution.eigenvalues[n_modes - 1])
        lam_hi = float(solution.eigenvalues[n_modes])
        cutoff_gap = abs(lam_hi - lam_lo) / max(abs(lam_hi), abs(lam_lo), 1e-30)
        if cutoff_gap <= degeneracy_relative_gap:
            raise RuntimeError(
                "stored mode cutoff splits a near-degenerate eigenspace; "
                "change n_modes or the dataset convention before training"
            )

    grid = make_material_grid(geometry, grid_size)
    rho = np.asarray(grid.rho, dtype=np.float64)
    # Points exactly on the piecewise-linear Gmsh boundary can be classified
    # outside by tiny inverse-map roundoff.  Simply-supported displacement is
    # zero at rho=1, so evaluate only strict interior points and set the edge 0.
    sample_mask = np.asarray(grid.mask, dtype=bool) & (rho < 1.0 - boundary_margin)
    points = np.column_stack(
        (grid.physical_x[sample_mask], grid.physical_y[sample_mask])
    )

    shapes = np.zeros((n_modes, grid_size, grid_size), dtype=np.float64)
    for k in range(n_modes):
        interpolator = solution.basis.interpolator(solution.eigenvectors[:, k])
        sampled = np.asarray(interpolator(points.T), dtype=np.float64).reshape(-1)
        mode = np.zeros((grid_size, grid_size), dtype=np.float64)
        mode[sample_mask] = sampled

        norm2 = float(np.sum(grid.inner_product_weights * mode * mode))
        if not np.isfinite(norm2) or norm2 <= 1e-20:
            raise RuntimeError(f"sampled mode {k} has zero/invalid material-grid norm")
        mode /= np.sqrt(norm2)
        mode = canonicalize_mode_sign(mode, grid.mask)
        shapes[k] = mode

    lam = np.asarray(solution.eigenvalues[:n_modes], dtype=np.float64)
    return SampledModes(
        modal_factors=np.ascontiguousarray(solution.modal_factors[:n_modes]),
        eigenvalues=np.ascontiguousarray(lam),
        mode_shapes=np.ascontiguousarray(shapes, dtype=np.float64),
        area_weights=np.ascontiguousarray(grid.area_weights, dtype=np.float64),
        inner_product_weights=np.ascontiguousarray(
            grid.inner_product_weights, dtype=np.float64
        ),
        degenerate_group_id=np.ascontiguousarray(
            detect_degenerate_groups(lam, degeneracy_relative_gap), dtype=np.int16
        ),
        residuals=np.ascontiguousarray(solution.residuals[:n_modes]),
    )
