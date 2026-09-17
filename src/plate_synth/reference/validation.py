from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mode_sampling import SampledModes
from .plate_solver import PlateEigenSolution


@dataclass(frozen=True)
class SampleQualityReport:
    accepted: bool
    reasons: tuple[str, ...]
    max_residual: float
    mass_orthogonality_error: float
    max_grid_norm_error: float


def modal_assurance_matrix(
    modes_a: np.ndarray,
    modes_b: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted MAC between two sets of material-grid mode shapes."""
    a = np.asarray(modes_a, dtype=np.float64).reshape(modes_a.shape[0], -1)
    b = np.asarray(modes_b, dtype=np.float64).reshape(modes_b.shape[0], -1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if a.shape[1] != b.shape[1] or a.shape[1] != w.size:
        raise ValueError("mode grids and weights must have matching point count")

    aw = a * w[None, :]
    bw = b * w[None, :]
    cross = aw @ b.T
    na = np.sum(aw * a, axis=1)
    nb = np.sum(bw * b, axis=1)
    denom = np.maximum(na[:, None] * nb[None, :], 1e-30)
    return np.abs(cross) ** 2 / denom


def subspace_projection_score(
    modes_a: np.ndarray,
    modes_b: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Basis-invariant overlap score for two equal-dimensional modal subspaces."""
    a = np.asarray(modes_a, dtype=np.float64).reshape(modes_a.shape[0], -1).T
    b = np.asarray(modes_b, dtype=np.float64).reshape(modes_b.shape[0], -1).T
    if a.shape[1] != b.shape[1]:
        raise ValueError("subspaces must have equal dimension")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    sqrt_w = np.sqrt(np.maximum(w, 0.0))[:, None]
    qa, _ = np.linalg.qr(sqrt_w * a)
    qb, _ = np.linalg.qr(sqrt_w * b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0) ** 2))


def validate_generated_sample(
    solution: PlateEigenSolution,
    sampled: SampledModes,
    *,
    max_residual: float = 1e-7,
    max_mass_orthogonality_error: float = 1e-7,
    max_grid_norm_error: float = 5e-5,
) -> SampleQualityReport:
    """Apply hard QA gates before a generated sample enters the dataset."""
    reasons: list[str] = []
    lam = np.asarray(sampled.eigenvalues)
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        reasons.append("non-positive or non-finite eigenvalue")
    if np.any(np.diff(lam) < -1e-10):
        reasons.append("eigenvalues are not sorted")
    if not np.all(np.isfinite(sampled.mode_shapes)):
        reasons.append("mode shapes contain NaN/Inf")

    residual = float(np.max(sampled.residuals))
    if residual > max_residual:
        reasons.append(f"eigensolver residual {residual:.3e} exceeds {max_residual:.3e}")

    orth_error = float(solution.mass_orthogonality_error)
    if orth_error > max_mass_orthogonality_error:
        reasons.append(
            f"FEM mass orthogonality error {orth_error:.3e} exceeds "
            f"{max_mass_orthogonality_error:.3e}"
        )

    norms = np.sum(
        sampled.inner_product_weights[None, :, :] * sampled.mode_shapes**2,
        axis=(1, 2),
    )
    grid_norm_error = float(np.max(np.abs(norms - 1.0)))
    if grid_norm_error > max_grid_norm_error:
        reasons.append(
            f"material-grid norm error {grid_norm_error:.3e} exceeds "
            f"{max_grid_norm_error:.3e}"
        )

    return SampleQualityReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        max_residual=residual,
        mass_orthogonality_error=orth_error,
        max_grid_norm_error=grid_norm_error,
    )
