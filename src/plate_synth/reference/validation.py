from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import ReferenceMesh
from .mode_sampling import SampledModes
from .plate_solver import PlateEigenSolution


@dataclass(frozen=True)
class SampleQualityReport:
    accepted: bool
    reasons: tuple[str, ...]
    max_residual: float
    mass_orthogonality_error: float
    max_grid_norm_error: float
    modal_factor_relation_error: float
    area_weight_error: float
    inner_product_weight_error: float
    raw_grid_area_relative_error: float
    mesh_area_relative_error: float
    boundary_hausdorff_approx: float


@dataclass(frozen=True)
class ReferenceModeMatch:
    """One-to-one FEM -> analytical/reference mode assignment.

    ``fem_for_reference[j]`` is the FEM mode index assigned to reference mode
    ``j``.  The assignment is frequency-led, with weighted MAC used only as a
    small shape tie-breaker so close numerical crossings do not make validation
    depend on raw eigensolver ordering.
    """

    fem_for_reference: np.ndarray
    pair_relative_error: np.ndarray
    mac_matrix: np.ndarray


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
    if a.shape[0] != np.asarray(weights).size:
        raise ValueError("subspaces and weights must have matching point count")

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    sqrt_w = np.sqrt(np.maximum(w, 0.0))[:, None]
    qa, _ = np.linalg.qr(sqrt_w * a)
    qb, _ = np.linalg.qr(sqrt_w * b)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0) ** 2))


def match_modes_to_reference(
    fem_factors: np.ndarray,
    fem_modes: np.ndarray,
    reference_factors: np.ndarray,
    reference_modes: np.ndarray,
    weights: np.ndarray,
    *,
    shape_tiebreak_weight: float = 1e-3,
) -> ReferenceModeMatch:
    """Match FEM modes to a reference basis without trusting raw mode order.

    The Hungarian assignment minimizes

        relative_frequency_error + eps * (1 - MAC)

    where ``eps`` is intentionally small.  Frequency therefore defines modal
    identity for the rectangle validation while MAC only resolves close/tied
    choices.  Repeated reference eigenspaces are evaluated later as complete
    subspaces; the within-group assignment is physically irrelevant.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - phase-3 dependency
        raise RuntimeError("mode matching requires scipy") from exc

    fem_f = np.asarray(fem_factors, dtype=np.float64)
    ref_f = np.asarray(reference_factors, dtype=np.float64)
    fem_shapes = np.asarray(fem_modes, dtype=np.float64)
    ref_shapes = np.asarray(reference_modes, dtype=np.float64)

    if fem_f.ndim != 1 or ref_f.ndim != 1:
        raise ValueError("modal factors must be 1D")
    if fem_shapes.ndim != 3 or ref_shapes.ndim != 3:
        raise ValueError("mode maps must have shape (N,H,W)")
    if fem_shapes.shape[0] != fem_f.size or ref_shapes.shape[0] != ref_f.size:
        raise ValueError("factor and mode-map counts must match")
    if fem_f.size < ref_f.size:
        raise ValueError("need at least as many FEM modes as reference modes")
    if np.any(ref_f <= 0.0) or not np.all(np.isfinite(ref_f)):
        raise ValueError("reference factors must be positive and finite")
    if np.any(fem_f <= 0.0) or not np.all(np.isfinite(fem_f)):
        raise ValueError("FEM factors must be positive and finite")
    if shape_tiebreak_weight < 0.0:
        raise ValueError("shape_tiebreak_weight must be non-negative")

    mac = modal_assurance_matrix(fem_shapes, ref_shapes, weights)
    relative = np.abs(fem_f[:, None] / ref_f[None, :] - 1.0)
    cost = relative + float(shape_tiebreak_weight) * (1.0 - mac)
    rows, cols = linear_sum_assignment(cost)

    assignment = np.full(ref_f.size, -1, dtype=np.int64)
    assignment[cols] = rows
    if np.any(assignment < 0):
        raise RuntimeError("Hungarian assignment did not cover every reference mode")

    pair_error = relative[assignment, np.arange(ref_f.size)]
    return ReferenceModeMatch(
        fem_for_reference=np.ascontiguousarray(assignment),
        pair_relative_error=np.ascontiguousarray(pair_error),
        mac_matrix=np.ascontiguousarray(mac),
    )


def reorder_match_within_reference_groups(
    fem_factors: np.ndarray,
    fem_for_reference: np.ndarray,
    reference_group_ids: np.ndarray,
) -> np.ndarray:
    """Make factor ordering deterministic inside repeated/near-repeated groups.

    Subspace scores do not care which vector represents which member of a
    repeated eigenspace, but mesh-convergence tables compare scalar modal
    factors.  Sorting the selected FEM factors inside each reference group
    prevents arbitrary eigensolver basis/order changes from creating fake
    convergence jumps.
    """
    factors = np.asarray(fem_factors, dtype=np.float64)
    assignment = np.asarray(fem_for_reference, dtype=np.int64).copy()
    groups = np.asarray(reference_group_ids)
    if assignment.ndim != 1 or groups.shape != assignment.shape:
        raise ValueError("assignment and reference_group_ids must be matching 1D arrays")
    if np.any(assignment < 0) or np.any(assignment >= factors.size):
        raise ValueError("assignment contains invalid FEM indices")

    for group_id in np.unique(groups):
        ref_indices = np.flatnonzero(groups == group_id)
        selected = assignment[ref_indices]
        order = np.argsort(factors[selected], kind="stable")
        assignment[ref_indices] = selected[order]
    return np.ascontiguousarray(assignment)


def validate_generated_sample(
    solution: PlateEigenSolution,
    sampled: SampledModes,
    mesh: ReferenceMesh,
    *,
    max_residual: float = 1e-7,
    max_mass_orthogonality_error: float = 1e-7,
    max_grid_norm_error: float = 5e-5,
    max_modal_factor_relation_error: float = 1e-10,
    max_area_weight_error: float = 1e-8,
    max_inner_product_weight_error: float = 1e-8,
    max_raw_grid_area_relative_error: float = 0.03,
    max_mesh_area_relative_error: float = 5e-3,
    max_boundary_hausdorff: float = 1e-2,
) -> SampleQualityReport:
    """Apply hard QA gates before a generated sample enters the dataset."""
    reasons: list[str] = []

    lam = np.asarray(sampled.eigenvalues, dtype=np.float64)
    mu = np.asarray(sampled.modal_factors, dtype=np.float64)
    shapes = np.asarray(sampled.mode_shapes, dtype=np.float64)
    area_weights = np.asarray(sampled.area_weights, dtype=np.float64)
    inner_weights = np.asarray(sampled.inner_product_weights, dtype=np.float64)

    if lam.ndim != 1 or mu.shape != lam.shape:
        reasons.append("modal factor/eigenvalue arrays have inconsistent shape")
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        reasons.append("non-positive or non-finite eigenvalue")
    if not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
        reasons.append("non-positive or non-finite modal factor")
    if np.any(np.diff(lam) < -1e-10):
        reasons.append("eigenvalues are not sorted")
    if not np.all(np.isfinite(shapes)):
        reasons.append("mode shapes contain NaN/Inf")

    factor_relation = float(
        np.max(np.abs(mu * mu - lam) / np.maximum(np.abs(lam), 1e-30))
    )
    if factor_relation > max_modal_factor_relation_error:
        reasons.append(
            f"modal_factor^2/eigenvalue error {factor_relation:.3e} exceeds "
            f"{max_modal_factor_relation_error:.3e}"
        )

    residual = float(np.max(sampled.residuals))
    if not np.isfinite(residual) or residual > max_residual:
        reasons.append(f"eigensolver residual {residual:.3e} exceeds {max_residual:.3e}")

    orth_error = float(solution.mass_orthogonality_error)
    if not np.isfinite(orth_error) or orth_error > max_mass_orthogonality_error:
        reasons.append(
            f"FEM mass orthogonality error {orth_error:.3e} exceeds "
            f"{max_mass_orthogonality_error:.3e}"
        )

    if not np.all(np.isfinite(area_weights)) or np.any(area_weights < 0.0):
        reasons.append("area weights contain negative or non-finite values")
    if not np.all(np.isfinite(inner_weights)) or np.any(inner_weights < 0.0):
        reasons.append("inner-product weights contain negative or non-finite values")

    target_area = float(mesh.target_area)
    area_weight_error = abs(float(np.sum(area_weights)) - target_area) / max(
        abs(target_area), 1e-30
    )
    if area_weight_error > max_area_weight_error:
        reasons.append(
            f"area-weight integral error {area_weight_error:.3e} exceeds "
            f"{max_area_weight_error:.3e}"
        )

    inner_weight_error = abs(float(np.sum(inner_weights)) - 1.0)
    if inner_weight_error > max_inner_product_weight_error:
        reasons.append(
            f"inner-product weight sum error {inner_weight_error:.3e} exceeds "
            f"{max_inner_product_weight_error:.3e}"
        )

    norms = np.sum(inner_weights[None, :, :] * shapes**2, axis=(1, 2))
    grid_norm_error = float(np.max(np.abs(norms - 1.0)))
    if grid_norm_error > max_grid_norm_error:
        reasons.append(
            f"material-grid norm error {grid_norm_error:.3e} exceeds "
            f"{max_grid_norm_error:.3e}"
        )

    raw_grid_area_error = float(sampled.raw_grid_area_relative_error)
    if not np.isfinite(raw_grid_area_error) or raw_grid_area_error > max_raw_grid_area_relative_error:
        reasons.append(
            f"raw material-grid area error {raw_grid_area_error:.3e} exceeds "
            f"{max_raw_grid_area_relative_error:.3e}"
        )

    mesh_area_error = float(mesh.relative_area_error)
    if not np.isfinite(mesh_area_error) or mesh_area_error > max_mesh_area_relative_error:
        reasons.append(
            f"mesh/target area error {mesh_area_error:.3e} exceeds "
            f"{max_mesh_area_relative_error:.3e}"
        )

    hausdorff = float(mesh.boundary_hausdorff_approx)
    if not np.isfinite(hausdorff) or hausdorff > max_boundary_hausdorff:
        reasons.append(
            f"mesh/target boundary distance {hausdorff:.3e} exceeds "
            f"{max_boundary_hausdorff:.3e}"
        )

    return SampleQualityReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        max_residual=residual,
        mass_orthogonality_error=orth_error,
        max_grid_norm_error=grid_norm_error,
        modal_factor_relation_error=factor_relation,
        area_weight_error=float(area_weight_error),
        inner_product_weight_error=float(inner_weight_error),
        raw_grid_area_relative_error=raw_grid_area_error,
        mesh_area_relative_error=mesh_area_error,
        boundary_hausdorff_approx=hausdorff,
    )
