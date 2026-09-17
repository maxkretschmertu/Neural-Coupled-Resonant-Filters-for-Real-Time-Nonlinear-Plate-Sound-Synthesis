from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import SUPPORTED_BOUNDARY_CONDITIONS
from .mesh import ReferenceMesh


@dataclass(frozen=True)
class PlateSolverConfig:
    """Dimensionless Kirchhoff--Love reference eigenproblem settings."""

    n_modes: int = 32
    n_solve: int = 40
    poisson_ratio: float = 0.30
    boundary_condition: str = "simply_supported"
    integration_order: int = 4
    eigensolver_tolerance: float = 1e-10
    eigensolver_maxiter: int = 30_000
    positive_eigenvalue_floor: float = 1e-10

    def validate(self) -> None:
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if self.n_solve < self.n_modes:
            raise ValueError("n_solve must be >= n_modes")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        if self.boundary_condition not in SUPPORTED_BOUNDARY_CONDITIONS:
            raise ValueError(f"unsupported boundary condition: {self.boundary_condition}")
        if self.boundary_condition != "simply_supported":
            raise ValueError("Phase 3 currently implements simply_supported only")
        if self.integration_order < 2:
            raise ValueError("integration_order must be >= 2")


@dataclass(frozen=True)
class PlateEigenSolution:
    """Mass-normalized eigenpairs on the Morley finite-element space."""

    eigenvalues: np.ndarray          # lambda, (Nsolve,)
    modal_factors: np.ndarray        # sqrt(lambda), (Nsolve,)
    eigenvectors: np.ndarray         # full Morley coefficients, (Ndof, Nsolve)
    residuals: np.ndarray            # generalized eigen residual on free DOFs
    mass_orthogonality_error: float
    constrained_dofs: np.ndarray
    reference_mesh: ReferenceMesh
    fem_mesh: Any
    basis: Any

    @property
    def n_solved(self) -> int:
        return int(self.eigenvalues.shape[0])


def solve_plate_modes(
    reference_mesh: ReferenceMesh,
    config: PlateSolverConfig | None = None,
) -> PlateEigenSolution:
    """Solve the dimensionless Kirchhoff--Love eigenproblem with Morley FEM.

    We assemble

        a(u,v) = int [(1-nu) H(u):H(v) + nu Delta(u) Delta(v)] dA
        m(u,v) = int u v dA

    after factoring out the physical flexural rigidity D.  For the simply
    supported contract only the displacement DOFs `u` on the boundary are
    essential; the bending-moment condition is natural in the weak form.

    The resulting eigenvalues satisfy lambda = modal_factor**2, matching the
    runtime convention

        omega = modal_factor * sqrt(D/(rho*H)) / size**2.
    """
    cfg = config or PlateSolverConfig()
    cfg.validate()

    try:
        from scipy.sparse.linalg import eigsh
        from skfem import Basis, BilinearForm, ElementTriMorley, MeshTri, asm
        from skfem.helpers import dd, ddot, trace
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Phase 3 solving requires scipy and scikit-fem. "
            "Install requirements-phase3.txt."
        ) from exc

    mesh = MeshTri(
        np.asarray(reference_mesh.points, dtype=np.float64).T,
        np.asarray(reference_mesh.triangles, dtype=np.int32).T,
    )
    basis = Basis(mesh, ElementTriMorley(), intorder=cfg.integration_order)
    nu = float(cfg.poisson_ratio)

    @BilinearForm
    def stiffness(u, v, _):
        hu = dd(u)
        hv = dd(v)
        return (1.0 - nu) * ddot(hu, hv) + nu * trace(hu) * trace(hv)

    @BilinearForm
    def mass(u, v, _):
        return u * v

    K = asm(stiffness, basis).tocsr()
    M = asm(mass, basis).tocsr()

    boundary_facets = mesh.boundary_facets()
    boundary_view = basis.get_dofs(facets=boundary_facets)
    if "u" not in boundary_view.nodal:
        raise RuntimeError("Morley basis does not expose the expected nodal 'u' DOF")
    constrained = np.unique(np.asarray(boundary_view.nodal["u"], dtype=np.int64))
    all_dofs = np.arange(basis.N, dtype=np.int64)
    free = np.setdiff1d(all_dofs, constrained, assume_unique=True)

    if free.size <= cfg.n_modes:
        raise RuntimeError(
            f"mesh is too small: {free.size} free DOFs for {cfg.n_modes} requested modes"
        )

    k = min(int(cfg.n_solve), int(free.size - 1))
    if k < cfg.n_modes:
        raise RuntimeError("mesh does not provide enough free DOFs for n_modes")

    Kf = K[free][:, free]
    Mf = M[free][:, free]

    # Shift-invert around zero is robust for the smallest positive generalized
    # eigenpairs of the positive simply-supported plate problem.
    eigenvalues, vectors_free = eigsh(
        Kf,
        k=k,
        M=Mf,
        sigma=0.0,
        which="LM",
        tol=float(cfg.eigensolver_tolerance),
        maxiter=int(cfg.eigensolver_maxiter),
    )

    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    vectors_free = np.asarray(vectors_free[:, order], dtype=np.float64)

    positive = eigenvalues > float(cfg.positive_eigenvalue_floor)
    eigenvalues = eigenvalues[positive]
    vectors_free = vectors_free[:, positive]
    if eigenvalues.size < cfg.n_modes:
        raise RuntimeError(
            f"only {eigenvalues.size} positive eigenpairs found; need {cfg.n_modes}"
        )

    # Mass-normalize in the reduced space.  This is the authoritative FEM
    # normalization; material-grid targets are normalized again after sampling.
    for j in range(eigenvalues.size):
        v = vectors_free[:, j]
        norm2 = float(v @ (Mf @ v))
        if not np.isfinite(norm2) or norm2 <= 0.0:
            raise RuntimeError("non-positive FEM modal mass")
        vectors_free[:, j] = v / np.sqrt(norm2)

    full_vectors = np.zeros((basis.N, eigenvalues.size), dtype=np.float64)
    full_vectors[free, :] = vectors_free

    residuals = np.empty(eigenvalues.size, dtype=np.float64)
    for j, lam in enumerate(eigenvalues):
        v = vectors_free[:, j]
        kv = Kf @ v
        mv = Mf @ v
        numerator = np.linalg.norm(kv - lam * mv)
        denominator = max(np.linalg.norm(kv), abs(lam) * np.linalg.norm(mv), 1e-30)
        residuals[j] = numerator / denominator

    gram = vectors_free.T @ (Mf @ vectors_free)
    orth_error = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))

    return PlateEigenSolution(
        eigenvalues=np.ascontiguousarray(eigenvalues),
        modal_factors=np.ascontiguousarray(np.sqrt(eigenvalues)),
        eigenvectors=np.ascontiguousarray(full_vectors),
        residuals=np.ascontiguousarray(residuals),
        mass_orthogonality_error=orth_error,
        constrained_dofs=np.ascontiguousarray(constrained, dtype=np.int64),
        reference_mesh=reference_mesh,
        fem_mesh=mesh,
        basis=basis,
    )
