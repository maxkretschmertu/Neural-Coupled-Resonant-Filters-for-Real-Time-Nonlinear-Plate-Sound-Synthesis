import numpy as np
import triangle as tr

from scipy.sparse.linalg import eigsh

from skfem import (
    MeshTri,
    Basis,
    ElementTriMorley,
    BilinearForm,
    asm,
)

from skfem.helpers import (
    dd,
    ddot,
    trace,
)


# =========================================================
# FEM forms
# =========================================================

@BilinearForm
def mass(u, v, w):
    return u * v


# =========================================================
# Mesh
# =========================================================

def mesh_area(mesh):
    """Total area of a triangular mesh."""
    triangles = mesh.p[:, mesh.t]

    x0, y0 = triangles[:, 0]
    x1, y1 = triangles[:, 1]
    x2, y2 = triangles[:, 2]

    areas = 0.5 * np.abs(
        (x1 - x0) * (y2 - y0)
        - (x2 - x0) * (y1 - y0)
    )

    return areas.sum()


def mesh_from_contour(
    contour,
    max_area=0.00025,
):
    """Triangulate a closed 2-D contour."""
    vertices = np.asarray(
        contour,
        dtype=np.float64,
    )

    if (
        len(vertices) > 1
        and np.allclose(
            vertices[0],
            vertices[-1],
        )
    ):
        vertices = vertices[:-1]

    if len(vertices) < 3:
        raise ValueError(
            "Contour needs at least three vertices."
        )

    indices = np.arange(
        len(vertices)
    )

    segments = np.column_stack(
        (
            indices,
            np.roll(indices, -1),
        )
    )

    result = tr.triangulate(
        {
            "vertices": vertices,
            "segments": segments,
        },
        f"pq28a{max_area}",
    )

    if "triangles" not in result:
        raise RuntimeError(
            "Triangle failed to create a mesh."
        )

    return MeshTri(
        np.ascontiguousarray(
            result["vertices"].T
        ),
        np.ascontiguousarray(
            result["triangles"].T
        ),
    )


# =========================================================
# Plate eigenproblem
# =========================================================

def solve_plate(
    mesh,
    n_modes=32,
    poisson=0.3,
):
    """
    Solve the simply-supported Kirchhoff plate eigenproblem.

    Returns:
        modal_factors
        eigenvalues
        eigenvectors
        basis
        free_dofs
    """
    basis = Basis(
        mesh,
        ElementTriMorley(),
    )

    @BilinearForm
    def stiffness(u, v, w):
        Hu = dd(u)
        Hv = dd(v)

        return (
            (1.0 - poisson)
            * ddot(Hu, Hv)
            + poisson
            * trace(Hu)
            * trace(Hv)
        )

    K = asm(
        stiffness,
        basis,
    )

    M = asm(
        mass,
        basis,
    )

    # Simply-supported boundary:
    # displacement u = 0,
    # bending moment remains natural.
    fixed_dofs = (
        basis
        .get_dofs()
        .nodal["u"]
    )

    free_dofs = np.setdiff1d(
        np.arange(basis.N),
        fixed_dofs,
    )

    K_free = K[
        free_dofs
    ][:, free_dofs]

    M_free = M[
        free_dofs
    ][:, free_dofs]

    eigenvalues, eigenvectors = eigsh(
        K_free,
        k=n_modes,
        M=M_free,
        sigma=0.0,
        which="LM",
    )

    order = np.argsort(
        eigenvalues
    )

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    # Remove uniform geometry scale.
    modal_factors = (
        np.sqrt(eigenvalues)
        * mesh_area(mesh)
    )

    return (
        modal_factors,
        eigenvalues,
        eigenvectors,
        basis,
        free_dofs,
    )


# =========================================================
# Mode normalization
# =========================================================

def mass_normalize_modes(
    basis,
    eigenvectors,
    free_dofs,
):
    """Normalize each eigenvector to unit modal mass."""
    M = asm(
        mass,
        basis,
    )

    M_free = M[
        free_dofs
    ][:, free_dofs]

    modes = np.asarray(
        eigenvectors,
        dtype=np.float64,
    ).copy()

    modal_mass = np.sum(
        modes
        * (M_free @ modes),
        axis=0,
    )

    if np.any(modal_mass <= 0.0):
        raise RuntimeError(
            "Invalid modal mass."
        )

    modes /= np.sqrt(
        modal_mass
    )[None, :]

    return modes


# =========================================================
# Mode values at arbitrary points
# =========================================================

def evaluate_mode_gains(
    basis,
    eigenvectors,
    free_dofs,
    points,
):
    """
    Evaluate mass-normalized FEM modes at arbitrary XY points.

    points:
        (N, 2)

    returns:
        (N, n_modes)
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim == 1:
        points = points[None, :]

    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):
        raise ValueError(
            "points must have shape (N, 2)"
        )

    n_modes = eigenvectors.shape[1]

    # eigsh returns only coefficients for free DOFs.
    # Reconstruct complete Morley coefficient vectors.
    full_modes = np.zeros(
        (
            basis.N,
            n_modes,
        ),
        dtype=np.float64,
    )

    full_modes[
        free_dofs,
        :
    ] = eigenvectors

    probe = basis.probes(
        points.T
    )

    return np.asarray(
        probe @ full_modes
    )