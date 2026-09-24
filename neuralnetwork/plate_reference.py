import numpy as np
import triangle as tr

from skimage.measure import (
    find_contours,
    approximate_polygon,
)

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

from shapes import (
    morph_sdf,
)


@BilinearForm
def mass(u, v, w):
    return u * v


def mesh_area(mesh):
    triangles = mesh.p[:, mesh.t]

    x0 = triangles[0, 0]
    y0 = triangles[1, 0]

    x1 = triangles[0, 1]
    y1 = triangles[1, 1]

    x2 = triangles[0, 2]
    y2 = triangles[1, 2]

    triangle_areas = 0.5 * np.abs(
        (x1 - x0) * (y2 - y0)
        - (x2 - x0) * (y1 - y0)
    )

    return np.sum(
        triangle_areas
    )


def solve_plate(
    mesh,
    n_modes=32,
    poisson=0.3,
):
    basis = Basis(
        mesh,
        ElementTriMorley(),
    )

    @BilinearForm
    def stiffness(u, v, w):
        hessian_u = dd(u)
        hessian_v = dd(v)

        return (
            (1.0 - poisson)
            * ddot(
                hessian_u,
                hessian_v,
            )
            + poisson
            * trace(hessian_u)
            * trace(hessian_v)
        )

    K = asm(
        stiffness,
        basis,
    )

    M = asm(
        mass,
        basis,
    )

    boundary_displacement_dofs = (
        basis
        .get_dofs()
        .nodal["u"]
    )

    all_dofs = np.arange(
        basis.N
    )

    free_dofs = np.setdiff1d(
        all_dofs,
        boundary_displacement_dofs,
    )

    K_free = K[
        free_dofs
    ][:, free_dofs]

    M_free = M[
        free_dofs
    ][:, free_dofs]

    beta, eigenvectors = eigsh(
        K_free,
        k=n_modes,
        M=M_free,
        sigma=0.0,
        which="LM",
    )

    order = np.argsort(
        beta
    )

    beta = beta[order]

    eigenvectors = (
        eigenvectors[:, order]
    )

    area = mesh_area(
        mesh
    )

    modal_factors = (
        np.sqrt(beta)
        * area
    )

    return (
        modal_factors,
        beta,
        eigenvectors,
        basis,
        free_dofs,
    )


def mesh_from_mask(
    mask,
    threshold=0.5,
):
    ny, nx = mask.shape

    if nx != ny:
        raise ValueError(
            "Mask must currently be square."
        )

    resolution = nx

    coords = np.linspace(
        -1.0,
        1.0,
        resolution + 1,
    )

    mesh = MeshTri.init_tensor(
        coords,
        coords,
    )

    centers = (
        mesh.p[:, mesh.t]
        .mean(axis=1)
    )

    x = centers[0]
    y = centers[1]

    ix = (
        (x + 1.0)
        * 0.5
        * resolution
    ).astype(int)

    iy = (
        (y + 1.0)
        * 0.5
        * resolution
    ).astype(int)

    ix = np.clip(
        ix,
        0,
        resolution - 1,
    )

    iy = np.clip(
        iy,
        0,
        resolution - 1,
    )

    inside = (
        mask[iy, ix]
        >= threshold
    )

    remove = np.flatnonzero(
        ~inside
    )

    mesh = mesh.remove_elements(
        remove
    )

    return mesh


def mesh_from_sdf(
    sdf,
    max_area=0.00025,
    simplify=0.002,
):
    ny, nx = sdf.shape

    contours = find_contours(
        sdf,
        level=0.0,
    )

    if len(contours) == 0:
        raise ValueError(
            "No contour found in SDF."
        )

    contour = max(
        contours,
        key=len,
    )

    rows = contour[:, 0]
    cols = contour[:, 1]

    # Convert pixel coordinates to
    # physical [-1, 1] coordinates FIRST.
    x = (
        -1.0
        + (cols + 0.5)
        * 2.0
        / nx
    )

    y = (
        -1.0
        + (rows + 0.5)
        * 2.0
        / ny
    )

    vertices = np.column_stack(
        (
            x,
            y,
        )
    )

    # simplify is now measured in the actual
    # [-1, 1] geometry coordinates and therefore
    # independent of SDF resolution.
    vertices = approximate_polygon(
        vertices,
        tolerance=simplify,
    )

    if np.allclose(
        vertices[0],
        vertices[-1],
    ):
        vertices = vertices[:-1]

    n_vertices = len(
        vertices
    )

    if n_vertices < 3:
        raise ValueError(
            "Contour has fewer than 3 vertices."
        )

    indices = np.arange(
        n_vertices
    )

    segments = np.column_stack(
        (
            indices,
            np.roll(
                indices,
                -1,
            ),
        )
    )

    geometry = {
        "vertices": vertices,
        "segments": segments,
    }

    options = (
        "pq28"
        f"a{max_area}"
    )

    triangulation = tr.triangulate(
        geometry,
        options,
    )

    if "triangles" not in triangulation:
        raise RuntimeError(
            "Triangle failed to create mesh."
        )

    mesh = MeshTri(
        triangulation[
            "vertices"
        ].T,
        triangulation[
            "triangles"
        ].T,
    )

    return mesh

def mesh_from_contour(
    vertices,
    max_area=0.00025,
):
    vertices = np.asarray(
        vertices,
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

    n_vertices = len(
        vertices
    )

    if n_vertices < 3:
        raise ValueError(
            "Contour has fewer than 3 vertices."
        )

    indices = np.arange(
        n_vertices
    )

    segments = np.column_stack(
        (
            indices,
            np.roll(
                indices,
                -1,
            ),
        )
    )

    geometry = {
        "vertices": vertices,
        "segments": segments,
    }

    options = (
        "pq28"
        f"a{max_area}"
    )

    triangulation = tr.triangulate(
        geometry,
        options,
    )

    if "triangles" not in triangulation:
        raise RuntimeError(
            "Triangle failed to create mesh."
        )

    mesh = MeshTri(
        np.ascontiguousarray(
            triangulation[
                "vertices"
            ].T
        ),
        np.ascontiguousarray(
            triangulation[
                "triangles"
            ].T
        ),
    )

    return mesh


if __name__ == "__main__":
    N_MODES = 32

    SDF_RESOLUTION = 512
    MAX_AREA = 0.00025
    SIMPLIFY = 0.002

    aspects = [
        0.7,
        1.0,
        1.4,
    ]

    for aspect in aspects:
        sdf = morph_sdf(
            morph=0.0,
            aspect=aspect,
            resolution=SDF_RESOLUTION,
        )

        mesh = mesh_from_sdf(
            sdf,
            max_area=MAX_AREA,
            simplify=SIMPLIFY,
        )

        (
            modal_factors,
            beta,
            eigenvectors,
            basis,
            free_dofs,
        ) = solve_plate(
            mesh,
            n_modes=N_MODES,
            poisson=0.3,
        )

        candidate_factors = []

        for m in range(1, 16):
            for n in range(1, 16):
                mu = np.pi**2 * (
                    m**2 / aspect
                    + aspect * n**2
                )

                candidate_factors.append(
                    mu
                )

        analytic = np.sort(
            np.asarray(
                candidate_factors
            )
        )[:N_MODES]

        relative_error = (
            np.abs(
                modal_factors
                - analytic
            )
            / analytic
        )

        print(
            f"\nAspect = {aspect}"
        )

        print(
            f"vertices={mesh.p.shape[1]} | "
            f"triangles={mesh.t.shape[1]} | "
            f"area={mesh_area(mesh):.6f}"
        )

        print(
            f"mean error = "
            f"{relative_error.mean() * 100.0:.4f}%"
        )

        print(
            f"max error = "
            f"{relative_error.max() * 100.0:.4f}%"
        )

        print(
            f"mode 1 error = "
            f"{relative_error[0] * 100.0:.4f}%"
        )

        print(
            "\nNumerical:"
        )

        print(
            modal_factors
        )

        print(
            "\nAnalytical:"
        )

        print(
            analytic
        )