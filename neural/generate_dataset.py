from pathlib import Path
import time

import numpy as np

from shapes import (
    make_morph_contour,
    contour_to_mask,
)

from plate_reference import (
    mesh_from_contour,
    solve_plate,
    mass_normalize_modes,
    evaluate_mode_gains,
)


# =========================================================
# Settings
# =========================================================

N_GEOMETRIES = 500
N_MODES = 32
N_POINTS = 32

RESOLUTION = 64
CONTOUR_POINTS = 512
MAX_AREA = 0.00025
POISSON = 0.3

MORPH_RANGE = (0.0, 1.0)
ASPECT_RANGE = (0.5, 2.0)

TRAIN_FRACTION = 0.8

GEOMETRY_SEED = 0
POINT_SEED = 1
SPLIT_SEED = 2

OUTPUT = (
    Path(__file__).parent
    / "plate_dataset.npz"
)


# =========================================================
# Geometry sampling
# =========================================================

def sample_geometry_parameters(n, rng):
    """
    Stratified sampling over morph and aspect.
    """

    morph = (
        np.arange(n) + rng.random(n)
    ) / n

    aspect = (
        np.arange(n) + rng.random(n)
    ) / n

    rng.shuffle(aspect)

    morph = (
        MORPH_RANGE[0]
        + morph
        * (MORPH_RANGE[1] - MORPH_RANGE[0])
    )

    aspect = (
        ASPECT_RANGE[0]
        + aspect
        * (ASPECT_RANGE[1] - ASPECT_RANGE[0])
    )

    order = rng.permutation(n)

    return (
        morph[order],
        aspect[order],
    )


# =========================================================
# Uniform point sampling inside triangular mesh
# =========================================================

def sample_mesh_points(mesh, n, rng):
    triangles = mesh.p[:, mesh.t]

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]

    areas = 0.5 * np.abs(
        (v1[0] - v0[0]) * (v2[1] - v0[1])
        - (v2[0] - v0[0]) * (v1[1] - v0[1])
    )

    triangle_indices = rng.choice(
        len(areas),
        size=n,
        p=areas / areas.sum(),
    )

    a = v0[:, triangle_indices]
    b = v1[:, triangle_indices]
    c = v2[:, triangle_indices]

    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)

    points = (
        (1.0 - r1) * a
        + r1 * (1.0 - r2) * b
        + r1 * r2 * c
    )

    return points.T


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    geometry_rng = np.random.default_rng(
        GEOMETRY_SEED
    )

    point_rng = np.random.default_rng(
        POINT_SEED
    )

    split_rng = np.random.default_rng(
        SPLIT_SEED
    )


    morphs, aspects = sample_geometry_parameters(
        N_GEOMETRIES,
        geometry_rng,
    )


    masks = np.empty(
        (N_GEOMETRIES, RESOLUTION, RESOLUTION),
        dtype=np.float32,
    )

    factors = np.empty(
        (N_GEOMETRIES, N_MODES),
        dtype=np.float32,
    )

    points = np.empty(
        (N_GEOMETRIES, N_POINTS, 2),
        dtype=np.float32,
    )

    point_gains = np.empty(
        (N_GEOMETRIES, N_POINTS, N_MODES),
        dtype=np.float32,
    )


    print(
        f"Generating {N_GEOMETRIES} geometries..."
    )

    start = time.perf_counter()


    for i, (morph, aspect) in enumerate(
        zip(morphs, aspects)
    ):

        contour = make_morph_contour(
            morph=morph,
            aspect=aspect,
            n_points=CONTOUR_POINTS,
        )


        masks[i] = contour_to_mask(
            contour,
            resolution=RESOLUTION,
        )


        mesh = mesh_from_contour(
            contour,
            max_area=MAX_AREA,
        )


        (
            modal_factors,
            _,
            eigenvectors,
            basis,
            free_dofs,
        ) = solve_plate(
            mesh,
            n_modes=N_MODES,
            poisson=POISSON,
        )


        eigenvectors = mass_normalize_modes(
            basis,
            eigenvectors,
            free_dofs,
        )


        sample_points = sample_mesh_points(
            mesh,
            N_POINTS,
            point_rng,
        )


        gains = evaluate_mode_gains(
            basis,
            eigenvectors,
            free_dofs,
            sample_points,
        )


        factors[i] = modal_factors
        points[i] = sample_points
        point_gains[i] = gains


        elapsed = time.perf_counter() - start
        avg = elapsed / (i + 1)
        eta = avg * (N_GEOMETRIES - i - 1)


        print(
            f"{i + 1:3d}/{N_GEOMETRIES}"
            f" | morph={morph:.3f}"
            f" | aspect={aspect:.3f}"
            f" | mu1={modal_factors[0]:7.3f}"
            f" | ETA={eta / 60:4.1f} min"
        )


    # =====================================================
    # Geometry-level train / validation split
    # =====================================================

    indices = np.arange(
        N_GEOMETRIES
    )

    split_rng.shuffle(
        indices
    )

    n_train = round(
        TRAIN_FRACTION
        * N_GEOMETRIES
    )

    train_indices = np.sort(
        indices[:n_train]
    )

    val_indices = np.sort(
        indices[n_train:]
    )


    # =====================================================
    # Save
    # =====================================================

    np.savez_compressed(
        OUTPUT,

        masks=masks,
        factors=factors,

        points=points,
        point_gains=point_gains,

        morphs=morphs.astype(np.float32),
        aspects=aspects.astype(np.float32),

        train_indices=train_indices,
        val_indices=val_indices,

        n_modes=N_MODES,
        n_points=N_POINTS,
    )


    elapsed = time.perf_counter() - start


    print()
    print("Done.")
    print("Saved:", OUTPUT)

    print()
    print("masks:      ", masks.shape)
    print("factors:    ", factors.shape)
    print("points:     ", points.shape)
    print("point_gains:", point_gains.shape)

    print()
    print(
        "Train geometries:",
        len(train_indices),
    )

    print(
        "Validation geometries:",
        len(val_indices),
    )

    print()
    print(
        "Possible point pairs / geometry:",
        N_POINTS ** 2,
    )

    print(
        "Possible train pairs:",
        len(train_indices) * N_POINTS ** 2,
    )

    print()
    print(
        f"Generation time: "
        f"{elapsed / 60:.2f} min"
    )