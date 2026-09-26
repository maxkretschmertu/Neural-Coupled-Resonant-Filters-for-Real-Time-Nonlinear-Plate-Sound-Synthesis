import time
from pathlib import Path

import numpy as np

from skfem import asm
from skimage.measure import points_in_poly

from shapes import (
    make_morph_contour,
    contour_to_mask,
)

from plate_reference import (
    mesh_from_contour,
    solve_plate,
    evaluate_mode_gains,
    mass,
)


# =========================================================
# Dataset settings
# =========================================================

N_MODES = 32

NN_RESOLUTION = 64
CONTOUR_POINTS = 512

MAX_AREA = 0.00025
POISSON = 0.3


# ---------------------------------------------------------
# Pilot dataset
# ---------------------------------------------------------

N_GEOMETRIES = 500
N_POINT_PAIRS = 16

TRAIN_FRACTION = 0.80


# ---------------------------------------------------------
# Geometry parameter range
# ---------------------------------------------------------

MORPH_MIN = 0.0
MORPH_MAX = 1.0

ASPECT_MIN = 0.5
ASPECT_MAX = 2.0


# ---------------------------------------------------------
# Separate seeds
#
# Geometry parameters, point sampling and split are
# deliberately independent.
# ---------------------------------------------------------

GEOMETRY_SEED = 0
POINT_SEED = 1
SPLIT_SEED = 2


OUTPUT_PATH = (
    Path(__file__).parent
    / "plate_dataset_pilot_500.npz"
)


# =========================================================
# Mass-normalize FEM eigenvectors
#
# phi_k.T @ M @ phi_k = 1
# =========================================================

def mass_normalize_eigenvectors(
    basis,
    eigenvectors,
    free_dofs,
):

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


    modal_masses = np.sum(
        modes
        * (
            M_free
            @ modes
        ),
        axis=0,
    )


    if np.any(
        modal_masses <= 0.0
    ):

        raise RuntimeError(
            "Non-positive modal mass."
        )


    modes /= np.sqrt(
        modal_masses
    )[None, :]


    return (
        modes,
        modal_masses,
    )


# =========================================================
# Stratified geometry parameters
#
# Similar to a simple Latin-hypercube sample:
#
# - every part of the morph range is represented
# - every part of the aspect range is represented
# - aspect ordering is independently shuffled
#
# This gives better coverage than 500 purely random draws.
# =========================================================

def generate_geometry_parameters(
    n_samples,
    rng,
):

    # -----------------------------------------------------
    # One jittered sample per interval
    # -----------------------------------------------------

    morph_unit = (
        np.arange(
            n_samples,
            dtype=np.float64,
        )
        + rng.random(
            n_samples
        )
    ) / n_samples


    aspect_unit = (
        np.arange(
            n_samples,
            dtype=np.float64,
        )
        + rng.random(
            n_samples
        )
    ) / n_samples


    # -----------------------------------------------------
    # Decouple both dimensions
    # -----------------------------------------------------

    rng.shuffle(
        aspect_unit
    )


    # -----------------------------------------------------
    # Scale into requested ranges
    # -----------------------------------------------------

    morphs = (
        MORPH_MIN
        + morph_unit
        * (
            MORPH_MAX
            - MORPH_MIN
        )
    )


    aspects = (
        ASPECT_MIN
        + aspect_unit
        * (
            ASPECT_MAX
            - ASPECT_MIN
        )
    )


    # -----------------------------------------------------
    # Shuffle final geometry order
    #
    # This avoids the dataset itself being sorted by morph.
    # -----------------------------------------------------

    order = rng.permutation(
        n_samples
    )


    return (
        morphs[order],
        aspects[order],
    )


# =========================================================
# Sample points uniformly inside FEM mesh
# =========================================================

def sample_points_inside_mesh(
    contour,
    mesh,
    rng,
    n_points,
):

    contour = np.asarray(
        contour,
        dtype=np.float64,
    )


    min_xy = np.min(
        contour,
        axis=0,
    )


    max_xy = np.max(
        contour,
        axis=0,
    )


    finder = mesh.element_finder()


    accepted = []


    while len(
        accepted
    ) < n_points:

        n_missing = (
            n_points
            - len(
                accepted
            )
        )


        n_candidates = max(
            4 * n_missing,
            32,
        )


        candidates = rng.uniform(
            low=min_xy,
            high=max_xy,
            size=(
                n_candidates,
                2,
            ),
        )


        inside = points_in_poly(
            candidates,
            contour,
        )


        candidates = candidates[
            inside
        ]


        for point in candidates:

            x = np.array(
                [
                    point[0]
                ],
                dtype=np.float64,
            )


            y = np.array(
                [
                    point[1]
                ],
                dtype=np.float64,
            )


            try:

                finder(
                    x,
                    y,
                )

            except ValueError:

                continue


            accepted.append(
                point
            )


            if len(
                accepted
            ) >= n_points:

                break


    return np.asarray(
        accepted,
        dtype=np.float64,
    )


# =========================================================
# Generate one geometry
# =========================================================

def generate_geometry_sample(
    morph,
    aspect,
    point_rng,
):

    # -----------------------------------------------------
    # Shared analytic geometry
    # -----------------------------------------------------

    contour = make_morph_contour(
        morph=morph,
        aspect=aspect,
        n_points=CONTOUR_POINTS,
    )


    # -----------------------------------------------------
    # NN input mask
    # -----------------------------------------------------

    mask = contour_to_mask(
        contour=contour,
        resolution=NN_RESOLUTION,
        supersample=4,
    )


    # -----------------------------------------------------
    # FEM mesh
    # -----------------------------------------------------

    mesh = mesh_from_contour(
        contour,
        max_area=MAX_AREA,
    )


    # -----------------------------------------------------
    # FEM eigenproblem
    # -----------------------------------------------------

    (
        modal_factors,
        beta,
        eigenvectors,
        basis,
        free_dofs,
    ) = solve_plate(
        mesh,
        n_modes=N_MODES,
        poisson=POISSON,
    )


    # -----------------------------------------------------
    # Mass normalization
    # -----------------------------------------------------

    (
        normalized_eigenvectors,
        original_modal_masses,
    ) = mass_normalize_eigenvectors(
        basis=basis,
        eigenvectors=eigenvectors,
        free_dofs=free_dofs,
    )


    # -----------------------------------------------------
    # Strike + pickup positions
    #
    # 2 * P unique random points
    # -----------------------------------------------------

    points = sample_points_inside_mesh(
        contour=contour,
        mesh=mesh,
        rng=point_rng,
        n_points=(
            2
            * N_POINT_PAIRS
        ),
    )


    strike_points = points[
        :N_POINT_PAIRS
    ]


    pickup_points = points[
        N_POINT_PAIRS:
    ]


    # -----------------------------------------------------
    # Evaluate every point in one FEM probe call
    #
    # gains:
    #
    # (2 * P, N_MODES)
    # -----------------------------------------------------

    gains = evaluate_mode_gains(
        basis=basis,
        eigenvectors=normalized_eigenvectors,
        free_dofs=free_dofs,
        points=points,
        normalize=False,
    )


    strike_gains = gains[
        :N_POINT_PAIRS
    ]


    pickup_gains = gains[
        N_POINT_PAIRS:
    ]


    # -----------------------------------------------------
    # Modal residues
    #
    # r_k =
    #
    # phi_k(strike)
    # *
    # phi_k(pickup)
    #
    # Eigenvector sign cancels.
    # -----------------------------------------------------

    modal_weights = (
        strike_gains
        * pickup_gains
    )


    return {
        "mask":
            mask,

        "modal_factors":
            modal_factors,

        "strike_points":
            strike_points,

        "pickup_points":
            pickup_points,

        "modal_weights":
            modal_weights,

        "modal_masses":
            original_modal_masses,

        "mesh_vertices":
            mesh.p.shape[1],

        "mesh_triangles":
            mesh.t.shape[1],
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    geometry_rng = (
        np.random.default_rng(
            GEOMETRY_SEED
        )
    )


    point_rng = (
        np.random.default_rng(
            POINT_SEED
        )
    )


    split_rng = (
        np.random.default_rng(
            SPLIT_SEED
        )
    )


    # =====================================================
    # Generate well-distributed geometry parameters
    # =====================================================

    (
        morph_values,
        aspect_values,
    ) = generate_geometry_parameters(
        n_samples=N_GEOMETRIES,
        rng=geometry_rng,
    )


    print(
        "=" * 72
    )

    print(
        "PILOT DATASET GENERATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Geometries: "
        f"{N_GEOMETRIES}"
    )

    print(
        f"Point pairs / geometry: "
        f"{N_POINT_PAIRS}"
    )

    print(
        f"Total transfer examples: "
        f"{N_GEOMETRIES * N_POINT_PAIRS}"
    )

    print()

    print(
        f"Morph range: "
        f"{morph_values.min():.4f}"
        f" ... "
        f"{morph_values.max():.4f}"
    )

    print(
        f"Aspect range: "
        f"{aspect_values.min():.4f}"
        f" ... "
        f"{aspect_values.max():.4f}"
    )

    print()


    # =====================================================
    # Output lists
    # =====================================================

    masks = []
    factors = []

    morphs = []
    aspects = []

    strike_points_all = []
    pickup_points_all = []

    modal_weights_all = []


    # Optional diagnostics
    mesh_vertices_all = []
    mesh_triangles_all = []


    start_total = (
        time.perf_counter()
    )


    # =====================================================
    # FEM generation
    # =====================================================

    for i in range(
        N_GEOMETRIES
    ):

        morph = float(
            morph_values[i]
        )


        aspect = float(
            aspect_values[i]
        )


        start = (
            time.perf_counter()
        )


        sample = generate_geometry_sample(
            morph=morph,
            aspect=aspect,
            point_rng=point_rng,
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        # -------------------------------------------------
        # Collect
        # -------------------------------------------------

        masks.append(
            sample[
                "mask"
            ]
        )


        factors.append(
            sample[
                "modal_factors"
            ]
        )


        morphs.append(
            morph
        )


        aspects.append(
            aspect
        )


        strike_points_all.append(
            sample[
                "strike_points"
            ]
        )


        pickup_points_all.append(
            sample[
                "pickup_points"
            ]
        )


        modal_weights_all.append(
            sample[
                "modal_weights"
            ]
        )


        mesh_vertices_all.append(
            sample[
                "mesh_vertices"
            ]
        )


        mesh_triangles_all.append(
            sample[
                "mesh_triangles"
            ]
        )


        # -------------------------------------------------
        # Diagnostics
        # -------------------------------------------------

        modal_masses = sample[
            "modal_masses"
        ]


        weights = sample[
            "modal_weights"
        ]


        elapsed_total = (
            time.perf_counter()
            - start_total
        )


        average_time = (
            elapsed_total
            / (
                i + 1
            )
        )


        remaining = (
            average_time
            * (
                N_GEOMETRIES
                - i
                - 1
            )
        )


        print(
            f"{i + 1:03d}/"
            f"{N_GEOMETRIES}"
            f" | "
            f"morph="
            f"{morph:.3f}"
            f" | "
            f"aspect="
            f"{aspect:.3f}"
            f" | "
            f"mu1="
            f"{sample['modal_factors'][0]:7.3f}"
            f" | "
            f"max|w|="
            f"{np.max(np.abs(weights)):6.3f}"
            f" | "
            f"mesh="
            f"{sample['mesh_vertices']}v/"
            f"{sample['mesh_triangles']}t"
            f" | "
            f"{elapsed:5.2f} s"
            f" | "
            f"ETA="
            f"{remaining / 60.0:5.1f} min"
        )


        # -------------------------------------------------
        # Check eigensolver normalization
        # -------------------------------------------------

        if (
            np.min(
                modal_masses
            ) < 0.99
            or np.max(
                modal_masses
            ) > 1.01
        ):

            print(
                "    modal masses before "
                "explicit normalization: "
                f"{modal_masses.min():.6f}"
                f" ... "
                f"{modal_masses.max():.6f}"
            )


    # =====================================================
    # Convert to arrays
    # =====================================================

    masks = np.asarray(
        masks,
        dtype=np.float32,
    )


    factors = np.asarray(
        factors,
        dtype=np.float32,
    )


    morphs = np.asarray(
        morphs,
        dtype=np.float32,
    )


    aspects = np.asarray(
        aspects,
        dtype=np.float32,
    )


    strike_points_all = np.asarray(
        strike_points_all,
        dtype=np.float32,
    )


    pickup_points_all = np.asarray(
        pickup_points_all,
        dtype=np.float32,
    )


    modal_weights_all = np.asarray(
        modal_weights_all,
        dtype=np.float32,
    )


    mesh_vertices_all = np.asarray(
        mesh_vertices_all,
        dtype=np.int32,
    )


    mesh_triangles_all = np.asarray(
        mesh_triangles_all,
        dtype=np.int32,
    )


    # =====================================================
    # Geometry-level Train / Validation split
    #
    # IMPORTANT:
    #
    # We split geometry indices.
    #
    # All 16 point pairs belonging to one geometry stay
    # entirely inside either train or validation.
    # =====================================================

    all_indices = np.arange(
        N_GEOMETRIES,
        dtype=np.int64,
    )


    split_rng.shuffle(
        all_indices
    )


    n_train = int(
        round(
            TRAIN_FRACTION
            * N_GEOMETRIES
        )
    )


    train_indices = np.sort(
        all_indices[
            :n_train
        ]
    )


    val_indices = np.sort(
        all_indices[
            n_train:
        ]
    )


    # =====================================================
    # Validation
    # =====================================================

    expected_shapes = {

        "masks":
            (
                N_GEOMETRIES,
                NN_RESOLUTION,
                NN_RESOLUTION,
            ),

        "factors":
            (
                N_GEOMETRIES,
                N_MODES,
            ),

        "strike_points":
            (
                N_GEOMETRIES,
                N_POINT_PAIRS,
                2,
            ),

        "pickup_points":
            (
                N_GEOMETRIES,
                N_POINT_PAIRS,
                2,
            ),

        "modal_weights":
            (
                N_GEOMETRIES,
                N_POINT_PAIRS,
                N_MODES,
            ),
    }


    actual_shapes = {

        "masks":
            masks.shape,

        "factors":
            factors.shape,

        "strike_points":
            strike_points_all.shape,

        "pickup_points":
            pickup_points_all.shape,

        "modal_weights":
            modal_weights_all.shape,
    }


    for name in expected_shapes:

        if (
            actual_shapes[name]
            != expected_shapes[name]
        ):

            raise RuntimeError(
                f"Unexpected shape for "
                f"{name}: "
                f"{actual_shapes[name]}, "
                f"expected "
                f"{expected_shapes[name]}"
            )


    arrays_to_check = [
        masks,
        factors,
        morphs,
        aspects,
        strike_points_all,
        pickup_points_all,
        modal_weights_all,
    ]


    if not all(
        np.all(
            np.isfinite(
                array
            )
        )
        for array
        in arrays_to_check
    ):

        raise RuntimeError(
            "Dataset contains NaN or Inf."
        )


    # -----------------------------------------------------
    # Verify split
    # -----------------------------------------------------

    if (
        len(
            np.intersect1d(
                train_indices,
                val_indices,
            )
        )
        != 0
    ):

        raise RuntimeError(
            "Train/validation overlap detected."
        )


    if (
        len(
            train_indices
        )
        + len(
            val_indices
        )
        != N_GEOMETRIES
    ):

        raise RuntimeError(
            "Invalid train/validation split."
        )


    # =====================================================
    # Summary
    # =====================================================

    elapsed_total = (
        time.perf_counter()
        - start_total
    )


    print()

    print(
        "=" * 72
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        "masks:",
        masks.shape,
    )


    print(
        "factors:",
        factors.shape,
    )


    print(
        "morphs:",
        morphs.shape,
    )


    print(
        "aspects:",
        aspects.shape,
    )


    print(
        "strike_points:",
        strike_points_all.shape,
    )


    print(
        "pickup_points:",
        pickup_points_all.shape,
    )


    print(
        "modal_weights:",
        modal_weights_all.shape,
    )


    print()

    print(
        f"Train geometries: "
        f"{len(train_indices)}"
    )


    print(
        f"Validation geometries: "
        f"{len(val_indices)}"
    )


    print(
        f"Train transfer examples: "
        f"{len(train_indices) * N_POINT_PAIRS}"
    )


    print(
        f"Validation transfer examples: "
        f"{len(val_indices) * N_POINT_PAIRS}"
    )


    print()

    print(
        f"Morph range: "
        f"{morphs.min():.4f}"
        f" ... "
        f"{morphs.max():.4f}"
    )


    print(
        f"Aspect range: "
        f"{aspects.min():.4f}"
        f" ... "
        f"{aspects.max():.4f}"
    )


    print()

    print(
        f"Mask range: "
        f"{masks.min():.3f}"
        f" ... "
        f"{masks.max():.3f}"
    )


    print(
        f"Modal-factor range: "
        f"{factors.min():.3f}"
        f" ... "
        f"{factors.max():.3f}"
    )


    point_min = min(
        strike_points_all.min(),
        pickup_points_all.min(),
    )


    point_max = max(
        strike_points_all.max(),
        pickup_points_all.max(),
    )


    print(
        f"Point range: "
        f"{point_min:.3f}"
        f" ... "
        f"{point_max:.3f}"
    )


    print(
        f"Modal-weight range: "
        f"{modal_weights_all.min():.3f}"
        f" ... "
        f"{modal_weights_all.max():.3f}"
    )


    print()

    print(
        f"Mesh vertices: "
        f"{mesh_vertices_all.min()}"
        f" ... "
        f"{mesh_vertices_all.max()}"
    )


    print(
        f"Mesh triangles: "
        f"{mesh_triangles_all.min()}"
        f" ... "
        f"{mesh_triangles_all.max()}"
    )


    print()

    print(
        f"Total generation time: "
        f"{elapsed_total:.2f} s"
    )


    print(
        f"Total generation time: "
        f"{elapsed_total / 60.0:.2f} min"
    )


    print(
        f"Average per geometry: "
        f"{elapsed_total / N_GEOMETRIES:.2f} s"
    )


    # =====================================================
    # Save
    # =====================================================

    np.savez_compressed(

        OUTPUT_PATH,

        # -------------------------------------------------
        # Main data
        # -------------------------------------------------

        masks=masks,

        factors=factors,

        morphs=morphs,

        aspects=aspects,

        strike_points=(
            strike_points_all
        ),

        pickup_points=(
            pickup_points_all
        ),

        modal_weights=(
            modal_weights_all
        ),


        # -------------------------------------------------
        # Geometry-level split
        # -------------------------------------------------

        train_indices=(
            train_indices
        ),

        val_indices=(
            val_indices
        ),


        # -------------------------------------------------
        # Mesh diagnostics
        # -------------------------------------------------

        mesh_vertices=(
            mesh_vertices_all
        ),

        mesh_triangles=(
            mesh_triangles_all
        ),


        # -------------------------------------------------
        # Dataset metadata
        # -------------------------------------------------

        n_modes=np.int32(
            N_MODES
        ),

        n_point_pairs=np.int32(
            N_POINT_PAIRS
        ),

        resolution=np.int32(
            NN_RESOLUTION
        ),

        poisson=np.float32(
            POISSON
        ),

        max_area=np.float32(
            MAX_AREA
        ),

        morph_min=np.float32(
            MORPH_MIN
        ),

        morph_max=np.float32(
            MORPH_MAX
        ),

        aspect_min=np.float32(
            ASPECT_MIN
        ),

        aspect_max=np.float32(
            ASPECT_MAX
        ),

        geometry_seed=np.int32(
            GEOMETRY_SEED
        ),

        point_seed=np.int32(
            POINT_SEED
        ),

        split_seed=np.int32(
            SPLIT_SEED
        ),
    )


    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )