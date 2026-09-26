from pathlib import Path
import time

import numpy as np

from skfem import asm
from skimage.measure import points_in_poly

from shapes import (
    make_morph_contour,
    contour_to_mask,
)

from plate_reference import (
    mass,
    mesh_from_contour,
    solve_plate,
    evaluate_mode_gains,
)


# =========================================================
# Settings
# =========================================================

N_MODES = 32

N_POINTS = 32

NN_RESOLUTION = 64
CONTOUR_POINTS = 512

MAX_AREA = 0.00025
POISSON = 0.3

POINT_SEED = 1234


# =========================================================
# Files
# =========================================================

ROOT = Path(__file__).parent


SOURCE_DATASET_PATH = (
    ROOT
    / "plate_dataset_pilot_500.npz"
)


OUTPUT_PATH = (
    ROOT
    / "plate_dataset_points_500.npz"
)


# =========================================================
# Mass-normalize generalized FEM eigenvectors
#
# phi.T @ M @ phi = 1
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
    ][
        :,
        free_dofs
    ]


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
            "Non-positive modal mass encountered."
        )


    modes /= np.sqrt(
        modal_masses
    )[
        None,
        :
    ]


    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    normalized_masses = np.sum(
        modes
        * (
            M_free
            @ modes
        ),
        axis=0,
    )


    max_mass_error = np.max(
        np.abs(
            normalized_masses
            - 1.0
        )
    )


    return (
        modes,
        max_mass_error,
    )


# =========================================================
# Sample points inside actual FEM geometry
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

        missing = (
            n_points
            - len(
                accepted
            )
        )


        # Oversample to avoid many tiny loops for triangles.
        n_candidates = max(
            8 * missing,
            64,
        )


        candidates = rng.uniform(
            low=min_xy,
            high=max_xy,
            size=(
                n_candidates,
                2,
            ),
        )


        # -------------------------------------------------
        # First cheap polygon check
        # -------------------------------------------------

        inside_polygon = points_in_poly(
            candidates,
            contour,
        )


        candidates = candidates[
            inside_polygon
        ]


        if len(
            candidates
        ) == 0:

            continue


        # -------------------------------------------------
        # Verify points against actual triangular FEM mesh
        # -------------------------------------------------

        for point in candidates:

            x = np.asarray(
                [
                    point[
                        0
                    ]
                ],
                dtype=np.float64,
            )


            y = np.asarray(
                [
                    point[
                        1
                    ]
                ],
                dtype=np.float64,
            )


            try:

                element_index = finder(
                    x,
                    y,
                )

            except ValueError:

                continue


            if (
                element_index.size == 0
                or element_index[
                    0
                ] < 0
            ):

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
# Generate point data for one geometry
# =========================================================

def generate_geometry_point_data(
    morph,
    aspect,
    rng,
):

    # -----------------------------------------------------
    # Exact analytic contour
    # -----------------------------------------------------

    contour = make_morph_contour(
        morph=morph,
        aspect=aspect,
        n_points=CONTOUR_POINTS,
    )


    # -----------------------------------------------------
    # NN mask reconstructed from the exact same contour
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
    # Eigenproblem
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
    # Explicit mass normalization
    # -----------------------------------------------------

    (
        normalized_eigenvectors,
        max_mass_error,
    ) = mass_normalize_eigenvectors(
        basis=basis,
        eigenvectors=eigenvectors,
        free_dofs=free_dofs,
    )


    # -----------------------------------------------------
    # Common pool of points
    #
    # These points can later be used independently as
    # strike or pickup positions.
    # -----------------------------------------------------

    points = sample_points_inside_mesh(
        contour=contour,
        mesh=mesh,
        rng=rng,
        n_points=N_POINTS,
    )


    # -----------------------------------------------------
    # Evaluate all modes at all points
    #
    # Shape:
    #
    #     (N_POINTS, N_MODES)
    #
    # IMPORTANT:
    #
    # normalize=False because the eigenvectors have already
    # been mass-normalized.
    # -----------------------------------------------------

    point_gains = evaluate_mode_gains(
        basis=basis,
        eigenvectors=normalized_eigenvectors,
        free_dofs=free_dofs,
        points=points,
        normalize=False,
    )


    return {
        "mask":
            mask,

        "modal_factors":
            modal_factors,

        "points":
            points,

        "point_gains":
            point_gains,

        "max_mass_error":
            max_mass_error,

        "mesh_vertices":
            mesh.p.shape[
                1
            ],

        "mesh_triangles":
            mesh.t.shape[
                1
            ],
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # Load existing pilot dataset
    #
    # We deliberately reuse:
    #
    # - morph
    # - aspect
    # - masks
    # - geometry order
    # - train indices
    # - validation indices
    #
    # so Frequency V2/V2.1 and PointNet later use exactly
    # the same geometry split.
    # =====================================================

    source = np.load(
        SOURCE_DATASET_PATH
    )


    source_masks = np.asarray(
        source[
            "masks"
        ],
        dtype=np.float32,
    )


    source_factors = np.asarray(
        source[
            "factors"
        ],
        dtype=np.float32,
    )


    morphs = np.asarray(
        source[
            "morphs"
        ],
        dtype=np.float32,
    )


    aspects = np.asarray(
        source[
            "aspects"
        ],
        dtype=np.float32,
    )


    train_indices = np.asarray(
        source[
            "train_indices"
        ],
        dtype=np.int64,
    )


    val_indices = np.asarray(
        source[
            "val_indices"
        ],
        dtype=np.int64,
    )


    N_GEOMETRIES = len(
        morphs
    )


    if N_GEOMETRIES != 500:

        raise RuntimeError(
            f"Expected 500 geometries, "
            f"found {N_GEOMETRIES}."
        )


    if source_factors.shape != (
        N_GEOMETRIES,
        N_MODES,
    ):

        raise RuntimeError(
            "Unexpected source factor shape: "
            f"{source_factors.shape}"
        )


    # =====================================================
    # Header
    # =====================================================

    print(
        "=" * 78
    )

    print(
        "POINT DATASET GENERATION"
    )

    print(
        "=" * 78
    )

    print()


    print(
        "Source:",
        SOURCE_DATASET_PATH,
    )


    print(
        "Output:",
        OUTPUT_PATH,
    )


    print()


    print(
        "Geometries:",
        N_GEOMETRIES,
    )


    print(
        "Modes:",
        N_MODES,
    )


    print(
        "Points / geometry:",
        N_POINTS,
    )


    print(
        "Possible ordered point pairs / geometry:",
        N_POINTS
        * N_POINTS,
    )


    print(
        "Possible total ordered pairs:",
        N_GEOMETRIES
        * N_POINTS
        * N_POINTS,
    )


    print()


    print(
        "Train geometries:",
        len(
            train_indices
        ),
    )


    print(
        "Validation geometries:",
        len(
            val_indices
        ),
    )


    # =====================================================
    # Output arrays
    # =====================================================

    masks = np.empty(
        (
            N_GEOMETRIES,
            NN_RESOLUTION,
            NN_RESOLUTION,
        ),
        dtype=np.float32,
    )


    factors = np.empty(
        (
            N_GEOMETRIES,
            N_MODES,
        ),
        dtype=np.float32,
    )


    points_all = np.empty(
        (
            N_GEOMETRIES,
            N_POINTS,
            2,
        ),
        dtype=np.float32,
    )


    point_gains_all = np.empty(
        (
            N_GEOMETRIES,
            N_POINTS,
            N_MODES,
        ),
        dtype=np.float32,
    )


    mesh_vertices = np.empty(
        N_GEOMETRIES,
        dtype=np.int32,
    )


    mesh_triangles = np.empty(
        N_GEOMETRIES,
        dtype=np.int32,
    )


    mass_normalization_errors = np.empty(
        N_GEOMETRIES,
        dtype=np.float64,
    )


    factor_recompute_errors = np.empty(
        N_GEOMETRIES,
        dtype=np.float64,
    )


    mask_recompute_errors = np.empty(
        N_GEOMETRIES,
        dtype=np.float64,
    )


    # =====================================================
    # Generate
    # =====================================================

    start_total = time.perf_counter()


    for geometry_index in range(
        N_GEOMETRIES
    ):

        morph = float(
            morphs[
                geometry_index
            ]
        )


        aspect = float(
            aspects[
                geometry_index
            ]
        )


        # -------------------------------------------------
        # Independent deterministic RNG per geometry.
        #
        # This means point sampling does not change for
        # later geometries if an earlier geometry changes.
        # -------------------------------------------------

        seed_sequence = np.random.SeedSequence(
            [
                POINT_SEED,
                geometry_index,
            ]
        )


        point_rng = np.random.default_rng(
            seed_sequence
        )


        start = time.perf_counter()


        result = generate_geometry_point_data(
            morph=morph,
            aspect=aspect,
            rng=point_rng,
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        # =================================================
        # Verify that reconstructed geometry matches the
        # existing pilot dataset.
        # =================================================

        regenerated_mask = np.asarray(
            result[
                "mask"
            ],
            dtype=np.float32,
        )


        mask_error = np.max(
            np.abs(
                regenerated_mask
                - source_masks[
                    geometry_index
                ]
            )
        )


        mask_recompute_errors[
            geometry_index
        ] = mask_error


        # -------------------------------------------------
        # This should be deterministic.
        #
        # If not, shapes.py has changed since the pilot
        # dataset was generated.
        # -------------------------------------------------

    # ---------------------------------------------------------
    # The original pilot masks were generated from float64
    # morph/aspect parameters and the parameters were later
    # stored as float32.
    #
    # Reconstructing the contour from the stored float32 values
    # can therefore move an isolated supersample across the
    # boundary.
    #
    # With supersample=4, one such sample corresponds to:
    #
    #     1 / 16 = 0.0625
    #
    # This is harmless because we keep the exact original mask
    # as NN input below.
    # ---------------------------------------------------------

    if mask_error > 0.125:

        raise RuntimeError(
            "\nLarge geometry mismatch detected.\n"
            f"geometry index: {geometry_index}\n"
            f"morph: {morph}\n"
            f"aspect: {aspect}\n"
            f"mask max error: {mask_error}\n"
        )


    if mask_error > 1e-6:

        differing_pixels = np.count_nonzero(
            np.abs(
                regenerated_mask
                - source_masks[
                    geometry_index
                ]
            )
            > 1e-6
        )


        mean_mask_error = np.mean(
            np.abs(
                regenerated_mask
                - source_masks[
                    geometry_index
                ]
            )
        )


        print(
            f"    mask rounding difference:"
            f" max={mask_error:.4f}"
            f" | pixels={differing_pixels}"
            f" | mean={mean_mask_error:.8f}"
        )


        # =================================================
        # Compare newly solved modal factors to source
        # =================================================

        modal_factors = np.asarray(
            result[
                "modal_factors"
            ],
            dtype=np.float64,
        )


        source_modal_factors = np.asarray(
            source_factors[
                geometry_index
            ],
            dtype=np.float64,
        )


        factor_relative_error = np.max(
            np.abs(
                modal_factors
                - source_modal_factors
            )
            / source_modal_factors
        )


        factor_recompute_errors[
            geometry_index
        ] = factor_relative_error


        # FEM eigensolver should reproduce the source
        # eigenvalues extremely closely.
        #
        # 0.5 % is deliberately a loose hard failure limit.
        if factor_relative_error > 0.005:

            raise RuntimeError(
                "\nFEM modal-factor mismatch detected.\n"
                f"geometry index: {geometry_index}\n"
                f"maximum relative difference: "
                f"{factor_relative_error * 100.0:.4f}%"
            )


        # =================================================
        # Store
        # =================================================

        # Keep the exact existing NN mask.
        masks[
            geometry_index
        ] = source_masks[
            geometry_index
        ]


        # Store modal factors from THIS FEM solve because
        # these correspond exactly to point_gains.
        factors[
            geometry_index
        ] = modal_factors.astype(
            np.float32
        )


        points_all[
            geometry_index
        ] = np.asarray(
            result[
                "points"
            ],
            dtype=np.float32,
        )


        point_gains_all[
            geometry_index
        ] = np.asarray(
            result[
                "point_gains"
            ],
            dtype=np.float32,
        )


        mesh_vertices[
            geometry_index
        ] = result[
            "mesh_vertices"
        ]


        mesh_triangles[
            geometry_index
        ] = result[
            "mesh_triangles"
        ]


        mass_normalization_errors[
            geometry_index
        ] = result[
            "max_mass_error"
        ]


        # =================================================
        # Progress
        # =================================================

        total_elapsed = (
            time.perf_counter()
            - start_total
        )


        average_time = (
            total_elapsed
            / (
                geometry_index
                + 1
            )
        )


        remaining_time = (
            average_time
            * (
                N_GEOMETRIES
                - geometry_index
                - 1
            )
        )


        max_gain = np.max(
            np.abs(
                result[
                    "point_gains"
                ]
            )
        )


        print(
            f"{geometry_index + 1:03d}/"
            f"{N_GEOMETRIES}"
            f" | "
            f"morph={morph:.3f}"
            f" | "
            f"aspect={aspect:.3f}"
            f" | "
            f"mu1={modal_factors[0]:7.3f}"
            f" | "
            f"max|g|={max_gain:6.3f}"
            f" | "
            f"mesh="
            f"{result['mesh_vertices']}v/"
            f"{result['mesh_triangles']}t"
            f" | "
            f"{elapsed:5.2f} s"
            f" | "
            f"ETA="
            f"{remaining_time / 60.0:5.1f} min"
        )


    # =====================================================
    # Dataset validation
    # =====================================================

    print()

    print(
        "=" * 78
    )

    print(
        "VALIDATING DATASET"
    )

    print(
        "=" * 78
    )


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

        "points":
            (
                N_GEOMETRIES,
                N_POINTS,
                2,
            ),

        "point_gains":
            (
                N_GEOMETRIES,
                N_POINTS,
                N_MODES,
            ),
    }


    actual_shapes = {

        "masks":
            masks.shape,

        "factors":
            factors.shape,

        "points":
            points_all.shape,

        "point_gains":
            point_gains_all.shape,
    }


    for name in expected_shapes:

        if (
            actual_shapes[
                name
            ]
            != expected_shapes[
                name
            ]
        ):

            raise RuntimeError(
                f"{name}: "
                f"got {actual_shapes[name]}, "
                f"expected {expected_shapes[name]}"
            )


    arrays_to_check = [
        masks,
        factors,
        points_all,
        point_gains_all,
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
            "NaN or Inf detected in dataset."
        )


    # -----------------------------------------------------
    # Modal factors must be strictly positive
    # -----------------------------------------------------

    if np.any(
        factors <= 0.0
    ):

        raise RuntimeError(
            "Non-positive modal factor detected."
        )


    # -----------------------------------------------------
    # Modal factors must be ordered
    # -----------------------------------------------------

    modal_differences = (
        factors[
            :,
            1:
        ]
        - factors[
            :,
            :-1
        ]
    )


    if np.any(
        modal_differences < 0.0
    ):

        raise RuntimeError(
            "Unordered modal factors detected."
        )


    # -----------------------------------------------------
    # Geometry split must remain disjoint
    # -----------------------------------------------------

    overlap = np.intersect1d(
        train_indices,
        val_indices,
    )


    if len(
        overlap
    ) != 0:

        raise RuntimeError(
            "Train/validation geometry overlap."
        )


    # =====================================================
    # Pair statistics
    #
    # Build all products only for statistics.
    #
    # Do NOT store them.
    # =====================================================

    gain_min = float(
        point_gains_all.min()
    )


    gain_max = float(
        point_gains_all.max()
    )


    max_absolute_residue = 0.0


    for geometry_index in range(
        N_GEOMETRIES
    ):

        gains = point_gains_all[
            geometry_index
        ]


        # Shape:
        #
        # point_a, point_b, mode
        pair_residues = (
            gains[
                :,
                None,
                :
            ]
            * gains[
                None,
                :,
                :
            ]
        )


        max_absolute_residue = max(
            max_absolute_residue,
            float(
                np.max(
                    np.abs(
                        pair_residues
                    )
                )
            ),
        )


    # =====================================================
    # Summary
    # =====================================================

    total_elapsed = (
        time.perf_counter()
        - start_total
    )


    print()

    print(
        "=" * 78
    )

    print(
        "POINT DATASET SUMMARY"
    )

    print(
        "=" * 78
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
        "points:",
        points_all.shape,
    )


    print(
        "point_gains:",
        point_gains_all.shape,
    )


    print()

    print(
        "Train geometries:",
        len(
            train_indices
        ),
    )


    print(
        "Validation geometries:",
        len(
            val_indices
        ),
    )


    print()

    print(
        "Point range:",
        f"{points_all.min():.4f}",
        "...",
        f"{points_all.max():.4f}",
    )


    print(
        "Gain range:",
        f"{gain_min:.4f}",
        "...",
        f"{gain_max:.4f}",
    )


    print(
        "Maximum |pair residue|:",
        f"{max_absolute_residue:.4f}",
    )


    print()

    print(
        "Modal-factor range:",
        f"{factors.min():.3f}",
        "...",
        f"{factors.max():.3f}",
    )


    print(
        "Maximum source/recomputed "
        "modal-factor difference:",
        f"{factor_recompute_errors.max() * 100.0:.6f}%",
    )


    print(
        "Maximum mask reconstruction error:",
        f"{mask_recompute_errors.max():.8f}",
    )


    print(
        "Maximum mass-normalization error:",
        f"{mass_normalization_errors.max():.3e}",
    )


    print()

    print(
        "Mesh vertices:",
        int(
            mesh_vertices.min()
        ),
        "...",
        int(
            mesh_vertices.max()
        ),
    )


    print(
        "Mesh triangles:",
        int(
            mesh_triangles.min()
        ),
        "...",
        int(
            mesh_triangles.max()
        ),
    )


    print()

    print(
        "Possible ordered pair combinations "
        "per geometry:",
        N_POINTS
        * N_POINTS,
    )


    print(
        "Possible TRAIN pair combinations:",
        len(
            train_indices
        )
        * N_POINTS
        * N_POINTS,
    )


    print(
        "Possible VALIDATION pair combinations:",
        len(
            val_indices
        )
        * N_POINTS
        * N_POINTS,
    )


    print()

    print(
        "Total generation time:",
        f"{total_elapsed:.2f} s",
    )


    print(
        "Total generation time:",
        f"{total_elapsed / 60.0:.2f} min",
    )


    print(
        "Average per geometry:",
        f"{total_elapsed / N_GEOMETRIES:.2f} s",
    )


    # =====================================================
    # Save
    # =====================================================

    np.savez_compressed(

        OUTPUT_PATH,

        # -------------------------------------------------
        # Geometry
        # -------------------------------------------------

        masks=masks,

        morphs=morphs,

        aspects=aspects,


        # -------------------------------------------------
        # FEM modal data
        # -------------------------------------------------

        factors=factors,


        # -------------------------------------------------
        # Common spatial point pool
        #
        # points:
        #     geometry, point, xy
        #
        # point_gains:
        #     geometry, point, mode
        # -------------------------------------------------

        points=points_all,

        point_gains=point_gains_all,


        # -------------------------------------------------
        # Exact same geometry split as pilot dataset
        # -------------------------------------------------

        train_indices=train_indices,

        val_indices=val_indices,


        # -------------------------------------------------
        # Diagnostics
        # -------------------------------------------------

        mesh_vertices=mesh_vertices,

        mesh_triangles=mesh_triangles,

        mass_normalization_errors=(
            mass_normalization_errors
        ),

        factor_recompute_errors=(
            factor_recompute_errors
        ),


        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        n_modes=np.int32(
            N_MODES
        ),

        n_points=np.int32(
            N_POINTS
        ),

        resolution=np.int32(
            NN_RESOLUTION
        ),

        contour_points=np.int32(
            CONTOUR_POINTS
        ),

        max_area=np.float32(
            MAX_AREA
        ),

        poisson=np.float32(
            POISSON
        ),

        point_seed=np.int32(
            POINT_SEED
        ),

        source_dataset=np.asarray(
            SOURCE_DATASET_PATH.name
        ),
    )


    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )