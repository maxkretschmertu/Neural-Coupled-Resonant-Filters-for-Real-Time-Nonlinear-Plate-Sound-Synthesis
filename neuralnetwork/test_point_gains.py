import numpy as np

from shapes import (
    make_morph_contour,
)

from plate_reference import (
    mesh_from_contour,
    solve_plate,
    evaluate_mode_gains,
)


ASPECT = 1.6

N_MODES = 32
N_TEST_MODES = 12

MAX_AREA = 0.00025
POISSON = 0.3

MAX_HALF_SIZE = 0.9


# ---------------------------------------------------------
# Rectangle dimensions
# ---------------------------------------------------------

if ASPECT >= 1.0:
    half_width = MAX_HALF_SIZE
    half_height = (
        MAX_HALF_SIZE / ASPECT
    )
else:
    half_height = MAX_HALF_SIZE
    half_width = (
        MAX_HALF_SIZE * ASPECT
    )


# ---------------------------------------------------------
# FEM geometry
# ---------------------------------------------------------

contour = make_morph_contour(
    morph=0.0,
    aspect=ASPECT,
    n_points=512,
)

mesh = mesh_from_contour(
    contour,
    max_area=MAX_AREA,
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
    poisson=POISSON,
)


# ---------------------------------------------------------
# Analytical rectangle modes
#
# mu_mn =
# pi^2 * (
#     m^2 / aspect
#     + aspect * n^2
# )
# ---------------------------------------------------------

analytic_modes = []

for m in range(1, 12):
    for n in range(1, 12):
        mu = (
            np.pi**2
            * (
                m**2 / ASPECT
                + ASPECT * n**2
            )
        )

        analytic_modes.append(
            (
                mu,
                m,
                n,
            )
        )

analytic_modes.sort(
    key=lambda item: item[0]
)

analytic_modes = (
    analytic_modes[:N_MODES]
)


# ---------------------------------------------------------
# Test points inside rectangle
#
# Stay somewhat away from boundary so that
# nodal zeros don't dominate the comparison.
# ---------------------------------------------------------

x_values = np.linspace(
    -0.8 * half_width,
    +0.8 * half_width,
    9,
)

y_values = np.linspace(
    -0.8 * half_height,
    +0.8 * half_height,
    9,
)

x_grid, y_grid = np.meshgrid(
    x_values,
    y_values,
)

points = np.column_stack(
    (
        x_grid.ravel(),
        y_grid.ravel(),
    )
)


# ---------------------------------------------------------
# FEM mode shapes at all test points
#
# Don't normalize here.
# Eigenvectors have arbitrary amplitude and sign.
# We fit one scalar per mode below.
# ---------------------------------------------------------

numerical_gains = evaluate_mode_gains(
    basis=basis,
    eigenvectors=eigenvectors,
    free_dofs=free_dofs,
    points=points,
    normalize=False,
)


# ---------------------------------------------------------
# Convert physical coordinates to
# X,Y in [0,1]
# ---------------------------------------------------------

X = (
    points[:, 0]
    + half_width
) / (
    2.0 * half_width
)

Y = (
    points[:, 1]
    + half_height
) / (
    2.0 * half_height
)


# ---------------------------------------------------------
# Compare modes
# ---------------------------------------------------------

print(
    f"Aspect = {ASPECT}"
)

print(
    f"Mesh vertices = "
    f"{mesh.p.shape[1]}"
)

print(
    f"Mesh triangles = "
    f"{mesh.t.shape[1]}"
)

print()

print(
    "mode | (m,n) | "
    "mu FEM | mu exact | "
    "mu err [%] | "
    "|corr| | shape RMSE [%]"
)

print(
    "-" * 88
)


for k in range(N_TEST_MODES):

    (
        exact_mu,
        m,
        n,
    ) = analytic_modes[k]

    numerical = (
        numerical_gains[:, k]
    )

    analytical = (
        np.sin(
            m * np.pi * X
        )
        * np.sin(
            n * np.pi * Y
        )
    )

    # -----------------------------------------------------
    # Eigenvectors have arbitrary sign AND amplitude.
    #
    # Find the scalar which maps the numerical mode
    # as closely as possible onto the analytical mode.
    # This automatically fixes both.
    # -----------------------------------------------------

    denominator = np.dot(
        numerical,
        numerical,
    )

    scale = (
        np.dot(
            numerical,
            analytical,
        )
        / denominator
    )

    numerical_aligned = (
        scale
        * numerical
    )


    # -----------------------------------------------------
    # Shape error
    # -----------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            (
                numerical_aligned
                - analytical
            ) ** 2
        )
    )

    shape_rmse_percent = (
        rmse
        / np.max(
            np.abs(
                analytical
            )
        )
        * 100.0
    )


    # -----------------------------------------------------
    # Shape correlation
    # -----------------------------------------------------

    correlation = np.corrcoef(
        numerical,
        analytical,
    )[0, 1]

    correlation = abs(
        correlation
    )


    # -----------------------------------------------------
    # Eigenvalue / modal-factor error
    # -----------------------------------------------------

    mu_error = (
        abs(
            modal_factors[k]
            - exact_mu
        )
        / exact_mu
        * 100.0
    )


    print(
        f"{k + 1:4d} | "
        f"({m:1d},{n:1d}) | "
        f"{modal_factors[k]:7.3f} | "
        f"{exact_mu:8.3f} | "
        f"{mu_error:10.4f} | "
        f"{correlation:6.4f} | "
        f"{shape_rmse_percent:14.4f}"
    )