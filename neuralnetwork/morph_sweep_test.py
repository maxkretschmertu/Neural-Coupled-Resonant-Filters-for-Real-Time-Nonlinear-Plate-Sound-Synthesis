import numpy as np
import matplotlib.pyplot as plt

from shapes import make_morph_contour

from plate_reference import (
    mesh_from_contour,
    solve_plate,
)


N_MODES = 32

MAX_AREA = 0.00025
POISSON = 0.3
CONTOUR_POINTS = 512


aspects = [
    0.6,
    1.0,
    1.5,
    1.9,
]

morph_values = np.linspace(
    0.0,
    1.0,
    41,
)


for aspect in aspects:

    all_factors = []

    print(
        f"\nAspect = {aspect}"
    )

    for i, morph in enumerate(
        morph_values
    ):
        contour = make_morph_contour(
            morph=morph,
            aspect=aspect,
            n_points=CONTOUR_POINTS,
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

        all_factors.append(
            modal_factors
        )

        print(
            f"{i + 1:02d}/"
            f"{len(morph_values)} | "
            f"morph={morph:.3f} | "
            f"mu1={modal_factors[0]:.3f}"
        )

    all_factors = np.asarray(
        all_factors
    )

    plt.figure()

    for mode in range(8):
        plt.plot(
            morph_values,
            all_factors[:, mode],
            label=f"Mode {mode + 1}",
        )

    plt.xlabel(
        "Morph"
    )

    plt.ylabel(
        "Modal factor"
    )

    plt.title(
        f"Modal factors vs morph "
        f"(aspect={aspect})"
    )

    plt.legend()

    plt.grid(
        True
    )


plt.show()