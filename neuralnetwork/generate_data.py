import time
import numpy as np

from shapes import (
    make_morph_mask,
    morph_sdf,
)

from plate_reference import (
    mesh_from_sdf,
    solve_plate,
)


N_MODES = 32

NN_RESOLUTION = 64
SDF_RESOLUTION = 512

MAX_AREA = 0.00025
SIMPLIFY = 0.002
POISSON = 0.3


def generate_sample(
    morph,
    aspect,
):
    # Kleine Maske:
    # das sieht später das Neural Network.
    mask = make_morph_mask(
        morph=morph,
        aspect=aspect,
        resolution=NN_RESOLUTION,
    )

    # Hochauflösende Geometrie:
    # nur für die Ground-Truth-Berechnung.
    sdf = morph_sdf(
        morph=morph,
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
        poisson=POISSON,
    )

    return (
        mask,
        modal_factors,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(
        0
    )

    N_SAMPLES = 20

    masks = []
    factors = []
    morphs = []
    aspects = []

    start_total = time.perf_counter()

    for i in range(N_SAMPLES):
        morph = rng.uniform(
            0.0,
            1.0,
        )

        aspect = rng.uniform(
            0.5,
            2.0,
        )

        start = time.perf_counter()

        mask, modal_factors = (
            generate_sample(
                morph=morph,
                aspect=aspect,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        masks.append(
            mask
        )

        factors.append(
            modal_factors
        )

        morphs.append(
            morph
        )

        aspects.append(
            aspect
        )

        print(
            f"{i + 1:02d}/{N_SAMPLES} | "
            f"morph={morph:.3f} | "
            f"aspect={aspect:.3f} | "
            f"mu1={modal_factors[0]:.3f} | "
            f"{elapsed:.2f} s"
        )

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

    elapsed_total = (
        time.perf_counter()
        - start_total
    )

    print()
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
        f"total time: "
        f"{elapsed_total:.2f} s"
    )

    print(
        f"average per sample: "
        f"{elapsed_total / N_SAMPLES:.2f} s"
    )

    np.savez_compressed(
        "neuralnetwork/"
        "plate_dataset_test.npz",
        masks=masks,
        factors=factors,
        morphs=morphs,
        aspects=aspects,
    )