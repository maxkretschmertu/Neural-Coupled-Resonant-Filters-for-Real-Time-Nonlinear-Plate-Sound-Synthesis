import numpy as np
from scipy.ndimage import distance_transform_edt


DEFAULT_RESOLUTION = 64


def make_rectangle_mask(
    aspect,
    resolution=DEFAULT_RESOLUTION,
    supersample=4,
):
    high_res = resolution * supersample

    coords = (
        (np.arange(high_res) + 0.5)
        / high_res
        * 2.0
        - 1.0
    )

    x, y = np.meshgrid(
        coords,
        coords,
    )

    max_half_size = 0.9

    if aspect >= 1.0:
        half_width = max_half_size
        half_height = (
            max_half_size / aspect
        )
    else:
        half_height = max_half_size
        half_width = (
            max_half_size * aspect
        )

    high_res_mask = (
        (np.abs(x) <= half_width)
        & (np.abs(y) <= half_height)
    ).astype(np.float32)

    mask = high_res_mask.reshape(
        resolution,
        supersample,
        resolution,
        supersample,
    ).mean(axis=(1, 3))

    return mask


def make_ellipse_mask(
    aspect,
    resolution=DEFAULT_RESOLUTION,
    supersample=4,
):
    high_res = resolution * supersample

    coords = (
        (np.arange(high_res) + 0.5)
        / high_res
        * 2.0
        - 1.0
    )

    x, y = np.meshgrid(
        coords,
        coords,
    )

    max_half_size = 0.9

    if aspect >= 1.0:
        radius_x = max_half_size
        radius_y = (
            max_half_size / aspect
        )
    else:
        radius_y = max_half_size
        radius_x = (
            max_half_size * aspect
        )

    high_res_mask = (
        (x / radius_x) ** 2
        + (y / radius_y) ** 2
        <= 1.0
    ).astype(np.float32)

    mask = high_res_mask.reshape(
        resolution,
        supersample,
        resolution,
        supersample,
    ).mean(axis=(1, 3))

    return mask


def make_triangle_mask(
    aspect,
    resolution=DEFAULT_RESOLUTION,
    supersample=4,
):
    high_res = resolution * supersample

    coords = (
        (np.arange(high_res) + 0.5)
        / high_res
        * 2.0
        - 1.0
    )

    x, y = np.meshgrid(
        coords,
        coords,
    )

    max_half_size = 0.9

    if aspect >= 1.0:
        half_width = max_half_size
        half_height = (
            max_half_size / aspect
        )
    else:
        half_height = max_half_size
        half_width = (
            max_half_size * aspect
        )

    inside_y = (
        (y >= -half_height)
        & (y <= half_height)
    )

    allowed_half_width = (
        half_width
        * (half_height - y)
        / (2.0 * half_height)
    )

    high_res_mask = (
        inside_y
        & (
            np.abs(x)
            <= allowed_half_width
        )
    ).astype(np.float32)

    mask = high_res_mask.reshape(
        resolution,
        supersample,
        resolution,
        supersample,
    ).mean(axis=(1, 3))

    return mask


def mask_to_sdf(
    mask,
    threshold=0.5,
):
    inside = (
        mask >= threshold
    )

    distance_inside = (
        distance_transform_edt(
            inside
        )
    )

    distance_outside = (
        distance_transform_edt(
            ~inside
        )
    )

    sdf = (
        distance_inside
        - distance_outside
    )

    return sdf.astype(
        np.float32
    )


def morph_masks(
    mask_a,
    mask_b,
    amount,
):
    amount = float(
        np.clip(
            amount,
            0.0,
            1.0,
        )
    )

    sdf_a = mask_to_sdf(
        mask_a
    )

    sdf_b = mask_to_sdf(
        mask_b
    )

    sdf = (
        (1.0 - amount) * sdf_a
        + amount * sdf_b
    )

    return (
        sdf >= 0.0
    ).astype(np.float32)


def morph_sdf(
    morph,
    aspect=1.0,
    resolution=DEFAULT_RESOLUTION,
):
    morph = float(
        np.clip(
            morph,
            0.0,
            1.0,
        )
    )

    rectangle = make_rectangle_mask(
        aspect=aspect,
        resolution=resolution,
    )

    ellipse = make_ellipse_mask(
        aspect=aspect,
        resolution=resolution,
    )

    triangle = make_triangle_mask(
        aspect=aspect,
        resolution=resolution,
    )

    rectangle_sdf = mask_to_sdf(
        rectangle
    )

    ellipse_sdf = mask_to_sdf(
        ellipse
    )

    triangle_sdf = mask_to_sdf(
        triangle
    )

    if morph <= 0.5:
        amount = (
            morph * 2.0
        )

        sdf = (
            (1.0 - amount)
            * rectangle_sdf
            + amount
            * ellipse_sdf
        )

    else:
        amount = (
            (morph - 0.5)
            * 2.0
        )

        sdf = (
            (1.0 - amount)
            * ellipse_sdf
            + amount
            * triangle_sdf
        )

    return sdf.astype(
        np.float32
    )


def make_morph_mask(
    morph,
    aspect=1.0,
    resolution=DEFAULT_RESOLUTION,
):
    sdf = morph_sdf(
        morph=morph,
        aspect=aspect,
        resolution=resolution,
    )

    return (
        sdf >= 0.0
    ).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    aspect = 1.0
    resolution = 128

    morph_values = [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

    for morph in morph_values:
        sdf = morph_sdf(
            morph=morph,
            aspect=aspect,
            resolution=resolution,
        )

        mask = make_morph_mask(
            morph=morph,
            aspect=aspect,
            resolution=resolution,
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8, 4),
        )

        axes[0].imshow(
            sdf,
            origin="lower",
        )

        axes[0].set_title(
            f"SDF\nMorph = {morph:.2f}"
        )

        axes[0].axis("off")

        axes[1].imshow(
            mask,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )

        axes[1].set_title(
            f"Mask\nMorph = {morph:.2f}"
        )

        axes[1].axis("off")

        plt.tight_layout()

    plt.show()