import numpy as np


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

    x, y = np.meshgrid(coords, coords)

    max_half_size = 0.9

    if aspect >= 1.0:
        half_width = max_half_size
        half_height = max_half_size / aspect
    else:
        half_height = max_half_size
        half_width = max_half_size * aspect

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mask = make_rectangle_mask(1.0)


    plt.imshow(mask, origin="lower")
    plt.title("aspect = 1.0")
    plt.show()