import numpy as np
from skimage.measure import points_in_poly


DEFAULT_RESOLUTION = 64
MAX_HALF_SIZE = 0.9


def _shape_half_extents(aspect):
    """Return half width and height for a given aspect ratio."""
    if aspect <= 0.0:
        raise ValueError("aspect must be positive")

    if aspect >= 1.0:
        return MAX_HALF_SIZE, MAX_HALF_SIZE / aspect

    return MAX_HALF_SIZE * aspect, MAX_HALF_SIZE


def _rectangle_radius(theta, half_width, half_height):
    """Distance from origin to rectangle boundary at angle theta."""
    dx = np.cos(theta)
    dy = np.sin(theta)

    rx = np.divide(
        half_width,
        np.abs(dx),
        out=np.full_like(theta, np.inf),
        where=np.abs(dx) > 1e-12,
    )

    ry = np.divide(
        half_height,
        np.abs(dy),
        out=np.full_like(theta, np.inf),
        where=np.abs(dy) > 1e-12,
    )

    return np.minimum(rx, ry)


def _ellipse_radius(theta, radius_x, radius_y):
    """Distance from origin to ellipse boundary at angle theta."""
    dx = np.cos(theta)
    dy = np.sin(theta)

    return 1.0 / np.sqrt(
        (dx / radius_x) ** 2
        + (dy / radius_y) ** 2
    )


def _triangle_radius(theta, half_width, half_height):
    """Distance from origin to an upright triangular boundary."""
    dx = np.cos(theta)
    dy = np.sin(theta)

    # Triangle vertices:
    #
    #          (0, +h)
    #             /\
    #            /  \
    #           /    \
    #   (-w,-h)------(w,-h)

    coefficients = (
        2.0 * dx / half_width + dy / half_height,
        -2.0 * dx / half_width + dy / half_height,
        -dy / half_height,
    )

    radius = np.full_like(theta, np.inf)

    for coefficient in coefficients:
        candidate = np.divide(
            1.0,
            coefficient,
            out=np.full_like(theta, np.inf),
            where=coefficient > 1e-12,
        )

        radius = np.minimum(radius, candidate)

    return radius


def make_morph_contour(
    morph,
    aspect=1.0,
    n_points=512,
):
    """
    Generate the normalized plate contour.

    morph:
        0.0 = rectangle
        0.5 = ellipse
        1.0 = triangle
    """
    morph = float(np.clip(morph, 0.0, 1.0))

    half_width, half_height = _shape_half_extents(aspect)

    # Uniform angular samples.
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_points,
        endpoint=False,
    )

    # Explicitly include corner directions so polygonal
    # endpoints are represented exactly.
    critical_vertices = np.array(
        [
            [half_width, half_height],
            [-half_width, half_height],
            [-half_width, -half_height],
            [half_width, -half_height],
            [0.0, half_height],
            [-half_width, -half_height],
            [half_width, -half_height],
        ],
        dtype=np.float64,
    )

    critical_theta = np.mod(
        np.arctan2(
            critical_vertices[:, 1],
            critical_vertices[:, 0],
        ),
        2.0 * np.pi,
    )

    theta = np.sort(
        np.unique(
            np.concatenate(
                (theta, critical_theta)
            )
        )
    )

    rectangle = _rectangle_radius(
        theta,
        half_width,
        half_height,
    )

    ellipse = _ellipse_radius(
        theta,
        half_width,
        half_height,
    )

    triangle = _triangle_radius(
        theta,
        half_width,
        half_height,
    )

    if morph <= 0.5:
        amount = 2.0 * morph
        radius = (
            (1.0 - amount) * rectangle
            + amount * ellipse
        )
    else:
        amount = 2.0 * (morph - 0.5)
        radius = (
            (1.0 - amount) * ellipse
            + amount * triangle
        )

    return np.column_stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
        )
    ).astype(np.float64)


def contour_to_mask(
    contour,
    resolution=DEFAULT_RESOLUTION,
    supersample=4,
):
    """Rasterize a contour into an antialiased square mask."""
    high_res = resolution * supersample

    coords = (
        (np.arange(high_res) + 0.5)
        / high_res
        * 2.0
        - 1.0
    )

    x, y = np.meshgrid(coords, coords)

    points = np.column_stack(
        (
            x.ravel(),
            y.ravel(),
        )
    )

    inside = points_in_poly(
        points,
        contour,
    )

    high_res_mask = inside.reshape(
        high_res,
        high_res,
    ).astype(np.float32)

    return high_res_mask.reshape(
        resolution,
        supersample,
        resolution,
        supersample,
    ).mean(
        axis=(1, 3)
    ).astype(np.float32)


def make_morph_mask(
    morph,
    aspect=1.0,
    resolution=DEFAULT_RESOLUTION,
    supersample=4,
    contour_points=512,
):
    """Convenience function used by the NN/runtime."""
    contour = make_morph_contour(
        morph=morph,
        aspect=aspect,
        n_points=contour_points,
    )

    return contour_to_mask(
        contour=contour,
        resolution=resolution,
        supersample=supersample,
    )