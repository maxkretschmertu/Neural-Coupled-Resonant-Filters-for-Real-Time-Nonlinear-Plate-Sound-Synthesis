from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# 360 boundary samples are sufficient for interactive control-rate geometry
# updates at the current 64x64 SDF resolution. Dataset generation may request
# a larger value explicitly (e.g. 720 or 1440).
DEFAULT_BOUNDARY_SAMPLES = 360
DEFAULT_SDF_GRID = 64
DEFAULT_DOMAIN_HALF_EXTENT = 1.75
DEFAULT_MAX_STRETCH = 4.0


@dataclass(frozen=True)
class GeometryDescription:
    """Unit-area star-shaped plate geometry in canonical physical coordinates."""

    morph: float
    shape_mod: float
    boundary_theta: np.ndarray
    boundary_radius: np.ndarray
    boundary_xy: np.ndarray
    sdf: np.ndarray
    mask: np.ndarray
    domain_half_extent: float
    area: float

    @property
    def grid_size(self) -> int:
        return int(self.sdf.shape[0])


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _shape_aspect(
    shape_mod: float,
    base_aspect: float = 1.0,
    max_stretch: float = DEFAULT_MAX_STRETCH,
) -> float:
    s = float(np.clip(shape_mod, 0.0, 1.0))
    return float(base_aspect * np.exp(np.log(max_stretch) * s))


def _rectangle_radius(theta: np.ndarray, shape_mod: float) -> np.ndarray:
    aspect = _shape_aspect(shape_mod)
    width = np.sqrt(aspect)
    height = 1.0 / np.sqrt(aspect)
    hx, hy = 0.5 * width, 0.5 * height
    c = np.abs(np.cos(theta))
    s = np.abs(np.sin(theta))
    tx = np.full(theta.shape, np.inf, dtype=np.float64)
    ty = np.full(theta.shape, np.inf, dtype=np.float64)
    np.divide(hx, c, out=tx, where=c > 1e-14)
    np.divide(hy, s, out=ty, where=s > 1e-14)
    return np.minimum(tx, ty)


def _ellipse_radius(theta: np.ndarray, shape_mod: float) -> np.ndarray:
    aspect = _shape_aspect(shape_mod)
    a = np.sqrt(aspect / np.pi)
    b = np.sqrt(1.0 / (np.pi * aspect))
    c = np.cos(theta)
    s = np.sin(theta)
    return 1.0 / np.sqrt((c / a) ** 2 + (s / b) ** 2)


def _triangle_vertices(shape_mod: float) -> np.ndarray:
    equilateral_aspect = 2.0 / np.sqrt(3.0)
    aspect = _shape_aspect(shape_mod, base_aspect=equilateral_aspect)
    height = np.sqrt(2.0 / aspect)
    width = np.sqrt(2.0 * aspect)
    return np.array(
        [
            [0.0, 2.0 * height / 3.0],
            [-width / 2.0, -height / 3.0],
            [width / 2.0, -height / 3.0],
        ],
        dtype=np.float64,
    )


def _polygon_radial_radius(
    theta: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    radius = np.full(theta.shape, np.inf, dtype=np.float64)

    for i in range(vertices.shape[0]):
        a = vertices[i]
        b = vertices[(i + 1) % vertices.shape[0]]
        e = b - a
        den = directions[:, 0] * e[1] - directions[:, 1] * e[0]
        valid_den = np.abs(den) > 1e-14
        cross_ae = a[0] * e[1] - a[1] * e[0]
        cross_ad = a[0] * directions[:, 1] - a[1] * directions[:, 0]
        t = np.full(theta.shape, np.inf, dtype=np.float64)
        u = np.full(theta.shape, np.inf, dtype=np.float64)
        t[valid_den] = cross_ae / den[valid_den]
        u[valid_den] = cross_ad[valid_den] / den[valid_den]
        hit = (
            valid_den
            & (t > 0.0)
            & (u >= -1e-12)
            & (u <= 1.0 + 1e-12)
        )
        radius[hit] = np.minimum(radius[hit], t[hit])

    if not np.all(np.isfinite(radius)):
        raise RuntimeError("triangle radial intersection failed")
    return radius


def _triangle_radius(theta: np.ndarray, shape_mod: float) -> np.ndarray:
    return _polygon_radial_radius(theta, _triangle_vertices(shape_mod))


def _periodic_interp(
    theta_query: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    period = 2.0 * np.pi
    tq = np.mod(theta_query, period)
    theta_ext = np.concatenate((theta, [period]))
    values_ext = np.concatenate((values, [values[0]]))
    return np.interp(tq, theta_ext, values_ext)


def _unit_area_normalize(
    theta: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    dtheta = 2.0 * np.pi / theta.size
    area = 0.5 * dtheta * float(np.sum(radius * radius))
    if area <= 0.0:
        raise ValueError("geometry has non-positive area")
    return radius / np.sqrt(area)


def _morphed_radius(
    theta: np.ndarray,
    morph: float,
    shape_mod: float,
) -> np.ndarray:
    m = float(np.clip(morph, 0.0, 1.0))
    rect = _rectangle_radius(theta, shape_mod)
    ellipse = _ellipse_radius(theta, shape_mod)
    triangle = _triangle_radius(theta, shape_mod)

    if m <= 0.5:
        t = _smoothstep01(2.0 * m)
        radius = (1.0 - t) * rect + t * ellipse
    else:
        t = _smoothstep01(2.0 * m - 1.0)
        radius = (1.0 - t) * ellipse + t * triangle

    return _unit_area_normalize(theta, radius)


def _signed_distance_to_boundary(
    boundary_xy: np.ndarray,
    boundary_theta: np.ndarray,
    boundary_radius: np.ndarray,
    grid_size: int,
    domain_half_extent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact segment-distance SDF on the requested regular grid.

    This path is intended for control-rate NN input and offline dataset labels,
    not for the audio callback. The runtime default uses 360 contour segments;
    callers generating training data can request more samples explicitly.
    """
    axis = np.linspace(
        -domain_half_extent,
        domain_half_extent,
        grid_size,
        dtype=np.float64,
    )
    xg, yg = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((xg.ravel(), yg.ravel()))

    a = boundary_xy
    b = np.roll(boundary_xy, -1, axis=0)
    e = b - a
    e2 = np.sum(e * e, axis=1)
    min_dist2 = np.full(points.shape[0], np.inf, dtype=np.float64)

    # Larger chunks reduce Python-loop overhead without creating a large
    # persistent allocation. 1024 x 360 segment work is modest at 64x64.
    chunk = 1024
    for start in range(0, points.shape[0], chunk):
        p = points[start : start + chunk]
        pa = p[:, None, :] - a[None, :, :]
        t = np.sum(pa * e[None, :, :], axis=2) / np.maximum(
            e2[None, :],
            1e-20,
        )
        t = np.clip(t, 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * e[None, :, :]
        d2 = np.sum((p[:, None, :] - closest) ** 2, axis=2)
        min_dist2[start : start + p.shape[0]] = np.min(d2, axis=1)

    radial_angle = np.arctan2(points[:, 1], points[:, 0])
    point_radius = np.sqrt(np.sum(points * points, axis=1))
    local_boundary = _periodic_interp(
        radial_angle,
        boundary_theta,
        boundary_radius,
    )
    inside = point_radius <= local_boundary
    distance = np.sqrt(min_dist2)
    signed = np.where(inside, -distance, distance)

    return (
        np.ascontiguousarray(signed.reshape(grid_size, grid_size)),
        np.ascontiguousarray(
            inside.reshape(grid_size, grid_size),
            dtype=bool,
        ),
    )


def make_geometry(
    morph: float,
    shape_mod: float,
    grid_size: int = DEFAULT_SDF_GRID,
    boundary_samples: int = DEFAULT_BOUNDARY_SAMPLES,
    domain_half_extent: float = DEFAULT_DOMAIN_HALF_EXTENT,
) -> GeometryDescription:
    """Generate the full rectangle -> ellipse -> triangle morph geometry."""
    if grid_size < 16:
        raise ValueError("grid_size must be >= 16")
    if boundary_samples < 64:
        raise ValueError("boundary_samples must be >= 64")

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        boundary_samples,
        endpoint=False,
        dtype=np.float64,
    )
    radius = _morphed_radius(theta, morph, shape_mod)
    boundary = np.column_stack(
        (radius * np.cos(theta), radius * np.sin(theta))
    )
    sdf, mask = _signed_distance_to_boundary(
        boundary,
        theta,
        radius,
        int(grid_size),
        float(domain_half_extent),
    )

    dtheta = 2.0 * np.pi / boundary_samples
    area = 0.5 * dtheta * float(np.sum(radius * radius))
    return GeometryDescription(
        morph=float(np.clip(morph, 0.0, 1.0)),
        shape_mod=float(np.clip(shape_mod, 0.0, 1.0)),
        boundary_theta=np.ascontiguousarray(theta),
        boundary_radius=np.ascontiguousarray(radius),
        boundary_xy=np.ascontiguousarray(boundary),
        sdf=sdf,
        mask=mask,
        domain_half_extent=float(domain_half_extent),
        area=float(area),
    )


def boundary_radius_at(
    geometry: GeometryDescription,
    theta: np.ndarray | float,
) -> np.ndarray:
    return np.asarray(
        _periodic_interp(
            np.asarray(theta, dtype=np.float64),
            geometry.boundary_theta,
            geometry.boundary_radius,
        ),
        dtype=np.float64,
    )
