from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import GeometryDescription, boundary_radius_at


@dataclass(frozen=True)
class MaterialGrid:
    """Fixed canonical unit-disk grid mapped onto a current plate geometry.

    `area_weights` approximate dA on the normalized plate geometry and are
    rescaled to integrate to `geometry.area` exactly. `inner_product_weights`
    are the same quadrature normalized to sum to one; they are convenient for
    mode normalization, MAC-like comparisons and basis projection. They are
    not physical mass weights.
    """

    u: np.ndarray
    v: np.ndarray
    rho: np.ndarray
    theta: np.ndarray
    mask: np.ndarray
    physical_x: np.ndarray
    physical_y: np.ndarray
    area_weights: np.ndarray
    inner_product_weights: np.ndarray

    @property
    def grid_size(self) -> int:
        return int(self.u.shape[0])


def make_material_grid(geometry: GeometryDescription, grid_size: int = 64) -> MaterialGrid:
    if grid_size < 16:
        raise ValueError("grid_size must be >= 16")

    axis = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float64)
    u, v = np.meshgrid(axis, axis, indexing="xy")
    rho = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)
    mask = rho <= 1.0

    rb = boundary_radius_at(geometry, theta)
    x = rho * rb * np.cos(theta)
    y = rho * rb * np.sin(theta)
    x = np.where(mask, x, 0.0)
    y = np.where(mask, y, 0.0)

    # For x = R(theta) u and y = R(theta) v, the Jacobian from the
    # canonical disk to the physical normalized plate is R(theta)^2.
    du = 2.0 / (grid_size - 1)
    raw_area_weights = np.where(mask, rb * rb * du * du, 0.0)
    raw_total = float(np.sum(raw_area_weights))
    if raw_total <= 0.0:
        raise ValueError("material grid has zero area")

    area_weights = raw_area_weights * (float(geometry.area) / raw_total)
    inner_product_weights = area_weights / float(geometry.area)

    return MaterialGrid(
        u=np.ascontiguousarray(u),
        v=np.ascontiguousarray(v),
        rho=np.ascontiguousarray(rho),
        theta=np.ascontiguousarray(theta),
        mask=np.ascontiguousarray(mask),
        physical_x=np.ascontiguousarray(x),
        physical_y=np.ascontiguousarray(y),
        area_weights=np.ascontiguousarray(area_weights),
        inner_product_weights=np.ascontiguousarray(inner_product_weights),
    )


def material_to_physical(
    geometry: GeometryDescription,
    u: np.ndarray | float,
    v: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    rho = np.sqrt(u_arr * u_arr + v_arr * v_arr)
    theta = np.arctan2(v_arr, u_arr)
    rb = boundary_radius_at(geometry, theta)
    return rho * rb * np.cos(theta), rho * rb * np.sin(theta)


def physical_to_material(
    geometry: GeometryDescription,
    x: np.ndarray | float,
    y: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    theta = np.arctan2(y_arr, x_arr)
    r = np.sqrt(x_arr * x_arr + y_arr * y_arr)
    rb = np.maximum(boundary_radius_at(geometry, theta), 1e-12)
    rho = np.clip(r / rb, 0.0, 1.0)
    return rho * np.cos(theta), rho * np.sin(theta)


def clamp_material_point(
    u: float,
    v: float,
    max_rho: float = 0.999,
) -> tuple[float, float]:
    r = float(np.hypot(u, v))
    if r <= max_rho:
        return float(u), float(v)
    scale = max_rho / max(r, 1e-12)
    return float(u * scale), float(v * scale)
