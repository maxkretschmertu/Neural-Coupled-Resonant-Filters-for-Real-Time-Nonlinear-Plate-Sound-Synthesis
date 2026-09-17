from __future__ import annotations

import numpy as np
from numba import njit

from .modal_backend import ModalBasis


PROJECTION_CONVENTION = "material_coordinates_new_mass_inner_product"


def projection_matrix(
    new_basis: ModalBasis,
    old_basis: ModalBasis,
    regularization: float = 1e-8,
) -> np.ndarray:
    """Weighted least-squares mapping from old to new modal coordinates.

    Convention used throughout the project:

    1. Old and new mode shapes are both represented on the same canonical
       material-coordinate disk.
    2. The old displacement/velocity field is transported with those material
       coordinates as the geometry changes.
    3. That transported field is projected onto the *new* modal basis using
       the new geometry's normalized mass/area inner product.

    For uniform rho and H, the scalar physical mass factor cancels from the
    least-squares normal equations, so normalized geometric weights are
    sufficient. This is a quasistatic basis-transfer model, not the exact PDE
    for a continuously moving boundary (which would also contain basis-time-
    derivative terms).
    """
    old_shapes = np.asarray(old_basis.mode_shapes, dtype=np.float64)
    new_shapes = np.asarray(new_basis.mode_shapes, dtype=np.float64)
    if old_shapes.shape[1:] != new_shapes.shape[1:]:
        raise ValueError("old/new modal grids must match")

    weights = np.asarray(
        new_basis.inner_product_weights,
        dtype=np.float64,
    ).reshape(-1)
    phi_old = old_shapes.reshape(old_basis.n_modes, -1).T
    phi_new = new_shapes.reshape(new_basis.n_modes, -1).T

    weighted_new = phi_new * weights[:, None]
    gram = phi_new.T @ weighted_new
    cross = weighted_new.T @ phi_old

    scale = max(float(np.trace(gram)) / max(new_basis.n_modes, 1), 1.0)
    gram = gram + (regularization * scale) * np.eye(new_basis.n_modes)
    return np.ascontiguousarray(
        np.linalg.solve(gram, cross),
        dtype=np.float64,
    )


def material_field_projection_error(
    new_basis: ModalBasis,
    old_basis: ModalBasis,
    transform: np.ndarray,
    old_coordinates: np.ndarray,
) -> float:
    """Relative RMS error of the projected material-coordinate displacement.

    The error is measured with the new geometry's inner product, matching the
    projection convention above. A zero old field returns zero error.
    """
    q_old = np.asarray(old_coordinates, dtype=np.float64)
    if q_old.shape != (old_basis.n_modes,):
        raise ValueError("old_coordinates has wrong shape")

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (new_basis.n_modes, old_basis.n_modes):
        raise ValueError("transform has wrong shape")

    phi_old = old_basis.mode_shapes.reshape(old_basis.n_modes, -1).T
    phi_new = new_basis.mode_shapes.reshape(new_basis.n_modes, -1).T
    q_new = transform @ q_old
    field_old = phi_old @ q_old
    field_new = phi_new @ q_new
    weights = new_basis.inner_product_weights.reshape(-1)

    residual_energy = float(np.sum(weights * (field_new - field_old) ** 2))
    reference_energy = float(np.sum(weights * field_old**2))
    if reference_energy <= 1e-30:
        return 0.0
    return float(np.sqrt(residual_energy / reference_energy))


def damping_rates(
    frequencies_hz: np.ndarray,
    alpha_g: float,
    alpha_r: float,
) -> np.ndarray:
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    return np.ascontiguousarray(
        np.exp(alpha_g + alpha_r * freqs),
        dtype=np.float64,
    )


@njit(cache=True, fastmath=True)
def apply_state_projection(
    old_states: np.ndarray,
    transform: np.ndarray,
    old_frequencies_hz: np.ndarray,
    old_damping: np.ndarray,
    new_frequencies_hz: np.ndarray,
    new_damping: np.ndarray,
    out_states: np.ndarray,
) -> None:
    old_n = old_states.shape[0]
    new_n = out_states.shape[0]
    q_old = np.empty(old_n, dtype=np.float64)
    v_old = np.empty(old_n, dtype=np.float64)

    for i in range(old_n):
        zr = old_states[i].real
        zi = old_states[i].imag
        omega = 2.0 * np.pi * old_frequencies_hz[i]
        q_old[i] = zi
        v_old[i] = omega * zr - old_damping[i] * zi

    for k in range(new_n):
        q = 0.0
        v = 0.0
        for j in range(old_n):
            weight = transform[k, j]
            q += weight * q_old[j]
            v += weight * v_old[j]

        omega_new = 2.0 * np.pi * new_frequencies_hz[k]
        real = 0.0
        if omega_new > 1e-12:
            real = (v + new_damping[k] * q) / omega_new
        out_states[k] = real + 1j * q
