from __future__ import annotations

import numpy as np
from numba import njit

from .modal_backend import ModalBasis


def projection_matrix(new_basis: ModalBasis, old_basis: ModalBasis, regularization: float = 1e-8) -> np.ndarray:
    """Weighted least-squares mapping from old to new modal coordinates."""
    old_shapes = np.asarray(old_basis.mode_shapes, dtype=np.float64)
    new_shapes = np.asarray(new_basis.mode_shapes, dtype=np.float64)
    if old_shapes.shape[1:] != new_shapes.shape[1:]:
        raise ValueError("old/new modal grids must match")
    weights = np.asarray(new_basis.integration_weights, dtype=np.float64).reshape(-1)
    phi_old = old_shapes.reshape(old_basis.n_modes, -1).T
    phi_new = new_shapes.reshape(new_basis.n_modes, -1).T
    weighted_new = phi_new * weights[:, None]
    gram = phi_new.T @ weighted_new
    cross = weighted_new.T @ phi_old
    scale = max(float(np.trace(gram)) / max(new_basis.n_modes, 1), 1.0)
    gram = gram + (regularization * scale) * np.eye(new_basis.n_modes)
    return np.ascontiguousarray(np.linalg.solve(gram, cross), dtype=np.float64)


def damping_rates(frequencies_hz: np.ndarray, alpha_g: float, alpha_r: float) -> np.ndarray:
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    return np.ascontiguousarray(np.exp(alpha_g + alpha_r * freqs), dtype=np.float64)


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
