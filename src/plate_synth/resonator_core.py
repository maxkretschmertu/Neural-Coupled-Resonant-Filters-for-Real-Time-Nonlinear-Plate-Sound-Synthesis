from __future__ import annotations

import numpy as np
from numba import njit

from .config import SynthParameters


def build_distribution_matrix(frequencies_hz: np.ndarray, eta: float, lamb: float) -> np.ndarray:
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    n = len(freqs)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)
    diff = np.abs(freqs[:, None] - freqs[None, :])
    a = 1.0 - diff / np.mean(freqs)
    np.fill_diagonal(a, 0.0)
    denom = np.sum(a, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return np.ascontiguousarray(eta * lamb * a / denom - lamb * np.eye(n), dtype=np.float64)


def build_resonator_poles(frequencies_hz: np.ndarray, alpha_g: float, alpha_r: float, sample_rate: float) -> np.ndarray:
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    alphas = np.exp(alpha_g + alpha_r * freqs)
    poles = np.exp(-alphas / sample_rate) * np.exp(1j * 2.0 * np.pi * freqs / sample_rate)
    return np.ascontiguousarray(poles, dtype=np.complex128)


def make_excitation_block(sample_positions: np.ndarray, impact_start: int | None, mode_weights: np.ndarray, params: SynthParameters) -> np.ndarray:
    positions = np.asarray(sample_positions, dtype=np.float64)
    weights = np.asarray(mode_weights, dtype=np.float64)
    n_modes, n_frames = weights.shape[0], positions.shape[0]
    if impact_start is None:
        return np.zeros((n_modes, n_frames), dtype=np.float64)
    relative = positions - float(impact_start)
    if params.excitation_mode == 0:
        pulse = np.zeros_like(relative)
        length = int(params.excitation_length_samples)
        inside = (relative >= 0.0) & (relative < length)
        scale = 2.0 / length if length > 0 else 1.0
        pulse[inside] = params.excitation_amplitude * scale * np.sin(np.pi * relative[inside] / length) ** 2
    else:
        pulse = ((relative >= 0.0) & (relative.astype(np.int64) % int(params.sample_rate) == 0)).astype(np.float64)
        pulse *= params.excitation_amplitude
    return np.ascontiguousarray(weights[:, None] * pulse[None, :], dtype=np.float64)


@njit(cache=True, fastmath=True)
def process_block(states, poles, coupling_matrix, excitation, pickup_weights, tau, max_state_magnitude, output, last_transfer):
    n_modes = states.shape[0]
    n_frames = excitation.shape[1]
    rectified = np.empty(n_modes, dtype=np.float64)
    transfer = np.empty(n_modes, dtype=np.float64)
    for i in range(n_frames):
        for k in range(n_modes):
            zr, zi = states[k].real, states[k].imag
            r = 0.5 * (zr * zr + zi * zi) - tau
            rectified[k] = r if r > 0.0 else 0.0
        for k in range(n_modes):
            acc = 0.0
            for j in range(n_modes):
                acc += coupling_matrix[k, j] * rectified[j]
            transfer[k] = acc if acc > 0.0 else 0.0
        for k in range(n_modes):
            zr, zi = states[k].real, states[k].imag
            mag2 = zr * zr + zi * zi
            if mag2 < 1e-20:
                new_state = np.sqrt(2.0 * transfer[k]) * poles[k] + excitation[k, i]
            else:
                gain = np.sqrt(1.0 + 2.0 * transfer[k] / mag2)
                new_state = gain * poles[k] * states[k] + excitation[k, i]
            mag = np.sqrt(new_state.real * new_state.real + new_state.imag * new_state.imag)
            if mag > max_state_magnitude:
                compressed = max_state_magnitude * np.tanh(mag / max_state_magnitude)
                new_state = new_state * (compressed / mag)
            states[k] = new_state
        sample = 0.0
        for k in range(n_modes):
            sample += pickup_weights[k] * states[k].imag
        output[i] = sample
        if i == n_frames - 1:
            for k in range(n_modes):
                last_transfer[k] = transfer[k]


def warm_up_numba(n_modes: int) -> None:
    n = max(int(n_modes), 1)
    process_block(
        np.zeros(n, np.complex128), np.ones(n, np.complex128), np.zeros((n, n)),
        np.zeros((n, 4)), np.ones(n), 1.0, 10.0, np.zeros(4), np.zeros(n)
    )
