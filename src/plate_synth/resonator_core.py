from __future__ import annotations

import numpy as np
from numba import njit

from .config import SynthParameters


def build_distribution_matrix(
    frequencies_hz: np.ndarray,
    eta: float,
    lamb: float,
) -> np.ndarray:
    """Build the same nonlinear coupling matrix used in v12.

    The existing mathematics is intentionally kept unchanged in Phase 1.
    """
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    n = len(freqs)

    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    diff = np.abs(freqs[:, None] - freqs[None, :])
    a = 1.0 - diff / np.mean(freqs)
    np.fill_diagonal(a, 0.0)

    denom = np.sum(a, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0

    matrix = eta * lamb * a / denom - lamb * np.eye(n)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def build_resonator_poles(
    frequencies_hz: np.ndarray,
    alpha_g: float,
    alpha_r: float,
    sample_rate: float,
) -> np.ndarray:
    """Build complex one-pole modal resonators exactly as in v12."""
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    alphas = np.exp(alpha_g + alpha_r * freqs)
    poles = np.exp(-alphas / sample_rate) * np.exp(
        1j * 2.0 * np.pi * freqs / sample_rate
    )
    return np.ascontiguousarray(poles, dtype=np.complex128)


def make_excitation_block(
    sample_positions: np.ndarray,
    impact_start: int | None,
    mode_weights: np.ndarray,
    params: SynthParameters,
) -> np.ndarray:
    """Create the per-mode excitation matrix used by the Numba kernel."""
    positions = np.asarray(sample_positions, dtype=np.float64)
    weights = np.asarray(mode_weights, dtype=np.float64)
    n_modes = weights.shape[0]
    n_frames = positions.shape[0]

    if impact_start is None:
        return np.zeros((n_modes, n_frames), dtype=np.float64)

    relative = positions - float(impact_start)

    if params.excitation_mode == 0:
        pulse = np.zeros_like(relative)
        inside = (relative >= 0.0) & (relative < params.excitation_length_samples)
        length = int(params.excitation_length_samples)
        scale = 2.0 / length if length > 0 else 1.0
        pulse[inside] = (
            params.excitation_amplitude
            * scale
            * np.sin(np.pi * relative[inside] / length) ** 2
        )
    else:
        pulse = (
            (relative >= 0.0)
            & (relative.astype(np.int64) % int(params.sample_rate) == 0)
        ).astype(np.float64)
        pulse *= params.excitation_amplitude

    excitation = weights[:, None] * pulse[None, :]
    return np.ascontiguousarray(excitation, dtype=np.float64)


@njit(cache=True, fastmath=True)
def process_block(
    states: np.ndarray,
    poles: np.ndarray,
    coupling_matrix: np.ndarray,
    excitation: np.ndarray,
    pickup_weights: np.ndarray,
    tau: float,
    max_state_magnitude: float,
    output: np.ndarray,
    last_transfer: np.ndarray,
) -> None:
    """Run the nonlinear coupled-resonator core for one audio block.

    The nonlinear state update is the v12 algorithm. The only Phase-1
    extension is the pickup-weighted readout. Passing all-one pickup weights
    reproduces the equal-weight v12 output.
    """
    n_modes = states.shape[0]
    n_frames = excitation.shape[1]

    rectified = np.empty(n_modes, dtype=np.float64)
    transfer = np.empty(n_modes, dtype=np.float64)

    for i in range(n_frames):
        # p_k = 0.5 |z_k|^2 and [p_k - tau]_+
        for k in range(n_modes):
            zr = states[k].real
            zi = states[k].imag
            p = 0.5 * (zr * zr + zi * zi)
            r = p - tau
            rectified[k] = r if r > 0.0 else 0.0

        # T_k = [sum_j M_kj r_j]_+
        for k in range(n_modes):
            acc = 0.0
            for j in range(n_modes):
                acc += coupling_matrix[k, j] * rectified[j]
            transfer[k] = acc if acc > 0.0 else 0.0

        # Complex modal state update
        for k in range(n_modes):
            zr = states[k].real
            zi = states[k].imag
            magnitude_squared = zr * zr + zi * zi

            if magnitude_squared < 1e-20:
                amplitude = np.sqrt(2.0 * transfer[k])
                new_state = amplitude * poles[k] + excitation[k, i]
            else:
                gain = np.sqrt(
                    1.0 + (2.0 * transfer[k]) / magnitude_squared
                )
                new_state = (
                    gain * poles[k] * states[k] + excitation[k, i]
                )

            # Preserve the v12 soft magnitude protection.
            nr = new_state.real
            ni = new_state.imag
            magnitude = np.sqrt(nr * nr + ni * ni)
            if magnitude > max_state_magnitude:
                compressed = (
                    max_state_magnitude
                    * np.tanh(magnitude / max_state_magnitude)
                )
                new_state = new_state * (compressed / magnitude)

            states[k] = new_state

        # Spatial pickup readout.
        sample = 0.0
        for k in range(n_modes):
            sample += pickup_weights[k] * states[k].imag
        output[i] = sample

        if i == n_frames - 1:
            for k in range(n_modes):
                last_transfer[k] = transfer[k]


def warm_up_numba(n_modes: int) -> None:
    """Compile the Numba kernel before the audio stream starts."""
    n = max(int(n_modes), 1)
    states = np.zeros(n, dtype=np.complex128)
    poles = np.ones(n, dtype=np.complex128)
    coupling = np.zeros((n, n), dtype=np.float64)
    excitation = np.zeros((n, 4), dtype=np.float64)
    pickup = np.ones(n, dtype=np.float64)
    output = np.zeros(4, dtype=np.float64)
    transfer = np.zeros(n, dtype=np.float64)

    process_block(
        states,
        poles,
        coupling,
        excitation,
        pickup,
        1.0,
        10.0,
        output,
        transfer,
    )
