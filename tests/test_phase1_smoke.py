"""Minimal regression checks for the Phase-1 refactor.

Run from the repository root:
    python tests/test_phase1_smoke.py

These tests deliberately avoid opening an audio device.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plate_synth.config import SynthParameters
from plate_synth.modal_backend import LegacyRectangleBackend
from plate_synth.resonator_core import (
    build_distribution_matrix,
    build_resonator_poles,
    process_block,
)


def test_legacy_mode_count_and_frequency_formula() -> None:
    params = SynthParameters(
        legacy_mode_order=10,
        length_x_m=1.0,
        length_y_m=1.0,
        flexural_rigidity=18_300.0,
        density=7_800.0,
        thickness_m=0.01,
    )
    basis = LegacyRectangleBackend(shape_grid_size=32).predict(params)

    assert basis.n_modes == 10 * 9

    first_factor = 1.0**2 + 1.0 * 1.0**2
    omega = (
        np.pi**2
        / (params.length_x_m * params.length_y_m)
        * np.sqrt(
            params.flexural_rigidity
            / (params.density * params.thickness_m)
        )
        * first_factor
    )
    expected_hz = omega / (2.0 * np.pi)

    np.testing.assert_allclose(
        basis.frequencies_hz[0],
        expected_hz,
        rtol=1e-13,
        atol=0.0,
    )


def test_legacy_spatial_weight_matches_v12_expression() -> None:
    params = SynthParameters(legacy_mode_order=4)
    basis = LegacyRectangleBackend(shape_grid_size=32).predict(params)

    x = 0.31
    y = 0.67
    actual = basis.sample_weights(x, y)

    l = basis.legacy_mode_indices[:, 0]
    m = basis.legacy_mode_indices[:, 1]
    expected = np.sin(l * np.pi * x) * np.sin(m * np.pi * y)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_pickup_compatibility_path_is_equal_weight_sum() -> None:
    params = SynthParameters(legacy_mode_order=3)
    basis = LegacyRectangleBackend(shape_grid_size=16).predict(params)
    n = basis.n_modes

    poles = build_resonator_poles(
        basis.frequencies_hz,
        params.alpha_g,
        params.alpha_r,
        params.sample_rate,
    )
    coupling = build_distribution_matrix(
        basis.frequencies_hz,
        params.eta,
        params.lamb,
    )

    states = np.zeros(n, dtype=np.complex128)
    excitation = np.zeros((n, 8), dtype=np.float64)
    excitation[:, 0] = 0.01
    output = np.zeros(8, dtype=np.float64)
    transfer = np.zeros(n, dtype=np.float64)
    pickup = np.ones(n, dtype=np.float64)

    process_block(
        states,
        poles,
        coupling,
        excitation,
        pickup,
        params.tau,
        params.max_state_magnitude,
        output,
        transfer,
    )

    assert np.all(np.isfinite(output))
    assert output.shape == (8,)


if __name__ == "__main__":
    test_legacy_mode_count_and_frequency_formula()
    test_legacy_spatial_weight_matches_v12_expression()
    test_pickup_compatibility_path_is_equal_weight_sum()
    print("Phase-1 smoke tests passed.")
