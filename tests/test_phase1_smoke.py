"""Regression and architecture checks for the Phase-1 refactor.

Run from repository root:
    python tests/test_phase1_smoke.py
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plate_synth.audio_engine import AudioEngine
from plate_synth.config import SynthParameters
from plate_synth.geometry import make_rectangle_geometry
from plate_synth.modal_backend import LegacyRectangleBackend, resolve_modal_frequencies_hz
from plate_synth.resonator_core import (
    build_distribution_matrix,
    build_resonator_poles,
    process_block,
)


class CountingBackend:
    def __init__(self) -> None:
        self.inner = LegacyRectangleBackend(shape_grid_size=32)
        self.calls = 0

    def predict(self, geometry, n_modes):
        self.calls += 1
        return self.inner.predict(geometry, n_modes)


def _v12_reference_process(states, poles, coupling, excitation, tau, max_mag):
    """Small pure-Python transcription of the historical v12 DSP loop."""
    states = states.copy()
    n, frames = excitation.shape
    out = np.zeros(frames, dtype=np.float64)
    for i in range(frames):
        p = 0.5 * np.abs(states) ** 2
        rectified = np.maximum(p - tau, 0.0)
        transfer = np.maximum(coupling @ rectified, 0.0)
        for k in range(n):
            mag2 = states[k].real**2 + states[k].imag**2
            if mag2 < 1e-20:
                new_z = np.sqrt(2.0 * transfer[k]) * poles[k] + excitation[k, i]
            else:
                gain = np.sqrt(1.0 + 2.0 * transfer[k] / mag2)
                new_z = gain * poles[k] * states[k] + excitation[k, i]
            mag = abs(new_z)
            if mag > max_mag:
                compressed = max_mag * np.tanh(mag / max_mag)
                new_z *= compressed / mag
            states[k] = new_z
        out[i] = np.sum(states.imag)
    return states, out


def test_geometry_only_backend_and_v12_frequency_formula() -> None:
    params = SynthParameters(n_modes=90)
    geometry = make_rectangle_geometry(params.length_x_m, params.length_y_m, 32)
    basis = LegacyRectangleBackend(shape_grid_size=32).predict(geometry, params.n_modes)
    assert basis.n_modes == 90
    assert basis.legacy_mode_indices.shape == (90, 2)

    freqs = resolve_modal_frequencies_hz(basis, params)
    l, m = basis.legacy_mode_indices[0]
    v = params.length_x_m / params.length_y_m
    omega = (
        np.pi**2 / (params.length_x_m * params.length_y_m)
        * np.sqrt(params.flexural_rigidity / (params.density * params.thickness_m))
        * (l**2 + v * m**2)
    )
    np.testing.assert_allclose(freqs[0], omega / (2*np.pi), rtol=1e-13)


def test_rectangle_field_is_actual_sdf_not_dummy_constant() -> None:
    geometry = make_rectangle_geometry(2.0, 1.0, 33)
    sdf = geometry.sdf
    assert sdf[16, 16] < 0.0
    assert sdf[0, 0] > 0.0
    assert np.ptp(sdf) > 0.1


def test_legacy_spatial_weights_match_v12() -> None:
    params = SynthParameters(n_modes=12)
    geometry = make_rectangle_geometry(1.0, 1.0, 32)
    basis = LegacyRectangleBackend(32).predict(geometry, params.n_modes)
    x, y = 0.31, 0.67
    actual = basis.sample_weights(x, y)
    l = basis.legacy_mode_indices[:, 0]
    m = basis.legacy_mode_indices[:, 1]
    expected = np.sin(l*np.pi*x) * np.sin(m*np.pi*y)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_numba_core_matches_v12_reference_render() -> None:
    params = SynthParameters(n_modes=6, tau=0.1)
    geometry = make_rectangle_geometry(1.0, 1.0, 16)
    basis = LegacyRectangleBackend(16).predict(geometry, params.n_modes)
    freqs = resolve_modal_frequencies_hz(basis, params)
    poles = build_resonator_poles(freqs, params.alpha_g, params.alpha_r, params.sample_rate)
    coupling = build_distribution_matrix(freqs, params.eta, params.lamb)

    frames = 64
    excitation = np.zeros((basis.n_modes, frames), dtype=np.float64)
    excitation[:, :4] = np.array([0.02, 0.03, 0.01, 0.0])[None, :]
    initial = np.zeros(basis.n_modes, dtype=np.complex128)

    ref_states, ref_out = _v12_reference_process(
        initial, poles, coupling, excitation, params.tau, params.max_state_magnitude
    )
    states = initial.copy()
    out = np.zeros(frames, dtype=np.float64)
    transfer = np.zeros(basis.n_modes, dtype=np.float64)
    process_block(
        states, poles, coupling, excitation, np.ones(basis.n_modes),
        params.tau, params.max_state_magnitude, out, transfer,
    )
    np.testing.assert_allclose(out, ref_out, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(states, ref_states, rtol=1e-12, atol=1e-12)


def test_dependency_layers_do_not_rerun_backend_for_material_or_damping() -> None:
    backend = CountingBackend()
    engine = AudioEngine(SynthParameters(n_modes=8), modal_backend=backend, shape_grid_size=32)
    assert backend.calls == 1

    engine.update_parameters(flexural_rigidity=20_000.0)
    assert backend.calls == 1
    engine.update_parameters(alpha_g=0.5)
    assert backend.calls == 1
    engine.update_parameters(eta=0.02)
    assert backend.calls == 1
    engine.update_parameters(strike_x=0.4)
    assert backend.calls == 1

    # Aspect-ratio change is a true geometry change.
    engine.update_parameters(length_x_m=1.2)
    assert backend.calls == 2


def test_uniform_size_change_reuses_geometry_backend() -> None:
    backend = CountingBackend()
    engine = AudioEngine(SynthParameters(n_modes=8), modal_backend=backend, shape_grid_size=32)
    engine.update_parameters(length_x_m=2.0, length_y_m=2.0)
    assert backend.calls == 1


def test_mode_count_is_explicit_and_resets_are_preallocated() -> None:
    engine = AudioEngine(SynthParameters(n_modes=8), shape_grid_size=16)
    engine.update_parameters(n_modes=12)
    pending = engine._pending_update
    assert pending is not None
    assert pending.snapshot.n_modes == 12
    assert pending.reset_states is not None
    assert pending.reset_states.shape == (12,)


def test_strike_is_scheduled_at_next_block_start() -> None:
    engine = AudioEngine(SynthParameters(n_modes=4), shape_grid_size=16)
    engine.strike()
    outdata = np.zeros((256, 1), dtype=np.float64)
    engine._callback(outdata, 256, None, None)
    assert engine._impact_start == 0
    assert engine.sample_position == 256


def test_sample_rate_cannot_change_while_engine_exists() -> None:
    engine = AudioEngine(SynthParameters(n_modes=4), shape_grid_size=16)
    try:
        engine.update_parameters(sample_rate=44_100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("sample_rate update should have been rejected")


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} Phase-1 tests passed.")


if __name__ == "__main__":
    main()
