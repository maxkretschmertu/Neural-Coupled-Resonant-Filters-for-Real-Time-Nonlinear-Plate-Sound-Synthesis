from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plate_synth.config import SynthParameters
from plate_synth.geometry import make_geometry
from plate_synth.material_coordinates import (
    make_material_grid,
    material_to_physical,
    physical_to_material,
)
from plate_synth.modal_backend import (
    AnalyticRectangleBackend,
    ModalBasis,
    UnsupportedGeometryError,
    physical_mass_weights,
    resolve_modal_frequencies_hz,
)
from plate_synth.state_projection import (
    material_field_projection_error,
    projection_matrix,
)
from plate_synth.audio_engine import AudioEngine


def test_endpoints_and_area():
    for m in (0.0, 0.5, 1.0):
        g = make_geometry(m, 0.0, grid_size=32, boundary_samples=360)
        assert abs(g.area - 1.0) < 2e-3
        assert g.sdf[g.grid_size // 2, g.grid_size // 2] < 0
        assert g.sdf[0, 0] > 0


def test_shape_mod_changes_slenderness():
    g0 = make_geometry(0.0, 0.0, grid_size=32, boundary_samples=360)
    g1 = make_geometry(0.0, 1.0, grid_size=32, boundary_samples=360)
    w0 = np.ptp(g0.boundary_xy[:, 0])
    h0 = np.ptp(g0.boundary_xy[:, 1])
    w1 = np.ptp(g1.boundary_xy[:, 0])
    h1 = np.ptp(g1.boundary_xy[:, 1])
    assert w1 / h1 > w0 / h0 * 3.0


def test_material_mapping_roundtrip():
    g = make_geometry(0.77, 0.63, grid_size=32, boundary_samples=360)
    for u, v in ((0.0, 0.0), (0.2, 0.3), (-0.5, 0.2), (0.7, -0.1)):
        x, y = material_to_physical(g, u, v)
        uu, vv = physical_to_material(g, x, y)
        np.testing.assert_allclose([uu, vv], [u, v], atol=1e-8)


def test_material_weight_semantics_and_physical_mass():
    g = make_geometry(0.31, 0.8, grid_size=32, boundary_samples=360)
    mg = make_material_grid(g, 64)
    np.testing.assert_allclose(np.sum(mg.area_weights), g.area, atol=1e-12)
    np.testing.assert_allclose(np.sum(mg.inner_product_weights), 1.0, atol=1e-12)

    params = SynthParameters(
        morph=0.0,
        shape_mod=0.3,
        n_modes=8,
        size_m=0.7,
        density=7800.0,
        thickness_m=0.01,
    )
    rect = make_geometry(0.0, 0.3, grid_size=32, boundary_samples=360)
    basis = AnalyticRectangleBackend(32).predict(
        rect,
        params.n_modes,
        params.boundary_condition,
    )
    mass = physical_mass_weights(basis, params)
    expected = params.density * params.thickness_m * params.size_m**2 * rect.area
    np.testing.assert_allclose(np.sum(mass), expected, rtol=1e-12, atol=1e-12)


def _brute_rectangle_factors(shape_mod: float, n_modes: int, max_index: int = 64):
    aspect = 4.0 ** shape_mod
    lx = np.sqrt(aspect)
    ly = 1.0 / np.sqrt(aspect)
    factors = [
        np.pi**2 * (m*m/(lx*lx) + n*n/(ly*ly))
        for m in range(1, max_index + 1)
        for n in range(1, max_index + 1)
    ]
    return np.sort(np.asarray(factors, dtype=np.float64))[:n_modes]


def test_rectangle_backend_returns_globally_lowest_modes():
    backend = AnalyticRectangleBackend(32)
    for shape_mod in (0.0, 0.5, 1.0):
        g = make_geometry(0.0, shape_mod, grid_size=32, boundary_samples=360)
        basis = backend.predict(g, 64, "simply_supported")
        expected = _brute_rectangle_factors(shape_mod, 64)
        np.testing.assert_allclose(
            basis.modal_factors,
            expected,
            rtol=1e-13,
            atol=1e-13,
        )


def test_rectangle_backend_is_sorted_and_frequency_positive():
    g = make_geometry(0.0, 0.4, grid_size=32, boundary_samples=360)
    b = AnalyticRectangleBackend(32).predict(g, 16, "simply_supported")
    assert b.n_modes == 16
    assert np.all(np.diff(b.modal_factors) >= -1e-12)
    assert b.mode_shapes.shape == (16, 32, 32)
    params = SynthParameters(morph=0.0, shape_mod=0.4, n_modes=16)
    f = resolve_modal_frequencies_hz(b, params)
    assert np.all(f > 0)


def test_nonrectangle_not_faked():
    g = make_geometry(0.5, 0.0, grid_size=32, boundary_samples=360)
    try:
        AnalyticRectangleBackend(32).predict(g, 8, "simply_supported")
    except UnsupportedGeometryError:
        return
    raise AssertionError("non-rectangle geometry must not receive fake modes")


def test_projection_identity():
    g = make_geometry(0.0, 0.2, grid_size=32, boundary_samples=360)
    b = AnalyticRectangleBackend(32).predict(g, 8, "simply_supported")
    p = projection_matrix(b, b)
    np.testing.assert_allclose(p, np.eye(8), atol=2e-6, rtol=2e-6)


def test_projection_exact_under_mode_permutation():
    g = make_geometry(0.0, 0.2, grid_size=32, boundary_samples=360)
    base = AnalyticRectangleBackend(32).predict(g, 8, "simply_supported")
    permutation = np.array([3, 0, 7, 2, 5, 1, 6, 4])
    permuted = ModalBasis(
        modal_factors=base.modal_factors[permutation],
        mode_shapes=base.mode_shapes[permutation],
        area_weights=base.area_weights,
        inner_product_weights=base.inner_product_weights,
        geometry=base.geometry,
        boundary_condition=base.boundary_condition,
    )
    p = projection_matrix(permuted, base)
    q = np.array([0.7, -0.2, 0.1, 0.4, -0.5, 0.8, -0.1, 0.3])
    error = material_field_projection_error(permuted, base, p, q)
    assert error < 3e-6


def test_projection_between_rectangle_shapes_reduces_material_field_error():
    backend = AnalyticRectangleBackend(32)
    old = backend.predict(
        make_geometry(0.0, 0.0, 32, 360),
        12,
        "simply_supported",
    )
    new = backend.predict(
        make_geometry(0.0, 0.7, 32, 360),
        12,
        "simply_supported",
    )
    p = projection_matrix(new, old)
    q = np.linspace(-0.6, 0.8, old.n_modes)
    error = material_field_projection_error(new, old, p, q)
    assert np.isfinite(error)
    # Weighted least squares cannot be worse than representing the field by zero,
    # whose normalized relative RMS error is exactly one.
    assert error <= 1.0001


def test_engine_backend_dependency_and_preview_freeze():
    class CountingBackend:
        def __init__(self):
            self.calls = 0
            self.inner = AnalyticRectangleBackend(32)

        def predict(self, geometry, n_modes, boundary_condition):
            self.calls += 1
            return self.inner.predict(geometry, n_modes, boundary_condition)

    cb = CountingBackend()
    e = AudioEngine(
        SynthParameters(n_modes=8),
        modal_backend=cb,
        geometry_grid_size=32,
        mode_grid_size=32,
    )
    assert cb.calls == 1

    baseline = e.current_frequencies_hz.copy()
    e.update_parameters(size_m=1.2)
    assert cb.calls == 1
    e.update_parameters(alpha_g=0.5)
    assert cb.calls == 1
    e.update_parameters(shape_mod=0.5)
    assert cb.calls == 2

    # Consume the valid rectangle update so the active-audio geometry is clear.
    out = np.zeros((32, 1), dtype=np.float64)
    e._callback(out, 32, None, None)
    assert abs(e.active_audio_geometry.shape_mod - 0.5) < 1e-12

    valid_freqs = e.current_frequencies_hz.copy()
    e.update_parameters(morph=0.5)
    assert cb.calls == 3
    assert not e.control_model_valid
    assert abs(e.current_geometry.morph - 0.5) < 1e-12
    assert abs(e.active_audio_geometry.morph - 0.0) < 1e-12
    np.testing.assert_allclose(e.current_frequencies_hz, valid_freqs)

    # Audio-affecting controls are accepted as preview state but must not be
    # applied to the old rectangle audio while the preview has no modal model.
    e.update_parameters(size_m=1.5, eta=0.03, strike_u=0.2)
    np.testing.assert_allclose(e.current_frequencies_hz, valid_freqs)
    assert not e.control_model_valid

    # Returning to a supported geometry publishes all accumulated controls.
    e.update_parameters(morph=0.0)
    assert e.control_model_valid
    restored = e.current_frequencies_hz
    assert np.max(np.abs(restored - valid_freqs)) > 1e-6


def test_boundary_condition_is_explicit_and_fixed():
    try:
        AudioEngine(
            SynthParameters(boundary_condition="free", n_modes=4),
            geometry_grid_size=32,
            mode_grid_size=32,
        )
    except ValueError:
        return
    raise AssertionError("unsupported boundary condition must be rejected")


def main():
    tests = [
        v
        for k, v in globals().items()
        if k.startswith("test_") and callable(v)
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(len(tests), "tests passed")


if __name__ == "__main__":
    main()
