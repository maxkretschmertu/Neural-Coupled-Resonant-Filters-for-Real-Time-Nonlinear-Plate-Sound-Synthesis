from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plate_synth.config import SynthParameters
from plate_synth.geometry import make_geometry
from plate_synth.material_coordinates import make_material_grid, material_to_physical, physical_to_material
from plate_synth.modal_backend import AnalyticRectangleBackend, UnsupportedGeometryError, resolve_modal_frequencies_hz
from plate_synth.state_projection import projection_matrix
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
    w0 = np.ptp(g0.boundary_xy[:, 0]); h0 = np.ptp(g0.boundary_xy[:, 1])
    w1 = np.ptp(g1.boundary_xy[:, 0]); h1 = np.ptp(g1.boundary_xy[:, 1])
    assert w1 / h1 > w0 / h0 * 3.0


def test_material_mapping_roundtrip():
    g = make_geometry(0.77, 0.63, grid_size=32, boundary_samples=360)
    for u, v in ((0.0, 0.0), (0.2, 0.3), (-0.5, 0.2), (0.7, -0.1)):
        x, y = material_to_physical(g, u, v)
        uu, vv = physical_to_material(g, x, y)
        np.testing.assert_allclose([uu, vv], [u, v], atol=1e-8)


def test_material_weights_integrate_unit_area():
    g = make_geometry(0.31, 0.8, grid_size=32, boundary_samples=360)
    mg = make_material_grid(g, 64)
    np.testing.assert_allclose(np.sum(mg.integration_weights), 1.0, atol=1e-12)


def test_rectangle_backend_is_clean_and_sorted():
    g = make_geometry(0.0, 0.4, grid_size=32, boundary_samples=360)
    b = AnalyticRectangleBackend(32).predict(g, 16)
    assert b.n_modes == 16
    assert np.all(np.diff(b.modal_factors) >= -1e-12)
    assert b.mode_shapes.shape == (16, 32, 32)
    params = SynthParameters(morph=0.0, shape_mod=0.4, n_modes=16)
    f = resolve_modal_frequencies_hz(b, params)
    assert np.all(f > 0)


def test_nonrectangle_not_faked():
    g = make_geometry(0.5, 0.0, grid_size=32, boundary_samples=360)
    try:
        AnalyticRectangleBackend(32).predict(g, 8)
    except UnsupportedGeometryError:
        return
    raise AssertionError("non-rectangle geometry must not receive fake modes")


def test_projection_identity():
    g = make_geometry(0.0, 0.2, grid_size=32, boundary_samples=360)
    b = AnalyticRectangleBackend(32).predict(g, 8)
    p = projection_matrix(b, b)
    np.testing.assert_allclose(p, np.eye(8), atol=2e-6, rtol=2e-6)


def test_projection_between_rectangle_shapes():
    backend = AnalyticRectangleBackend(32)
    a = backend.predict(make_geometry(0.0, 0.0, 32, 360), 8)
    b = backend.predict(make_geometry(0.0, 0.7, 32, 360), 8)
    p = projection_matrix(b, a)
    assert p.shape == (8, 8)
    assert np.all(np.isfinite(p))


def test_engine_geometry_preview_and_backend_dependency():
    class CountingBackend:
        def __init__(self):
            self.calls = 0
            self.inner = AnalyticRectangleBackend(32)
        def predict(self, geometry, n_modes):
            self.calls += 1
            return self.inner.predict(geometry, n_modes)

    cb = CountingBackend()
    e = AudioEngine(SynthParameters(n_modes=8), modal_backend=cb, geometry_grid_size=32, mode_grid_size=32)
    assert cb.calls == 1
    e.update_parameters(size_m=1.2)
    assert cb.calls == 1
    e.update_parameters(alpha_g=0.5)
    assert cb.calls == 1
    e.update_parameters(shape_mod=0.5)
    assert cb.calls == 2
    e.update_parameters(morph=0.5)
    assert cb.calls == 3
    assert abs(e.current_geometry.morph - 0.5) < 1e-12
    assert "No trained neural modal backend" in e.model_status


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(len(tests), "tests passed")


if __name__ == "__main__":
    main()
