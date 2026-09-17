from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plate_synth.geometry import make_geometry
from plate_synth.material_coordinates import make_material_grid
from plate_synth.modal_backend import AnalyticRectangleBackend
from plate_synth.reference.dataset import (
    DatasetConfig,
    _validate_resume_compatibility,
    build_sample_specs,
)
from plate_synth.reference.mode_sampling import canonicalize_mode_sign, detect_degenerate_groups
from plate_synth.reference.validation import modal_assurance_matrix, subspace_projection_score


def test_dataset_split_is_deterministic_disjoint_and_contains_family_anchors() -> None:
    cfg = DatasetConfig(train_count=12, val_count=5, test_count=5)
    a = build_sample_specs(cfg)
    b = build_sample_specs(cfg)
    assert a == b
    train = [(x.morph, x.shape_mod) for x in a if x.split == "train"]
    val = {(x.morph, x.shape_mod) for x in a if x.split == "val"}
    test = {(x.morph, x.shape_mod) for x in a if x.split == "test"}
    for anchor in ((0.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 1.0), (1.0, 0.0), (1.0, 1.0)):
        assert anchor in train
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert val.isdisjoint(test)
    assert all(0.0 < m < 1.0 and 0.0 < s < 1.0 for m, s in test)


def test_dataset_fingerprint_tracks_numerical_contract_not_execution_details() -> None:
    cfg = DatasetConfig(train_count=12, val_count=5, test_count=5)
    assert replace(cfg, workers=8).fingerprint() == cfg.fingerprint()
    assert replace(cfg, shard_size=17).fingerprint() == cfg.fingerprint()
    assert replace(cfg, compression_level=1).fingerprint() == cfg.fingerprint()
    assert replace(cfg, mesh_edge_length=0.04).fingerprint() != cfg.fingerprint()
    assert replace(cfg, poisson_ratio=0.25).fingerprint() != cfg.fingerprint()
    assert replace(cfg, n_modes=16, n_solve=24).fingerprint() != cfg.fingerprint()


def test_resume_rejects_incompatible_shard() -> None:
    try:
        import h5py
    except ImportError:
        return
    cfg = DatasetConfig(train_count=1, val_count=0, test_count=0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with h5py.File(root / "train_00000.h5", "w") as f:
            f.attrs["config_fingerprint"] = "wrong"
            f.create_dataset("sample_id", data=np.asarray([b"train-000000"]))
        try:
            _validate_resume_compatibility(root, cfg)
        except RuntimeError as exc:
            assert "fingerprint" in str(exc)
        else:
            raise AssertionError("incompatible shard must be rejected")


def test_degenerate_group_detection() -> None:
    lam = np.array([1.0, 1.0001, 2.0, 3.0, 3.0002])
    groups = detect_degenerate_groups(lam, relative_gap=5e-4)
    np.testing.assert_array_equal(groups, np.array([0, 0, 1, 2, 2], dtype=np.int16))


def test_canonical_mode_sign() -> None:
    mode = np.zeros((5, 5))
    mask = np.ones((5, 5), dtype=bool)
    mode[2, 3] = -4.0
    mode[1, 1] = 2.0
    fixed = canonicalize_mode_sign(mode, mask)
    assert fixed[2, 3] == 4.0


def test_mac_is_sign_invariant() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 8, 8))
    weights = np.ones((8, 8)) / 64.0
    mac = modal_assurance_matrix(x, -x, weights)
    np.testing.assert_allclose(mac[0, 0], 1.0, atol=1e-12)


def test_subspace_score_is_basis_rotation_invariant() -> None:
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(2, 12, 12))
    weights = np.ones((12, 12), dtype=np.float64) / 144.0
    # Weighted Gram-Schmidt gives a clean two-dimensional reference subspace.
    a = raw.reshape(2, -1)
    a[0] /= np.sqrt(np.sum(weights.ravel() * a[0] ** 2))
    a[1] -= np.sum(weights.ravel() * a[1] * a[0]) * a[0]
    a[1] /= np.sqrt(np.sum(weights.ravel() * a[1] ** 2))
    a = a.reshape(2, 12, 12)
    angle = 0.71
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    b = (rotation @ a.reshape(2, -1)).reshape(2, 12, 12)
    score = subspace_projection_score(a, b, weights)
    np.testing.assert_allclose(score, 1.0, atol=1e-12)


def test_material_grid_exposes_unrenormalized_quadrature_error() -> None:
    geometry = make_geometry(0.83, 0.91, grid_size=32, boundary_samples=360)
    grid = make_material_grid(geometry, 64)
    assert grid.raw_area_integral > 0.0
    expected = abs(grid.raw_area_integral - geometry.area) / geometry.area
    np.testing.assert_allclose(grid.raw_area_relative_error, expected, atol=1e-15)
    np.testing.assert_allclose(np.sum(grid.area_weights), geometry.area, atol=1e-12)
    np.testing.assert_allclose(np.sum(grid.inner_product_weights), 1.0, atol=1e-12)


def test_optional_fem_rectangle_aspect4_and_nonrectangle() -> None:
    if os.environ.get("PLATE_SYNTH_RUN_FEM_TESTS") != "1":
        return
    from plate_synth.reference.mesh import mesh_geometry
    from plate_synth.reference.mode_sampling import sample_modes_on_material_grid
    from plate_synth.reference.plate_solver import PlateSolverConfig, solve_plate_modes

    for morph, shape_mod in ((0.0, 1.0), (0.5, 0.4)):
        geometry = make_geometry(morph, shape_mod, grid_size=32, boundary_samples=180)
        mesh = mesh_geometry(geometry, target_edge_length=0.13)
        assert mesh.relative_area_error < 0.03
        assert mesh.boundary_hausdorff_approx < 0.05
        solution = solve_plate_modes(
            mesh,
            PlateSolverConfig(n_modes=4, n_solve=6, eigensolver_tolerance=1e-9),
        )
        sampled = sample_modes_on_material_grid(solution, geometry, n_modes=4, grid_size=32)
        assert sampled.mode_shapes.shape == (4, 32, 32)
        assert np.all(np.isfinite(sampled.mode_shapes))

        if morph == 0.0:
            analytic = AnalyticRectangleBackend(32).predict(geometry, 4, "simply_supported")
            rel = np.abs(sampled.modal_factors / analytic.modal_factors - 1.0)
            assert np.max(rel) < 0.10  # coarse-mesh integration smoke tolerance


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(len(tests), "Phase-3 tests passed")


if __name__ == "__main__":
    main()
