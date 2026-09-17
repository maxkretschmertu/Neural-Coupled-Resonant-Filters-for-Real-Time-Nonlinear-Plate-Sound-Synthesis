from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plate_synth.geometry import make_geometry
from plate_synth.modal_backend import AnalyticRectangleBackend
from plate_synth.reference.dataset import DatasetConfig, build_sample_specs
from plate_synth.reference.mode_sampling import canonicalize_mode_sign, detect_degenerate_groups
from plate_synth.reference.validation import modal_assurance_matrix


def test_dataset_split_is_deterministic_and_contains_family_anchors() -> None:
    cfg = DatasetConfig(train_count=12, val_count=5, test_count=5)
    a = build_sample_specs(cfg)
    b = build_sample_specs(cfg)
    assert a == b
    train = [(x.morph, x.shape_mod) for x in a if x.split == "train"]
    for anchor in ((0.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 1.0), (1.0, 0.0), (1.0, 1.0)):
        assert anchor in train
    test = [x for x in a if x.split == "test"]
    assert all(0.0 < x.morph < 1.0 and 0.0 < x.shape_mod < 1.0 for x in test)


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


def test_optional_fem_rectangle_against_analytic() -> None:
    if os.environ.get("PLATE_SYNTH_RUN_FEM_TESTS") != "1":
        return
    from plate_synth.reference.mesh import mesh_geometry
    from plate_synth.reference.mode_sampling import sample_modes_on_material_grid
    from plate_synth.reference.plate_solver import PlateSolverConfig, solve_plate_modes

    geometry = make_geometry(0.0, 0.0, grid_size=32, boundary_samples=180)
    mesh = mesh_geometry(geometry, target_edge_length=0.13)
    solution = solve_plate_modes(
        mesh,
        PlateSolverConfig(n_modes=4, n_solve=6, eigensolver_tolerance=1e-9),
    )
    sampled = sample_modes_on_material_grid(solution, geometry, n_modes=4, grid_size=32)
    analytic = AnalyticRectangleBackend(32).predict(geometry, 4, "simply_supported")
    rel = np.abs(sampled.modal_factors / analytic.modal_factors - 1.0)
    assert np.max(rel) < 0.08  # deliberately loose coarse-mesh smoke tolerance


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(len(tests), "Phase-3 tests passed")


if __name__ == "__main__":
    main()
