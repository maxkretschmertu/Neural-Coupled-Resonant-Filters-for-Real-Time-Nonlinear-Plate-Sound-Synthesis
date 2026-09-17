#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plate_synth.geometry import make_geometry
from plate_synth.modal_backend import AnalyticRectangleBackend
from plate_synth.reference.mesh import mesh_geometry
from plate_synth.reference.mode_sampling import sample_modes_on_material_grid
from plate_synth.reference.plate_solver import PlateSolverConfig, solve_plate_modes
from plate_synth.reference.validation import modal_assurance_matrix


def shape_mod_for_aspect(aspect: float) -> float:
    if not 1.0 <= aspect <= 4.0:
        raise ValueError("validation aspect must lie in [1, 4]")
    return float(np.log(aspect) / np.log(4.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Morley plate FEM against analytic rectangles")
    parser.add_argument("--output", type=Path, default=Path("validation/phase3"))
    parser.add_argument("--n-modes", type=int, default=16)
    parser.add_argument("--n-solve", type=int, default=24)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--boundary-samples", type=int, default=720)
    parser.add_argument("--poisson", type=float, default=0.30)
    parser.add_argument("--aspects", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--h", type=float, nargs="+", default=[0.12, 0.08, 0.055])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    freq_rows: list[dict[str, float | int]] = []
    convergence_rows: list[dict[str, float | int]] = []

    for aspect in args.aspects:
        shape_mod = shape_mod_for_aspect(aspect)
        geometry = make_geometry(0.0, shape_mod, args.grid, args.boundary_samples)
        analytic = AnalyticRectangleBackend(args.grid).predict(
            geometry, args.n_modes, "simply_supported"
        )
        by_h: dict[float, np.ndarray] = {}

        for h in args.h:
            mesh = mesh_geometry(geometry, h)
            solution = solve_plate_modes(
                mesh,
                PlateSolverConfig(
                    n_modes=args.n_modes,
                    n_solve=args.n_solve,
                    poisson_ratio=args.poisson,
                    boundary_condition="simply_supported",
                ),
            )
            sampled = sample_modes_on_material_grid(
                solution, geometry, n_modes=args.n_modes, grid_size=args.grid
            )
            by_h[h] = sampled.modal_factors.copy()

            mac = modal_assurance_matrix(
                sampled.mode_shapes,
                analytic.mode_shapes,
                sampled.inner_product_weights,
            )
            # Hungarian assignment handles exact/near degeneracies without
            # assuming the eigensolver returns the same basis ordering.
            from scipy.optimize import linear_sum_assignment

            rel = np.abs(
                sampled.modal_factors[:, None] / analytic.modal_factors[None, :] - 1.0
            )
            cost = rel + 0.15 * (1.0 - mac)
            rows, cols = linear_sum_assignment(cost)
            order = np.argsort(cols)
            rows, cols = rows[order], cols[order]

            for fem_i, analytic_i in zip(rows, cols):
                freq_rows.append(
                    {
                        "aspect": aspect,
                        "h_target": h,
                        "mode": int(analytic_i),
                        "analytic_factor": float(analytic.modal_factors[analytic_i]),
                        "fem_factor": float(sampled.modal_factors[fem_i]),
                        "relative_error": float(rel[fem_i, analytic_i]),
                        "mac": float(mac[fem_i, analytic_i]),
                        "residual": float(sampled.residuals[fem_i]),
                        "mesh_vertices": mesh.n_vertices,
                        "mesh_triangles": mesh.n_triangles,
                    }
                )

        finest_h = min(by_h)
        finest = by_h[finest_h]
        for h, factors in by_h.items():
            if h == finest_h:
                continue
            rel = np.abs(factors / finest - 1.0)
            for mode, error in enumerate(rel):
                convergence_rows.append(
                    {
                        "aspect": aspect,
                        "h_target": h,
                        "reference_h": finest_h,
                        "mode": mode,
                        "relative_change": float(error),
                    }
                )

    for name, rows in (
        ("rectangle_frequency_mac.csv", freq_rows),
        ("mesh_convergence.csv", convergence_rows),
    ):
        path = args.output / name
        if not rows:
            continue
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(path)

    errors = np.asarray([row["relative_error"] for row in freq_rows], dtype=float)
    macs = np.asarray([row["mac"] for row in freq_rows], dtype=float)
    print(f"frequency relative error: median={np.median(errors):.3e}, max={np.max(errors):.3e}")
    print(f"MAC: median={np.median(macs):.6f}, min={np.min(macs):.6f}")
