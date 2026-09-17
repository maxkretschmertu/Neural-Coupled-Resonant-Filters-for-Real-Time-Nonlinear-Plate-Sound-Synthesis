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
from plate_synth.reference.mode_sampling import detect_degenerate_groups, sample_modes_on_material_grid
from plate_synth.reference.plate_solver import PlateSolverConfig, solve_plate_modes
from plate_synth.reference.validation import (
    match_modes_to_reference,
    reorder_match_within_reference_groups,
    subspace_projection_score,
)


def shape_mod_for_aspect(aspect: float) -> float:
    if not 1.0 <= aspect <= 4.0:
        raise ValueError("validation aspect must lie in [1, 4]")
    return float(np.log(aspect) / np.log(4.0))


def _group_indices(group_ids: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(group_ids == gid) for gid in np.unique(group_ids)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Morley plate FEM against analytic rectangles")
    parser.add_argument("--output", type=Path, default=Path("validation/phase3"))
    parser.add_argument("--n-modes", type=int, default=32)
    parser.add_argument("--n-solve", type=int, default=40)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--boundary-samples", type=int, default=720)
    parser.add_argument("--poisson", type=float, default=0.30)
    parser.add_argument("--degeneracy-gap", type=float, default=5e-4)
    parser.add_argument(
        "--comparison-group-gap",
        type=float,
        default=0.02,
        help=(
            "relative modal-factor gap below which neighbouring analytical modes "
            "inside the stored target range are judged as one validation subspace"
        ),
    )
    parser.add_argument("--matching-shape-weight", type=float, default=0.25)
    parser.add_argument("--aspects", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--h", type=float, nargs="+", default=[0.12, 0.08, 0.055])
    parser.add_argument("--low-mode-count", type=int, default=16)
    parser.add_argument("--max-rel-error-low", type=float, default=0.005)
    parser.add_argument("--max-rel-error-high", type=float, default=0.01)
    parser.add_argument("--min-shape-score", type=float, default=0.95)
    parser.add_argument("--max-convergence-change", type=float, default=0.02)
    parser.add_argument("--max-mesh-area-error", type=float, default=0.005)
    parser.add_argument("--max-boundary-distance", type=float, default=0.01)
    args = parser.parse_args()

    if args.n_solve <= args.n_modes:
        raise SystemExit("--n-solve must exceed --n-modes so cutoff degeneracy is observable")
    if len(args.h) < 2:
        raise SystemExit("provide at least two mesh sizes for convergence validation")
    if args.matching_shape_weight < 0.0:
        raise SystemExit("--matching-shape-weight must be non-negative")
    if args.comparison_group_gap < args.degeneracy_gap:
        raise SystemExit("--comparison-group-gap must be >= --degeneracy-gap")

    args.output.mkdir(parents=True, exist_ok=True)
    freq_rows: list[dict[str, float | int | str]] = []
    convergence_rows: list[dict[str, float | int]] = []
    finest_rows: list[dict[str, float | int | str]] = []
    failures: list[str] = []

    h_values = sorted(set(float(h) for h in args.h), reverse=True)
    finest_h = min(h_values)
    second_finest_h = sorted(h_values)[1]

    for aspect in args.aspects:
        shape_mod = shape_mod_for_aspect(aspect)
        geometry = make_geometry(0.0, shape_mod, args.grid, args.boundary_samples)

        # Probe one analytical mode beyond the FEM solve horizon.  This lets us
        # prove that a physical eigenspace intersecting the stored N-mode cutoff
        # is complete.  The square at N=32 is the important example: reference
        # modes 30, 31 and 32 form one three-dimensional eigenspace.  Comparing
        # only modes 30 and 31 against an arbitrary two-dimensional slice of the
        # FEM eigenspace creates a false low subspace score.
        analytic_probe = AnalyticRectangleBackend(args.grid).predict(
            geometry, args.n_solve + 1, "simply_supported"
        )
        probe_factors = np.asarray(analytic_probe.modal_factors, dtype=np.float64)
        probe_physical_groups = detect_degenerate_groups(
            probe_factors**2, args.degeneracy_gap
        )
        cutoff_group_id = probe_physical_groups[args.n_modes - 1]
        cutoff_group_indices = np.flatnonzero(
            probe_physical_groups == cutoff_group_id
        )
        cutoff_group_end = int(cutoff_group_indices[-1])
        if cutoff_group_end >= args.n_solve:
            raise SystemExit(
                f"aspect {aspect:g}: physical eigenspace crossing the N={args.n_modes} "
                f"cutoff extends beyond n_solve={args.n_solve}; increase --n-solve"
            )

        comparison_count = max(args.n_modes, cutoff_group_end + 1)
        analytic_factors = probe_factors[:comparison_count]
        analytic_modes = np.asarray(
            analytic_probe.mode_shapes[:comparison_count], dtype=np.float64
        )
        physical_groups = probe_physical_groups[:comparison_count]
        target_comparison_groups = detect_degenerate_groups(
            analytic_factors[: args.n_modes], args.comparison_group_gap
        )
        cutoff_extended = cutoff_group_end >= args.n_modes

        by_h: dict[float, np.ndarray] = {}

        for h in h_values:
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
            if solution.n_solved < comparison_count:
                raise RuntimeError(
                    f"only {solution.n_solved} FEM modes available, but validation "
                    f"needs {comparison_count} to complete the cutoff eigenspace"
                )

            # Sample the extra solved FEM modes as well. They are validation-only
            # guard modes and are not part of the N-mode dataset target.
            sampled = sample_modes_on_material_grid(
                solution,
                geometry,
                n_modes=solution.n_solved,
                grid_size=args.grid,
                degeneracy_relative_gap=args.degeneracy_gap,
            )

            match = match_modes_to_reference(
                sampled.modal_factors,
                sampled.mode_shapes,
                analytic_factors,
                analytic_modes,
                sampled.inner_product_weights,
                shape_tiebreak_weight=args.matching_shape_weight,
            )

            # Only true/near-true physical degeneracies are reordered for scalar
            # factor comparison.  Near-but-distinct validation bands keep their
            # shape-led Hungarian identity; their span is used only for the shape
            # score itself.
            assignment = reorder_match_within_reference_groups(
                sampled.modal_factors,
                match.fem_for_reference,
                physical_groups,
            )
            ordered_factors = np.asarray(sampled.modal_factors)[
                assignment[: args.n_modes]
            ]
            by_h[h] = ordered_factors.copy()
            rel = np.abs(
                ordered_factors / analytic_factors[: args.n_modes] - 1.0
            )

            shape_score = np.empty(args.n_modes, dtype=np.float64)
            score_kind = np.empty(args.n_modes, dtype=object)
            comparison_group_size = np.ones(args.n_modes, dtype=np.int32)
            physical_group_size = np.ones(args.n_modes, dtype=np.int32)
            cutoff_group_extended = np.zeros(args.n_modes, dtype=np.uint8)

            for indices in _group_indices(physical_groups):
                target_indices = indices[indices < args.n_modes]
                if target_indices.size:
                    physical_group_size[target_indices] = indices.size

            cutoff_target_members = cutoff_group_indices[
                cutoff_group_indices < args.n_modes
            ]

            for target_indices in _group_indices(target_comparison_groups):
                eval_indices = target_indices
                kind = "mac" if target_indices.size == 1 else "band_subspace"

                # If the fixed output cutoff slices through a genuine physical
                # eigenspace, validate the COMPLETE eigenspace using the guard
                # modes above N.  A truncated subspace has no unique orientation
                # and must not be used as a shape-quality gate.
                if cutoff_extended and np.intersect1d(
                    target_indices, cutoff_target_members
                ).size:
                    eval_indices = cutoff_group_indices
                    kind = "cutoff_physical_subspace"
                    cutoff_group_extended[target_indices] = 1

                fem_indices = assignment[eval_indices]
                if eval_indices.size == 1:
                    reference_i = int(eval_indices[0])
                    fem_i = int(fem_indices[0])
                    score = float(match.mac_matrix[fem_i, reference_i])
                else:
                    score = subspace_projection_score(
                        sampled.mode_shapes[fem_indices],
                        analytic_modes[eval_indices],
                        sampled.inner_product_weights,
                    )

                shape_score[target_indices] = score
                score_kind[target_indices] = kind
                comparison_group_size[target_indices] = eval_indices.size

            for i in range(args.n_modes):
                fem_i = int(assignment[i])
                row = {
                    "aspect": float(aspect),
                    "h_target": float(h),
                    "mode": int(i),
                    "fem_mode": fem_i,
                    "analytic_factor": float(analytic_factors[i]),
                    "fem_factor": float(ordered_factors[i]),
                    "relative_error": float(rel[i]),
                    "shape_score": float(shape_score[i]),
                    "score_kind": str(score_kind[i]),
                    "comparison_group_size": int(comparison_group_size[i]),
                    "physical_degenerate_group_size": int(physical_group_size[i]),
                    "cutoff_group_extended": int(cutoff_group_extended[i]),
                    "residual": float(sampled.residuals[fem_i]),
                    "mesh_vertices": mesh.n_vertices,
                    "mesh_triangles": mesh.n_triangles,
                    "mesh_area_error": float(mesh.relative_area_error),
                    "boundary_hausdorff_approx": float(mesh.boundary_hausdorff_approx),
                }
                freq_rows.append(row)
                if h == finest_h:
                    finest_rows.append(row)

            if mesh.relative_area_error > args.max_mesh_area_error:
                failures.append(
                    f"aspect {aspect:g}, h={h:g}: mesh area error {mesh.relative_area_error:.3e}"
                )
            if mesh.boundary_hausdorff_approx > args.max_boundary_distance:
                failures.append(
                    f"aspect {aspect:g}, h={h:g}: boundary distance {mesh.boundary_hausdorff_approx:.3e}"
                )

        finest = by_h[finest_h]
        for h, factors in by_h.items():
            if h == finest_h:
                continue
            rel_change = np.abs(factors / finest - 1.0)
            for mode, error in enumerate(rel_change):
                convergence_rows.append(
                    {
                        "aspect": float(aspect),
                        "h_target": float(h),
                        "reference_h": float(finest_h),
                        "mode": int(mode),
                        "relative_change": float(error),
                    }
                )
            if h == second_finest_h and float(np.max(rel_change)) > args.max_convergence_change:
                failures.append(
                    f"aspect {aspect:g}: second-finest→finest max modal change "
                    f"{np.max(rel_change):.3e} exceeds {args.max_convergence_change:.3e}"
                )

    for name, rows in (
        ("rectangle_frequency_shape.csv", freq_rows),
        ("mesh_convergence.csv", convergence_rows),
    ):
        path = args.output / name
        if rows:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(path)

    if not finest_rows:
        raise SystemExit("validation produced no finest-mesh rows")

    finest_errors = np.asarray([float(row["relative_error"]) for row in finest_rows])
    finest_scores = np.asarray([float(row["shape_score"]) for row in finest_rows])
    finest_modes = np.asarray([int(row["mode"]) for row in finest_rows])
    low = finest_modes < min(args.low_mode_count, args.n_modes)
    high = ~low

    low_max = float(np.max(finest_errors[low])) if np.any(low) else 0.0
    high_max = float(np.max(finest_errors[high])) if np.any(high) else 0.0
    min_score = float(np.min(finest_scores))

    if low_max > args.max_rel_error_low:
        failures.append(
            f"finest mesh low-mode max relative error {low_max:.3e} exceeds {args.max_rel_error_low:.3e}"
        )
    if np.any(high) and high_max > args.max_rel_error_high:
        failures.append(
            f"finest mesh high-mode max relative error {high_max:.3e} exceeds {args.max_rel_error_high:.3e}"
        )
    if min_score < args.min_shape_score:
        failures.append(
            f"finest mesh minimum MAC/subspace score {min_score:.6f} below {args.min_shape_score:.6f}"
        )

    print(f"finest low-mode relative error max={low_max:.3e}")
    if np.any(high):
        print(f"finest high-mode relative error max={high_max:.3e}")
    print(f"finest MAC/subspace score min={min_score:.6f}")

    print("\nWorst finest-mesh shape comparisons:")
    for row in sorted(finest_rows, key=lambda item: float(item["shape_score"]))[:8]:
        suffix = " +guard" if int(row["cutoff_group_extended"]) else ""
        print(
            " - aspect={aspect:g}, ref={mode}, fem={fem_mode}, "
            "score={shape_score:.6f} ({score_kind}, band={comparison_group_size}{suffix}), "
            "freq_err={relative_error:.3e}".format(suffix=suffix, **row)
        )

    if failures:
        print("\nVALIDATION FAILED:")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
