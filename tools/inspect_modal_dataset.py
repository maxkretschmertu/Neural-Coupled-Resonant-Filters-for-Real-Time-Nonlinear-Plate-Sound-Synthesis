#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Phase-3 HDF5 modal shards")
    parser.add_argument("path", type=Path, help="Dataset directory or one .h5 shard")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required; install requirements-phase3.txt") from exc

    shards = sorted(args.path.glob("*.h5")) if args.path.is_dir() else [args.path]
    shards = [p for p in shards if not p.name.endswith(".partial.h5")]
    if not shards:
        raise SystemExit("no finalized HDF5 shards found")

    total = 0
    fingerprints: set[str] = set()
    maxima = {
        "residual": 0.0,
        "mass_orth": 0.0,
        "raw_grid_area": 0.0,
        "mesh_area": 0.0,
        "boundary_distance": 0.0,
    }

    for path in shards:
        with h5py.File(path, "r") as f:
            n = int(f["morph"].shape[0])
            total += n
            raw_fp = f.attrs.get("config_fingerprint", "")
            if isinstance(raw_fp, bytes):
                raw_fp = raw_fp.decode("utf-8")
            fingerprints.add(str(raw_fp))

            if n:
                maxima["residual"] = max(maxima["residual"], float(np.max(f["eigensolver_residual"][:])))
                maxima["mass_orth"] = max(maxima["mass_orth"], float(np.max(f["mass_orthogonality_error"][:])))
                if "raw_grid_area_relative_error" in f:
                    maxima["raw_grid_area"] = max(
                        maxima["raw_grid_area"], float(np.max(f["raw_grid_area_relative_error"][:]))
                    )
                if "mesh_relative_area_error" in f:
                    maxima["mesh_area"] = max(
                        maxima["mesh_area"], float(np.max(f["mesh_relative_area_error"][:]))
                    )
                if "mesh_boundary_hausdorff_approx" in f:
                    maxima["boundary_distance"] = max(
                        maxima["boundary_distance"], float(np.max(f["mesh_boundary_hausdorff_approx"][:]))
                    )

            print(
                f"{path.name}: {n} samples, split={f.attrs.get('split')}, "
                f"modes={f.attrs.get('n_modes')}, fingerprint={str(raw_fp)[:12]}"
            )

    if len(fingerprints) != 1 or "" in fingerprints:
        raise SystemExit("dataset contains missing or inconsistent config fingerprints")

    print("total samples:", total)
    print("config fingerprint:", next(iter(fingerprints)))
    print(
        "QA maxima: "
        f"residual={maxima['residual']:.3e}, "
        f"mass_orth={maxima['mass_orth']:.3e}, "
        f"raw_grid_area={maxima['raw_grid_area']:.3e}, "
        f"mesh_area={maxima['mesh_area']:.3e}, "
        f"boundary_distance={maxima['boundary_distance']:.3e}"
    )

    with h5py.File(shards[0], "r") as f:
        if f["morph"].shape[0] == 0:
            return
        i = min(max(args.sample, 0), f["morph"].shape[0] - 1)
        sid = f["sample_id"][i]
        if isinstance(sid, bytes):
            sid = sid.decode()
        factors = f["modal_factors"][i]
        extra = ""
        if "mesh_relative_area_error" in f:
            extra += f", mesh area err={f['mesh_relative_area_error'][i]:.3e}"
        if "mesh_boundary_hausdorff_approx" in f:
            extra += f", boundary dist={f['mesh_boundary_hausdorff_approx'][i]:.3e}"
        if "raw_grid_area_relative_error" in f:
            extra += f", raw grid area err={f['raw_grid_area_relative_error'][i]:.3e}"
        print(
            f"sample={sid}, morph={f['morph'][i]:.4f}, shape={f['shape_mod'][i]:.4f}, "
            f"factor range={factors[0]:.4f}..{factors[-1]:.4f}, "
            f"max residual={np.max(f['eigensolver_residual'][i]):.3e}{extra}"
        )

        if args.plot:
            import matplotlib.pyplot as plt

            mode = f["mode_shapes"][i, 0]
            sdf = f["sdf"][i]
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(sdf, origin="lower")
            axes[0].contour(sdf, levels=[0.0])
            axes[0].set_title("SDF")
            axes[1].imshow(mode, origin="lower")
            axes[1].set_title("Mode 1 on material grid")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
