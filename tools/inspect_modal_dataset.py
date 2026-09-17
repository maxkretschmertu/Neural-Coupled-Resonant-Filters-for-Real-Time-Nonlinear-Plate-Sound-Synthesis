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
    if not shards:
        raise SystemExit("no HDF5 shards found")

    total = 0
    for path in shards:
        with h5py.File(path, "r") as f:
            n = int(f["morph"].shape[0])
            total += n
            print(
                f"{path.name}: {n} samples, split={f.attrs.get('split')}, "
                f"modes={f.attrs.get('n_modes')}"
            )
    print("total samples:", total)

    with h5py.File(shards[0], "r") as f:
        if f["morph"].shape[0] == 0:
            return
        i = min(max(args.sample, 0), f["morph"].shape[0] - 1)
        sid = f["sample_id"][i]
        if isinstance(sid, bytes):
            sid = sid.decode()
        factors = f["modal_factors"][i]
        print(
            f"sample={sid}, morph={f['morph'][i]:.4f}, shape={f['shape_mod'][i]:.4f}, "
            f"factor range={factors[0]:.4f}..{factors[-1]:.4f}, "
            f"max residual={np.max(f['eigensolver_residual'][i]):.3e}"
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
