#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from plate_synth.reference.dataset import DatasetConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase-3 FEM modal dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_phase3_pilot.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N split specs")
    parser.add_argument("--workers", type=int, default=None, help="Override config workers; 1 = serial")
    args = parser.parse_args()

    config = DatasetConfig.from_json(args.config)
    counts = generate_dataset(config, limit=args.limit, workers=args.workers)
    print("Phase-3 generation complete:", counts)


if __name__ == "__main__":
    main()
