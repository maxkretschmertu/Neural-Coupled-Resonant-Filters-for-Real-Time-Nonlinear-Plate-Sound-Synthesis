from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import ceil, log2, sqrt
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from ..geometry import make_geometry
from .mesh import ReferenceMesh, mesh_geometry
from .mode_sampling import SampledModes, sample_modes_on_material_grid
from .plate_solver import PlateSolverConfig, solve_plate_modes
from .validation import SampleQualityReport, validate_generated_sample


@dataclass(frozen=True)
class DatasetConfig:
    version: str = "phase3-v1"
    output_dir: str = "data/modal_dataset"
    seed: int = 20260917
    train_count: int = 8000
    val_count: int = 1000
    test_count: int = 1000
    geometry_grid_size: int = 64
    boundary_samples: int = 720
    mode_grid_size: int = 64
    n_modes: int = 32
    n_solve: int = 40
    poisson_ratio: float = 0.30
    boundary_condition: str = "simply_supported"
    mesh_edge_length: float = 0.055
    integration_order: int = 4
    eigensolver_tolerance: float = 1e-10
    eigensolver_maxiter: int = 30_000
    degeneracy_relative_gap: float = 5e-4
    max_residual: float = 1e-7
    max_mass_orthogonality_error: float = 1e-7
    max_grid_norm_error: float = 5e-5
    shard_size: int = 250
    compression_level: int = 4
    workers: int = 0
    store_mesh: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def validate(self) -> None:
        if min(self.train_count, self.val_count, self.test_count) < 0:
            raise ValueError("split counts must be non-negative")
        if self.n_modes < 1 or self.n_solve < self.n_modes:
            raise ValueError("require 1 <= n_modes <= n_solve")
        if self.geometry_grid_size < 16 or self.mode_grid_size < 16:
            raise ValueError("geometry/mode grid size must be >= 16")
        if self.boundary_samples < 64:
            raise ValueError("boundary_samples must be >= 64")
        if self.mesh_edge_length <= 0.0 or self.shard_size < 1:
            raise ValueError("mesh_edge_length and shard_size must be positive")
        if not (-1.0 < self.poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        if self.boundary_condition != "simply_supported":
            raise ValueError("Phase 3 currently supports simply_supported only")
        if self.integration_order < 2 or self.eigensolver_maxiter < 1:
            raise ValueError("invalid integration/eigensolver settings")
        if self.eigensolver_tolerance <= 0.0 or self.degeneracy_relative_gap < 0.0:
            raise ValueError("invalid eigensolver/degeneracy tolerance")
        if min(
            self.max_residual,
            self.max_mass_orthogonality_error,
            self.max_grid_norm_error,
        ) <= 0.0:
            raise ValueError("quality thresholds must be positive")


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    split: str
    morph: float
    shape_mod: float


@dataclass(frozen=True)
class GeneratedSample:
    spec: SampleSpec
    sdf: np.ndarray
    mask: np.ndarray
    sampled: SampledModes
    report: SampleQualityReport
    mesh: ReferenceMesh
    mass_orthogonality_error: float


@dataclass(frozen=True)
class FailedSample:
    spec: SampleSpec
    error: str
    traceback_text: str


def _sobol_points(count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dataset sampling requires scipy") from exc
    exponent = int(ceil(log2(max(count, 1))))
    sampler = qmc.Sobol(d=2, scramble=True, seed=int(seed))
    return np.asarray(sampler.random_base2(exponent)[:count], dtype=np.float64)


def _test_grid_points(count: int) -> np.ndarray:
    """Deterministic interior grid reserved from random train/validation data."""
    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    side = int(ceil(sqrt(count)))
    axis = (np.arange(side, dtype=np.float64) + 0.5) / side
    m, s = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((m.ravel(), s.ravel()))[:count]


def build_sample_specs(config: DatasetConfig) -> list[SampleSpec]:
    """Build a deterministic split before any expensive FEM work is run."""
    config.validate()
    train = _sobol_points(config.train_count, config.seed)
    val = _sobol_points(config.val_count, config.seed + 101)
    test = _test_grid_points(config.test_count)

    # Ensure exact family endpoints and shape-mod extremes are represented in
    # the training set without increasing its requested size.
    anchors = np.asarray(
        [
            (0.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ],
        dtype=np.float64,
    )
    n_anchor = min(train.shape[0], anchors.shape[0])
    if n_anchor:
        train[:n_anchor] = anchors[:n_anchor]

    specs: list[SampleSpec] = []
    for split, points in (("train", train), ("val", val), ("test", test)):
        for i, (morph, shape_mod) in enumerate(points):
            specs.append(
                SampleSpec(
                    sample_id=f"{split}-{i:06d}",
                    split=split,
                    morph=float(morph),
                    shape_mod=float(shape_mod),
                )
            )
    return specs


def _solver_config(config: DatasetConfig) -> PlateSolverConfig:
    return PlateSolverConfig(
        n_modes=config.n_modes,
        n_solve=config.n_solve,
        poisson_ratio=config.poisson_ratio,
        boundary_condition=config.boundary_condition,
        integration_order=config.integration_order,
        eigensolver_tolerance=config.eigensolver_tolerance,
        eigensolver_maxiter=config.eigensolver_maxiter,
    )


def _generate_one(
    payload: tuple[DatasetConfig, SampleSpec],
) -> GeneratedSample | FailedSample:
    config, spec = payload
    try:
        geometry = make_geometry(
            spec.morph,
            spec.shape_mod,
            grid_size=config.geometry_grid_size,
            boundary_samples=config.boundary_samples,
        )
        mesh = mesh_geometry(geometry, config.mesh_edge_length)
        solution = solve_plate_modes(mesh, _solver_config(config))
        sampled = sample_modes_on_material_grid(
            solution,
            geometry,
            n_modes=config.n_modes,
            grid_size=config.mode_grid_size,
            degeneracy_relative_gap=config.degeneracy_relative_gap,
        )
        report = validate_generated_sample(
            solution,
            sampled,
            max_residual=config.max_residual,
            max_mass_orthogonality_error=config.max_mass_orthogonality_error,
            max_grid_norm_error=config.max_grid_norm_error,
        )
        if not report.accepted:
            raise RuntimeError("; ".join(report.reasons))
        return GeneratedSample(
            spec=spec,
            sdf=np.asarray(geometry.sdf, dtype=np.float32),
            mask=np.asarray(geometry.mask, dtype=np.uint8),
            sampled=sampled,
            report=report,
            mesh=mesh,
            mass_orthogonality_error=float(solution.mass_orthogonality_error),
        )
    except Exception as exc:  # keep batch generation alive; details go to JSONL
        return FailedSample(
            spec=spec,
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )


class _ShardWriter:
    def __init__(self, root: Path, split: str, config: DatasetConfig) -> None:
        self.root = root
        self.split = split
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        # Only finalized .h5 files participate in resume. A process crash may
        # leave a .partial.h5; discard it instead of treating it as valid data.
        for partial in self.root.glob(f"{split}_*.partial.h5"):
            partial.unlink(missing_ok=True)
        existing = sorted(self.root.glob(f"{split}_*.h5"))
        self.next_index = 0
        if existing:
            self.next_index = max(
                int(path.stem.split("_")[-1]) for path in existing
            ) + 1
        self._file = None
        self._final_path: Path | None = None
        self._partial_path: Path | None = None
        self._datasets: dict[str, Any] = {}
        self._count = 0

    def _open(self) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("HDF5 output requires h5py") from exc

        final_path = self.root / f"{self.split}_{self.next_index:05d}.h5"
        partial_path = self.root / f"{self.split}_{self.next_index:05d}.partial.h5"
        self.next_index += 1
        f = h5py.File(partial_path, "w")
        f.attrs["dataset_version"] = self.config.version
        f.attrs["split"] = self.split
        f.attrs["boundary_condition"] = self.config.boundary_condition
        f.attrs["poisson_ratio"] = self.config.poisson_ratio
        f.attrs["n_modes"] = self.config.n_modes
        f.attrs["n_solve"] = self.config.n_solve
        f.attrs["modal_factor_convention"] = (
            "sqrt(kirchhoff_dimensionless_eigenvalue)"
        )
        self._file = f
        self._final_path = final_path
        self._partial_path = partial_path
        self._datasets = {}
        self._count = 0

    def _dataset(
        self,
        name: str,
        sample_shape: tuple[int, ...],
        dtype: Any,
        *,
        compress: bool = True,
    ):
        if name in self._datasets:
            return self._datasets[name]
        assert self._file is not None
        kwargs: dict[str, Any] = {
            "shape": (0, *sample_shape),
            "maxshape": (None, *sample_shape),
            "chunks": (1, *sample_shape),
            "dtype": dtype,
        }
        if compress and sample_shape:
            kwargs.update(
                compression="gzip",
                compression_opts=int(self.config.compression_level),
                shuffle=True,
            )
        ds = self._file.create_dataset(name, **kwargs)
        self._datasets[name] = ds
        return ds

    def _append_array(self, name: str, value: np.ndarray, dtype: Any) -> None:
        array = np.asarray(value, dtype=dtype)
        ds = self._dataset(name, array.shape, dtype)
        ds.resize(self._count + 1, axis=0)
        ds[self._count] = array

    def _append_scalar(self, name: str, value: Any, dtype: Any) -> None:
        ds = self._dataset(name, (), dtype, compress=False)
        ds.resize(self._count + 1, axis=0)
        ds[self._count] = value

    def append(self, sample: GeneratedSample) -> None:
        if self._file is None:
            self._open()
        sampled = sample.sampled
        self._append_scalar("sample_id", sample.spec.sample_id, "S32")
        self._append_scalar("morph", sample.spec.morph, np.float32)
        self._append_scalar("shape_mod", sample.spec.shape_mod, np.float32)
        self._append_array("sdf", sample.sdf, np.float32)
        self._append_array("mask", sample.mask, np.uint8)
        self._append_array("modal_factors", sampled.modal_factors, np.float32)
        self._append_array("eigenvalues", sampled.eigenvalues, np.float64)
        self._append_array("mode_shapes", sampled.mode_shapes, np.float32)
        self._append_array("area_weights", sampled.area_weights, np.float32)
        self._append_array(
            "inner_product_weights", sampled.inner_product_weights, np.float32
        )
        self._append_array(
            "degenerate_group_id", sampled.degenerate_group_id, np.int16
        )
        self._append_array("mode_loss_mask", sampled.mode_loss_mask, np.float32)
        self._append_scalar(
            "cutoff_group_complete", int(sampled.cutoff_group_complete), np.uint8
        )
        self._append_array(
            "eigensolver_residual", sampled.residuals, np.float64
        )
        self._append_scalar(
            "mass_orthogonality_error",
            sample.mass_orthogonality_error,
            np.float64,
        )
        self._append_scalar("mesh_n_vertices", sample.mesh.n_vertices, np.int32)
        self._append_scalar(
            "mesh_n_triangles", sample.mesh.n_triangles, np.int32
        )
        self._append_scalar(
            "mesh_h_min", sample.mesh.min_edge_length, np.float32
        )
        self._append_scalar(
            "mesh_h_mean", sample.mesh.mean_edge_length, np.float32
        )
        self._append_scalar(
            "mesh_h_max", sample.mesh.max_edge_length, np.float32
        )

        if self.config.store_mesh:
            try:
                import h5py
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("mesh storage requires h5py") from exc
            for name, array, dtype in (
                ("mesh_points_flat", sample.mesh.points.ravel(), np.float64),
                ("mesh_triangles_flat", sample.mesh.triangles.ravel(), np.int32),
            ):
                if name not in self._datasets:
                    assert self._file is not None
                    self._datasets[name] = self._file.create_dataset(
                        name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=h5py.vlen_dtype(np.dtype(dtype)),
                    )
                ds = self._datasets[name]
                ds.resize(self._count + 1, axis=0)
                ds[self._count] = np.asarray(array, dtype=dtype)

        self._count += 1
        if self._count >= self.config.shard_size:
            self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            if self._partial_path is not None and self._final_path is not None:
                self._partial_path.replace(self._final_path)
        self._file = None
        self._final_path = None
        self._partial_path = None
        self._datasets = {}
        self._count = 0


def _completed_ids(root: Path) -> set[str]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("resume scanning requires h5py") from exc
    done: set[str] = set()
    for path in root.glob("*_*.h5"):
        if path.name.endswith(".partial.h5"):
            continue
        with h5py.File(path, "r") as f:
            if "sample_id" not in f:
                continue
            for raw in f["sample_id"][:]:
                done.add(
                    raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                )
    return done


def _write_manifest(
    root: Path,
    config: DatasetConfig,
    extra: dict[str, Any],
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        **extra,
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def generate_dataset(
    config: DatasetConfig,
    *,
    limit: int | None = None,
    workers: int | None = None,
) -> dict[str, int]:
    """Generate/resume sharded FEM labels for the full Phase-2 shape space."""
    config.validate()
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    specs = build_sample_specs(config)
    if limit is not None:
        specs = specs[: max(int(limit), 0)]
    completed = _completed_ids(root)
    pending = [spec for spec in specs if spec.sample_id not in completed]

    writers = {
        split: _ShardWriter(root, split, config)
        for split in ("train", "val", "test")
    }
    failed_path = root / "failed_samples.jsonl"
    counts = {
        "written": 0,
        "failed": 0,
        "skipped": len(specs) - len(pending),
    }

    requested_workers = config.workers if workers is None else workers
    if requested_workers is None or requested_workers <= 0:
        import os

        requested_workers = max(1, (os.cpu_count() or 2) - 1)

    _write_manifest(
        root,
        config,
        {
            "status": "running",
            "requested_samples": len(specs),
            "remaining_at_start": len(pending),
        },
    )

    def consume(result: GeneratedSample | FailedSample) -> None:
        if isinstance(result, FailedSample):
            counts["failed"] += 1
            record = {
                "sample_id": result.spec.sample_id,
                "split": result.spec.split,
                "morph": result.spec.morph,
                "shape_mod": result.spec.shape_mod,
                "error": result.error,
                "traceback": result.traceback_text,
            }
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            return
        writers[result.spec.split].append(result)
        counts["written"] += 1

    try:
        if requested_workers == 1:
            for spec in pending:
                consume(_generate_one((config, spec)))
        else:
            with ProcessPoolExecutor(max_workers=requested_workers) as pool:
                futures = [
                    pool.submit(_generate_one, (config, spec)) for spec in pending
                ]
                for future in as_completed(futures):
                    consume(future.result())
    finally:
        for writer in writers.values():
            writer.close()

    final_status = "complete" if counts["failed"] == 0 else "complete_with_failures"
    _write_manifest(
        root,
        config,
        {
            "status": final_status,
            "requested_samples": len(specs),
            **counts,
        },
    )
    return counts
