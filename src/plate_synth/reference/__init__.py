"""Offline Kirchhoff--Love reference solver and dataset tooling.

This package is intentionally separate from the realtime synthesizer.  Heavy
optional dependencies (Gmsh, scikit-fem, h5py) are imported lazily so the
runtime synth does not require them.
"""

from .dataset import DatasetConfig, SampleSpec, build_sample_specs, generate_dataset
from .mesh import ReferenceMesh, mesh_geometry
from .mode_sampling import SampledModes, sample_modes_on_material_grid
from .plate_solver import PlateEigenSolution, PlateSolverConfig, solve_plate_modes
from .validation import SampleQualityReport, validate_generated_sample

__all__ = [
    "DatasetConfig",
    "PlateEigenSolution",
    "PlateSolverConfig",
    "ReferenceMesh",
    "SampleQualityReport",
    "SampleSpec",
    "SampledModes",
    "build_sample_specs",
    "generate_dataset",
    "mesh_geometry",
    "sample_modes_on_material_grid",
    "solve_plate_modes",
    "validate_generated_sample",
]
