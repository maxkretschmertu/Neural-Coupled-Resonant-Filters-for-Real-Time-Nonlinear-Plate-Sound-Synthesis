"""Neural modal plate synthesizer core."""

from .config import SynthParameters, SUPPORTED_BOUNDARY_CONDITIONS
from .geometry import GeometryDescription, make_geometry
from .modal_backend import (
    AnalyticRectangleBackend,
    ModalBackend,
    ModalBasis,
    UnsupportedBoundaryConditionError,
    UnsupportedGeometryError,
    physical_mass_weights,
)

__all__ = [
    "AnalyticRectangleBackend",
    "GeometryDescription",
    "ModalBackend",
    "ModalBasis",
    "SUPPORTED_BOUNDARY_CONDITIONS",
    "SynthParameters",
    "UnsupportedBoundaryConditionError",
    "UnsupportedGeometryError",
    "make_geometry",
    "physical_mass_weights",
]
