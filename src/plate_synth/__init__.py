"""Neural modal plate synthesizer core."""

from .config import SynthParameters
from .geometry import GeometryDescription, make_geometry
from .modal_backend import AnalyticRectangleBackend, ModalBackend, ModalBasis, UnsupportedGeometryError

__all__ = [
    "AnalyticRectangleBackend",
    "GeometryDescription",
    "ModalBackend",
    "ModalBasis",
    "SynthParameters",
    "UnsupportedGeometryError",
    "make_geometry",
]
