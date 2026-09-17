"""Plate synth package for the neural modal refactor."""

from .config import SynthParameters
from .geometry import GeometryDescription, make_rectangle_geometry
from .modal_backend import (
    LegacyRectangleBackend,
    ModalBackend,
    ModalBasis,
    resolve_modal_frequencies_hz,
)

__all__ = [
    "GeometryDescription",
    "LegacyRectangleBackend",
    "ModalBackend",
    "ModalBasis",
    "SynthParameters",
    "make_rectangle_geometry",
    "resolve_modal_frequencies_hz",
]
