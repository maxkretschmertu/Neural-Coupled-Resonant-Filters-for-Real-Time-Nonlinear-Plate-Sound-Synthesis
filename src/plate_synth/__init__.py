"""Plate synth package for the neural modal refactor.

Phase 1 keeps the existing nonlinear resonator synthesis model intact while
introducing a clean modal-backend boundary for the future neural model.

`AudioEngine` is intentionally not imported here so modal/data tooling can use
this package on machines without an audio device or the optional sounddevice
package installed.
"""

from .config import SynthParameters
from .modal_backend import LegacyRectangleBackend, ModalBackend, ModalBasis

__all__ = [
    "LegacyRectangleBackend",
    "ModalBackend",
    "ModalBasis",
    "SynthParameters",
]
