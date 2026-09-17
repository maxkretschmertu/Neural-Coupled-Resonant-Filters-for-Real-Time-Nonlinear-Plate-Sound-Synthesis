from __future__ import annotations

from dataclasses import dataclass, replace


SUPPORTED_BOUNDARY_CONDITIONS = ("simply_supported",)


@dataclass(frozen=True)
class SynthParameters:
    """Immutable control state for the neural modal plate synthesizer."""

    sample_rate: float = 48_000.0
    n_modes: int = 16
    max_state_magnitude: float = 10.0

    # Canonical geometry has unit area. size_m = sqrt(physical plate area).
    morph: float = 0.0
    shape_mod: float = 0.0
    size_m: float = 1.0

    # Boundary condition is part of the modal-model / dataset contract.
    # Phase 2 deliberately supports one condition only; Phase 3 labels must
    # use the same convention.
    boundary_condition: str = "simply_supported"

    flexural_rigidity: float = 18_300.0
    density: float = 7_800.0
    thickness_m: float = 0.01
    alpha_g: float = 0.3322
    alpha_r: float = 4e-5

    frequency_scale: float = 1.0

    tau: float = 1.0
    eta: float = 0.01
    lamb: float = 0.01

    # Strike/pickup live on the canonical material disk (u^2 + v^2 <= 1).
    strike_u: float = -0.30
    strike_v: float = 0.20
    pickup_u: float = 0.35
    pickup_v: float = 0.10
    pickup_enabled: bool = True

    excitation_length_samples: int = 192
    excitation_mode: int = 0
    excitation_amplitude: float = 1.0

    def updated(self, **changes: object) -> "SynthParameters":
        return replace(self, **changes)
