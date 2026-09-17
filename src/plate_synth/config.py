from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SynthParameters:
    """Immutable control state for the realtime synth.

    Geometry, physical/material scaling, damping/tuning, spatial controls and
    nonlinear coupling are kept explicit so the future neural backend only has
    to learn geometry -> modal basis.
    """

    # Runtime. sample_rate is intentionally immutable after AudioEngine init.
    sample_rate: float = 48_000.0
    n_modes: int = 90
    max_state_magnitude: float = 10.0

    # Phase-1 rectangle geometry. Absolute scale and aspect are resolved
    # outside the modal backend; the backend receives a normalized geometry.
    length_x_m: float = 1.0
    length_y_m: float = 1.0

    # Material / damping
    flexural_rigidity: float = 18_300.0  # D [N m]
    density: float = 7_800.0             # rho [kg/m^3]
    thickness_m: float = 0.01            # H [m]
    alpha_g: float = 0.3322
    alpha_r: float = 4e-5

    # Musical tuning, applied after physical frequency scaling.
    frequency_scale: float = 1.0

    # Nonlinear coupling
    tau: float = 1.0
    eta: float = 0.01
    lamb: float = 0.01

    # Excitation
    strike_x: float = 0.3
    strike_y: float = 0.3
    excitation_length_samples: int = 192
    excitation_mode: int = 0
    excitation_amplitude: float = 1.0

    # Output pickup. Disabled reproduces the v12 equal-weight readout.
    pickup_enabled: bool = False
    pickup_x: float = 0.70
    pickup_y: float = 0.70

    def updated(self, **changes: object) -> "SynthParameters":
        return replace(self, **changes)
