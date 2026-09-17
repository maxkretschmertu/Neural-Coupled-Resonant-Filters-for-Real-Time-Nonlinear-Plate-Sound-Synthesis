from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SynthParameters:
    """Immutable control state shared between the GUI and audio engine.

    The fields deliberately separate physical/material controls from the
    nonlinear resonator controls. The future neural network will only replace
    the geometry -> modal-basis step; these parameters remain ordinary code.
    """

    # Runtime
    sample_rate: float = 48_000.0
    legacy_mode_order: int = 10
    max_state_magnitude: float = 10.0

    # Rectangle reference geometry used by the Phase-1 legacy backend
    length_x_m: float = 1.0
    length_y_m: float = 1.0

    # Material / damping
    flexural_rigidity: float = 18_300.0  # D [N m]
    density: float = 7_800.0             # rho [kg/m^3]
    thickness_m: float = 0.01            # H [m]
    alpha_g: float = 0.3322
    alpha_r: float = 4e-5

    # Musical tuning kept outside the future NN
    frequency_scale: float = 1.0

    # Nonlinear coupling
    tau: float = 1.0
    eta: float = 0.01
    lamb: float = 0.01

    # Excitation
    strike_x: float = 0.3
    strike_y: float = 0.3
    excitation_length_samples: int = 192
    excitation_mode: int = 0  # 0 = short sin^2 impact, 1 = 1 Hz impulses
    excitation_amplitude: float = 1.0

    # Output pickup. Disabled reproduces the v12 equal-weight output.
    pickup_enabled: bool = False
    pickup_x: float = 0.70
    pickup_y: float = 0.70

    def updated(self, **changes: object) -> "SynthParameters":
        """Return a new immutable parameter object with selected fields changed."""
        return replace(self, **changes)
