from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import sounddevice as sd

from .config import SynthParameters
from .modal_backend import LegacyRectangleBackend, ModalBackend, ModalBasis
from .resonator_core import (
    build_distribution_matrix,
    build_resonator_poles,
    make_excitation_block,
    process_block,
    warm_up_numba,
)


@dataclass(frozen=True)
class ModelSnapshot:
    """Immutable control-rate data consumed by the audio callback."""

    basis: ModalBasis
    poles: np.ndarray
    coupling_matrix: np.ndarray
    strike_weights: np.ndarray
    pickup_weights: np.ndarray

    @property
    def n_modes(self) -> int:
        return self.basis.n_modes


class AudioEngine:
    """Realtime engine with control-rate model building outside the callback.

    Expensive work (modal backend, pole construction, coupling matrix, spatial
    sampling) happens in the GUI/control thread. A complete immutable snapshot
    is then published and swapped at the next audio block boundary.
    """

    MODEL_FIELDS = frozenset(
        {
            "legacy_mode_order",
            "length_x_m",
            "length_y_m",
            "flexural_rigidity",
            "density",
            "thickness_m",
            "alpha_g",
            "alpha_r",
            "frequency_scale",
            "eta",
            "lamb",
        }
    )
    SPATIAL_FIELDS = frozenset(
        {
            "strike_x",
            "strike_y",
            "pickup_enabled",
            "pickup_x",
            "pickup_y",
        }
    )

    def __init__(
        self,
        params: SynthParameters | None = None,
        modal_backend: ModalBackend | None = None,
    ) -> None:
        self._params = params or SynthParameters()
        self._backend = modal_backend or LegacyRectangleBackend()

        initial = self._build_model_snapshot(self._params)
        self._snapshot = initial
        self._pending_snapshot: ModelSnapshot | None = None

        self._states = np.zeros(initial.n_modes, dtype=np.complex128)
        self._last_transfer = np.zeros(initial.n_modes, dtype=np.float64)

        self._impact_start: int | None = None
        self._sample_position = 0
        self._stream: sd.OutputStream | None = None
        self.last_status = ""

        warm_up_numba(initial.n_modes)

    @property
    def params(self) -> SynthParameters:
        return self._params

    @property
    def sample_position(self) -> int:
        return int(self._sample_position)

    @property
    def current_basis(self) -> ModalBasis:
        pending = self._pending_snapshot
        return pending.basis if pending is not None else self._snapshot.basis

    @property
    def current_frequencies_hz(self) -> np.ndarray:
        return self.current_basis.frequencies_hz.copy()

    @property
    def last_transfer(self) -> np.ndarray:
        return self._last_transfer.copy()

    def _build_model_snapshot(self, params: SynthParameters) -> ModelSnapshot:
        basis = self._backend.predict(params)

        if params.frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be > 0")
        scaled_frequencies = (
            basis.frequencies_hz * float(params.frequency_scale)
        )
        basis = basis.with_frequencies(scaled_frequencies)

        poles = build_resonator_poles(
            frequencies_hz=basis.frequencies_hz,
            alpha_g=params.alpha_g,
            alpha_r=params.alpha_r,
            sample_rate=params.sample_rate,
        )
        coupling = build_distribution_matrix(
            frequencies_hz=basis.frequencies_hz,
            eta=params.eta,
            lamb=params.lamb,
        )

        strike_weights = basis.sample_weights(params.strike_x, params.strike_y)

        if params.pickup_enabled:
            pickup_weights = basis.sample_weights(
                params.pickup_x,
                params.pickup_y,
            )
        else:
            # Compatibility path: identical modal readout to v12.
            pickup_weights = np.ones(basis.n_modes, dtype=np.float64)

        return ModelSnapshot(
            basis=basis,
            poles=poles,
            coupling_matrix=coupling,
            strike_weights=np.ascontiguousarray(
                strike_weights, dtype=np.float64
            ),
            pickup_weights=np.ascontiguousarray(
                pickup_weights, dtype=np.float64
            ),
        )

    def _rebuild_spatial_snapshot(
        self,
        snapshot: ModelSnapshot,
        params: SynthParameters,
    ) -> ModelSnapshot:
        strike = snapshot.basis.sample_weights(params.strike_x, params.strike_y)
        pickup = (
            snapshot.basis.sample_weights(params.pickup_x, params.pickup_y)
            if params.pickup_enabled
            else np.ones(snapshot.n_modes, dtype=np.float64)
        )
        return replace(
            snapshot,
            strike_weights=np.ascontiguousarray(strike, dtype=np.float64),
            pickup_weights=np.ascontiguousarray(pickup, dtype=np.float64),
        )

    def update_parameters(self, **changes: object) -> None:
        """Update controls and prepare any required model data off the RT thread."""
        unknown = set(changes) - set(SynthParameters.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown parameter(s): {sorted(unknown)}")

        changed_names = {
            name
            for name, value in changes.items()
            if getattr(self._params, name) != value
        }
        if not changed_names:
            return

        new_params = self._params.updated(**changes)
        self._validate_params(new_params)

        # Publish control values first. The object is immutable, so the audio
        # callback always observes either the old or complete new state.
        self._params = new_params

        if changed_names & self.MODEL_FIELDS:
            self._pending_snapshot = self._build_model_snapshot(new_params)
        elif changed_names & self.SPATIAL_FIELDS:
            base = self._pending_snapshot or self._snapshot
            self._pending_snapshot = self._rebuild_spatial_snapshot(
                base,
                new_params,
            )

    @staticmethod
    def _validate_params(params: SynthParameters) -> None:
        if params.sample_rate <= 0.0:
            raise ValueError("sample_rate must be > 0")
        if params.legacy_mode_order < 1:
            raise ValueError("legacy_mode_order must be >= 1")
        if params.excitation_length_samples < 1:
            raise ValueError("excitation_length_samples must be >= 1")
        if params.max_state_magnitude <= 0.0:
            raise ValueError("max_state_magnitude must be > 0")
        for name in ("strike_x", "strike_y", "pickup_x", "pickup_y"):
            value = float(getattr(params, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def strike(self) -> None:
        """Schedule a strike at the current audio block boundary."""
        self._impact_start = int(self._sample_position)

    def clear(self) -> None:
        """Clear all resonator states without rebuilding the model."""
        self._states.fill(0.0)
        self._last_transfer.fill(0.0)
        self._impact_start = None

    def _swap_pending_snapshot(self) -> ModelSnapshot:
        pending = self._pending_snapshot
        if pending is None:
            return self._snapshot

        if pending.n_modes != self._states.shape[0]:
            # Mode-count changes cannot preserve one-to-one state in Phase 1.
            # True basis projection is implemented in the later morph phase.
            self._states = np.zeros(pending.n_modes, dtype=np.complex128)
            self._last_transfer = np.zeros(
                pending.n_modes, dtype=np.float64
            )

        self._snapshot = pending
        self._pending_snapshot = None
        return self._snapshot

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            # Never print or touch Tk from the realtime callback.
            self.last_status = str(status)

        snapshot = self._swap_pending_snapshot()
        params = self._params

        positions = self._sample_position + np.arange(frames)
        excitation = make_excitation_block(
            sample_positions=positions,
            impact_start=self._impact_start,
            mode_weights=snapshot.strike_weights,
            params=params,
        )

        output = np.empty(frames, dtype=np.float64)
        process_block(
            states=self._states,
            poles=snapshot.poles,
            coupling_matrix=snapshot.coupling_matrix,
            excitation=excitation,
            pickup_weights=snapshot.pickup_weights,
            tau=float(params.tau),
            max_state_magnitude=float(params.max_state_magnitude),
            output=output,
            last_transfer=self._last_transfer,
        )

        self._sample_position += frames

        n_modes = max(snapshot.n_modes, 1)
        output = output / np.sqrt(n_modes)
        output = np.nan_to_num(
            output,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        output = np.clip(output, -1.0, 1.0)
        outdata[:] = output.reshape(-1, 1)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            channels=1,
            samplerate=int(self._params.sample_rate),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def __enter__(self) -> "AudioEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
