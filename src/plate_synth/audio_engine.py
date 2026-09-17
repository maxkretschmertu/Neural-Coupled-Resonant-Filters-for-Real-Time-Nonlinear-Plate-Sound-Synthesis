from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # Tests/data tools must work without an audio device.
    sd = None

from .config import SynthParameters
from .geometry import make_rectangle_geometry
from .modal_backend import (
    LegacyRectangleBackend,
    ModalBackend,
    ModalBasis,
    resolve_modal_frequencies_hz,
)
from .resonator_core import (
    build_distribution_matrix,
    build_resonator_poles,
    make_excitation_block,
    process_block,
    warm_up_numba,
)


@dataclass(frozen=True)
class ModelSnapshot:
    """Immutable control-rate model consumed by the audio callback."""

    basis: ModalBasis
    frequencies_hz: np.ndarray
    poles: np.ndarray
    coupling_matrix: np.ndarray
    strike_weights: np.ndarray
    pickup_weights: np.ndarray

    @property
    def n_modes(self) -> int:
        return self.basis.n_modes


@dataclass(frozen=True)
class PendingModelUpdate:
    snapshot: ModelSnapshot
    reset_states: np.ndarray | None = None
    reset_transfer: np.ndarray | None = None


class AudioEngine:
    """Realtime engine with all modal/model construction off the RT thread."""

    def __init__(
        self,
        params: SynthParameters | None = None,
        modal_backend: ModalBackend | None = None,
        shape_grid_size: int = 64,
    ) -> None:
        initial_params = params or SynthParameters()
        self._validate_params(initial_params)

        self._backend = modal_backend or LegacyRectangleBackend(shape_grid_size)
        self._shape_grid_size = int(shape_grid_size)
        self._control_lock = threading.Lock()

        geometry = make_rectangle_geometry(
            initial_params.length_x_m,
            initial_params.length_y_m,
            self._shape_grid_size,
        )
        basis = self._backend.predict(geometry, initial_params.n_modes)
        initial = self._build_full_snapshot(basis, initial_params)

        # _control_params is the newest GUI/control state. _active_params is
        # only changed at an audio block boundary together with pending model
        # data, so the callback never sees new parameters with an old model.
        self._control_params = initial_params
        self._active_params = initial_params
        self._pending_params: SynthParameters | None = None

        self._snapshot = initial
        self._pending_update: PendingModelUpdate | None = None
        self._states = np.zeros(initial.n_modes, dtype=np.complex128)
        self._last_transfer = np.zeros(initial.n_modes, dtype=np.float64)

        self._impact_start: int | None = None
        self._pending_strike = False
        self._pending_clear = False
        self._sample_position = 0
        self._stream = None
        self.last_status = ""

        warm_up_numba(initial.n_modes)

    @property
    def params(self) -> SynthParameters:
        return self._control_params

    @property
    def sample_position(self) -> int:
        return int(self._sample_position)

    def _latest_snapshot(self) -> ModelSnapshot:
        with self._control_lock:
            pending = self._pending_update
            return pending.snapshot if pending is not None else self._snapshot

    @property
    def current_basis(self) -> ModalBasis:
        return self._latest_snapshot().basis

    @property
    def current_frequencies_hz(self) -> np.ndarray:
        return self._latest_snapshot().frequencies_hz.copy()

    @property
    def last_transfer(self) -> np.ndarray:
        return self._last_transfer.copy()

    def _build_full_snapshot(self, basis: ModalBasis, params: SynthParameters) -> ModelSnapshot:
        frequencies = resolve_modal_frequencies_hz(basis, params)
        poles = build_resonator_poles(
            frequencies, params.alpha_g, params.alpha_r, params.sample_rate
        )
        coupling = build_distribution_matrix(frequencies, params.eta, params.lamb)
        strike = basis.sample_weights(params.strike_x, params.strike_y)
        pickup = (
            basis.sample_weights(params.pickup_x, params.pickup_y)
            if params.pickup_enabled
            else np.ones(basis.n_modes, dtype=np.float64)
        )
        return ModelSnapshot(
            basis=basis,
            frequencies_hz=np.ascontiguousarray(frequencies, dtype=np.float64),
            poles=np.ascontiguousarray(poles, dtype=np.complex128),
            coupling_matrix=np.ascontiguousarray(coupling, dtype=np.float64),
            strike_weights=np.ascontiguousarray(strike, dtype=np.float64),
            pickup_weights=np.ascontiguousarray(pickup, dtype=np.float64),
        )

    def _publish_control_state(
        self,
        params: SynthParameters,
        snapshot: ModelSnapshot | None,
    ) -> None:
        """Publish complete immutable control/model objects atomically.

        Mode-count-dependent buffers are allocated on the control thread before
        publishing; the callback only swaps references.
        """
        reset_states = None
        reset_transfer = None
        if snapshot is not None:
            with self._control_lock:
                active_n = self._snapshot.n_modes
            if snapshot.n_modes != active_n:
                reset_states = np.zeros(snapshot.n_modes, dtype=np.complex128)
                reset_transfer = np.zeros(snapshot.n_modes, dtype=np.float64)
            update = PendingModelUpdate(snapshot, reset_states, reset_transfer)
        else:
            update = None

        with self._control_lock:
            self._control_params = params
            self._pending_params = params
            if update is not None:
                self._pending_update = update

    def update_parameters(self, **changes: object) -> None:
        """Rebuild only dependency layers affected by changed controls."""
        unknown = set(changes) - set(SynthParameters.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown parameter(s): {sorted(unknown)}")

        old_params = self._control_params
        if "sample_rate" in changes and float(changes["sample_rate"]) != old_params.sample_rate:
            raise ValueError("sample_rate is immutable after AudioEngine construction")

        changed = {k for k, v in changes.items() if getattr(old_params, k) != v}
        if not changed:
            return

        new_params = old_params.updated(**changes)
        self._validate_params(new_params)
        base = self._latest_snapshot()

        old_aspect = old_params.length_x_m / old_params.length_y_m
        new_aspect = new_params.length_x_m / new_params.length_y_m
        geometry_changed = (
            "n_modes" in changed
            or not np.isclose(old_aspect, new_aspect, rtol=1e-12, atol=1e-15)
        )

        if geometry_changed:
            geometry = make_rectangle_geometry(
                new_params.length_x_m,
                new_params.length_y_m,
                self._shape_grid_size,
            )
            basis = self._backend.predict(geometry, new_params.n_modes)
        else:
            basis = base.basis

        frequency_changed = geometry_changed or bool(
            changed
            & {
                "length_x_m",
                "length_y_m",
                "flexural_rigidity",
                "density",
                "thickness_m",
                "frequency_scale",
            }
        )
        frequencies = (
            resolve_modal_frequencies_hz(basis, new_params)
            if frequency_changed
            else base.frequencies_hz
        )

        poles_changed = frequency_changed or bool(changed & {"alpha_g", "alpha_r"})
        poles = (
            build_resonator_poles(
                frequencies,
                new_params.alpha_g,
                new_params.alpha_r,
                new_params.sample_rate,
            )
            if poles_changed
            else base.poles
        )

        coupling_changed = frequency_changed or bool(changed & {"eta", "lamb"})
        coupling = (
            build_distribution_matrix(frequencies, new_params.eta, new_params.lamb)
            if coupling_changed
            else base.coupling_matrix
        )

        spatial_changed = geometry_changed or bool(
            changed & {"strike_x", "strike_y", "pickup_enabled", "pickup_x", "pickup_y"}
        )
        if spatial_changed:
            strike = basis.sample_weights(new_params.strike_x, new_params.strike_y)
            pickup = (
                basis.sample_weights(new_params.pickup_x, new_params.pickup_y)
                if new_params.pickup_enabled
                else np.ones(basis.n_modes, dtype=np.float64)
            )
        else:
            strike = base.strike_weights
            pickup = base.pickup_weights

        snapshot_changed = any(
            (geometry_changed, frequency_changed, poles_changed, coupling_changed, spatial_changed)
        )
        snapshot = None
        if snapshot_changed:
            snapshot = ModelSnapshot(
                basis=basis,
                frequencies_hz=np.ascontiguousarray(frequencies, dtype=np.float64),
                poles=np.ascontiguousarray(poles, dtype=np.complex128),
                coupling_matrix=np.ascontiguousarray(coupling, dtype=np.float64),
                strike_weights=np.ascontiguousarray(strike, dtype=np.float64),
                pickup_weights=np.ascontiguousarray(pickup, dtype=np.float64),
            )

        # tau/excitation/max_state-only updates publish parameters without any
        # backend, frequency, pole or coupling rebuild.
        self._publish_control_state(new_params, snapshot)

    @staticmethod
    def _validate_params(params: SynthParameters) -> None:
        if params.sample_rate <= 0.0:
            raise ValueError("sample_rate must be > 0")
        if params.n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if params.length_x_m <= 0.0 or params.length_y_m <= 0.0:
            raise ValueError("plate lengths must be > 0")
        if min(params.flexural_rigidity, params.density, params.thickness_m) <= 0.0:
            raise ValueError("material parameters must be > 0")
        if params.frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be > 0")
        if params.excitation_length_samples < 1:
            raise ValueError("excitation_length_samples must be >= 1")
        if params.max_state_magnitude <= 0.0:
            raise ValueError("max_state_magnitude must be > 0")
        for name in ("strike_x", "strike_y", "pickup_x", "pickup_y"):
            if not 0.0 <= float(getattr(params, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def strike(self) -> None:
        """Queue a strike exactly at the beginning of the next audio block."""
        with self._control_lock:
            self._pending_strike = True

    def clear(self) -> None:
        """Queue a safe state clear for the next audio block boundary."""
        with self._control_lock:
            self._pending_clear = True

    def _consume_block_boundary(self) -> tuple[ModelSnapshot, SynthParameters, bool]:
        """Atomically consume control-thread publications for one audio block."""
        with self._control_lock:
            pending = self._pending_update
            self._pending_update = None
            pending_params = self._pending_params
            self._pending_params = None
            strike_now = self._pending_strike
            self._pending_strike = False
            clear_now = self._pending_clear
            self._pending_clear = False

        if pending is not None:
            self._snapshot = pending.snapshot
            if pending.reset_states is not None:
                self._states = pending.reset_states
                self._last_transfer = pending.reset_transfer

        if pending_params is not None:
            self._active_params = pending_params

        if clear_now:
            self._states.fill(0.0)
            self._last_transfer.fill(0.0)
            self._impact_start = None

        return self._snapshot, self._active_params, strike_now

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.last_status = str(status)

        snapshot, params, strike_now = self._consume_block_boundary()
        block_start = self._sample_position
        if strike_now:
            self._impact_start = block_start

        positions = block_start + np.arange(frames)
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
        output /= np.sqrt(max(snapshot.n_modes, 1))
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(output, -1.0, 1.0, out=output)
        outdata[:] = output.reshape(-1, 1)

    def start(self) -> None:
        if self._stream is not None:
            return
        if sd is None:
            raise RuntimeError("sounddevice is required to start realtime audio")
        self._stream = sd.OutputStream(
            channels=1,
            samplerate=int(self._active_params.sample_rate),
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
