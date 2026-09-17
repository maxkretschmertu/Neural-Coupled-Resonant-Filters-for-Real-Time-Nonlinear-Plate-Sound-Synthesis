from __future__ import annotations

from dataclasses import dataclass
import threading
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from .config import SynthParameters
from .geometry import GeometryDescription, make_geometry
from .material_coordinates import clamp_material_point
from .modal_backend import (
    AnalyticRectangleBackend,
    ModalBackend,
    ModalBasis,
    UnsupportedGeometryError,
    resolve_modal_frequencies_hz,
)
from .resonator_core import (
    build_distribution_matrix,
    build_resonator_poles,
    make_excitation_block,
    process_block,
    warm_up_numba,
)
from .state_projection import apply_state_projection, damping_rates, projection_matrix


@dataclass(frozen=True)
class ModelSnapshot:
    basis: ModalBasis
    frequencies_hz: np.ndarray
    damping: np.ndarray
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
    transform: np.ndarray | None
    projected_states: np.ndarray
    projected_transfer: np.ndarray


class AudioEngine:
    """Realtime engine with geometry/model work performed off the audio thread."""

    def __init__(
        self,
        params: SynthParameters | None = None,
        modal_backend: ModalBackend | None = None,
        geometry_grid_size: int = 64,
        mode_grid_size: int = 64,
    ) -> None:
        initial = params or SynthParameters()
        self._validate_params(initial)
        self._geometry_grid_size = int(geometry_grid_size)
        self._backend = modal_backend or AnalyticRectangleBackend(mode_grid_size=mode_grid_size)
        self._control_lock = threading.Lock()

        geometry = make_geometry(initial.morph, initial.shape_mod, self._geometry_grid_size)
        basis = self._backend.predict(geometry, initial.n_modes)
        snapshot = self._build_snapshot(basis, initial)

        self._control_params = initial
        self._active_params = initial
        self._control_geometry = geometry
        self._control_basis = basis
        self._snapshot = snapshot
        self._pending_update: PendingModelUpdate | None = None
        self._pending_params: SynthParameters | None = None
        self._states = np.zeros(snapshot.n_modes, dtype=np.complex128)
        self._last_transfer = np.zeros(snapshot.n_modes, dtype=np.float64)
        self._pending_strike = False
        self._pending_clear = False
        self._impact_start: int | None = None
        self._sample_position = 0
        self._stream = None
        self.last_status = ""
        self.model_status = "rectangle analytic backend active"
        warm_up_numba(snapshot.n_modes)

    @property
    def params(self) -> SynthParameters:
        return self._control_params

    @property
    def current_geometry(self) -> GeometryDescription:
        return self._control_geometry

    @property
    def current_frequencies_hz(self) -> np.ndarray:
        with self._control_lock:
            pending = self._pending_update
            snapshot = pending.snapshot if pending is not None else self._snapshot
        return snapshot.frequencies_hz.copy()

    @property
    def sample_position(self) -> int:
        return int(self._sample_position)

    def _build_snapshot(self, basis: ModalBasis, params: SynthParameters) -> ModelSnapshot:
        frequencies = resolve_modal_frequencies_hz(basis, params)
        damping = damping_rates(frequencies, params.alpha_g, params.alpha_r)
        poles = build_resonator_poles(frequencies, params.alpha_g, params.alpha_r, params.sample_rate)
        coupling = build_distribution_matrix(frequencies, params.eta, params.lamb)
        strike = basis.sample_weights(params.strike_u, params.strike_v)
        pickup = (
            basis.sample_weights(params.pickup_u, params.pickup_v)
            if params.pickup_enabled
            else np.ones(basis.n_modes, dtype=np.float64)
        )
        return ModelSnapshot(
            basis=basis,
            frequencies_hz=np.ascontiguousarray(frequencies),
            damping=np.ascontiguousarray(damping),
            poles=np.ascontiguousarray(poles),
            coupling_matrix=np.ascontiguousarray(coupling),
            strike_weights=np.ascontiguousarray(strike),
            pickup_weights=np.ascontiguousarray(pickup),
        )

    def _make_pending_update(self, new_snapshot: ModelSnapshot) -> PendingModelUpdate:
        # Always project from the active basis/state. Rapid GUI updates replace
        # an older pending transition instead of chaining stale bases.
        with self._control_lock:
            old_snapshot = self._snapshot
        same_basis = old_snapshot.basis is new_snapshot.basis
        same_frequency_state = (
            same_basis
            and old_snapshot.n_modes == new_snapshot.n_modes
            and np.array_equal(old_snapshot.frequencies_hz, new_snapshot.frequencies_hz)
            and np.array_equal(old_snapshot.damping, new_snapshot.damping)
        )
        transform = None
        if not same_frequency_state:
            transform = (
                np.eye(new_snapshot.n_modes, dtype=np.float64)
                if same_basis
                else projection_matrix(new_snapshot.basis, old_snapshot.basis)
            )
        return PendingModelUpdate(
            snapshot=new_snapshot,
            transform=transform,
            projected_states=np.empty(new_snapshot.n_modes, dtype=np.complex128),
            projected_transfer=np.zeros(new_snapshot.n_modes, dtype=np.float64),
        )

    def update_parameters(self, **changes: object) -> None:
        unknown = set(changes) - set(SynthParameters.__dataclass_fields__)
        if unknown:
            raise KeyError(f"Unknown parameter(s): {sorted(unknown)}")
        old_params = self._control_params
        if "sample_rate" in changes and float(changes["sample_rate"]) != old_params.sample_rate:
            raise ValueError("sample_rate is immutable after AudioEngine construction")
        changed = {k for k, v in changes.items() if getattr(old_params, k) != v}
        if not changed:
            return

        if "strike_u" in changes or "strike_v" in changes:
            u = float(changes.get("strike_u", old_params.strike_u))
            v = float(changes.get("strike_v", old_params.strike_v))
            changes["strike_u"], changes["strike_v"] = clamp_material_point(u, v)
        if "pickup_u" in changes or "pickup_v" in changes:
            u = float(changes.get("pickup_u", old_params.pickup_u))
            v = float(changes.get("pickup_v", old_params.pickup_v))
            changes["pickup_u"], changes["pickup_v"] = clamp_material_point(u, v)

        new_params = old_params.updated(**changes)
        self._validate_params(new_params)
        geometry_changed = bool(changed & {"morph", "shape_mod"})
        geometry = (
            make_geometry(new_params.morph, new_params.shape_mod, self._geometry_grid_size)
            if geometry_changed
            else self._control_geometry
        )
        self._control_geometry = geometry

        backend_needed = geometry_changed or "n_modes" in changed
        basis = self._control_basis
        if backend_needed:
            try:
                basis = self._backend.predict(geometry, new_params.n_modes)
            except UnsupportedGeometryError as exc:
                # Geometry preview is fully live in Phase 2. Audio stays on the
                # last valid modal snapshot until the neural backend is trained.
                self._control_params = new_params
                self.model_status = str(exc)
                return
            self._control_basis = basis
            self.model_status = "modal backend active"

        model_changed = backend_needed or bool(
            changed & {
                "size_m", "flexural_rigidity", "density", "thickness_m",
                "frequency_scale", "alpha_g", "alpha_r", "eta", "lamb",
                "strike_u", "strike_v", "pickup_u", "pickup_v", "pickup_enabled",
            }
        )
        pending = self._make_pending_update(self._build_snapshot(basis, new_params)) if model_changed else None

        with self._control_lock:
            self._control_params = new_params
            self._pending_params = new_params
            if pending is not None:
                self._pending_update = pending

    @staticmethod
    def _validate_params(params: SynthParameters) -> None:
        if params.sample_rate <= 0.0 or params.n_modes < 1 or params.size_m <= 0.0:
            raise ValueError("sample_rate, n_modes and size_m must be positive")
        if min(params.flexural_rigidity, params.density, params.thickness_m, params.frequency_scale) <= 0.0:
            raise ValueError("material and frequency scale must be positive")
        if not (0.0 <= params.morph <= 1.0 and 0.0 <= params.shape_mod <= 1.0):
            raise ValueError("morph and shape_mod must be in [0,1]")
        if params.excitation_length_samples < 1 or params.max_state_magnitude <= 0.0:
            raise ValueError("invalid excitation length or state limit")
        for u, v, name in (
            (params.strike_u, params.strike_v, "strike"),
            (params.pickup_u, params.pickup_v, "pickup"),
        ):
            if u * u + v * v > 1.000001:
                raise ValueError(f"{name} material position must lie in the unit disk")

    def strike(self) -> None:
        with self._control_lock:
            self._pending_strike = True

    def clear(self) -> None:
        with self._control_lock:
            self._pending_clear = True

    def _consume_block_boundary(self) -> tuple[ModelSnapshot, SynthParameters, bool]:
        with self._control_lock:
            pending = self._pending_update
            self._pending_update = None
            params = self._pending_params
            self._pending_params = None
            strike = self._pending_strike
            self._pending_strike = False
            clear = self._pending_clear
            self._pending_clear = False

        if pending is not None:
            if pending.transform is not None:
                old_snapshot = self._snapshot
                apply_state_projection(
                    self._states,
                    pending.transform,
                    old_snapshot.frequencies_hz,
                    old_snapshot.damping,
                    pending.snapshot.frequencies_hz,
                    pending.snapshot.damping,
                    pending.projected_states,
                )
                self._states = pending.projected_states
                self._last_transfer = pending.projected_transfer
            elif pending.snapshot.n_modes != self._states.shape[0]:
                self._states = pending.projected_states
                self._last_transfer = pending.projected_transfer
            self._snapshot = pending.snapshot
        if params is not None:
            self._active_params = params
        if clear:
            self._states.fill(0.0)
            self._last_transfer.fill(0.0)
            self._impact_start = None
        return self._snapshot, self._active_params, strike

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            self.last_status = str(status)
        snapshot, params, strike_now = self._consume_block_boundary()
        block_start = self._sample_position
        if strike_now:
            self._impact_start = block_start
        positions = block_start + np.arange(frames)
        excitation = make_excitation_block(positions, self._impact_start, snapshot.strike_weights, params)
        output = np.empty(frames, dtype=np.float64)
        process_block(
            self._states, snapshot.poles, snapshot.coupling_matrix, excitation,
            snapshot.pickup_weights, float(params.tau), float(params.max_state_magnitude),
            output, self._last_transfer,
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
        self._stream = sd.OutputStream(channels=1, samplerate=int(self._active_params.sample_rate), callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
