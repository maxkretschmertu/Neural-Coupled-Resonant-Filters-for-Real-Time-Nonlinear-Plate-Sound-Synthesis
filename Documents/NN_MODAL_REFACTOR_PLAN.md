# Neural modal synth refactor plan

Base: `momofilter` / `filter_resonator_plate_numba_v12.py`
Working branch: `neural-modal-refactor`

## Goal

Keep the current nonlinear coupled-resonator synthesis model, but replace the current rectangle-only modal setup by a generic modal backend that can later be driven by a trained neural surrogate for arbitrary learned plate geometries.

The neural network is not an audio generator. It will predict the *linear modal basis* of the current geometry:

- dimensionless modal eigenvalue / frequency factor for each mode,
- spatial mode shape for each mode on a canonical domain.

Material, absolute size, frequency scaling, damping and the nonlinear filter coupling remain outside the network.

Target signal path:

```text
Morph + Shape Mod
        |
        v
Geometry Generator ---- Size / Material / Freq Scale
        |                         |
        v                         |
Neural Modal Backend              |
  -> modal eigenvalues -----------+
  -> mode shapes
        |
        +--> strike weights phi_k(x_strike, y_strike)
        +--> pickup weights phi_k(x_pickup, y_pickup)
        +--> modal basis projection during live morphing
        |
        v
frequencies -> resonator poles Z -> nonlinear coupled resonator core -> pickup-weighted output
                  |
                  +-> coupling matrix M remains in the existing nonlinear model
```

The existing nonlinear process is preserved: modal energy `p`, thresholding by `tau`, coupling through `M`, computation of `T`, nonlinear state gain and resonator update remain the synthesis core.

---

# Phase 1 — Refactor v12 into a clean synth architecture

Phase 1 changes structure, not the synthesis concept. The existing v12 behaviour must remain available as a legacy/reference backend for A/B testing.

## 1.1 Keep v12 untouched as reference

Do not continue adding features directly to `filter_resonator_plate_numba_v12.py`.

Create a package-based implementation beside it. The v12 file remains the baseline to verify that refactoring has not unintentionally changed the sound or nonlinear dynamics.

Proposed layout:

```text
src/
  plate_synth/
    __init__.py
    config.py
    types.py
    geometry.py
    modal_backend.py
    spatial.py
    resonator_core.py
    audio_engine.py
    state_projection.py
    ui.py
    app.py
run_synth.py

tests/
  test_legacy_modal_backend.py
  test_resonator_core.py
  test_spatial_sampling.py
  test_geometry.py
  test_state_projection.py
```

## 1.2 Separate parameter groups

Replace the single global `params` dictionary by explicit parameter dataclasses.

Suggested groups:

- `GeometryParams`
  - `morph` in `[0, 1]`
  - `shape_mod` in `[0, 1]`
  - `size_m`
- `MaterialParams`
  - `D`
  - `rho`
  - `H`
  - `alpha_g`
  - `alpha_r`
- `TuningParams`
  - `frequency_scale`
- `ExcitationParams`
  - strike position
  - excitation duration
  - excitation amplitude/type
- `PickupParams`
  - pickup position
- `NonlinearParams`
  - `tau`
  - `eta`
  - `lamb`
- `RuntimeParams`
  - sample rate
  - number of modes
  - maximum state magnitude

This keeps learned geometry, physical material, musical tuning and nonlinear dynamics independent.

## 1.3 Introduce a generic modal data contract

All modal backends — current analytical rectangle and future neural model — must return the same type.

```python
@dataclass
class ModalBasis:
    modal_factors: np.ndarray      # (N,), dimensionless modal frequency/eigenvalue factors
    mode_shapes: np.ndarray        # (N, H, W), shapes on canonical coordinates
    frequencies_hz: np.ndarray     # (N,), after physical scaling
    geometry_sdf: np.ndarray       # (H, W)
    integration_weights: np.ndarray # (H, W), for projections / inner products
```

The exact naming of `modal_factors` will be fixed once the FEM eigenproblem convention is fixed. The intent is that the NN predicts geometry-dependent dimensionless modal information, while size/material scaling happens in ordinary code.

## 1.4 Introduce `ModalBackend`

Define one interface:

```python
class ModalBackend(Protocol):
    def predict(self, geometry, material, size, n_modes) -> ModalBasis:
        ...
```

First implementation:

`LegacyRectangleBackend`

- uses the current v12 `get_modes()` / `modes_to_freqs()` logic,
- produces analytical rectangle mode shapes,
- allows direct A/B comparison against v12,
- is not the final architecture; it is the regression reference.

Later implementation:

`NeuralModalBackend`

- receives the generated geometry representation,
- runs the trained network,
- returns learned modal factors and mode shapes,
- applies size/material/frequency scaling outside the network.

No heuristic circle or triangle modal approximations are to be added. Circle/triangle/intermediate geometries will become audible only through the trained modal backend.

## 1.5 Preserve nonlinear resonator core

Move the current Numba kernel into `resonator_core.py` with minimal mathematical changes.

Preserve:

- complex modal states,
- `p = 0.5 * |z|^2`,
- `[p - tau]_+`,
- coupling by `M`,
- positive `T`,
- nonlinear gain,
- damping / pole update,
- state magnitude protection.

The coupling matrix is kept as a separate function so its construction can later be improved without mixing it with the neural modal model.

## 1.6 Add pickup position now

The current v12 output sums every modal state equally. Replace this by a pickup-weighted readout:

```text
y[n] = sum_k pickup_weight[k] * Im(z_k[n])
```

where

```text
pickup_weight[k] = phi_k(x_pickup, y_pickup)
```

The strike uses the same spatial mode-shape data:

```text
strike_weight[k] = phi_k(x_strike, y_strike)
```

For the legacy backend, analytical rectangle shapes supply both weights. For the neural backend, learned mode-shape maps supply both weights.

For strict v12 regression, a compatibility mode can use pickup weights equal to one.

## 1.7 Move all non-realtime work out of the audio callback

The current v12 callback may rebuild frequencies / `Z` / `M` and also updates GUI state. The refactor must separate control-rate work from sample-rate work.

Control thread / GUI thread:

- geometry generation,
- neural inference,
- mode-shape sampling,
- `Z` construction,
- `M` construction,
- state projection,
- GUI drawing.

Audio callback:

- read a prepared immutable model snapshot,
- create/consume excitation signal,
- run `process_block`,
- write audio output.

Use a small thread-safe pending-model mechanism so a complete modal snapshot is swapped at a block boundary. Do not mutate a partially built model from the GUI thread while the callback is reading it.

## 1.8 Stable mode count and state size

The current `x` parameter creates `x * (x - 1)` modes. In the new architecture the runtime contract should use an explicit `n_modes`.

For the legacy backend, preserve the old enumeration internally when reproducing v12. The final neural backend will return exactly the requested first `N` modes.

State arrays, `Z`, `M`, strike weights and pickup weights must always have the same `N`.

### Phase 1 acceptance criteria

- v12 remains untouched.
- New package runs the rectangle synth without NN.
- Nonlinear coupling math is preserved.
- Strike position works from generic mode-shape sampling.
- Pickup position works.
- Material and frequency-scale controls are outside any NN.
- Audio callback contains no NN inference and no GUI calls.
- Modal backend can be swapped without changing the resonator core.
- A regression test compares legacy modal frequencies and one rendered impulse against the v12 reference within defined tolerances.

---

# Phase 2 — Geometry system and live modal-basis morph infrastructure

Phase 2 prepares the exact geometry space that the future dataset generator and neural network will learn. It does not fake circle/triangle modes before the trained network exists.

## 2.1 One geometry space for rectangle, circle and triangle

Use a centered canonical 2D domain and generate a signed-distance representation, e.g. `64 x 64`.

Primary shape families:

1. Rectangle
2. Circle / ellipse
3. Triangle

`shape_mod` has the same high-level semantic for each family: compact -> slender.

Examples:

- Rectangle: square -> long thin rectangle
- Circle family: circle -> elongated ellipse
- Triangle: compact/equilateral-like -> narrow/skinny triangle

Do not combine absolute size with `shape_mod`. Absolute physical size is a separate `size_m` parameter.

## 2.2 Morph control

A single morph knob defines a one-dimensional path:

```text
morph = 0.0   rectangle
morph = 0.5   circle
morph = 1.0   triangle
```

For `0 <= morph <= 0.5`, interpolate the rectangle and circle geometry representations.

For `0.5 <= morph <= 1`, interpolate circle and triangle.

Use smooth interpolation weights (e.g. smoothstep) to avoid derivative discontinuities at control points.

The result must be an actual intermediate geometry, not an audio crossfade and not an interpolation of finished spectra.

## 2.3 Geometry should be the neural input

The future network input is the generated geometry representation, not simply the two knob values.

```python
geometry = geometry_generator.make(
    morph=controls.morph,
    shape_mod=controls.shape_mod,
)

modal_basis = modal_backend.predict(
    geometry=geometry,
    material=material,
    size=controls.size_m,
    n_modes=runtime.n_modes,
)
```

This allows the neural model to learn geometry -> modes directly and keeps the UI parameterization independent from the network architecture.

## 2.4 Canonical material coordinates for strike, pickup and projection

Because the planned shapes are convex/star-shaped, use a common canonical material coordinate system for spatial controls and state transfer.

A useful mapping is based on normalized radius and angle:

```text
(rho, theta),  0 <= rho <= 1
```

For each shape, map the canonical point to the actual boundary radius at `theta`:

```text
r_actual = rho * r_boundary(theta)
```

This gives three benefits:

- strike remains at the same relative material location while the plate morphs,
- pickup remains attached to the same relative location,
- mode shapes from different geometries can be compared/projected on a common material domain.

The GUI can still display X/Y points; internally they are converted to/from canonical coordinates.

## 2.5 Modal-basis state projection for live morphing

When geometry changes while a sound is ringing, do not reset modal states and do not simply assume old mode `k` equals new mode `k`.

Convert the old complex states to displacement/velocity modal coordinates, reconstruct/project them into the new modal basis, and rebuild new complex states.

Conceptually:

```text
old modal state
   -> old physical/canonical vibration state
   -> projection into new learned basis
   -> new modal state
```

Using mode matrices `Phi_old` and `Phi_new`, solve a weighted least-squares projection on the canonical material grid:

```text
G_new = Phi_new^T W Phi_new
B     = Phi_new^T W Phi_old
q_new = solve(G_new, B q_old)
```

and analogously for modal velocity.

`W` contains integration / Jacobian weights for the canonical-to-physical mapping.

This infrastructure is implemented before neural training so the neural backend only has to supply consistent mode shapes.

## 2.6 Control-rate update pipeline

For a changed `morph`, `shape_mod` or `size`:

```text
1. generate new geometry
2. run modal backend
3. apply material + size + frequency scaling
4. build new resonator poles Z
5. build new coupling matrix M
6. compute strike/pickup spatial weights
7. project current state old basis -> new basis
8. atomically publish new model/state at audio block boundary
```

Material-only changes need not run the NN if geometry has not changed.

Frequency-scale-only changes need not run the NN.

Strike/pickup-only changes need not run the NN.

This distinction should be explicit in the parameter-update code.

## 2.7 UI controls after Phase 2

Geometry:

- Morph
- Shape Mod
- Size

Material:

- D / stiffness control
- rho
- H
- alpha_g
- alpha_r

Tuning:

- Frequency Scale

Excitation:

- Strike X/Y
- excitation length/amplitude

Pickup:

- Pickup X/Y

Nonlinear section:

- tau
- eta
- lambda

The sequencer can later store the musically useful subset per step. The architecture must not hard-wire sequencer state into the physical model classes.

### Phase 2 acceptance criteria

- Geometry generator produces continuous rectangle -> circle -> triangle geometry across the full morph range.
- Shape Mod works continuously for all three families.
- Absolute size remains independent of Shape Mod.
- Strike and pickup positions remain valid and spatially coherent during morphing.
- Generic `ModalBackend` accepts the generated geometry.
- No fake circle/triangle mode solver is introduced.
- State projection code can be unit-tested using two known analytical bases before the neural backend exists.
- Live model updates happen outside the real-time audio kernel and swap safely at block boundaries.

---

# After Phase 1–2

The next stages are intentionally separate:

- Phase 3: FEM/data generator for thousands of geometries from the exact Phase-2 geometry space.
- Phase 4: train neural geometry -> dimensionless modal factors + canonical mode shapes.
- Phase 5: integrate the trained model as `NeuralModalBackend`, validate against held-out FEM geometries, then enable full live rectangle/circle/triangle morphing.

This ordering ensures that the neural network is trained against the exact geometry/control semantics used by the final instrument, while the nonlinear synth and realtime architecture are already stable before ML is introduced.
