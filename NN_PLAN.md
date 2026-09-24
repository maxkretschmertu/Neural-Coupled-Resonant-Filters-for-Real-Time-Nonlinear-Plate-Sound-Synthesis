# NN Plan — V12 + Compact Neural Resonator

## Goal

Build the neural extension on top of `filter_resonator_plate_numba_v12.py` while keeping the codebase small and understandable.

The target is the full functionality discussed so far:

- continuous morphing between plate shapes
- `Morph` and `ShapeMod`
- rectangle / ellipse / triangle and intermediate shapes
- geometry-dependent modal frequencies
- geometry- and position-dependent strike response
- independent pickup position
- physical size and material scaling
- the existing V12 damping controls
- the existing V12 nonlinear mode coupling
- real-time operation
- no FEM/numerical solver in the audio callback
- no neural network that generates audio directly

The design follows the core idea of the Neural Resonator repository: use a physical reference model offline, let a compact network predict a low-dimensional resonator representation, and keep the runtime synthesizer simple.

---

## 1. Final signal flow

```text
Morph + ShapeMod
        |
        v
Shape Generator
        |
        v
64 x 64 shape mask
        |
        v
small CNN geometry encoder
        |
        +----------------------+
        |                      |
        v                      v
Frequency Head             Point Head
        |                      |
        |                 +----+----+
        |                 |         |
        v                 v         v
modal factors          Strike XY  Pickup XY
mu[0:N]                   |         |
                           v         v
                    strike gains   pickup gains
                        g_s[N]       g_p[N]

mu[N] + Size + D + rho + H
        |
        v
physical modal frequencies
        |
        v
V12 resonator bank
        |
        + strike gains
        + pickup gains
        |
        v
V12 nonlinear coupling
(tau, eta, lambda)
        |
        v
Audio
```

The neural network replaces only the rectangle-specific modal description. V12 remains the real-time synthesizer.

---

## 2. What the neural network takes over

### 2.1 Geometry-dependent modal structure

Current V12:

```text
Lx, Ly -> analytic rectangle mode formula -> frequencies
```

Final system:

```text
shape mask -> NN -> dimensionless modal factors mu[0:N]
```

The network learns how geometry changes the resonance pattern.

### 2.2 Spatial modal response

Current V12 excitation weighting:

```python
sin(l * pi * x) * sin(m * pi * y)
```

This is rectangle-specific.

Final system:

```text
shape latent + point (x, y) -> point head -> modal gains
```

The same point head is evaluated twice:

```python
strike_gains = point_head(shape_features, strike_xy)
pickup_gains = point_head(shape_features, pickup_xy)
```

This preserves independent strike and pickup positions without predicting full 64 x 64 mode-shape images.

---

## 3. Functionality retained at the end

| Function | Final system | Responsible part |
|---|---|---|
| Rectangle | yes | shape generator |
| Ellipse | yes | shape generator |
| Triangle | yes | shape generator |
| Continuous rectangle -> ellipse -> triangle morph | yes | shape generator + NN |
| Morph control | yes | shape generator |
| ShapeMod control | yes | shape generator |
| Intermediate shapes have their own resonance structure | yes | NN |
| Additional trained 2D shapes | yes | shape generator + NN |
| Geometry-dependent modal frequencies | yes | NN |
| Geometry-dependent modal density | yes | NN |
| Strike X/Y | yes | NN point head |
| Pickup X/Y | yes | NN point head |
| Independent strike and pickup | yes | point head queried twice |
| Size | yes | analytical scaling |
| Flexural rigidity D | yes | analytical scaling |
| Density rho | yes | analytical scaling |
| Thickness H | yes | analytical scaling |
| Frequency scale | yes | V12/runtime |
| alpha_g / alpha_r damping | yes | V12 |
| Excitation length | yes | V12 |
| Strike trigger | yes | V12 |
| tau | yes | V12 |
| eta | yes | V12 |
| lambda | yes | V12 |
| Existing nonlinear mode coupling | yes | V12 |
| Real-time morphing | yes | control-rate NN inference |
| Numerical solver during audio | no | offline only |
| NN directly generates audio | no | V12 generates audio |
| Full spatial mode-shape images as NN output | no | intentionally omitted |

The main functionality intentionally omitted from the earlier large design is full mode-shape-image prediction. The synthesizer only needs modal frequencies and modal sensitivity at the strike and pickup positions.

---

## 4. Shape system

The runtime shape generator produces a normalized 2D mask.

Initial morph path:

```text
Morph = 0.0      Rectangle
Morph = 0.0-0.5  Rectangle -> Ellipse
Morph = 0.5      Ellipse
Morph = 0.5-1.0  Ellipse -> Triangle
Morph = 1.0      Triangle
```

`ShapeMod` changes the proportions/deformation inside the current shape family.

Example runtime call:

```python
mask = make_shape_mask(morph, shape_mod, resolution=64)
```

The network receives the resulting mask, not just the two control values. This keeps the architecture open to additional trained shapes later.

---

## 5. Neural model

Keep the network deliberately small.

### Geometry encoder

Example:

```text
64 x 64 x 1
 -> Conv 1 -> 8
 -> ReLU
 -> Conv 8 -> 16
 -> ReLU
 -> Conv 16 -> 32
 -> ReLU
 -> AdaptiveAvgPool
 -> geometry latent z
```

### Frequency head

```text
z
 -> small MLP
 -> N dimensionless modal factors
```

### Point head

```text
z + x + y
 -> small MLP
 -> N modal gains
```

The point head is shared between strike and pickup.

Initial target mode count:

```text
N = 16
```

This can later be increased if necessary.

---

## 6. Physical scaling outside the neural network

The network should not learn material or size laws that are already known analytically.

The network predicts dimensionless geometry-dependent modal factors:

```text
mu_k
```

Runtime scaling:

```text
omega_k = mu_k * sqrt(D / (rho * H)) / L^2
f_k = omega_k / (2*pi)
```

Exact normalization will be fixed when the reference solver is implemented and validated.

Benefits:

- less information for the NN to learn
- better extrapolation over size/material parameters
- physical interpretation stays visible
- size and material controls remain independent of training

---

## 7. Offline reference model

Do not build a large FEM framework.

Use a compact regular-grid reference model for a Navier/simply-supported plate.

Pipeline:

```text
shape mask
 -> sparse discrete Laplacian
 -> scipy.sparse.linalg.eigsh
 -> lowest eigenpairs
 -> modal factors + mode shapes
```

Reference output per shape:

```python
modal_factors   # shape: (N,)
mode_shapes     # shape: (N, H, W)
```

Mode shapes are needed only for generating training targets. They are not predicted as full images by the NN.

The reference implementation should stay compact and live in one file.

---

## 8. Training examples

One physical solve can generate many training examples.

For one geometry:

```text
1 reference eigen-solve
 -> N mode shapes
 -> many random strike positions
 -> many random pickup positions
 -> many training examples
```

Example dataset scale:

```text
500 shapes
x 30 strike/pickup pairs per shape
= 15,000 training examples

but only 500 eigen-solves
```

Start much smaller during development.

---

## 9. Training targets

For reference mode k:

```text
strike gain  = phi_k(strike_x, strike_y)
pickup gain  = phi_k(pickup_x, pickup_y)
```

The linear modal transfer amplitude is proportional to:

```text
A_k = strike_gain_k * pickup_gain_k
```

A differentiable training resonator renders the predicted modal response.

Reference:

```text
reference modal factors
+ reference strike gains
+ reference pickup gains
 -> reference response
```

Prediction:

```text
NN modal factors
+ NN strike gains
+ NN pickup gains
 -> predicted response
```

Primary loss follows the Neural Resonator idea:

```text
log spectral loss between predicted and reference responses
```

For stability, add a small direct modal-factor loss.

Conceptually:

```text
L = L_spectral + w_f * L_frequency
```

No large mode-shape image loss is required.

---

## 10. V12 integration

Base file:

```text
filter_resonator_plate_numba_v12.py
```

The existing V12 DSP is retained as much as possible.

### Keep

- `distribution_matrix()`
- `process_block()`
- `_strike()`
- sounddevice callback structure
- damping calculation
- `tau`
- `eta`
- `lambda`
- nonlinear state transfer
- state limiting
- excitation envelope

### Replace or adapt

#### Current frequency generation

```python
get_modes(...)
modes_to_freqs(...)
```

becomes a common modal-model interface.

During the first stage:

```python
analytic_rectangle_model(...)
```

Later:

```python
predict_modal_model(...)
```

#### Current strike weighting

Current:

```python
weight = sin(l*pi*x) * sin(m*pi*y)
```

Final:

```python
u_ex[k, :] = strike_gains[k] * u
```

#### Output pickup

Current V12:

```python
sum(states[k].imag)
```

Final:

```python
sum(pickup_gains[k] * states[k].imag)
```

This adds an actual independent pickup position.

---

## 11. Runtime morphing

The neural network must never run inside the per-sample Numba loop.

Control path:

```text
Morph / ShapeMod / Strike / Pickup changes
 -> control/UI thread
 -> create new mask
 -> run NN inference
 -> compute new physical frequencies
 -> rebuild/update Z and M
 -> pass new modal parameters to audio engine
```

The current resonator states should be preserved where possible.

Parameter smoothing should be added so continuous morphing does not produce zipper noise or discontinuous jumps.

Target behavior:

```text
Morph 0.40 -> 0.41 -> 0.42 -> 0.43
 -> continuously moving resonances
 -> no re-trigger required
```

---

## 12. Minimal project structure

Keep the implementation small:

```text
filter_resonator_plate_numba_v12.py

neural/
    shapes.py
    reference.py
    model.py
    train.py

models/
    plate_nn.pt
```

Approximate new code budget:

```text
shapes.py       ~100 lines
reference.py    ~150 lines
model.py         ~80 lines
train.py        ~150-200 lines
V12 changes     ~100 lines
```

Target total: roughly 500-600 new relevant lines, not thousands.

No Hydra, Lightning, WandB, HDF5 shard framework, Gmsh, Morley FEM framework, or large package architecture unless a concrete need appears later.

---

## 13. Development phases

### Phase 1 — Freeze V12

Create a neural working copy based on V12.

Requirement:

```text
Before NN work, it must behave and sound like V12.
```

### Phase 2 — Common modal interface

Refactor the existing rectangle formulas into a small function such as:

```python
analytic_rectangle_model(...)
```

Return:

```python
modal_factors
strike_gains
pickup_gains
```

Initially the result must reproduce the original V12 behavior.

### Phase 3 — NN hello world: rectangle frequencies only

Input:

```text
rectangle mask
```

Target:

```text
N analytical rectangle modal factors
```

No numerical solver.
No strike/pickup training.
No audio loss.

Success condition:

The network predicts the resonance pattern of unseen rectangle aspect ratios.

### Phase 4 — Point head on analytical rectangles

Add random `x, y`.

Analytical target:

```text
sin(m*pi*x) * sin(n*pi*y)
```

Success condition:

The shared point head predicts modal sensitivity for unseen positions.

At this point the complete NN architecture is proven without any numerical arbitrary-shape solver.

### Phase 5 — Small arbitrary-shape reference solver

Implement:

```python
solve_reference_modes(mask)
```

First validation:

```text
numerical rectangle vs analytical rectangle
```

Do not continue until the low modes are sufficiently close for the synthesis task.

### Phase 6 — Shape generator

Implement:

```python
make_shape_mask(morph, shape_mod)
```

Generate rectangle, ellipse, triangle and continuous intermediate shapes.

### Phase 7 — Final training

Train with:

```text
shape mask
strike_xy
pickup_xy
 -> NN
 -> modal factors + point gains
 -> differentiable resonator
 -> spectral loss
```

Save:

```text
models/plate_nn.pt
```

### Phase 8 — Real-time V12 integration

Replace the analytical modal source with:

```python
predict_modal_model(...)
```

Add controls:

- Morph
- ShapeMod
- Size
- Strike X/Y
- Pickup X/Y

Keep V12 material, damping and nonlinear controls.

### Phase 9 — A/B validation

For rectangles compare:

```text
V12 analytical model
reference solver
NN prediction
```

Check:

- modal frequencies
- response spectrum
- strike position behavior
- pickup position behavior
- audio output

Then test ellipse, triangle and morph intermediates.

---

## 14. Acceptance criteria for the final instrument

The project is complete when:

1. V12 nonlinear resonator synthesis is still operational.
2. Rectangle behavior can be reproduced closely by the NN.
3. Morph can move continuously between rectangle, ellipse and triangle.
4. Intermediate shapes produce continuous modal changes.
5. Strike position changes which modes are excited.
6. Pickup position independently changes which modes are observed.
7. Size and material parameters scale frequencies physically outside the NN.
8. `tau`, `eta` and `lambda` still control the existing nonlinear coupling.
9. NN inference happens outside the audio sample loop.
10. Morphing does not require real-time FEM.
11. The neural implementation stays small enough to be understandable and explainable as part of the project.

---

## 15. Explicit non-goals

To keep the project compact, the first complete version will not include:

- NN prediction of complete mode-shape images
- general-purpose FEM infrastructure
- structural mesh import
- arbitrary boundary-condition families
- differentiable nonlinear V12 feedback during training
- end-to-end audio generation by a neural network
- large experiment-management frameworks

These can be added only if the finished compact system demonstrates a concrete need for them.

---

## Summary

The neural network learns the difficult geometry-dependent part:

```text
shape -> modal resonance structure
shape + point -> modal spatial sensitivity
```

Known physics remains explicit:

```text
modal factors + material + size -> physical frequencies
```

The existing V12 engine remains responsible for:

```text
resonators
damping
excitation
nonlinear coupling
real-time audio
```

This gives the intended full instrument behavior while keeping the neural extension close in spirit to Neural Resonator: a compact learned resonator representation rather than a large neural FEM replacement.
