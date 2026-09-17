# Phase 2 implementation

This phase defines the geometry/morph and state-transfer infrastructure that the future neural modal model will use. The new package does not preserve compatibility helpers from the earlier rectangle scripts.

## Geometry space

The controls are:

- `morph` in `[0,1]`: rectangle -> ellipse/circle -> triangle,
- `shape_mod` in `[0,1]`: compact -> slender inside each family,
- `size_m`: physical sqrt(area), completely separate from shape.

All canonical shapes are normalized to unit area. Rectangle, ellipse and triangle are represented as star-shaped radial contours `R(theta)`. Morphing interpolates radial contours with smoothstep weights and renormalizes area. A signed-distance field is generated on a fixed Cartesian grid for the future neural input.

The interactive default uses 360 contour samples at a 64x64 SDF resolution. Offline dataset generation can request 720/1440 contour samples explicitly; SDF construction is control-rate/offline work and is never part of the audio callback.

## Boundary-condition contract

The boundary condition is explicit model metadata. Phase 2 supports exactly:

```text
simply_supported
```

The analytical rectangle endpoint, future numerical labels and the trained neural model must use the same boundary condition. A future free/clamped model is a separate contract and must not silently reuse the current mode-shape edge sampling assumptions.

## Material coordinates

Mode shapes live on one canonical unit-disk material grid. A material point `(u,v)` maps to a physical point by preserving angle and normalized radial coordinate. This means strike and pickup stay attached to the same relative material position while geometry changes.

The radial map has Jacobian `R(theta)^2`. `MaterialGrid` stores two explicit quadratures:

- `area_weights`: normalized-geometry dA weights integrating to the canonical geometry area,
- `inner_product_weights`: the same weights normalized to sum to one for mode normalization, MAC-like comparisons and basis projection.

Neither contains physical mass. Physical lumped mass weights are computed separately as

```text
rho * H * size_m^2 * area_weights
```

so modal-geometry and physical/material scaling remain separate.

## Modal contract

`ModalBackend` receives geometry, exact requested `n_modes`, and the explicit boundary-condition identifier. It returns geometry-dependent, material-independent modal factors and mode shapes.

The modal-factor convention is

```text
omega_k = modal_factor_k * sqrt(D/(rho*H)) / size_m^2
```

and `frequency_scale` is applied afterwards. Material, size, damping, pickup/strike and nonlinear coupling are outside the future neural network.

`AnalyticRectangleBackend` is only a development backend for the rectangle endpoint. It uses the simply-supported Kirchhoff rectangle expression. Candidate indices are expanded adaptively until the returned modes are provably the globally lowest requested N modes, including slender rectangles. Morphed/circle/triangle geometries are rejected instead of receiving approximate fake modes.

## Live state projection

The project fixes one state-transfer convention:

1. old and new mode shapes are represented on the same canonical material disk,
2. the old displacement/velocity field is transported with those material coordinates as geometry changes,
3. that transported field is least-squares projected onto the new basis using the *new geometry's* inner product.

Thus

```text
P = (Phi_new^T W_new Phi_new)^-1 Phi_new^T W_new Phi_old
```

with a small numerical regularizer. For uniform `rho` and `H`, the scalar physical mass factor cancels and normalized geometric weights are sufficient.

This is a quasistatic modal-basis transfer. It preserves an instantaneous material-coordinate displacement/velocity field as well as the truncated new basis allows; it is not the exact time-dependent PDE of a continuously moving boundary, which would contain additional basis-derivative terms.

Complex resonator states are converted to displacement/velocity coordinates, projected, and reconstructed with the new modal frequencies/damping. The matrix solve is done outside the audio callback; only the small precomputed transform is applied at a block boundary.

## Preview versus active audio

Before the neural backend exists, the GUI can preview the entire rectangle/ellipse/triangle geometry space while only the rectangle endpoint has a modal model.

These states are now explicit:

- filled contour: current geometry preview,
- dashed contour (when different): currently audible geometry.

If a preview geometry is unsupported, the complete audio model is frozen at the last valid geometry. Material/tuning/spatial updates are kept as control state but are not applied to the stale audio basis. Returning to a supported geometry publishes the accumulated controls as one consistent update.

## Tests

Run:

```bash
python tests/test_phase2.py
```

The tests cover:

- unit-area rectangle/circle/triangle endpoints and SDF sign,
- shape modulation,
- material-coordinate roundtrip,
- area/inner-product/physical-mass weight semantics,
- globally correct lowest-N rectangle eigenmode ordering through aspect ratio 4,
- explicit simply-supported boundary contract,
- rejection of fake non-rectangle modes,
- identity and permutation-exact basis projection,
- material-field projection error for changing rectangle shapes,
- backend dependency behavior,
- separation/freeze of preview geometry and active audio geometry.
