# Phase 2 implementation

This phase defines the complete geometry/morph infrastructure that the future neural modal model will learn. The new code does not keep compatibility helpers from the earlier rectangle implementation.

## Geometry space

The controls are:

- `morph` in `[0,1]`: rectangle -> ellipse/circle -> triangle,
- `shape_mod` in `[0,1]`: compact -> slender inside each family,
- `size_m`: physical sqrt(area), completely separate from shape.

All canonical shapes are normalized to unit area. Rectangle, ellipse and triangle are represented as star-shaped radial contours `R(theta)`. Morphing interpolates radial contours with smoothstep weights and renormalizes area. A signed-distance field is generated on a fixed Cartesian grid for the future neural input.

## Material coordinates

Mode shapes live on one canonical unit-disk material grid. A material point `(u,v)` maps to a physical point by preserving angle and normalized radial coordinate. This means strike and pickup stay attached to the same relative material position while geometry changes.

The grid also stores Jacobian/integration weights for modal normalization, MAC comparisons and basis projection.

## Modal contract

`ModalBackend` receives only geometry and `n_modes`. It returns geometry-dependent, material-independent modal factors and mode shapes.

The modal-factor convention is

```
omega_k = modal_factor_k * sqrt(D/(rho*H)) / size_m^2
```

and `frequency_scale` is applied afterwards. Material, size, damping, pickup/strike and nonlinear coupling are outside the future neural network.

`AnalyticRectangleBackend` is only a development backend for the rectangle endpoint. It uses the correct simply-supported rectangular plate expression and sorted modes. It deliberately rejects all morphed/circle/triangle geometries instead of generating approximate fake modes. Those geometries become audible only when a trained neural backend is added.

## Live state projection

`state_projection.py` computes a weighted least-squares basis map

```
P = (Phi_new^T W Phi_new)^-1 Phi_new^T W Phi_old
```

on the canonical material grid. Complex resonator states are converted to displacement/velocity coordinates, projected and reconstructed with the new modal frequencies/damping. The matrix solve is done outside the audio callback; only the small precomputed transform is applied at the block boundary.

## UI

The plate canvas renders the actual morphed contour. Strike and pickup are stored in material coordinates and therefore move coherently with the geometry. Geometry preview works over the complete rectangle/ellipse/triangle morph range even before a neural backend exists.

## Tests

Run:

```bash
python tests/test_phase2.py
```

The tests cover unit-area endpoints, shape modulation, SDF sign, material-coordinate roundtrip, integration weights, sorted analytical rectangle modes, rejection of fake non-rectangle modes, identity/cross-basis projection and backend dependency behavior.
