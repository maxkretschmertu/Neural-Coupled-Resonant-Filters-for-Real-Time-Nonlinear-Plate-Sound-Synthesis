# Phase 3 — Kirchhoff–Love reference solver and modal dataset

Phase 3 is an **offline** pipeline.  It does not train the neural network and it
does not change the realtime resonator DSP.  Its only job is to create trusted
geometry → modal-basis labels for Phase 4.

## Fixed physical contract

The Phase-2 geometry is unit area and is the only geometry implementation used.
The current dataset contract is:

- Kirchhoff–Love thin plate,
- homogeneous/isotropic material with the material scale factored out,
- `boundary_condition = simply_supported`,
- fixed Poisson ratio `nu = 0.30`,
- geometry is the Phase-2 rectangle → ellipse → triangle morph space.

The dimensionless generalized eigenproblem is

```
K phi = lambda M phi
```

with the plate weak form

```
a(u,v) = ∫ [(1-nu) H(u):H(v) + nu Δu Δv] dA.
```

The stored runtime modal factor is

```
mu_k = sqrt(lambda_k)
```

so Phase 2 remains unchanged:

```
omega_k = mu_k * sqrt(D/(rho*H)) / size_m^2.
```

For simply-supported Morley FEM, only the displacement `u` DOF on the boundary
is constrained.  The bending-moment condition is natural in the weak form.

## Solver stack

```
Phase-2 GeometryDescription
        ↓
Gmsh first-order triangle mesh
        ↓
scikit-fem ElementTriMorley
        ↓
K, M sparse matrices
        ↓
scipy eigsh generalized eigenproblem
        ↓
FEM mass-normalized eigenvectors
        ↓
canonical material-grid interpolation
        ↓
64×64 mode maps + modal factors
```

Heavy dependencies are imported lazily under `plate_synth.reference`, so they
are not required by the realtime synth.

Install the offline stack with:

```bash
pip install -r requirements-phase3.txt
```

## Files

- `reference/mesh.py`: Gmsh polygon meshing and mesh statistics. Dense Phase-2
  contours are downsampled for the CAD boundary while sharp corners are kept,
  so 720-point SDF contours do not force tiny boundary mesh edges.
- `reference/plate_solver.py`: Morley Kirchhoff eigenproblem, simply-supported BC,
  mass normalization, residuals and orthogonality QA.
- `reference/mode_sampling.py`: evaluate FEM displacement modes on the exact
  Phase-2 canonical material grid, normalize targets, canonicalize sign and mark
  near-degenerate eigenvalue groups.
- `reference/validation.py`: MAC/subspace tools and hard QA gates.
- `reference/dataset.py`: deterministic split generation, multiprocessing,
  HDF5 sharding, resume, manifest and failed-sample logging.

## Solver validation must precede full generation

Run:

```bash
python tools/validate_plate_solver.py
```

This checks rectangle aspect ratios 1, 1.5, 2, 3 and 4 at several mesh sizes,
compares modal factors against the analytical simply-supported rectangle
backend, calculates MAC with Hungarian mode assignment, and writes:

```text
validation/phase3/rectangle_frequency_mac.csv
validation/phase3/mesh_convergence.csv
```

Do not start the 10k dataset until the validation report shows acceptable
frequency/MAC/convergence behaviour.  `mesh_edge_length=0.055` in the supplied
configs is a starting point, not a claim of converged accuracy; validation may
require changing it.

## Target representation

Each accepted sample stores:

```text
sdf                     float32 [64,64]
mask                    uint8   [64,64]
morph, shape_mod        float32
modal_factors           float32 [32]
eigenvalues             float64 [32]
mode_shapes             float32 [32,64,64]
area_weights            float32 [64,64]
inner_product_weights   float32 [64,64]
degenerate_group_id     int16   [32]
eigensolver_residual    float64 [32]
```

plus FEM/mesh quality metadata.  `store_mesh=true` optionally stores flattened
mesh coordinates/connectivity as variable-length HDF5 arrays for debugging; it
is off by default because retaining every training mesh is expensive.

The FEM eigenvectors are first mass-normalized with the FEM mass matrix.  After
interpolation they are normalized again with the Phase-2 material-grid inner
product.  A deterministic largest-absolute-sample sign convention makes
ordinary nondegenerate modes easier to inspect; `degenerate_group_id` is still
stored because sign canonicalization cannot make a repeated eigenspace unique.
Phase 4 must use subspace-aware loss/matching for repeated or near-repeated
modes.

## Split policy

The split is fixed before any FEM solve:

- training: scrambled Sobol points plus exact rectangle/ellipse/triangle family
  endpoints and shape-mod extremes,
- validation: independent scrambled Sobol sequence,
- test: deterministic interior grid that is never injected into training.

The final config contains 8000/1000/1000 train/validation/test geometries.  A
smaller pilot config contains 256/64/64.

## Pilot and final generation

Start with very small smoke work:

```bash
python tools/generate_modal_dataset.py \
    --config configs/dataset_phase3_pilot.json --limit 8 --workers 1
```

Then generate and inspect the full pilot:

```bash
python tools/generate_modal_dataset.py --config configs/dataset_phase3_pilot.json
python tools/inspect_modal_dataset.py data/modal_dataset_pilot --plot
```

Only after solver validation and pilot inspection:

```bash
python tools/generate_modal_dataset.py --config configs/dataset_phase3.json
```

Generation is resumable: existing `sample_id`s in HDF5 shards are skipped.
Failures are appended to `failed_samples.jsonl` rather than silently entering
the dataset.  Only the main process writes HDF5; FEM/Gmsh work is parallelized
across independent worker processes.

## Quality gates

A sample is rejected if it contains non-positive/unsorted eigenvalues,
non-finite mode maps, excessive generalized-eigen residual, excessive FEM mass
orthogonality error, or material-grid normalization error.  Thresholds are
configurable and recorded in the manifest.

## Phase-4 handoff

The neural training code should need no Gmsh or FEM knowledge.  Its core inputs
and targets are simply:

```
X = SDF [64,64]
Y_frequency = modal_factors [32]
Y_shapes = mode_shapes [32,64,64]
W = inner_product_weights [64,64]
groups = degenerate_group_id [32]
```
