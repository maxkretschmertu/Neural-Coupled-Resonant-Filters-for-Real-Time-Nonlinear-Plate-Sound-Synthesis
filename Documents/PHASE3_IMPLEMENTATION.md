# Phase 3 — Kirchhoff–Love reference solver and modal dataset

Phase 3 is an **offline** pipeline. It does not train the neural network and it does not change the realtime resonator DSP. Its only job is to create trusted geometry → modal-basis labels for Phase 4.

## Fixed physical contract

The Phase-2 geometry is unit area and is the only geometry implementation used. The dataset contract is:

- Kirchhoff–Love thin plate,
- homogeneous/isotropic material with material scale factored out,
- `boundary_condition = simply_supported`,
- fixed Poisson ratio `nu = 0.30`,
- Phase-2 rectangle → ellipse → triangle morph space.

The dimensionless generalized eigenproblem is

```text
K phi = lambda M phi
```

with weak form

```text
a(u,v) = ∫ [(1-nu) H(u):H(v) + nu Δu Δv] dA.
```

The stored runtime modal factor is

```text
mu_k = sqrt(lambda_k)
```

so the runtime remains

```text
omega_k = mu_k * sqrt(D/(rho*H)) / size_m^2.
```

For simply-supported Morley FEM, only the displacement `u` DOF on the boundary is constrained; the bending-moment condition is natural in the weak form.

## Solver stack

```text
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

Heavy dependencies are imported lazily under `plate_synth.reference`, so they are not required by the realtime synth.

```bash
pip install -r requirements-phase3.txt
```

## Files

- `reference/mesh.py`: Gmsh polygon meshing, edge statistics, mesh-area error and an approximate symmetric boundary Hausdorff distance against the Phase-2 target contour.
- `reference/plate_solver.py`: Morley Kirchhoff eigenproblem, simply-supported BC, FEM mass normalization, residuals and orthogonality QA.
- `reference/mode_sampling.py`: FEM displacement modes sampled on the canonical material grid, target normalization, sign convention, degeneracy groups, cutoff loss masks and raw grid-quadrature diagnostics.
- `reference/validation.py`: frequency-aware Hungarian reference matching, MAC/subspace tools and hard per-sample QA gates.
- `reference/dataset.py`: deterministic disjoint splits, multiprocessing, HDF5 sharding, config fingerprints, safe resume, manifest and failure logging.

## Solver validation must precede full generation

Run:

```bash
python tools/validate_plate_solver.py
```

The validation covers rectangle aspect ratios 1, 1.5, 2, 3 and 4 over several mesh sizes. Raw eigensolver indices are **not** trusted. FEM modes are assigned to analytical reference modes with a Hungarian solve whose cost is led by relative modal-factor error with only a small MAC tie-break term. This makes close numerical mode swaps harmless. Within repeated/near-repeated analytical groups, selected FEM factors are sorted before scalar convergence comparisons so arbitrary eigensolver order does not create fake convergence jumps.

Nondegenerate reference modes use weighted MAC after matching. Degenerate analytical eigenspaces use a basis-invariant subspace projection score on the full matched eigenspace, so a physically equivalent rotation inside a repeated eigenspace is not incorrectly penalized.

The tool writes:

```text
validation/phase3/rectangle_frequency_shape.csv
validation/phase3/mesh_convergence.csv
```

`rectangle_frequency_shape.csv` records both the reference mode and the matched FEM mode index.

The validator exits with code 1 if the configured hard gates fail. Default gates include:

- first 16 modal factors: max relative error 0.5%,
- modes above 16: max relative error 1%,
- minimum MAC/subspace score 0.95,
- second-finest → finest modal change 2%,
- mesh/target relative area error 0.5%,
- approximate boundary distance 0.01 in normalized coordinates.

These are acceptance gates, not claims that the supplied `mesh_edge_length=0.055` will automatically satisfy them. If validation fails, refine the mesh/config and rerun validation before generating labels. A useful focused convergence run is:

```bash
python tools/validate_plate_solver.py --aspects 1 --h 0.055 0.04 0.028 0.02
```

then, if that converges satisfactorily, repeat over all aspect ratios:

```bash
python tools/validate_plate_solver.py --h 0.04 0.028 0.02
```

## Target representation

Each accepted sample stores:

```text
sdf                            float32 [64,64]
mask                           uint8   [64,64]
morph, shape_mod               float32
modal_factors                  float32 [32]
eigenvalues                    float64 [32]
mode_shapes                    float32 [32,64,64]
area_weights                   float32 [64,64]
inner_product_weights          float32 [64,64]
degenerate_group_id            int16   [32]
mode_loss_mask                 float32 [32]
cutoff_group_complete          uint8
eigensolver_residual           float64 [32]
mass_orthogonality_error       float64
raw_grid_area_integral         float64
raw_grid_area_relative_error   float64
mesh_relative_area_error       float64
mesh_boundary_hausdorff_approx float64
```

plus mesh-size/count metadata. `store_mesh=true` optionally retains flattened mesh coordinates/connectivity for debugging.

The FEM eigenvectors are mass-normalized with the FEM mass matrix. After interpolation they are normalized again with the Phase-2 material-grid inner product. A deterministic largest-absolute-sample sign convention resolves only the sign ambiguity. Repeated eigenspaces remain marked by `degenerate_group_id`; Phase 4 must use subspace-aware loss/matching for them.

If the fixed output cutoff falls through a near-degenerate group, `mode_loss_mask` disables the incomplete stored portion from ordinary per-mode losses instead of deleting symmetric geometries from the dataset.

## Dataset-integrity contract

Every config receives a SHA-256 fingerprint covering all settings that can change sample identity, numerical labels, QA thresholds or stored schema. Execution-only settings such as worker count, compression level and shard size do not affect the fingerprint.

Every finalized HDF5 shard and `manifest.json` stores this fingerprint. Resume performs a hard compatibility check before reading any completed IDs. A changed mesh edge length, Poisson ratio, modal count, grid resolution, QA threshold, split definition, etc. therefore cannot silently mix old and new labels in one output directory. Older pre-fingerprint datasets must use a new output directory or be removed explicitly.

Shards are first written as `.partial.h5`, flushed/closed, then atomically renamed to `.h5`. Only finalized shards participate in resume. Duplicate `sample_id`s across finalized shards are treated as corruption.

Train/validation/test parameter pairs are also checked for duplicates within a split and for exact overlap between splits before any FEM work starts.

## Per-sample QA

A sample is rejected before HDF5 storage if any of these invariants fail:

- positive, finite, sorted eigenvalues,
- positive, finite modal factors,
- `modal_factor**2 ≈ eigenvalue`,
- finite mode maps,
- eigensolver residual threshold,
- FEM mass-orthogonality threshold,
- material-grid mode normalization threshold,
- nonnegative finite area/inner-product weights,
- area-weight and inner-product-weight sum consistency,
- raw (unrenormalized) material-grid area quadrature threshold,
- mesh-vs-target relative area threshold,
- mesh-vs-target approximate boundary Hausdorff threshold.

Failed samples are logged to `failed_samples.jsonl` with their config fingerprint and traceback.

## Split policy

The split is fixed before any FEM solve:

- training: scrambled Sobol points plus exact rectangle/ellipse/triangle family endpoints and shape-mod extremes,
- validation: independent scrambled Sobol sequence,
- test: deterministic interior grid never injected into training.

The final config is 8000/1000/1000 train/validation/test. The pilot config is 256/64/64.

## Pilot and final generation

First run the cheap/unit checks:

```bash
python tests/test_phase3.py
```

The optional real Gmsh/scikit-fem integration test is reported as `SKIP` unless `PLATE_SYNTH_RUN_FEM_TESTS=1` is set. Use the shell-appropriate syntax:

**Windows cmd.exe / Anaconda Prompt**

```bat
set PLATE_SYNTH_RUN_FEM_TESTS=1 && python tests/test_phase3.py
```

**PowerShell**

```powershell
$env:PLATE_SYNTH_RUN_FEM_TESTS="1"
python tests/test_phase3.py
```

**bash/zsh**

```bash
PLATE_SYNTH_RUN_FEM_TESTS=1 python tests/test_phase3.py
```

Then validate the solver:

```bash
python tools/validate_plate_solver.py
```

Start with a small smoke generation:

```bash
python tools/generate_modal_dataset.py \
    --config configs/dataset_phase3_pilot.json --limit 32 --workers 1
```

Then generate and inspect the full pilot:

```bash
python tools/generate_modal_dataset.py --config configs/dataset_phase3_pilot.json
python tools/inspect_modal_dataset.py data/modal_dataset_pilot --plot
```

Only after validation and pilot inspection:

```bash
python tools/generate_modal_dataset.py --config configs/dataset_phase3.json
```

## Phase-4 handoff

The neural training code should need no Gmsh/FEM knowledge. Its core data are:

```text
X = SDF [64,64]
Y_frequency = modal_factors [32]
Y_shapes = mode_shapes [32,64,64]
W = inner_product_weights [64,64]
groups = degenerate_group_id [32]
loss_mask = mode_loss_mask [32]
```
