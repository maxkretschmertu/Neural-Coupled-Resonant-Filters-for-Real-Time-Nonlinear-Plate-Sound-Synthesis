# Phase 1 implementation

This branch contains the first implementation of the modal-synth refactor
described in `NN_MODAL_REFACTOR_PLAN.md`.

## What changed

The historical `filter_resonator_plate_numba_v12.py` is intentionally left
untouched. The new implementation lives in `src/plate_synth/`.

Key boundaries:

- `config.py` — immutable synth controls.
- `modal_backend.py` — common `ModalBasis` / `ModalBackend` contract and the
  v12-compatible `LegacyRectangleBackend`.
- `spatial.py` — analytical legacy mode sampling plus generic bilinear
  mode-map sampling for the future neural backend.
- `resonator_core.py` — existing nonlinear Numba resonator model, separated
  from GUI and model construction.
- `audio_engine.py` — control-rate snapshot building and block-boundary model
  swapping.
- `ui.py` — Tk interface with separate strike and pickup positions.
- `run_synth.py` — repository-root entry point.

## Important compatibility decisions

The Phase-1 legacy backend intentionally preserves two historical v12 details:

1. `legacy_mode_order=x` still creates `x * (x - 1)` mode pairs.
2. The v12 rectangle frequency expression is preserved exactly.

These are regression choices, not endorsements of those conventions for the
future neural backend.

The pickup is new. With `pickup_enabled=False` (the default), pickup weights
are all one, which preserves the v12 equal-weight output sum. Enabling the
pickup changes the output to a spatially weighted readout.

## Realtime boundary

The audio callback no longer:

- rebuilds modal frequencies,
- rebuilds resonator poles,
- rebuilds the coupling matrix,
- samples GUI spatial controls,
- updates Tk widgets,
- prints callback status.

A complete `ModelSnapshot` is prepared on the control thread and swapped at
the beginning of an audio block. The callback still creates the current
excitation block and executes the Numba kernel.

## Run

From repository root:

```bash
python run_synth.py
```

Dependencies remain the same basic stack as v12:

```bash
pip install numpy numba sounddevice
```

Tkinter must be available in the Python installation.

## Smoke test

```bash
python tests/test_phase1_smoke.py
```

This checks:

- the legacy `x * (x - 1)` mode count,
- the first v12 modal frequency against the historical expression,
- analytical strike/pickup shape weighting,
- the Numba core with the v12-compatible equal-weight pickup path.

## Next

Phase 2 will add the shared geometry representation and rectangle -> circle ->
triangle morph space. No fake circle/triangle modal solver is introduced here;
those modes will later come from the trained neural modal backend.
