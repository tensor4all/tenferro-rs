# DB Replay Validator Design

**Date:** 2026-03-09

**Goal:** Validate the correctness of the JSON oracle database itself by replaying stored cases, recomputing observables and first-order derivatives from the materialized inputs, and checking that the stored references are reproducible.

## Motivation

`tensor-ad-oracles` already validates schema shape, duplicate `case_id` hygiene, and generator-time agreement between PyTorch AD and directional finite differences. That is necessary but not sufficient for database integrity.

If the generator contains a consistent bug, it can still emit a self-consistent but wrong database. The repository therefore needs a second line of defense that replays published JSON cases from disk and recomputes:

- forward observables
- PyTorch JVP
- PyTorch VJP
- finite-difference JVP
- expected ill-defined spectral errors

This replay validator is the closest Python analogue to the finite-difference helper style currently used in `tenferro-linalg/tests/linalg_tests.rs`.

## Chosen Approach

Implement a repository-local Python replay validator inside `tensor-ad-oracles` instead of depending on `tenferro-rs`.

The validator will:

1. load case JSONL records from `cases/`
2. decode stored tensors back to torch tensors
3. rebuild the op-specific observable
4. replay each paired probe
5. compare recomputed values against stored `pytorch_ref` and `fd_ref`

This keeps the database repository self-contained. Later, the same logic can be ported to `tenferro-rs` consumer tests.

## Rejected Alternatives

### Metadata-only validation

Checking only schema and saved scalar identities is too weak. It cannot detect a generator that wrote consistently incorrect references.

### Direct tenferro-rs integration first

Using Rust-side rules immediately would mix two goals:

- database self-validation
- downstream consumer validation

That coupling would make the new repository depend on tenferro internals before its own correctness contract is stabilized.

## Architecture

Add a replay layer under `validators/`:

- `validators/encoding.py`
  - decode JSON tensor objects into torch tensors
- `validators/replay.py`
  - replay a success or error case from disk
  - recompute observables and derivatives
  - apply tolerance checks
- `validators/case_loader.py`
  - load one case, one family, or all families

The validator will reuse existing generator-side helpers where appropriate:

- `generators/observables.py`
- `generators/fd.py`
- `generators/runtime.py`

This keeps the observable semantics identical between generation and replay.

## Replay Contract

For every `success` case, replay must verify:

1. stored observable shape and keys match recomputed observable shape and keys
2. recomputed `pytorch_ref.jvp` matches stored `pytorch_ref.jvp`
3. recomputed `pytorch_ref.vjp` matches stored `pytorch_ref.vjp`
4. recomputed `fd_ref.jvp` matches stored `fd_ref.jvp`
5. recomputed values still satisfy the case comparison tolerance
6. adjoint consistency holds:
   - `<bar_y, Jv_fd> ~= <J*bar_y_torch, v>`

For every `error` case, replay must verify that backward through the gauge-dependent observable raises an `"ill-defined"` runtime error.

## Scope

The first validator version will cover all currently materialized v1 families:

- `svd/u_abs`
- `svd/s`
- `svd/vh_abs`
- `svd/uvh_product`
- `svd/gauge_ill_defined`
- `eigh/values_vectors_abs`
- `eigh/gauge_ill_defined`
- `solve/identity`
- `cholesky/identity`
- `qr/identity`
- `pinv_singular/identity`

The validator will operate on existing stored probes. It will not generate new probes and will not widen the schema.

## Numerical Policy

Replay must not invent a second tolerance system. It should respect the per-case `comparison` already stored in JSON.

Directional finite-difference replay will use the stored `fd_ref.step` for each probe. This avoids drift from future changes to global FD policy code and directly validates the published artifact.

## Testing Strategy

Tests should be layered:

1. unit tests for tensor decoding and replay utilities
2. focused family tests, starting with `solve/identity`
3. an integration test that replays every JSONL file in `cases/`

The repository-level success condition is:

```bash
uv run python -m unittest discover -s tests -v
```

with replay validation included.

## Migration Path To tenferro-rs

After the replay validator is stable in `tensor-ad-oracles`, the logic can be transferred to `tenferro-rs` in one of two ways:

- Rust integration tests that read the same JSONL database
- a small cross-language harness that compares tenferro AD outputs against the stored references

That future migration should reuse the same case format and replay semantics defined here.
