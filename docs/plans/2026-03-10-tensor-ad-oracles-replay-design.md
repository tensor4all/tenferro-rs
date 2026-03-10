# tensor-ad-oracles Replay Design

## Goal

Integrate the published `tensor-ad-oracles` JSON database into `tenferro-rs`
so that `tenferro-linalg` replays the stored derivative probes against the
Rust implementation as part of normal workspace test execution.

## Context

- `tensor-ad-oracles` publishes machine-readable JSONL case families for
  `svd`, `eigh`, `solve`, `cholesky`, `qr`, and `pinv_singular`.
- The schema fixes tensor storage to `row_major`; `tenferro-rs` is
  column-major, so the replay harness must own the layout conversion.
- The oracle repository also carries schema, sample commands, and replay
  semantics. Vendoring only the case files would drop useful context.
- The current published case corpus is small enough to vendor comfortably.

## Decision

Vendor the full `tensor-ad-oracles` repository into `tenferro-rs` as a git
subtree under `third_party/tensor-ad-oracles/`.

Implement a native Rust replay harness under `tenferro-linalg/tests/oracle_db/`
and run it from the normal `cargo test --workspace --release` path.

## Architecture

### Vendored Oracle Snapshot

- Path: `third_party/tensor-ad-oracles/`
- Contents: full upstream repository snapshot, including:
  - `cases/`
  - `schema/`
  - sample commands and repository README
  - Python-side replay/reference code for maintenance cross-checks

The vendored subtree is read-only from `tenferro-rs`'s perspective. Updates
arrive by subtree pull, not by editing files in place.

### Replay Harness Placement

Place the replay harness in `tenferro-linalg/tests/oracle_db/`:

- `main.rs`: top-level integration tests and summary assertions
- `db.rs`: case discovery and JSON decoding into Rust structs
- `decode.rs`: row-major database tensor decoding into tenferro tensors
- `observables.rs`: mapping from oracle observable kinds to tenferro outputs
- `replay.rs`: forward/JVP/VJP replay logic and failure reporting

This keeps production code free of test-only oracle parsing while preserving
module-local structure inside the integration test tree.

## Data Contract

### Input Tensor Decoding

The oracle schema requires:

- `dtype`
- `shape`
- `order`
- `data`

`order` is fixed to `row_major`, so the decoder in `tenferro-rs` must:

1. parse the stored flat payload
2. reconstruct the logical tensor in row-major order
3. materialize an equivalent tenferro tensor in column-major storage

The decoder must preserve logical shape and values exactly. Probe directions
and cotangents use the same decode path as primals.

### Supported Oracle Surface

Initial replay scope:

- `success` families:
  - `solve/identity`
  - `cholesky/identity`
  - `qr/identity`
  - `svd/u_abs`
  - `svd/s`
  - `svd/vh_abs`
  - `svd/uvh_product`
  - `eigh/values_vectors_abs`
  - `pinv_singular/identity`
- `error` families:
  - `svd/gauge_ill_defined`
  - `eigh/gauge_ill_defined`

The replay harness must fail hard on unknown published observables, dtype
mismatches, or layout mismatches. It must not silently skip `success` cases.

## Replay Contract

### Success Cases

For each `success` case:

1. decode inputs
2. evaluate the oracle-defined observable with `tenferro-linalg`
3. replay each stored probe
4. compare tenferro JVP against `fd_ref.jvp`
5. check adjoint consistency using the stored cotangent and direction
6. where the mapping is direct, compare tenferro output against
   `pytorch_ref.{jvp,vjp}` as well

The harness should use the per-case `comparison` tolerances from the database.

### Error Cases

For each `error` case:

1. decode inputs
2. attempt the operation/observable path
3. assert that tenferro rejects it
4. classify the rejection against the stored `reason_code`

The gauge-ill-defined spectral cases should be treated as expected failures,
not as unsupported local skips.

## CI Behavior

This replay is intended to run in normal workspace tests.

That means:

- no sibling checkout dependency
- no environment-variable opt-in for the default happy path
- no network access during tests

If the corpus grows enough to make default workspace tests too slow, the
fallback is:

- keep a smoke replay in the default workspace test suite
- move the full replay into an additional required CI job

That is not the starting point.

## Maintenance Workflow

### Oracle Updates

When `tensor-ad-oracles` changes:

1. update the vendored subtree
2. rerun the tenferro replay harness
3. fix decode or observable mapping drift in the same PR if needed

### Local Debugging

The vendored full repository preserves:

- upstream README
- sample commands
- schema
- Python replay/reference code

This makes it possible to diagnose a Rust replay mismatch by comparing against
the upstream Python-side behavior without requiring a separate checkout.

## Risks

### Schema Drift

If upstream adds new observable kinds or tensor metadata, the replay harness
will start failing. This is desirable: the failure is the signal that tenferro
must adapt to the new published contract.

### Runtime Cost

Always-on replay increases test time. The current corpus size is small enough
that this is acceptable, but the harness should still report failures
succinctly and avoid duplicate decoding work where possible.

### Observable Mapping Ambiguity

The database validates derivative-relevant observables, not necessarily raw
decomposition outputs. The replay layer therefore needs explicit mapping code
per observable family rather than trying to genericize too early.
