# tenferro-linalg Oracle DB Replay Design

## Goal

Run `tenferro-linalg` AD rules against the published `tensor-ad-oracles` JSON database and report whether the current implementation matches the stored derivative oracles.

## Scope

- Add a Rust integration harness under `tenferro-linalg/tests/`.
- Read JSONL cases from `tensor-ad-oracles`.
- Replay all supported `float64` success records for:
  - `svd`
  - `eigen` (mapped from DB `eigh`)
  - `solve`
  - `cholesky`
  - `qr`
  - `pinv_singular`
- Explicitly account for unsupported DB records.

## Constraints

- `tenferro-rs` must not gain a hard CI dependency on a sibling checkout.
- `tenferro-linalg` AD APIs are real-only for the relevant spectral rules, so the two `complex128` gauge-error DB records are not replayable through the current Rust API.
- The DB compares processed observables, not raw decomposition outputs. The harness must implement the same observable semantics.

## Recommended Approach

### Option 1: `tenferro-linalg` integration test harness

Add a Rust integration test that:

- finds `tensor-ad-oracles` via `TENSOR_AD_ORACLES_ROOT` or `../tensor-ad-oracles`
- parses JSONL records into small Rust structs
- decodes DB tensors into `Tensor<f64>`
- maps DB observables to `tenferro-linalg` forward/rrule/frule APIs
- compares:
  - tenferro observable JVP vs stored `fd_ref.jvp`
  - tenferro VJP vs stored `pytorch_ref.vjp`
  - adjoint identity using stored probe data

Recommended because it tests the real Rust implementation directly and keeps the consumer-side contract next to `tenferro-linalg`.

### Option 2: Python harness calling Rust via subprocess

Run Rust from `tensor-ad-oracles` and parse stdout/json back in Python.

Rejected for now because it adds an unnecessary cross-language boundary and makes failures harder to debug.

### Option 3: Extend `tenferro-capi` and validate through FFI

Rejected for now because it increases scope and tests the C surface more than the native Rust AD rules.

## Design

### Test Layout

Create one integration test crate:

- `tenferro-linalg/tests/oracle_db/main.rs`

and split helpers into:

- `tenferro-linalg/tests/oracle_db/db.rs`
- `tenferro-linalg/tests/oracle_db/decode.rs`
- `tenferro-linalg/tests/oracle_db/replay.rs`

This keeps the test harness readable and matches the repository rule to avoid monolithic files.

### Data Model

Use small `serde` structs only for the DB subset that the Rust consumer needs:

- top-level case metadata
- success probes
- `comparison`
- tensor objects for `float64`

Complex error cases do not need full tensor decoding because the harness will mark them unsupported with an explicit reason.

### Observable Semantics

Implement DB-aligned observable adapters:

- `solve/identity`: raw solve output
- `cholesky/identity`: raw factor
- `qr/identity`: raw `(q, r)`
- `svd/u_abs`: `abs(u)`
- `svd/s`: `s`
- `svd/vh_abs`: `{ s, abs(vt) }`
- `svd/uvh_product`: `{ s, u @ vt }`
- `eigh/values_vectors_abs`: map DB `eigh` to Rust `eigen`, then `{ values, abs(vectors) }`
- `pinv_singular/identity`: replay `pinv(a @ b^T)` and push forward / pull back through the factorized input representation

### AD Mapping

For VJP:

- convert DB cotangents on processed observables into raw cotangents for `svd_rrule`, `qr_rrule`, `eigen_rrule`, `solve_rrule`, `cholesky_rrule`, `pinv_rrule`
- for `pinv_singular`, apply chain rule from matrix cotangent back to factor cotangents

For JVP:

- call the corresponding `*_frule`
- convert raw tangents to processed-observable tangents
- for `pinv_singular`, use `d(a b^T) = da b^T + a db^T`

### Compatibility Policy

Supported in v1:

- all 348 `success` records in `tensor-ad-oracles` at commit `b29d6a8`

Explicitly unsupported in v1:

- `svd_c128_gauge_ill_defined_001`
- `eigh_c128_gauge_ill_defined_001`

The test should assert that the unsupported set is exactly these two records, so future silent coverage loss is visible.

### Test Outcome

The integration test should produce a summary:

- number of validated records
- number of unsupported records
- first failure details if mismatches exist

If supported records disagree with DB references, the test should fail.

