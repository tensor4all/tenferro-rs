# Full-matrices SVD and least-squares solve (issue #1446)

Base: `origin/main` at `81031f21` (Add Provenance and Citation Policy).

## Summary

Adds two dense-linear-algebra capabilities to `tenferro-linalg`:

- **Full-matrices SVD** (`svd_full`): returns square `U (m x m)` and
  `Vh (n x n)` so the trailing `n - rank` rows of `Vh` span the input's right
  nullspace. `S` keeps `min(m, n)` singular values. Value-only; AD unsupported.
- **Least-squares** (`lstsq`): QR-based solve of tall/square, full-column-rank
  systems `argmin_x ||A x - b||`.

Both are exposed on the eager (`EagerTensorLinalgExt`) and traced
(`TracedTensorLinalgExt`) surfaces.

## Context read

- `extension.rs` (`LinalgOp`, `SvdOptions`/`SvdGauge`, `*_meta`,
  `payload_hash`, `prune_outputs`, `execute_linalg`), `backend.rs`
  (`LinalgBackend` trait + `Error::unsupported` defaults, `svd_with_options`),
  `cpu/backend.rs` (`linalg_provider_kind` dispatch), `cpu/linalg/faer_linalg.rs`
  (`svd_core`/`svd_2d`/`svd_view`, free `svd`), `cpu/linalg/lapack_linalg/svd.rs`.
- `eager_ext.rs` (direct-op surface), `eager_composites.rs` (`pinv`/`inv`/`solve`
  compositions — the template for `lstsq`), `eager_backend.rs` (eager
  `LinalgBackend` forwarder), `traced.rs` (`svd`, `solve`, helpers).
- AD: `ad.rs` (`linearize`/`linear_transpose` dispatch), `ad/support.rs`
  (machine-readable AD manifest with `jvp`/`vjp`/`route`/`caveats`) and tests.

The issue body predates the current API: it references an `SvdOptions` struct
exposing only `gauge`/`derivative_eps` and hardcoded `svd_meta` line numbers.
On the base commit `SvdOptions` now also drives a `svd_with_options` backend
entry and a `SvdGauge` canonicalization pass, and the AD manifest has been
rebuilt around JVP/VJP routes. Design was reconciled against the merged code.

## Design decisions

### Full SVD: separate `svd_full` entry point (not an `SvdOptions` field)

The issue offered two shapes. Even though `SvdOptions` now exists, a separate
op/entry point is the cleaner fit and matches the existing `svd_values` / `SvdVals`
sibling-op precedent:

- New value-only `LinalgOp::SvdFull` variant (no `derivative_eps`, no `gauge` —
  full SVD is raw, leaving thin defaults and gauge conventions untouched).
- Adding a `full_matrices` flag to `LinalgOp::Svd` would push a flag the thin AD
  rules, the `prune_outputs` SVD→SvdVals fusion, and the gauge pass must all
  special-case. A sibling op keeps every one of those paths unchanged.
- "AD unsupported for the full variant" becomes structural: `SvdFull`'s
  `linearize` emits no tangent (LuFactor precedent), `linear_transpose` is in
  the no-tangent group, and the AD manifest records every output `Unsupported`.
  This is the repo's existing mechanism for an unsupported route — no silent
  thin-SVD fallback.

Backend plumbing: a `svd_full` hook on `LinalgBackend` whose default returns
`Error::unsupported`, overridden only by the CPU faer provider. `svd_core` and
`svd_2d` gained a `full: bool` selecting `ComputeSvdVectors::Full` and square
`U`/`V` output shapes; `svd_view` and the thin free `svd` pass `false`.

### Provider policy: faer implements, LAPACK typed-unsupported

Per the issue's non-goal ("No GPU/BLAS backend requirement in the first slice;
typed unsupported is fine"), only the CPU faer provider computes full SVD; the
LAPACK provider and GPU backend return a typed error. Wiring LAPACK would
roughly double the FFI macro surface in `lapack_linalg/svd.rs` (gesdd/gesvd
job-char variants, `ldvt`, complex rwork) for a deferred capability. Faer is the
default provider, so the numeric acceptance tests run on it; they are gated
`#[cfg(feature = "cpu-faer")]` so the BLAS-only CI lane
(`--no-default-features --features cpu-blas`) skips them, and a dedicated
BLAS-lane test asserts the typed-unsupported boundary.

### `lstsq`: composition, not a new backend op

`lstsq` is composed from existing ops (`qr` → conj-transpose `dot_general` →
`triangular_solve`), mirroring `solve`/`pinv`/`inv` in `eager_composites.rs` and
the traced `solve`. Consequences:

- No new backend op, no new AD rule, no new op vocabulary.
- Because the component ops have AD rules, it stays differentiable through
  composition — it is *not* AD-unsupported; the issue's "AD unsupported"
  premise assumed a monolithic op. No unsupported route is added.
- Scope: tall/square, full column rank. Wide (`m < n`) inputs are rejected with
  a typed `InvalidArgument`. Rank deficiency is not detected (`R` singular ⇒
  ill-defined result); documented as a caller precondition, matching the issue's
  "documented error or min-norm" option.

## Alternatives rejected

- `full_matrices: bool` on `LinalgOp::Svd` / `SvdOptions`: entangles the thin AD
  rules, gauge pass, and output-pruning fusion. Rejected for a sibling op.
- Implementing LAPACK full SVD now: deferred per issue non-goal.
- A monolithic `lstsq` op with a hand-written AD rule: unnecessary; the QR
  composition reuses audited rules.

## Verification

- Default (faer) `--test integration -- svd_full lstsq`: 10 pass —
  tall/wide/batch real+complex reconstruction, `1x2` and random `3x5` nullspace
  recovery, `lstsq` vs known solution (real/complex/square), wide-input
  rejection.
- Whole crate `cargo test -p tenferro-linalg` (default) and `--features
  autodiff`: green, including the AD manifest (`SvdFull` `Unsupported` on all
  outputs, JVP/VJP unsupported) and eager/traced doctests.
- BLAS lane (`--no-default-features --features cpu-blas`): green — LAPACK
  typed-unsupported boundary test and `lstsq`; faer numeric tests skipped; lib
  compiles warning-free.

## Residual risks / follow-ups

- LAPACK and GPU full SVD remain unsupported (typed error) — future slices.
- `lstsq` uses unpivoted QR: no rank-revealing rank-deficiency detection.
