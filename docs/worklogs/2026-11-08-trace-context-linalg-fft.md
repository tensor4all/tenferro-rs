# Linalg and FFT on TraceContext (P3 of #2417)

## Decisions

- `TraceContextLinalgExt` and `TraceContextFftExt` take the context's own
  `TraceValue` inputs and return `TracedTensor`s, with `&self` receivers. The
  receiver is `&self` because every `TraceContext` method already takes `&self`
  and `TracedTensor::context()` hands back a shared reference, so a `&mut self`
  trait method could not be called from the delegating traced method without
  cloning the handle. `TraceContextEinsumExt` keeps its pre-existing `&mut self`
  receiver; aligning it is a separate change.
- Outputs stay `TracedTensor`s rather than raw values because the composites in
  these families are written against the typed standard-op surface
  (`a.abs()`, `abs.reduce_max(..)`, `matrix * &scale`). Returning values would
  need a second typed standard-op surface for the extension crates.
- The single implementation of every op remains the crate-private
  `(&TracedTensor)` function in `tenferro-linalg/src/traced.rs` (and the FFT
  builders in `tenferro-fft/src/lib.rs`). `TraceContextLinalgExt` derives the
  input tensor from the value and calls it; `TracedTensorLinalgExt` delegates to
  the context trait. Rejected alternative: rewriting the stringly independent
  composite bodies to take `(ctx, value)` directly. That relocates ~1200 lines
  for the same one-implementation result and would re-derive a tensor at every
  internal composite step (`lstsq` → `qr`, `pinv` → `svd`, …).
- Composites (`det`, `inv`, `lstsq`, `pinv`, `pinv_with_rtol`, `norm`,
  `eigvalsh`, `eigvals`) therefore have one body, reached only through the
  context entry point.
- `tenferro_runtime::extension::apply` stays. The memo's P3 row listed it as
  removed, but it is documented public API for out-of-tree extension authors
  (`docs/spec/extension-op.md`) and is used directly by the runtime's own
  integration tests and by `tenferro-linalg/src/householder.rs`; removing it is
  its own change with its own migration.
- The 25 implementation functions keep their `pub(crate)` visibility (they were
  `pub` in a private module) so the source-contract tests that pin their bodies
  keep working; only the intra-doc links to them moved to `Self::` paths.

## Verification conclusions and constraints

- `tenferro-linalg` (autodiff): 191 unit, 332 integration, 1 prelude, 3 + 1
  allocation and 209 doctests pass. `tenferro-fft` (autodiff): 31 unit, 115
  integration across its suites, 2 prelude and 37 doctests pass.
- The new parity tests pin that the context path and the traced path record the
  same program (`semantic_fingerprint`) for a direct decomposition, a composite
  (`pinv`), a two-value solve, and one- and n-dimensional FFTs.
- Two behaviours differ on the new path and are accepted, and both are pinned by
  tests in `crates/tenferro-linalg/tests/integration/trace_context_linalg.rs`:
  (a) a foreign value is reported as the value layer's
  `RuntimeStateSource(ForeignValue)` where the old two-tensor path reported
  `Validation` from `ensure_same_context`, because the derive happens before the
  op is appended — this also means ownership is decided before a composite's own
  dtype/rank/shape validation, so `lstsq` rejects a foreign `b` before it would
  reject an integer or wide `A`; (b) `norm` with an empty axis list returns a
  derived tensor, so the handle-local `attached_value()` cache is not carried
  (the program binding is unaffected).
- Not done here, by design: the memo's P3 also names einsum's `&mut self`
  receiver, `SparseCooTracedTensor` construction, and the sparse/tropical wrapper
  families. Issue #2525 scopes this phase to linalg and fft.
- Downstream (`tensor4all-rs`, `yao-rs`) was not built: `tensor4all-rs` pins
  tenferro `v0.8.0` at `a9f494cc` while this branch is `0.10.0`, so a local
  patch would be a version migration unrelated to #2525, and `yao-rs` is not
  checked out on this machine. The change only adds public items.
