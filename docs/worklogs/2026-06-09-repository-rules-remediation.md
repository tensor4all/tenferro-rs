# Repository Rules Remediation

Date: 2026-06-09

## Summary

This work fixed repository-rule violations found by a whole-code audit of
`tenferro-rs`. The audit used focused subagents for public surface drift,
documentation/snippets, AD/test organization, runtime/cache ownership, and
linalg AD support. The resulting changes intentionally break compatibility
where the previous surface exposed internals or stale names.

The only deferred item is full-pivot LU AD oracle coverage in
`third_party/tensor-ad-oracles/cases/**`; it needs upstream oracle-family work
and is tracked by <https://github.com/tensor4all/tenferro-rs/issues/987>.

## Code And Documents Read

- `REPOSITORY_RULES.md`
- `AGENTS.md`
- `README.md`
- `docs/api/index.md`
- `docs/getting-started/**`
- `docs/guides/tenferro-fft.md`
- `docs/design/gpu-backend-design.md`
- `docs/design/einsum-dyadtensor.md`
- `docs/design/einsum-cpu-porting-notes.md`
- `docs/design/testing.md`
- `crates/tenferro-ad/src/**`
- `crates/tenferro-runtime/src/**`
- `crates/tenferro-cpu/src/**`
- `crates/tenferro-gpu/src/**`
- `crates/tenferro-einsum/src/**`
- `crates/tenferro-linalg/src/**`
- `third_party/tensor-ad-oracles/cases/**`

## Reference Material Considered

- Existing `tensor-ad-oracles` families for related linalg operations such as
  `lu_factor`, `lu_factor_ex`, `solve`, `svd`, `eig`, and `eigh`.
- Existing repository design records for extension runtime restructuring, GPU
  backend design, einsum CPU porting, C-API planning, and testing strategy.
- Existing in-repository tests that encode public surface and source-contract
  expectations.

## Decisions

- Moved runtime compiler, exec, segmentation, and shape-inference internals out
  of the public root API. Sibling crates now use owner-scoped extension APIs
  where cross-crate access is genuinely needed.
- Removed eager extension fallback behavior for registered-runtime paths. A
  missing registered family now reports an explicit missing-runtime error.
- Kept CPU buffer-pool internals private and exposed only doc-hidden
  linalg interop hooks required by the operation crate.
- Reworked CPU elementwise and analytic read paths to avoid hidden
  materialization of borrowed views, and changed indexing hot loops to advance
  multi-indices incrementally instead of recomputing from flat offsets.
- Added rayon-backed CPU context installation so faer work can run under the
  configured CPU context.
- Renamed linalg allocation/execution APIs from `_view` to `_read` where they
  operate on `TensorRead` inputs rather than metadata-only views.
- Added machine-readable AD support manifests for primitive and linalg rules,
  including explicit support status for values-only/vector-output cases and the
  pending full-pivot LU oracle gap.
- Added finite-difference AD checks for linalg matrix operands and spectral
  vector-output observables. Full-pivot LU remains tracked as pending oracle
  coverage.
- Added bounded CubeCL extension caches with stats, clear, and capacity controls,
  and replaced raw CubeCL reach-through from linalg with a narrow interop module.
- Removed stale user-facing docs, synchronized README/core-concepts/FFT snippets
  with executable examples, and converted historical design notes so old names
  are not presented as current public API.
- Moved large inline unit-test blocks into module-local `src/**/tests*.rs`
  ownership locations.
- Added focused CPU `TensorRead`/view-dispatch tests after CI exposed that the
  remediated `elementwise.rs` read paths had fallen below the repository
  coverage threshold.

## Alternatives Rejected Or Deferred

- Backward-compatible aliases for renamed `_view` APIs were rejected because the
  requested remediation explicitly did not require backward compatibility.
- Keeping root runtime internals public for tests was rejected. Tests were moved
  to owner crates or through narrower owner-scoped APIs.
- Adding a local full-pivot LU oracle stub was deferred. The correct fix is an
  actual `tensor-ad-oracles` family; issue #987 tracks that work.
- Splitting `faer_linalg.rs` further was not done in this pass. The changes were
  concentrated around scratch reuse helpers and did not create a clearer
  operation-family extraction boundary.

## Verification

- `cargo fmt --check`
- `python3 scripts/check-doc-snippets.py --root-dir . --check`
- `cargo check --workspace`
- `cargo test --workspace`
- `cargo test -p tenferro-linalg --features autodiff`
- `cargo test -p tenferro-internal-ops --doc`
- `cargo check -p tenferro-gpu --features cuda`
- `cargo check -p tenferro-linalg --features cuda`
- `cargo test -p tenferro-einsum --features autodiff`
- `cargo test -p tenferro-cpu --lib`
- `cargo llvm-cov -p tenferro-cpu --release --lcov --output-path /tmp/tenferro-cpu.lcov`
  (`crates/tenferro-cpu/src/elementwise.rs`: 86.5%, 1575/1821 lines)

Additional focused checks were run during implementation for runtime, CPU,
einsum, AD, linalg manifest, and linalg finite-difference tests.

## Remaining Risks

- Full-pivot LU AD is still missing a `tensor-ad-oracles` family. The linalg AD
  manifest keeps that status explicit, and <https://github.com/tensor4all/tenferro-rs/issues/987>
  tracks the oracle work.
- CUDA paths were compile-checked only; no CUDA device execution tests were run
  in this local session.
