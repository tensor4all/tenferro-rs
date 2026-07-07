# Issue 1321 Root VJP Extra Roots Fix

## Summary

This session fixed the traced VJP regression where the generic
`linearize -> linear_transpose -> optimize` path could keep the original
strictly-linear tangent sweep alive as a materialization root.

The root cause had two parts:

- tidu returned one `LinearizedGraph` graph that mixed strictly-linear tangent
  values with residual values needed by later transposed graphs;
- tenferro transpose rules received raw value references, so downstream rules
  could accidentally treat a tangent-flow value as an ordinary fixed operand.

The fix splits tidu linearized graphs into a strictly-linear graph and a
residual graph, then changes tenferro to retain only the residual graph plus the
optimized transposed graph for cached VJP execution. Transpose rules now receive
typed inputs so residual, linear-with-primal, and linear-without-primal cases
remain explicit at the operation-family boundary.

## Context Read

- GitHub issue #1321 and review comments
- `HANDOFF.md`
- `docs/worklogs/2026-07-06-lu-generic-vjp.md`
- `docs/architecture/ad-pipeline.md`
- `docs/architecture/tidu.md`
- `docs/spec/ad-contract.md`
- `crates/tenferro-ad/src/traced.rs`
- `crates/tenferro-ad/src/transform_cache.rs`
- `crates/tenferro-internal-ops/src/ad/`
- `crates/tenferro-linalg/src/ad.rs`
- `crates/tenferro-einsum/src/extension.rs`
- `crates/tenferro-fft/src/lib.rs`

## Decisions

- Change tidu, not only tenferro call sites: `LinearizedGraph` now separates
  `linear` and `residual`, and `PrimitiveTransposeInput` classifies each
  transpose-rule input.
- Register metadata for graph parent chains so cached residual graphs and
  transposed graphs can recover symbolic shape sources without retaining the
  tangent sweep.
- Store traced VJP cache entries as residual graph plus optimized transposed
  graph. Cache hits re-register residual metadata; final VJP materialization
  pushes the residual graph as `extra_roots`, not the linear graph.
- Remove production helpers that collapsed transpose inputs to raw `ValueRef`s.
  Operation families must choose metadata-only, fixed-coefficient, or shape
  source behavior explicitly for each input.
- Keep primary extension VJP as an escape hatch for extension rules that still
  intentionally consume primal values directly, but make linearized extension
  transpose rules typed.

## Rejected Alternatives

- Do not reintroduce handwritten LU or Eigh VJP rules. The generic graph is
  structurally compact once the tangent sweep is not retained.
- Do not add LU- or Eigh-specific optimizer peepholes. The observed regression
  was a transform ownership bug, not an algebraic simplification gap.
- Do not patch `extra_roots` by filtering specific graph ids. The transform
  artifact itself must expose which graph is residual and which graph is the
  discardable tangent sweep.
- Do not allow a `Linear { primal: None, .. }` input to become an external
  tensor operand. Rules that need a fixed operand must have a residual value or
  a known primal counterpart.

## Benchmarks

Benchmarks were run from the local tenferro-benchmark worktree using
`PUBLICATION_GATE_FEATURES=cpu-faer`, trace mode, quick profile, and
`CPU_OPS_BENCHMARK_FILTER=grad_sum_lu_vjp,grad_sum_eigh_vjp`. The local machine
does not have the `/opt/openblas` setup used by the published Linux CPU report,
so these are same-machine before/after checks rather than exact published
report reproductions.

Large suite, 1 thread, 5 runs, 2 warmups:

- `grad_sum_eigh_vjp 256x256`: 10.487 ms -> 7.840 ms, 1.34x
- `grad_sum_eigh_vjp 512x512`: 63.715 ms -> 47.664 ms, 1.34x
- `grad_sum_lu_vjp 256x256`: 8.731 ms -> 5.395 ms, 1.62x
- `grad_sum_lu_vjp 512x512`: 64.154 ms -> 37.052 ms, 1.73x

Large suite, 4 threads, 5 runs, 2 warmups:

- `grad_sum_eigh_vjp 256x256`: 13.784 ms -> 9.640 ms, 1.43x
- `grad_sum_eigh_vjp 512x512`: 47.075 ms -> 32.738 ms, 1.44x
- `grad_sum_lu_vjp 256x256`: 9.447 ms -> 6.590 ms, 1.43x
- `grad_sum_lu_vjp 512x512`: 40.003 ms -> 26.311 ms, 1.52x

Small suite, 1 thread, 30 runs, 5 warmups:

- `grad_sum_eigh_vjp`: 1.40x to 2.30x faster across 2x2, 4x4, and 8x8
- `grad_sum_lu_vjp`: 1.27x to 1.36x faster for 4x4 and 8x8; 2x2 was 0.062 ms
  -> 0.069 ms with overlapping microbenchmark noise

## Verification Performed

- `cargo fmt --all --check`
- `git diff --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo check -p tenferro-internal-ops --features autodiff`
- `cargo check -p tenferro-einsum --features autodiff,cpu-faer`
- `cargo check -p tenferro-fft --features autodiff,cpu-faer`
- `cargo check -p tenferro-linalg --features autodiff,cpu-faer`
- `cargo test -p tenferro-internal-ops --features autodiff -- --nocapture`
- `cargo test -p tenferro-linalg --features autodiff,cpu-faer --test traced_ad_explicit sum_grad -- --nocapture`
- `cargo test -p tenferro-linalg --features autodiff,cpu-faer --lib ad::tests -- --nocapture`
- `cargo test -p tenferro-einsum --features autodiff,cpu-faer linear_transpose -- --nocapture`
- `cargo test -p tenferro-einsum --features autodiff,cpu-faer --test traced_ad_migration -- --nocapture`
- `cargo test -p tenferro-fft --features autodiff,cpu-faer -- --nocapture`
- `cargo test -p tenferro-ad --test extension_op -- --nocapture`
- `cargo test -p tenferro-ad --release --test ad`
- `cargo test -p tenferro-ad --release --test ad_structural_primitives`
- `cargo test -p tenferro-ad --release --test dynamic_truncate`
- `cargo test -p tenferro-einsum --release --test public_surface_contract`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

## Residual Risks

- Local benchmarks used `cpu-faer`, not the published `system-openblas` profile.
  The same root fix should be remeasured in the publication environment before
  regenerating public reports.
- The typed transpose input boundary now covers core, linalg, einsum, FFT, and
  extension tests touched by this change. New operation families must continue
  the same pattern instead of reintroducing raw transpose input collapse.
