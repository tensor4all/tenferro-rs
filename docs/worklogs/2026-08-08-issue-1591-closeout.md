# Issue #1591 closeout

Date: 2026-08-08

## Scope

Audit the tenferro side of the `strided-einsum2` retirement using
`origin/main` at `166abc167bb09b12b3a6a80761e817a92ec072f0`. No production Rust
code was changed.

The closeout covers tenferro-rs #1591 and its retirement dependency with
strided-rs #199/#202. tenferro-rs #1592 remains independent and was not changed.

## Ownership and provenance

The active Faer/BLAS dot-general path is owned by `tenferro-cpu`: validation,
GEMM analysis, checked batch offsets, provider dispatch, output allocation, and
writeback are local. The current tree has no `strided-einsum2` Cargo edge and no
`crates/tenferro-cpu/src/gemm/strided_dot.rs` adapter.

Earlier local implementation/adaptation commit
`eb689172666004ca70618757c62188181635429f` moved Faer preparation into
tenferro; its lineage is recorded in
`docs/worklogs/2026-06-23-strided-einsum2-removal.md`. Later PR #1553,
`6255590e76d21f3ec7ba2a7feaa7e160baecabc1`, removed stale dependency
manifests, feature wiring, and contract text. These are distinct steps.

The audit classified the upstream uninitialized-output API and its associated
`MaybeUninit`/injectivity concerns as not transferred because tenferro's active
dot-general route receives initialized `TensorWrite` output. Existing local
validation covers rank/config ordering, layout safety, batch overflow, Faer
parallelism, and canonical fallback for unfusable layouts.

## Verification

The following focused commands passed:

```text
cargo test -p tenferro-cpu --lib cpu_tensor_kernel_parallel_features_are_wired
cargo test -p tenferro-cpu --lib axis_groups_match_existing_rank_validation_through_rank_seventy
cargo test -p tenferro-cpu --lib dot_general_validation_accepts_checked_negative_stride_output
cargo test -p tenferro-cpu --lib checked_batch_offset_reports_batch_conversion_overflow
cargo test -p tenferro-cpu --lib test_dot_general_falls_back_for_unfusable_lhs_batch_layout
cargo test -p tenferro-cpu --lib faer_provider_covers_f32_c32_and_c64_conjugation
cargo test -p tenferro-cpu --lib faer_provider_executes_non_unit_strides_and_strided_batches
cargo test -p tenferro-cpu --no-default-features --features cpu-blas,provider-inject --test integration provider_inject_dot_general_uses_registered_blas
```

Additional gates:

- `cargo fmt --all --check` — passed.
- `cargo test --workspace` — passed, 0 failures.
- `cargo test --workspace --release` — passed, 0 failures.
- `cargo doc --workspace --no-deps` — passed.
- `scripts/build_docs_site.sh` and rendered docs checks — passed.
- The repository change policy classifies this final diff as `docs-only`; Rust,
  BLAS, GPU, and coverage CI lanes are therefore not required for this PR.
- A broad local coverage run was attempted; its checker reported existing
  below-threshold GPU/WebGPU/tutorial files, none of which are changed here.

## Remote disposition

- tenferro-rs #1591: closed with evidence comment and exact-command supplement.
- tenferro-rs #1637: open follow-up for the two exact regression fixtures not
  present in the current tree.
- strided-rs #199: tenferro Phase 1 dependency condition recorded.
- strided-rs #202: tenferro blocker cleared; issue remains open for retirement.
- strided-rs #198 and #201: no tenferro transfer required; comments posted.
- tenferro-rs #1592: unchanged.

## Residual risk

The exact mixed-batch-order and negative-destination-stride fixtures remain
tracked in tenferro-rs #1637. The closeout does not claim platform-specific
BLAS providers were exercised locally; the active tenferro dependency and
ownership evidence is complete.
