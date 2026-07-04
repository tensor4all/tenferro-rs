# Issue 1291 CPU Elementwise Fusion Via Strided

## Session Summary

Implemented the first CPU-side #1291 slice after `strided-rs` 0.3.0 was
published. The branch moves tenferro's strided dependencies from the old git
revision to crates.io 0.3.0, adds a CPU `execute_elementwise_fusion` adapter,
and extends segmented runtime planning so `BroadcastInDim` can be represented
as fusion input metadata instead of forcing eager materialization.

This is deliberately one PR with multiple measured phases. The final code keeps
the existing hand-written broadcast-multiply fast paths for the pure multiply
case because benchmarks showed they are still much faster than the generic
fused plan. Broadcast metadata fusion is used for chained elementwise work such
as `broadcast_mul_add`, where it removes a large materialization cost.

## Context Read

- Issue #1291 and its acceptance criteria around CPU elementwise fusion,
  broadcast view absorption, strided fused plans, and benchmark verification.
- Shared tensor4all common, Rust, performance, and docs/test rules.
- `REPOSITORY_RULES.md`, especially hidden materialization, CPU kernel
  ownership, benchmark, and worklog requirements.
- `strided-kernel` 0.3.0 `FusedPlan`, `FusedInst`, `FusedOp`, and
  `fused_elementwise_into`.
- Existing segmented execution and broadcast-multiply special cases in
  `crates/tenferro-runtime/src/segment.rs`.
- Existing CPU elementwise and structural broadcast implementations in
  `crates/tenferro-cpu/src/elementwise.rs` and `structural.rs`.

## Decisions

- Keep `ElementwiseFusionPlan` as the canonical hidden backend plan, but add
  hidden `ElementwiseFusionInputView` metadata for input-side identity and
  `BroadcastInDim` views.
- Use `strided_kernel::fused_elementwise_into` for supported CPU F32/F64/C32/C64
  plans, returning `None` for integer/bool and unsupported complex ordered ops
  so existing per-op semantics remain authoritative.
- Gate CPU fusion below 16K elements. The generic fused path has fixed overhead
  that regressed 4K-element add-multiply chains, so small plans fall back to the
  existing per-op path.
- Defer pure broadcast multiply plans to the existing broadcast-multiply special
  cases. Control benchmarks measured the special case at 15.8 us / 172.9 us for
  256x256 and 1024x1024, versus 111.6 us / 1.65 ms for generic broadcast fusion.
- Keep fusion input-view metadata in `Vec` rather than `SmallVec`. A/B
  benchmarks on the broadcast metadata path measured `SmallVec` about 6-7%
  slower than `Vec`, so the final code includes a source comment documenting
  why `Vec` is intentional.
- Add a local CPU fast path for the common `multiply -> add` fused chain. This
  avoids strided's generic fused interpreter for the broadcast chain benchmark
  and dispatches directly through `zip_map2_into`.

## Benchmarks

All benchmarks used:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
CARGO_TARGET_DIR=/tmp/tenferro-issue1291-target \
cargo bench -p tenferro-runtime --bench elementwise_fusion -- \
  --sample-size 10 --warm-up-time 0.2 --measurement-time 0.5
```

Baseline before dependency update, add-multiply chain:

- 4,096 elements: 8.302 us
- 65,536 elements: 92.241 us
- 1,048,576 elements: 1.664 ms

After updating strided dependencies to crates.io 0.3.0, before CPU fusion:

- 4,096 elements: 8.425 us
- 65,536 elements: 92.081 us
- 1,048,576 elements: 1.666 ms

Final add-multiply chain after CPU fusion and small-plan fallback:

- 4,096 elements: 8.534 us
- 65,536 elements: 26.314 us
- 1,048,576 elements: 332.75 us

Final simple broadcast multiply, which intentionally stays on the existing
special-case path:

- 256x256: 18.867 us
- 1024x1024: 179.66 us

Final broadcast multiply plus add chain, after keeping the segment intact and
using the local `multiply -> add` fast path:

- 256x256: 18.999 us
- 1024x1024: 164.35 us

Rejected alternatives measured during the session:

- Generic broadcast fused interpreter before the local fast path:
  938.47 us / 15.706 ms for 256x256 / 1024x1024 `broadcast_mul_add`.
- `SmallVec` metadata on the broadcast metadata path:
  118.54 us / 1.765 ms versus `Vec` control at 111.56 us / 1.655 ms.

## Verification

- Red test first confirmed CPU `execute_elementwise_fusion` returned `None` for
  a supported add-multiply plan.
- Targeted CPU fusion tests:
  `CARGO_TARGET_DIR=/tmp/tenferro-issue1291-target cargo test -p tenferro-cpu test_cpu_elementwise_fusion -- --nocapture`
- Runtime segment tests:
  `CARGO_TARGET_DIR=/tmp/tenferro-issue1291-target cargo test -p tenferro-runtime segment -- --nocapture`
- Benchmarks listed above.
- `cargo fmt --all --check`
- `git diff --check`
- `CARGO_TARGET_DIR=/tmp/tenferro-issue1291-target cargo test --workspace --release`
- `CARGO_TARGET_DIR=/tmp/tenferro-issue1291-target cargo clippy --workspace --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/tmp/tenferro-issue1291-target/tropical cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `CARGO_TARGET_DIR=/tmp/tenferro-issue1291-cov-target cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --worktree --output-json /tmp/repository-rules-review-worktree.json`

## Residual Risks

- The CPU fused adapter only covers F32/F64/C32/C64 and intentionally falls back
  for integer and bool plans.
- Broadcast metadata fusion currently handles metadata views on segment inputs.
  More general reshape/permute metadata and GPU broadcast-aware codegen remain
  follow-up work.
- The strided generic fused interpreter is not yet a good fit for every
  broadcast chain. This branch adds one high-value local specialization and
  keeps existing broadcast-multiply special cases where they still win.
