# 2026-08-15 — #1690: uninitialized full-overwrite provider destinations

## Session summary

Removed the wasted zero-fill on the two deferred sites from #1640: the
dot_runtime canonical-operand allocation and the exec_session dot-output
allocation (both fully overwritten by their consumers). Introduced a sound
opt-in unsafe-contract mechanism: two `unsafe trait`s
(`CpuUninitLayoutTransformProvider`, `CpuUninitGemmProvider`) whose
implementation asserts the full-write guarantee, enabled structurally via
`uninit_provider() -> Option<&dyn ...>` witnesses (default `None`; only an
`unsafe impl` can return `Some`), a public output-free
`CpuGemmUninitRequest`, and `PooledUninitOutput::assume_init` handoffs.

## Gate record (frontier review, per AGENTS.md)

- **Pre-implementation design gate** (reviewer-gpt on
  `docs/design/uninitialized-output-contract.md`): 3 rounds → **approved**.
  Round 1: 1 Critical (safe provider code could manufacture initialization
  proof) + 4 Important (request types didn't match; dot seam was the wrong
  provider; errors must not silently retry; reclaim didn't exist) → revision
  2 (unsafe traits + capability boolean). Round 2: 1 Critical (the boolean
  isn't structural proof → witness) + 2 Important (CpuDotGeneralRequest
  can't represent the call → output-free CpuGemmUninitRequest; the GEMM
  seam needs context + prepared request) → revision 3 (structural witness,
  CpuGemmUninitRequest with 2 lifetimes, context param). Round 3 →
  approved.
- **Post-implementation diff gate** (reviewer-gpt on the ~2340-line diff,
  SAFETY-CRITICAL): 3 rounds → **Correct-to-merge**. Round 1: unsafe
  contract sound (witness structural, destination stays MaybeUninit, full
  write before Executed verified incl. empty/batched/conj) but 1 blocking
  (tests not faer-gated for cpu-blas-only) + 1 later (BlasGemmProvider
  import under provider-inject). Both gating fixes verified.

## Design decisions

- Sound opt-in: the witness method returns `Option<&dyn CpuUninit*Provider>`;
  only an `unsafe impl` can construct that trait object, so a safe
  third-party provider is structurally unable to enable the uninit path. The
  destination travels only as `&mut [MaybeUninit<u8>]` until the `unsafe
  assume_init` handoff (dtype-agnostic byte carrier; alignment + length
  validated).
- Built-in impls satisfy the contract: layout transform replays every
  destination slot (incl. conj); faer `Accum::Replace` for beta == 0,
  `beta != 0` defensively refused, empty contractions write zeros.
- Fallback: `None` witness → zeroed path directly (no uninit checkout);
  opted-in `Unsupported` → drop checkout (frees) + zeroed fallback; `Err` →
  propagate (never silently retry). `beta != 0` dot paths untouched.
- Direct-plan-only routing: the uninit checkout holds the pool exclusively,
  so the canonical-operand path falls back to zeroed when a plan is needed
  (per the design's discard-and-retry pattern).

## Measurements (release, pinned core 40, 1 thread)

| op | before (zeroed) | after (uninit) | Δ |
|---|---|---|---|
| 128×128 f64 dot | 96.7–99.5 µs | 97.2–98.9 µs | ~0% (memcpy-bound, noise) |
| 2×2 f64 dot | 3.08–3.19 µs | 2.37 µs | ~24% faster |

Honest reading: mid-size is dominated by the GEMM memory traffic (the
zero-fill is ~2% of 128 KB); tiny cases win because the uninit branch also
skips the zeroed pool `resize`/`fill` and the redundant full re-validation of
the fresh compact output. The win is real but concentrated at small sizes.

## Verification

- `cargo test -p tenferro-cpu` (faer default): 520+1+46+2+188 passed
- `RUSTFLAGS="-l dylib=openblas -l dylib=lapack" cargo test -p tenferro-cpu
  --no-default-features --features cpu-blas`: 503+46+2+187 passed (the bare
  cpu-blas link failure reproduces on origin/main — pre-existing)
- `cargo check -p tenferro-cpu --features provider-inject`: clean
- `cargo build --workspace`, `cargo test -p tenferro-ad`: pass; fmt clean;
  clippy workspace 0 warnings
- 8 new tests: witness source-contract, provider-level parity (batched /
  empty-contraction / empty-output / nonzero-beta / layout+conj),
  opt-out and opted-in-Unsupported fallbacks with call counts, full
  dot/output value parity
- PR gates (`check-pr-fast.sh`, `repository-rules-review.py`) run at PR time

## Residual risks

- The unsafe contract relies on the built-in impls' full-write guarantee;
  future built-in provider changes must preserve it (the safety docs on the
  unsafe traits name the invariant).
- Third-party providers keep the zeroed path (default opt-out) — no behavior
  change for them.
