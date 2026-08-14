# 2026-08-14 — session-explicit concrete ops: `_in` prototype (issue #1673, PR B)

## Session summary

First implementation slice of the session-oriented concrete API (issue
#1673): `TensorSessionOpsExt` / `TypedTensorSessionOpsExt` with
`add_in`/`mul_in`/`exp_in`/`reduce_sum_in` (the `_in` suffix is a documented
transitional name — final canonical names drop it at the migration break).
One-shot `add`/`mul`/`exp`/`reduce_sum` now delegate to the `_in`
implementations, so a one-shot binary op goes from up to 5 session entries
(two broadcast + one op, `[1,2]+[2,1]` worst case) to exactly 1.

## Gate record (frontier review)

- **Pre-implementation design gate** (reviewer-gpt on
  `docs/design/session-oriented-concrete-apis.md` §"Prototype specification
  (PR B)"): 3 rounds. Round 1: 4 blocking (session count 3→5; `broadcast_error`
  not shared — dynamic uses `broadcast_error_to_validation`, typed has its own
  mapping; typed borrowed/read-based path must be separate from the owned
  dynamic helper; bench target + exact 10-op chain) + 2 non-blocking (dot
  exclusion rationale; doctest/test plan). Rounds 2-3: one-line name fix
  (`broadcast_shapes`) + stale 3-to-1 count → **approved for implementation**.
- **Post-implementation diff gate** (reviewer-gpt on the full diff): 1
  blocking (worklog + recorded performance evidence missing — this file) + 2
  non-blocking (rustdoc session-entry claims; parity tests could share a
  shared-wrong-result) + 1 nit (exp doctest exact equality). All fixed:
  rustdoc wording, independent value/dtype/`mul_in` test strengthening,
  exp tolerance.

## Design decisions

- **Owned dynamic surface**: `broadcast_to_in` preserves the `duplicate()`
  path for equal shapes / broadcast sources; add/mul = broadcast both + the
  session op, one borrowed session. Error mapping reuses
  `broadcast_error_to_validation`.
- **Borrowed typed surface**: read-based (`reshape_read` /
  `broadcast_in_dim_read` / `add_read` / `mul_read` / `exp_read` /
  `reduce_sum_read` + `into_typed_result`), reusing the existing `ReadInput`
  and the typed manual broadcast-error mapping. Not funneled through the
  owned helper (avoids copies/allocation-semantics drift).
- **Transitional naming**: `_in` exists only because the session form
  coexists with the one-shot `TensorOpsExt` (same receiver type + same method
  name on two in-scope traits = Rust ambiguous-method-resolution, and a
  prelude glob would break every call site). Final canonical names drop the
  suffix at a release-boundary migration break.
- Out of scope for this PR: dot/matmul (cache-ownership decision pending),
  convert/cast, einsum plans (PR C), GPU session changes, Send bounds.

## Measurements (release, pinned to idle core 40, criterion medians)

| arm | one_shot (10 entries) | one_session (1 entry) | ratio |
|---|---|---|---|
| no_broadcast (1×8) | 106.7 µs | 34.8 µs | 3.06× |
| broadcast (1×1 vs 1×8) | 122.1 µs | 44.9 µs | 2.72× |

- Gate `T_one_shot / T_one_session >= 2` passes (measured 3.06× / 2.72×;
  the ~6× prediction assumed managed-pinned entries; this bench uses default
  `CpuBackend::new()` with cheaper entries).
- **Pre-delegation one-shot baseline** (delegation reverted, bench + `_in`
  kept): no_broadcast 136.9 µs, broadcast 211.9 µs → delegation improves the
  one-shot arms by 19% / 43% (no regression; a side benefit of folding
  broadcast into the op session).
- **Large-op no-regression** (1024×1024 one-shot add, 50 iters, pinned):
  15017 µs current vs 14982 µs origin/main (+0.2%, noise).

## Verification

- `cargo build -p tenferro-runtime` — pass
- `cargo test -p tenferro-runtime` — 402 unit + 131 integration + 1 prelude + 425 doc, 0 failed
- `cargo test -p tenferro-runtime --doc` — 425 doctests (incl. 10 new `_in` examples), 0 failed
- `cargo test -p tenferro-runtime --test integration session_in` — 9/9 pass
- `cargo build --workspace` — pass; `cargo test -p tenferro-ad` — 571 pass (runtime is a dependency)
- `cargo fmt --all --check` — clean; `cargo clippy -p tenferro-runtime --all-targets` — 0 warnings
- Bench: `cargo bench -p tenferro-runtime --bench session_chain` (pinned core 40, release) — see Measurements
- PR gates: `check-pr-fast.sh` and `repository-rules-review.py` run at PR time (recorded in the PR body)

## Residual risks

- Predicted-vs-measured gate ratio differs (≈3 vs ≈6): the design's ~3 µs/
  entry figure was measured for managed-pinned CPU; default-entry benches are
  cheaper. The gate still passes; report the measured entry cost, not the
  predicted one, in follow-up work.
- The `_in` public names are transitional; renaming at the migration break is
  a breaking change to plan for (release boundary).
- GPU/CUDA/WebGPU nested-entry enforcement still open (PR A tracked it).
