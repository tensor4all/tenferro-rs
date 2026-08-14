# 2026-08-14 — #1680 Phase 2: single canonical core API (one-shot removal + rename)

## Session summary

The release-boundary breaking change. Removed the one-shot core concrete API
(`TensorOpsExt`, `TypedTensorOpsExt<T>`, `TypedTensorMaskOpsExt`) and renamed
the transitional `_in`/`_in_session` methods to the final plain names on the
session-explicit traits; migrated every workspace call site (48 files,
−2130 net lines). End state: one session-explicit concrete API for the
tenferro-runtime core vocabulary and `ConcreteEinsumPlan`. Follow-up surfaces
(einsum top-level traits' PUBLIC signatures, FFT, linalg, shape-packing) keep
their backend-taking signatures — their internals now route through the
canonical session path.

## Gate record (frontier review, per AGENTS.md)

- **Pre-implementation design gate** (reviewer-gpt on
  `docs/design/session-oriented-concrete-apis.md` §"Phase 2: single
  canonical core API"): 2 rounds → **approved**. Round 1: 4 blocking (scope
  overclaim → narrowed to core + plan-level einsum with explicit exclusions;
  typed mask `where_select` type contract → dedicated
  `TypedTensorMaskSessionOpsExt` for `TypedTensor<bool>` with generic branch
  scalar; einsum rename collides with the 7 backend-taking `execute*` →
  they are removed; grep/bench gates not executable → scoped exclusions +
  Phase-1-median comparison) + 1 minor + 1 nit.
- **Post-implementation diff gate** (reviewer-gpt on the full ~8.2k-line
  diff): 2 rounds → **Correct-to-merge**. Round 1: 5 blocking (doc-snippet
  generator still imported removed traits; einsum plan tests bypassed
  `with_backend_session`; top-level einsum internals bypassed the canonical
  plan path; prelude test + published docs taught one-shot calls; worklog +
  bench evidence missing). All fixed; Round 2 confirmed.

## Design decisions

- Scope boundary: runtime core traits + `ConcreteEinsumPlan` only; einsum
  top-level / FFT / linalg / shape-packing public surfaces stay (their
  internals migrate to the session form — einsum top-level adapters now
  prepare the plan and call `plan.execute*` inside one session, single
  execution path; error surface preserved via `into_tensor_error`).
- `TypedTensorMaskSessionOpsExt` (bool receiver, generic `U` branch scalars,
  `broadcast_ternary_in_read` + `select_read` + `into_typed_result`).
- Call-site rule: single-op wrap; group consecutive session-capable ops in
  one session; helpers take `&mut dyn BackendSession`; already-in-session
  sites call ops directly; closure `?`-chains get explicit result types.
- Benches dropped the one-shot arms (API gone); single-session arms are the
  benches.

## Measurements (release, pinned core 40; interleaved P1=main-with-Phase-1 vs P2=Phase-2, same window)

| chain (one_session) | P1 mean (2 runs) | P2 mean (2 runs) | Δ |
|---|---|---|---|
| no_broadcast | 33.475 µs (33.45/33.50) | 33.595 µs (33.63/33.56) | +0.36% |
| broadcast | 41.795 µs (41.86/41.73) | 42.050 µs (42.13/41.97) | +0.61% |
| phase1 | 44.105 µs (43.92/44.29) | 43.925 µs (43.71/44.14) | −0.41% |
| einsum B (10 calls) | — | 55.1 µs (after fast-path unification) | Phase-1 recorded 55.1 µs |
| einsum C (mixed) | — | 89.0 µs (after fast-path unification) | Phase-1 recorded 91.8 µs |

Interleaved runtime chains: **no regression (max |Δ| = 0.61%)**. Earlier
non-interleaved runs showed +4-10% which the interleaving proved to be
ambient machine drift (noisy shared host), not codegen. Einsum B/C match or
improve on the recorded Phase-1 medians after the top-level fast path was
unified through the plan (the plan's internal binary-dot optimization
preserves the kernel).

## Verification

- `cargo build --workspace` — pass
- Tests: runtime 402+146+1+413; einsum 161+24+1+87; ad 86+337+1+147; cpu
  512+1+46+2+185 — all pass (incl. doctests: renamed methods' examples run)
- `python3 scripts/check-guide-dependency-snippets.py` / `test-doc-consistency.py` /
  `check-doc-snippets.py` — OK
- fmt clean (incl. ext/tropical, ext/sparse); clippy workspace 0 findings
- Grep evidence: zero **live** `TensorOpsExt|TypedTensorOpsExt|TypedTensorMaskOpsExt`
  references in crates/scripts/tutorial-code (the only script hit is the
  intentional negative-check fixture at scripts/test-doc-consistency.py:404;
  remaining mentions are the design doc's migration-rule text + historical
  docs); zero one-shot `.add(&/matmul(&/exp(&/reduce_sum(&...&mut backend`
  in docs/scripts (outside historical docs/plans, worklogs, specs)
- PR gates (`check-pr-fast.sh`, `repository-rules-review.py`) run at PR time;
  recorded in the PR body

## Residual risks

- Follow-up one-shot surfaces remain (einsum top-level public signatures,
  FFT, linalg, shape-packing) — tracked as follow-up migration issues toward
  the same single-API end state.
- GPU (CUDA/WebGPU) nested-entry enforcement still open.
- The historical Phase-1 design narrative ("the one-shot API stays
  available") is superseded by the appended Phase-2 removals section; kept
  as history.
