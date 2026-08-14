# 2026-08-14 — #1680 Phase 1: complete the `_in` surface (full op vocabulary)

## Session summary

Extended the PR-B session-explicit prototype to the full concrete op
vocabulary (additive, non-breaking; transitional `_in`/`_in_session` names
kept — rename/removal is Phase 2). 26 dynamic (`TensorSessionOpsExt`) + 24
typed (`TypedTensorSessionOpsExt<T>`) new `_in` methods; every one-shot
method now delegates via `with_backend_session(|s| ..._in(...))`; dead
session-reentering broadcast helpers removed; session-safe ternary helpers
added for the dynamic and typed paths.

## Gate record (frontier review, per AGENTS.md)

- **Pre-implementation design gate** (reviewer-gpt on
  `docs/design/session-oriented-concrete-apis.md` §"Migration specification
  (issue #1680, Phase 1)"): 2 rounds → **approved**. Round 1: 2 blocking
  (op inventory missed `expm1`/`log1p` → 26/24 not 24/22; typed `clamp_in`
  needs a session-safe `broadcast_ternary_in_read`, the legacy typed
  `broadcast_ternary_read` re-enters sessions) + 1 minor (Phase-1 bench chain
  under-specified → exact [2,2] chain given).
- **Post-implementation diff gate** (reviewer-gpt on the full diff): 1
  blocking (Phase-1 worklog + fresh measurements missing — this file) → fixed.

## Design decisions

- Matmul cache ownership (resolves the PR-B deferral): `matmul_in` =
  `matmul_config_for_shapes` + `session.dot_general` on the plain entry;
  `with_backend_session_cached` stays internal to the runtime scheduler/
  eager paths. No `SessionCachedDot` in the concrete surface.
- Typed ternary: new `broadcast_ternary_in_read` built from
  `broadcast_shapes` + three `broadcast_to_in_read`; the legacy typed
  `broadcast_ternary_read` is retained ONLY for the mask-trait
  `where_select` one-shot (which legitimately enters a session).
- Spelling: `compare_in` takes `CompareDir` by value →
  `session.compare(..., &dir)`; `where_select_in` → `session.select`;
  typed compare returns `TypedTensor<bool>`; convert/cast →
  `session.convert`/`session.cast`; reshape/transpose → session ops.
- One-shot validation moves inside the session entry (accepted, PR-B/C
  precedent); error-parity tested.

## Measurements (release, pinned to idle core 40, criterion medians; 2 runs each)

| chain | arm | origin/main | Phase-1 | Δ (run1 / run2) |
|---|---|---|---|---|
| PR-B no_broadcast | one_shot | 107.7 / 118.8 µs | 112.7 / 117.3 µs | +4.6% / −1.2% |
| PR-B no_broadcast | one_session | 37.2 / 38.5 µs | 36.2 / 38.9 µs | −2.7% / +1.1% |
| PR-B broadcast | one_shot | 119.2 / 124.2 µs | 129.0 / 120.7 µs | +8.2% / −2.8% |
| PR-B broadcast | one_session | 45.3 / 46.0 µs | 43.7 / 44.6 µs | −3.5% / −3.2% |
| **Phase-1 chain** | one_shot | — | 120.8 / 117.3 µs | — |
| **Phase-1 chain** | one_session | — | 45.1 / 44.6 µs | **2.68× / 2.63×** |

One-shot arms differ by −2.8%..+8.2% across runs with no consistent sign →
**within machine noise (±4-6 µs), no regression**. One-session arms are
consistently equal-or-better. The Phase-1 chain (10 new ops) shows the same
≈2.6-2.7× session-reuse speedup as the PR-B gate chain (3.0-3.1×), and the
session-count test proves 5 ops in 1 entry vs 5 entries.

## Verification

- `cargo test -p tenferro-runtime` — 402 + 146 + 1 + 475 doctests, 0 failed
- `cargo test -p tenferro-runtime --doc` — 475 (58 new `_in` examples), 0
  failed
- `cargo build --workspace`, `cargo test -p tenferro-ad` — pass; fmt clean;
  clippy 0 warnings
- PR gates (`check-pr-fast.sh`, `repository-rules-review.py`) run at PR time;
  recorded in the PR body

## Residual risks

- Transitional `_in`/`_in_session` names remain (Phase 2 renames to plain
  names + removes the one-shot API — release-boundary breaking change).
- One-shot arms carry run-to-run noise on this shared machine; the
  no-regression conclusion rests on the two-run median and the consistent
  one-session direction, not a single measurement.
- Typed `where_select` (mask trait) still enters a session via the legacy
  ternary helper; converting it is a Phase-2 or follow-up item.
