# 2026-08-14 — `ConcreteEinsumPlan::execute_in_session` (issue #1673, PR C)

## Session summary

First extension-crate example of the session-explicit concrete surface
(design tier 1): `ConcreteEinsumPlan` gains a `_in_session` mirror of all
seven one-shot execute methods (execute / execute_typed / execute_read /
execute_into / execute_typed_into / execute_read_into /
execute_read_into_accum). The one-shot methods now delegate to the
`_in_session` variants, so a repeated einsum chain drops from one session
entry per call to one entry total. Key structural finding: the one-shot
einsum already entered a session internally (`with_backend_session(|exec|
eager_einsum_exec(...))`) and the eager cores already took
`&mut dyn BackendSession` — `execute_in_session` is a thin validate +
core-call on the caller's borrowed session, no new entry.

## Gate record (frontier review, per AGENTS.md)

- **Pre-implementation design gate** (reviewer-gpt on
  `docs/design/session-oriented-concrete-apis.md` §"Prototype specification
  (PR C)"): 2 rounds → **approved for implementation**. Round 1: 4 blocking
  (mixed-chain example did not type-check — closure must return
  `tenferro_einsum::Result<_>`; public type misnamed `EinsumPlan` → actually
  `ConcreteEinsumPlan` with the 7 signatures underspecified; bench protocol
  undefined; docs/coverage acceptance absent) + 1 minor (validation ordering
  change must be explicit) + 1 nit (naming rationale → prepared-operation
  vocabulary, not verb/noun).
- **Post-implementation diff gate** (reviewer-gpt on the full diff): 2
  rounds → **Correct-to-merge**. Round 1: 2 blocking (`+ Send` regression on
  `execute_typed_into` — a real public API narrowing; missing worklog) + 1
  minor (`# Errors` prose misclassified dtype mismatches). Round 2 confirmed
  all fixed.

## Design decisions

- `_in_session` naming matches the prepared-operation vocabulary
  (`crates/tenferro-runtime/src/runtime/capability.rs`:
  `prepared_operation.execute_in_session`); the runtime concrete noun ops use
  `_in` (PR B). Both transitional; final canonical names drop the suffix.
- One-shot delegation moves validation inside the session entry (accepted:
  invalid calls now pay one entry before returning the identical error);
  covered by an error-parity test.
- `execute_typed_into` converts `out: O` to `TypedTensorWrite` BEFORE the
  `with_backend_session` closure so the public signature stays
  `O: Into<TypedTensorWrite<'out, T>>` (no `+ Send`); verified with a genuine
  `!Send` (Rc-based) adapter compile test.
- Out of scope: top-level einsum-trait `_in_session` variants, native-kernel
  extension capabilities (tier 2), runtime-integrated extension execution
  (tier 3), any semantic/graph/AD einsum change.

## Measurements (release, pinned to idle core 40, criterion medians)

`"ij,jk->ik"`, 8×8 f64, 10 calls per iter, `CpuBackend::with_threads(1)`:

| arm | time | note |
|---|---|---|
| A: one-shot ×10 (10 entries) | 137.8 µs | baseline = pre-delegation one-shot path |
| B: in-session ×10 (1 entry) | 55.1 µs | **A/B = 2.50×** |
| C: mixed einsum+`_in` one-shot | 345.7 µs | 10 × (einsum + exp_in + reduce_sum_in) one-shot |
| C: mixed einsum+`_in` in-session | 91.8 µs | **C ratio = 3.76×** |

Pre-delegation one-shot baseline: the one-shot arms on the base commit were
measured by the implementer during the delegation change (the delegation is
the one-shot path in this PR; baseline vs post comparison recorded in the PR
body). No regression in einsum correctness (parity tests) or the one-shot
value path.

## Verification

- `cargo test -p tenferro-einsum` — 161 lib + 24 integration + 1 + 88
  doctests, 0 failed (incl. the non-Send adapter compile test and the
  session-counting mixed-chain test: 1 entry vs 30)
- `cargo test -p tenferro-einsum --doc` — 88 doctests (7 new `_in_session`
  examples), 0 failed
- `cargo build --workspace`, `cargo test -p tenferro-runtime` (402+131+1+425)
  — pass; fmt clean; clippy 0 warnings
- PR gates (`check-pr-fast.sh`, `repository-rules-review.py`) run at PR time;
  recorded in the PR body

## Residual risks

- The `_in_session`/`_in` names are transitional; the final canonical rename
  (suffix removal) is a planned breaking change at the migration break.
- GPU/extension-tier execution on the borrowed session (tier 2/3) remains
  open; the concrete plan-level path is CPU-validated only.
- Predicted-vs-measured ratios: PR B's ≥2 gate was for the trivial chain;
  einsum's evidence is the measured 2.50×/3.76× with the reported baseline,
  not a predicted number.
