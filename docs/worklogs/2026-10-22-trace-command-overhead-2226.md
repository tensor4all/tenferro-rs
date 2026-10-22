# Trace-mode command overhead: metadata commands and the per-run fixed cost (#2226)

Follow-up to [#2186](2026-10-21-cpu-prepared-command-dispatch-2186.md), whose
measurements this record continues. The issue has three work items; this batch
lands the first two and records why the third was not started.

## Decisions

- **A metadata-only operation with one reading consumer is not a scheduled
  command.** `ExecInstruction` carries `input_views`, sparse `(input, view)`
  pairs applied in order when the consumer reads the operand; `InputView`
  (`Transpose`, `Reshape`, `BroadcastInDim`, `Slice`) is the single definition
  of "metadata-only op" and `InputView::apply` the single place a view is
  derived, shared with the existing Read-slot path (`view_backend_op`). The
  fold is a plain `ExecProgram` pass (`fold_metadata_views`, beside
  `transpose_folding`), but it runs at root preparation, not in the staging
  pipeline: its rule needs the resolved placement so a value that crosses a
  transfer boundary keeps materializing (the issue's rule; a view derived on
  the consumer's engine would make the schedule transfer the source instead of
  the result). The preparation cache still identifies the placement-independent
  staged program; each root executes its own folded copy.
- **Fold rule, as the issue states it:** single consumer, owned outputs keep
  their command. The 2x2 reproducer keeps 2 of its 7 metadata commands by that
  rule (a `BroadcastInDim` with two consumers, a transposed program output), so
  the census is 25 → 20, not the 18 the issue estimated. Folding a stride-0
  broadcast into every consumer would read no extra bytes and is the obvious
  next census item; a transposed view read k times is a measured trade-off
  (k strided reads versus one copy plus k contiguous reads), which is why the
  single-consumer rule stays.
- **Owned-only accessors reject a folded view typed.** `get`/`ensure_owned`
  return `Error::Internal` when the input carries a view, and each backend
  dispatch entry declares `read`/`owned` beside its execute function. A
  misclassified consumer therefore fails loudly instead of reading the source
  value. Structural and indexing dispatch now takes the existing `*_read` hooks
  (#2221 reads views in place on CPU), so only `Convert` and the owned host ops
  (`DynamicTruncate`, `PadToMatch`) remain `owned`.
- **`Reshape` folds only from a non-view instruction result** with no pending
  view. "Instruction output" alone does not imply a compact layout: a kept
  metadata command over a borrowed input yields a strided `ExecSlot::Read`, and
  `reshape_view` of that can fail where the materializing entry succeeds
  (pre-implementation review finding). A non-view instruction result is an
  owned, compact engine result. `ExtractDiag` has no view form on `TensorView`
  and is not folded; adding one is a tensor-layer API, outside this batch.
- **Region planning keeps its boundaries.** A folded `BroadcastInDim` on a
  region input maps to the planner's `ElementwiseFusionInputView`; any other
  folded view declines the plan (today any `Transpose` in a candidate already
  declines the whole candidate, so no fusion is lost). The folded
  broadcast-multiply form is recognized by the segmenter so the triplet/pair
  boundary survives the fold.
- **Per-run fixed cost.** Facts about the immutable program are computed at
  prepare: `validate_exec_program` (on the folded program), coalescing
  eligibility, the flat location, the collective preflight, the region role of
  every node (`NodeRegion`, replacing two hash maps built per run), and the
  repeated-output positions. Extension caches are lent only to a program that
  has an extension command; a flat run collects its outputs straight from the
  slot array with no located-slot vector and no hash map. The per-command
  location proof is skipped for a flat schedule, which proved it at prepare.
- **Item 3 (prepared dispatch table) not started.** After items 1–2 the
  200-command chain costs 1.03–1.06 µs per command against 0.72–0.74 µs eager,
  a 0.3 µs gap that matches the issue's bound, but an O(1) dispatch-table index
  plus the flat location-proof skip moved the rows by less than the host noise
  (2x2 40.5 → 40.3 µs; chain 213 → 206–212 µs), so the remaining gap is not in
  the table scan or the classification. A `CommandStep` representation would
  add a second prepared structure (per handle, since engine slots follow the
  handle's target order, and after operation preparation, since the owner
  fallback depends on the executor) for a gain this host cannot measure. The
  skip of the flat location proof is kept as a simplification; the index was
  dropped.

## Verification conclusions and constraints

- Host: the Ryzen AI 9 HX 470 workstation (not the EPYC of #2186), 1T
  `CpuEngine::serial()` pinned with `taskset -c 0`, observer/builds on
  `4-11,16-23`, release, `TENFERRO_PROBE_THREADS=1`, 15 batches × 50 runs,
  three repeats; load average below 0.5 before the baseline, 2–3 during the
  later runs (builds on the other L3 domain). Acceptance gate declared before
  measuring: the 2x2 row improves by more than 5 % and no row regresses by more
  than 5 % (median), the host's noise band.

| row (1T, median over 15×50, min–max of 3 repeats) | baseline | after item 1 | after items 1–2 |
|---|---:|---:|---:|
| `grad_sum_qr_vjp` 2x2 (commands) | 47.8–48.1 µs (25) | 42.4–42.7 µs (20) | **40.5–40.7 µs (20)** |
| `grad_sum_qr_vjp` 8x8 | 51.7–51.8 µs | 45.9–46.5 µs | **44.1–44.4 µs** |
| 200-command `exp` chain | 279–281 µs | 276–300 µs | **213–216 µs** |
| one-command program | 4.36–4.40 µs | 4.20–4.25 µs | **3.42–3.46 µs** |
| eager `exp`, per op | 0.72–0.75 µs | 0.72–0.74 µs | 0.72–0.74 µs |

  Per command on the chain: 1.40 → 1.07 µs (eager 0.73). Per run fixed:
  4.36 → 3.43 µs. The 2x2 row is 15 % below the baseline; no row regressed.
- The 2x2 census (20 commands) is pinned by
  `tenferro-linalg/tests/integration/traced_command_census.rs`, which also
  checks the folded program's VJP against central finite differences.
  Executor parity for each folded form (transpose → add / reduce_sum,
  broadcast → multiply, slice/reshape → exp, a transposed program output, a
  view of a run input, borrowed and repeated outputs) is in
  `tenferro-runtime/tests/integration/metadata_views.rs`; the fold rule's
  cases, the planner's folded-view handling and the owned-accessor guard are
  module-local unit tests.
- Suites: tenferro-runtime (439 unit + integration), tenferro-cpu,
  tenferro-cpu-fused, tenferro-ad, tenferro-linalg with `autodiff`; the
  repository gate (`check-pr-fast.sh`, `repository-rules-review.py`).
- Not claimed: GPU behaviour (no device in this session). Folding is
  semantically safe on CUDA and WebGPU because every `*_read` entry either
  reads the view or stages it once (`stage_view`), the same single copy the
  removed command made; whether a backend reads a strided operand natively is
  a per-backend cost fact, verified here on CPU only. The JAX comparison of
  #2186 was not rerun.
