# Per-instruction session entries in the compiled path (issues #1926 / #1929)

**Date**: 2026-09-26
**Branch**: `refactor/1929-session-route-unification`
**Trigger**: the matched comparison in
`docs/worklogs/2026-09-26-session-route-recapture-and-matched-comparison.md`
showed `elementwise_fusion` up to +26.6% and `linalg_vjp_gate` +21…32%.

## What was wrong

Removing the owner route made three per-instruction steps construct a session
where they previously ran on the backend without one:

| step | before | after |
| --- | --- | --- |
| terminal-value probe | session (unchanged) | session (unchanged) |
| host instruction | `execute_host_instruction(backend, ..)` | `with_backend_session(\|exec\| execute_host_instruction_exec(..))` |
| ffi instruction | `execute_ffi_instruction_cached(backend, cache, ..)` | `with_backend_session_cached(cache, \|exec\| execute_ffi_instruction_exec(..))` |
| last-use reclaim | `reclaim_last_use_inputs_backend(slots, inst, backend)` | `reclaim_last_use_inputs_via_session(backend, ..)` = its own session |

A fresh CPU session pays the documented single-worker floor (`enter_managed_session`,
~5–8 µs, `docs/design/cpu-session-open-cost.md`), so a two-instruction compiled
graph paid roughly 2 × 7 µs ≈ 14 µs extra — independent of tensor size, which is
what the measurements showed: `add_mul/prepared_graph/4096` 53.23 → 67.37 µs and
`add_mul/unprepared_graph/4096` 123.01 → 137.28 µs, the **same** +14 µs on two
baselines that differ by 70 µs.

## The fix

1. `runtime/execution.rs::execute_slot_instruction` now runs the probe, the
   execution and the reclaim in **one** session per instruction, and only reaches
   for a second session in the owner-extension fallback, where the extension forms
   its own session between the probe and the reclaim.
2. `exec.rs` gains `instruction_may_be_terminal_value`, the session-free
   precondition that `try_execute_terminal_value_instruction` already used
   internally, and the callers in `execution.rs` and `segment.rs` now test it
   **before** opening a probe session. A probe that cannot succeed no longer costs
   an entry at all.
3. `segment.rs`'s two terminal-value tail branches fold the reclaim into the probe
   session when the probe handled the instruction.

## Measurements (cpu 1, 1T environment, per-case runs with idle checks)

| case | baseline | candidate before | candidate after |
| --- | --- | --- | --- |
| `add_mul/prepared_graph/4096` | 53.23 µs | 67.37 / 68.40 µs | **53.42 / 53.19 µs** |
| `add_mul/unprepared_graph/4096` | 122.15 µs | 137.28 µs | **120.98 / 120.53 µs** |
| `add_mul/prepared_graph/65536` | 1274.4 µs | — | **1272.7 µs** |
| `add_mul/prepared_graph/1048576` | 19732 µs | 19732 → 45823 µs (contended) | **19875 µs** |
| `broadcast_mul/prepared_graph/256x256` | 131.46 µs | 149.40 µs | **128.83 µs** |

Criterion's own same-directory comparison for `add_mul/prepared_graph/4096`
reported `change: [-24.7% -22.7% -20.6%] p = 0.00` when the fix landed, which is an
independent confirmation that the delta was real and is gone.

Verification: `cargo check --workspace --all-targets` is clean and
`cargo nextest run --workspace` reports 3513 passed / 0 failed / 208 skipped.

## Measurement methodology change (important for this host)

This host runs three unpinned foreign `julia` processes (100 days old) that
**migrate between cores**: `cpu 0`, `8`, `32` were saturated in the morning and
`cpu 0`, `8`, `40`, `56` in the evening. One full campaign-style run of the
elementwise target was therefore inflated uniformly by ~2.3× (up to +152%),
which is not a code effect. What worked instead:

* pin the measurement to one core and sample that core's `/proc/stat` busy
  fraction for a few seconds **immediately before and after** each measurement;
* prefer short, per-case runs (criterion `--sample-size 100` on one filtered case
  is ~7 s) over long target-wide runs;
* treat a *uniform* slowdown across sizes as contention, not as a regression:
  a per-operation fixed cost shows up as a constant absolute delta across sizes,
  while contention multiplies every case.

## What is still regressed, and why it is a different cause

The same measurements after the fix show two regressions that this change does
**not** touch:

* `linalg_vjp_gate`: `triangular_solve_vjp/{8,16}` +20.7% / +21.4%,
  `svd_values_vjp/{8,16}` +8.2% / +9.4%;
* `eager_dispatch_baseline` small ops: ~+3.5…7% (spot check after the fix:
  `materialized/reduce_sum_f64/1` 13.61 → 14.07 µs).

Both are the *eager/tape* path, where B3 removed the owner route and each tape
node now constructs its own session: a triangular-solve VJP is roughly eight to
ten linalg operations, which at ~7 µs per session predicts +56…70 µs — matching
the observed +68 µs. The compiled-path fixes here cannot help, because the cost is
one session per *operation*, not per instruction of a compiled program.

The fix direction is therefore a design change, not a local edit: run an
evaluation or a whole backward pass inside one `with_execution_scope`, so that the
per-operation entries reuse one permit instead of constructing a session each
(`session_chain` measures exactly that pattern and shows no regression). That
touches `tenferro-ad`'s eager evaluation and the scope-nesting contract, so it
needs its own decision and review rather than an unattended edit.
