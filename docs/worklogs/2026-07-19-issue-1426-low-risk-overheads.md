# Issue #1426 low-risk overhead cleanup

## Context

Issue #1426 records static performance risks found at commit `e4eb831f`.
Before editing, the current `origin/main` implementation was checked at
`52b07707`. This batch intentionally selects only internal overheads whose
ownership and behavior boundaries are already established. It does not add or
change public APIs.

The implementation follows `CONTRIBUTING.md`, `AGENTS.md`,
`REPOSITORY_RULES.md`, and the repository remediation workflow. The bug-fix
scope gate was applied because all selected changes preserve existing behavior
and remain inside the existing tensor and eager-AD architecture.

## Classification ledger

| Finding | Classification | Current evidence | Disposition | Residual risk |
|---|---|---|---|---|
| #1426 H6: per-element allocation in `dot_general` accumulation | Auto Fix | `accumulate_typed` allocated an index `Vec` and performed div/mod for every compact output element | Added a compact host-slice fast path; retained the checked indexed fallback for strided, backend, and mismatched-length output | Backend and arbitrary-stride outputs intentionally retain the existing cost |
| #1426 low: tensor metadata `SmallVec` round trips | Auto Fix | `types.rs` converted inline metadata through heap `Vec` at view/layout construction sites | Replaced the same-contract occurrences with direct `SmallVec` collection and direct fixed-rank slice-to-array conversion | Rank greater than inline capacity still allocates by design |
| #1426 low: unconditional eager profiling clock read | Auto Fix | `EagerTensor::nary_op` called `Instant::now()` before checking whether aggregate profiling was enabled | Clock acquisition now occurs lazily behind the existing enabled predicate | Profiling-enabled execution intentionally retains timing overhead |
| #1426 H1-H5, H7-H8, M1-M8, GPU notes | Verify First / Out of Scope | These findings involve separate planners, caches, synchronization, ownership, threading, algorithms, or backends | Deferred unchanged; this PR does not claim to close the umbrella | Each needs its own current-source validation and focused contract |

## Behavioral boundaries

- Compact accumulation still computes `out = alpha * dot + beta * out`.
- `beta == 0` does not read existing output values; a `NaN`-initialized
  regression case covers this contract.
- Noncompact output uses the original indexed implementation. A strided-output
  regression case confirms its physical gaps remain untouched.
- Shape, stride, offset, validation, error, dtype, and profiling-output
  semantics are unchanged.

## Benchmark

The added Criterion benchmark uses a compact 4096-element `f64` accumulation
with `alpha = beta = 1`. On this development host:

- compact fast path: median **2.90 us**;
- previous indexed path: median **75.83 us**;
- observed speedup: approximately **26x**.

The comparison was run in the same worktree and release target. The committed
implementation was restored afterward and checked byte-for-byte against
`HEAD` with `git diff --exit-code HEAD`.

## Verification

- `cargo fmt --all -- --check`
- `cargo test -p tenferro-tensor -p tenferro-ad`
- `cargo clippy -p tenferro-tensor -p tenferro-ad --all-targets -- -D warnings`
- `cargo bench -p tenferro-tensor --bench dot_accumulation --no-run`
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-tensor dot_general_accum' --test 'cargo test -p tenferro-ad eager_op_profile'`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review-1426.json` — pass, no findings

GPU checks were not run because this batch changes backend-independent host
accumulation, tensor metadata construction, and an eager profiling gate only.
Hosted CI remains responsible for workspace-wide backend variants.
