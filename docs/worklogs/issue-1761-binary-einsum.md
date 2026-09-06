# Binary einsum optimizer bypass (#1761, Phase 1 of #1771)

## Scope and decisions

Read the latest #1771 and #1761, AGENTS.md, REPOSITORY_RULES.md, shared common,
Rust/performance/numerical rules and contribution workflows. #1771 supersedes
#1758's infrastructure and multi-phase instructions. This is Phase 1 only.

Upstream main: `181cbadf` (fetched before branching). The two planning files
were unchanged relative to the parent of candidate `64de8474`. Extracted only
the tree/test changes from `64de8474` and `74ec1a64`; no mixed-branch probes,
measurement infrastructure, rule changes, eager/session investigation or guide.

`ContractionTree::optimize_with_options` retains option validation, and delegates
two operands to existing `from_pairs([(0, 1)])`. That retains size/broadcast/
diagonal validation and `compile_step_plans`, including final output order.
No kernel, cache, API, dtype promotion, device or AD contract changes.
Inspected callers in concrete preparation, eager shared planning, optimize
policy and traced extension planning; explicit paths and N-ary search remain.
Orientation is not assumed neutral: both operand orders and signed/transposed
host layouts have timing controls. Existing complex/read/output tests remain
numerical coverage, not a claim of GPU performance validation.

## Frozen comparison protocol (before any candidate timing)

- Host: Linux `primerose`, x86_64, AMD EPYC 7713P, 64 visible CPUs, one NUMA
  node; kernel 6.8.0-101-generic; Rust 1.97.1, LLVM 22.1.6.
- Base + experiment source: `d65a422c` (upstream main plus only the example).
  Candidate: `166ab813` (same example plus implementation/tests).
- Source: `crates/tenferro-einsum/examples/binary_planning.rs` at `d65a422c`.
  Cargo release, default `cpu-faer`, existing Cargo.lock, no custom RUSTFLAGS
  or compiler wrapper. Build sequentially; execute saved release binaries after
  both builds. Warm code/data, no compilation in timing.
- Two separate configurations: `default` (`CpuBackend::new`) and `1`
  (`CpuBackend::with_threads(1)`). No added process affinity, pool or provider
  policy. Backend/input/parsed-label/prepared-plan construction outside execution
  timing. Ordinary calls include public parsing/planning, session entry/exit,
  execution and observation of the full returned F64 slice. Prepare-only includes
  plan construction and destruction. No prepared/ordinary ratio requirement.
- All 36 named cases in the source: prepare string/labels, ordinary string/labels,
  prepared execution at 2/8/32/256; compact/transposed/reverse read layouts at
  8/256 in both operand orders; rank-3 batch prepare/ordinary; rank-3 N-ary
  prepare/ordinary. Explicit independent numerical checks precede timing.
- Seven complete paired processes per configuration, sequential, alternating
  base/candidate order. Each case: 100 warmup calls, five samples of at least
  50 ms, batches of 100 calls. Preserve all CSV samples, stderr, failures and
  host load observations. No unrelated jobs stopped or machine configuration
  changed. No selective case reruns or outlier removal.
- Analysis: median of five samples per process/case; paired log(candidate/base)
  ratios over seven pairs; geometric mean ratio and two-sided 95% Student-t
  interval (6 df, critical value 2.447). Primary per configuration is the equally
  weighted geometric mean of ordinary string calls at 2/8/32, with an interval
  across paired processes: upper bound < 1 demonstrates speedup, no minimum
  percentage. Also report each case separately.
- Non-regression: every other case's upper confidence bound < 1.10 (10% practical
  equivalence margin, not a speedup threshold). Primary size-specific ordinary
  cases must also satisfy that bound. An interval crossing the bound is
  INCONCLUSIVE, not proof of non-regression. A clearly slower control is reported
  as a regression, never hidden in the primary average.
- Host noise is handled by paired repetitions and confidence bounds, not an
  idle-core admission system. Record load and process-median CoV for context;
  wide intervals cannot pass by a median alone. Failed numerical checks or
  incomplete comparisons invalidate the attempt. After an inconclusive complete
  comparison, allow only one complete retry with the same protocol; retain both.
  No-improvement is a valid negative outcome, not Phase 1 completion.

## Verification before timing

- `cargo test -p tenferro-einsum`: passed (including integration tests and 102
  doctests); then new numerical cases / final restored bypass verified by
  `cargo test -p tenferro-einsum --lib`: 177 passed.
- Public execution counter test covers erased/typed, owned/read and output routes;
  counters are reset/read on the session's executing thread, with an N-ary
  positive control. Removing only the bypass made this test fail with 10 general
  optimizer calls instead of zero; restoring it passed. Instrumentation is
  test-only, not a TLS plan cache.
- Added explicit-value reduction, diagonal, output-permutation, outer-product,
  singleton broadcast, scalar, zero-contracting and zero-output cases.
- `cargo check -p tenferro-einsum --example binary_planning`: passed.

## Commands and results

To be appended after the frozen comparison. Performance is not yet verified.
