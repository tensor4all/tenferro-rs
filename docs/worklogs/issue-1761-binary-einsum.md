# Binary einsum optimizer bypass (#1761, Phase 1 of #1771)

## Adoption decision: explicit one worker

The maintainer authorized renewed investigation, then explicitly restricted
this overhead measurement to **one worker**, accepted the results below and
requested merge. This supersedes the earlier draft/stop decision recorded below;
the historical measurements are retained, not discarded or rewritten.

The independent fresh-target release builds reproduce the original executable
SHA-256 hashes exactly, excluding stale build artifacts. A renewed unpinned 1T
comparison does not reproduce the large reversed-input regression. Small
prepared controls remain noisy with an unpinned initiating thread.

The final complete comparison explicitly uses `CpuBackend::with_threads(1)` and
`taskset -c 32` for both executables. CPU affinity is separate from the worker
count: CPU 32 is the worker selected by the previous 1T configuration; pinning
also prevents the initiating thread from migrating. This is a pinned **1T**
result, not a default/all-core performance claim. Seven paired processes per
revision, alternating AB/BA, all 36 existing cases and five >=50 ms samples per
case; no case exclusions. The protocol was recorded before timing.

- Ordinary string 2/8/32 aggregate candidate/base ratio: **0.6851**, 95% CI
  **[0.6661, 0.7046]** (about 31.5% faster).
- All 36 cases pass the predeclared non-regression bound (upper CI < 1.10),
  including prepared, N-ary, layout/orientation and larger execution.
- Reversed 256 inputs: compact 0.9626 [0.9160, 1.0115], transposed 0.9936
  [0.9702, 1.0175], reverse view 0.9520 [0.9361, 0.9682].
- Prepared 2/8/32/256 upper confidence bounds: 1.0180 / 1.0668 / 1.0320 /
  1.0420. No dedicated binary optimizer or additional production changes needed.

The earlier all-core configuration used 64 workers. Using that as an overhead
acceptance condition was inappropriate. Its binary-to-binary timing difference
is not explained: both revisions selected (0,1), so an operand-order change
was **not** the cause. Same-process search/bypass and A/A diagnostics show
substantial variability; they do not establish a root cause for the 64-worker
difference. No claim that all-core performance is unchanged is made. The
maintainer chose not to pursue that separate question.

[New evidence archive](issue-1761-one-worker-data.tar.gz) contains the frozen
protocols, independent build logs and hashes, all recheck/A/A/pinned raw data,
analysis source and exploratory diagnostic logs/patch. Diagnostics are not
production changes or another supported API. The final diagnostic patch is
against d65a422c; earlier logs used the same probe before adding fresh-plan and
aggregate section counters. Their timings are diagnostic, not acceptance data.

Final measurement command per process (base/candidate are the independently
built copies of the unchanged example; see archive protocols for AB/BA order):

```bash
/usr/bin/time -v -o one-worker-pinned/1-$pair-$revision.time \
  taskset -c 32 ./$revision 1 \
  > one-worker-pinned/1-$pair-$revision.csv \
  2> one-worker-pinned/1-$pair-$revision.stderr
python3 analyze-one-worker.py one-worker-pinned
```

No Rust source changed after the previously passing correctness/local gate and
hosted CI at 3b0e4831. This update adds only the adoption record and evidence.
The full final 1T timing table follows at the end of this work log.

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

**STOPPED — Phase 1 is not complete; performance is unverified for promotion.**
Both complete comparisons retain inconclusive non-regression controls. The sole
retry additionally reports regressions for default-thread, reversed-operand
`read_compact_ba/256` (ratio 1.280, CI [1.148, 1.426]) and
`read_transpose_ba/256` (1.245, [1.158, 1.340]). No further measurements or
optimizations were attempted. Resumption requires a new maintainer decision.

The primary aggregate is faster in both attempts/configurations, but this is
**not** a successful speedup deliverable: an aggregate cannot override a failed
layout control or prove non-regression of noisy prepared execution. No causal
attribution of the large-case regression is claimed from these timings alone.

| Attempt | Default primary ratio [95% CI] | One-thread primary ratio [95% CI] | Overall |
|---|---|---|---|
| 1 | 0.715 [0.618, 0.829] | 0.607 [0.530, 0.695] | INCONCLUSIVE controls |
| 2 (sole retry) | 0.843 [0.752, 0.944] | 0.634 [0.528, 0.762] | REGRESSION and INCONCLUSIVE controls |

Full source, 10,080 raw samples (56 processes), empty measurement stderr files,
load observations, build/test logs and the small stdlib analysis script are
retained in [the evidence archive](issue-1761-binary-einsum-data.tar.gz).
The experiment source itself is tracked in the crate examples. Every process
completed, all samples met the 50 ms minimum, and numerical checks did not fail.
No timing sample was omitted. Tables below contain every case, with median
process latencies and paired-ratio confidence intervals; these are different
statistics, so the displayed median ratio need not equal the paired estimate.

Actual candidate build checkout was `4bf771b5f98ab7360d0ac42e59745fafc16f6437`,
which adds only this predeclared protocol to `166ab813`. Code/experiment bytes
are identical. Base is `d65a422ccba7b73b51b93f48c8407fe84742a936`.
The shared target directory initially reused a stale base executable on the
candidate build (identical checksum); detected **before timing**, refreshed
`tree.rs` mtime and rebuilt the einsum crate. Both build logs are retained.
Base/candidate executable SHA-256 values at the untracked copied-binary boundary
are in `binaries.sha256`. The candidate and base then had distinct executables.
All other workspace source/dependencies are identical. No compiler wrapper or
RUSTFLAGS were set; the captured RUST/CARGO/OMP/MKL/OPENBLAS/TENFERRO
variable listing was empty.

Build commands (run sequentially, not alongside measurements):

```bash
W=/home/shinaoka/tensor4all/.worktrees/tenferro-1761
B=/home/shinaoka/tensor4all/.worktrees/tenferro-1761-base
E=/tmp/tenferro-1761-evidence
(cd "$B" && CARGO_TARGET_DIR="$W/target" cargo build --release -p tenferro-einsum --example binary_planning)
cp "$W/target/release/examples/binary_planning" "$E/base"
(cd "$W" && touch crates/tenferro-einsum/src/planning/tree.rs && cargo build --release -p tenferro-einsum --example binary_planning)
cp "$W/target/release/examples/binary_planning" "$E/candidate"
```

Executed this loop for `attempt1`, then once for `attempt2` after classifying
attempt1 as inconclusive; `analyze.py` is in the archive (Python stdlib only):

```bash
cd /tmp/tenferro-1761-evidence
attempt=attempt1 # then attempt2, once only
mkdir -p "$attempt"
for threads in default 1; do
  for pair in 1 2 3 4 5 6 7; do
    order='base candidate'
    if [ $((pair % 2)) -eq 0 ]; then order='candidate base'; fi
    for revision in $order; do
      printf '%s threads=%s pair=%s revision=%s ' "$(date -Iseconds)" "$threads" "$pair" "$revision" >> "$attempt/host.log"
      uptime >> "$attempt/host.log"
      ./$revision "$threads" > "$attempt/$threads-$pair-$revision.csv" 2> "$attempt/$threads-$pair-$revision.stderr"
    done
  done
done
python3 analyze.py "$attempt" > "$attempt/summary.md"
```

## Final correctness and review

- `cargo test --release -p tenferro-einsum --lib`: 177 passed.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-einsum --features autodiff'`:
  passed, including repository formatting, root and standalone-extension clippy;
  einsum 193 unit + 41 integration + 2 prelude tests and 106 doctests passed.
  This includes actual eager/traced/AD CPU tests; not inferred from concrete tests.
- Coverage reviewed for the new two-input branch and unchanged option/shape
  validation paths; full numerical equality checks and positive optimizer
  controls cover the additions. Hosted CI owns measured line coverage and GPU /
  provider matrix; no GPU or BLAS performance claim.
- Self-reviewed the complete coherent diff against the repository rules and
  preflight checklist. No public surface, architecture, guide or skill changes.
  Known blocking finding: non-regression evidence above. Keep the PR **draft**,
  do not enable auto-merge or close #1761/#1771 as completed. This intentionally
  overrides the ordinary auto-merge default because the performance gate failed.
  The original checkout's arbiter edit and HANDOFF file were untouched.

## Complete timing tables

### attempt1

#### Threads: default
Primary ratio, 95% CI: (0.715498251035591, 0.6176233973541148, 0.8288833444913514)

| Case | Base us | Candidate us | Ratio [95% CI] | Max process CoV | Non-regression |
|---|---:|---:|---|---:|---|
| prepare_string/2 | 12.415 | 5.429 | 0.4473 [0.4318, 0.4633] | 4.2% | PASS |
| prepare_labels/2 | 10.510 | 4.321 | 0.4102 [0.3827, 0.4396] | 5.1% | PASS |
| ordinary_string/2 | 39.080 | 28.450 | 0.7281 [0.6783, 0.7815] | 6.8% | PASS |
| ordinary_labels/2 | 37.594 | 26.792 | 0.7106 [0.6590, 0.7662] | 5.6% | PASS |
| prepared/2 | 19.901 | 21.240 | 1.0082 [0.8821, 1.1524] | 13.0% | INCONCLUSIVE |
| prepare_string/8 | 12.325 | 5.399 | 0.4526 [0.4363, 0.4695] | 4.4% | PASS |
| prepare_labels/8 | 10.417 | 4.348 | 0.4126 [0.3980, 0.4277] | 4.3% | PASS |
| ordinary_string/8 | 46.297 | 30.102 | 0.6553 [0.6182, 0.6946] | 4.3% | PASS |
| ordinary_labels/8 | 44.382 | 31.144 | 0.6674 [0.6003, 0.7419] | 8.3% | PASS |
| prepared/8 | 20.060 | 24.336 | 1.0997 [0.9513, 1.2712] | 12.9% | INCONCLUSIVE |
| prepare_string/32 | 12.471 | 5.448 | 0.4513 [0.4333, 0.4700] | 5.0% | PASS |
| prepare_labels/32 | 10.702 | 4.207 | 0.4040 [0.3805, 0.4290] | 4.6% | PASS |
| ordinary_string/32 | 1063.263 | 728.595 | 0.7678 [0.4818, 1.2236] | 52.9% | INCONCLUSIVE |
| ordinary_labels/32 | 755.624 | 605.394 | 0.8624 [0.4661, 1.5959] | 71.3% | INCONCLUSIVE |
| prepared/32 | 539.732 | 615.634 | 1.0874 [0.6597, 1.7924] | 35.9% | INCONCLUSIVE |
| prepare_string/256 | 11.582 | 5.494 | 0.4583 [0.4362, 0.4815] | 4.8% | PASS |
| prepare_labels/256 | 10.510 | 4.375 | 0.4207 [0.3894, 0.4545] | 8.5% | PASS |
| ordinary_string/256 | 1009.856 | 854.443 | 0.9119 [0.6006, 1.3847] | 37.7% | INCONCLUSIVE |
| ordinary_labels/256 | 845.554 | 654.180 | 0.7025 [0.3619, 1.3640] | 39.8% | INCONCLUSIVE |
| prepared/256 | 772.080 | 746.483 | 0.8993 [0.5740, 1.4091] | 34.4% | INCONCLUSIVE |
| read_compact_ab/8 | 43.147 | 28.988 | 0.6895 [0.6269, 0.7583] | 7.9% | PASS |
| read_compact_ba/8 | 55.041 | 39.249 | 0.7264 [0.6840, 0.7713] | 6.1% | PASS |
| read_transpose_ab/8 | 43.099 | 30.975 | 0.7188 [0.6733, 0.7674] | 6.4% | PASS |
| read_transpose_ba/8 | 54.086 | 38.275 | 0.7166 [0.6891, 0.7452] | 6.2% | PASS |
| read_reverse_ab/8 | 43.416 | 31.025 | 0.7101 [0.6759, 0.7459] | 6.8% | PASS |
| read_reverse_ba/8 | 54.548 | 39.204 | 0.7357 [0.7008, 0.7725] | 5.9% | PASS |
| read_compact_ab/256 | 757.344 | 611.723 | 0.8541 [0.4029, 1.8108] | 72.5% | INCONCLUSIVE |
| read_compact_ba/256 | 3392.803 | 3159.082 | 0.9339 [0.7304, 1.1940] | 17.9% | INCONCLUSIVE |
| read_transpose_ab/256 | 973.802 | 1053.521 | 0.9875 [0.6274, 1.5542] | 60.9% | INCONCLUSIVE |
| read_transpose_ba/256 | 2891.384 | 3405.157 | 1.1425 [0.9555, 1.3661] | 15.1% | INCONCLUSIVE |
| read_reverse_ab/256 | 1013.633 | 1280.987 | 1.2701 [0.8464, 1.9060] | 37.3% | INCONCLUSIVE |
| read_reverse_ba/256 | 2845.735 | 3172.695 | 1.2078 [1.0328, 1.4124] | 10.9% | INCONCLUSIVE |
| batch_prepare/8 | 13.553 | 5.931 | 0.4461 [0.4358, 0.4566] | 2.7% | PASS |
| batch_ordinary/8 | 47.108 | 34.829 | 0.7545 [0.7007, 0.8125] | 5.5% | PASS |
| nary_prepare/8 | 25.996 | 25.979 | 1.0043 [0.9869, 1.0221] | 1.8% | PASS |
| nary_ordinary/8 | 78.767 | 78.360 | 1.0426 [0.9586, 1.1340] | 7.6% | INCONCLUSIVE |

#### Threads: 1
Primary ratio, 95% CI: (0.6072931125315971, 0.5304149221076447, 0.695314006368524)

| Case | Base us | Candidate us | Ratio [95% CI] | Max process CoV | Non-regression |
|---|---:|---:|---|---:|---|
| prepare_string/2 | 12.332 | 5.156 | 0.4197 [0.4038, 0.4363] | 3.3% | PASS |
| prepare_labels/2 | 10.646 | 4.113 | 0.3832 [0.3747, 0.3919] | 2.8% | PASS |
| ordinary_string/2 | 31.680 | 18.942 | 0.6129 [0.5278, 0.7116] | 11.5% | PASS |
| ordinary_labels/2 | 29.452 | 20.732 | 0.6722 [0.5506, 0.8206] | 12.7% | PASS |
| prepared/2 | 14.276 | 14.209 | 0.9577 [0.6533, 1.4040] | 25.5% | INCONCLUSIVE |
| prepare_string/8 | 11.922 | 5.141 | 0.4355 [0.4251, 0.4462] | 2.7% | PASS |
| prepare_labels/8 | 10.362 | 4.076 | 0.3951 [0.3853, 0.4052] | 1.8% | PASS |
| ordinary_string/8 | 33.100 | 19.073 | 0.5780 [0.4538, 0.7362] | 17.3% | PASS |
| ordinary_labels/8 | 31.931 | 18.643 | 0.6034 [0.4593, 0.7925] | 20.0% | PASS |
| prepared/8 | 14.843 | 11.059 | 0.8744 [0.6479, 1.1801] | 24.1% | INCONCLUSIVE |
| prepare_string/32 | 11.898 | 5.713 | 0.4864 [0.4711, 0.5023] | 4.6% | PASS |
| prepare_labels/32 | 10.184 | 4.765 | 0.4647 [0.4594, 0.4701] | 2.3% | PASS |
| ordinary_string/32 | 32.851 | 22.487 | 0.6323 [0.5155, 0.7754] | 15.9% | PASS |
| ordinary_labels/32 | 30.588 | 20.644 | 0.6425 [0.5101, 0.8093] | 20.6% | PASS |
| prepared/32 | 14.063 | 13.183 | 0.9713 [0.8157, 1.1565] | 16.7% | INCONCLUSIVE |
| prepare_string/256 | 11.685 | 5.182 | 0.4429 [0.4187, 0.4684] | 5.0% | PASS |
| prepare_labels/256 | 10.267 | 4.109 | 0.4047 [0.3872, 0.4231] | 3.8% | PASS |
| ordinary_string/256 | 745.133 | 729.350 | 0.9773 [0.9465, 1.0092] | 3.5% | PASS |
| ordinary_labels/256 | 740.633 | 732.212 | 0.9904 [0.9812, 0.9998] | 2.2% | PASS |
| prepared/256 | 728.537 | 727.276 | 1.0069 [0.9783, 1.0363] | 3.0% | PASS |
| read_compact_ab/8 | 30.611 | 19.826 | 0.6447 [0.6003, 0.6924] | 5.6% | PASS |
| read_compact_ba/8 | 41.332 | 26.807 | 0.6666 [0.6449, 0.6890] | 3.3% | PASS |
| read_transpose_ab/8 | 29.586 | 19.087 | 0.6347 [0.6010, 0.6702] | 4.4% | PASS |
| read_transpose_ba/8 | 40.854 | 27.244 | 0.6709 [0.6452, 0.6977] | 3.0% | PASS |
| read_reverse_ab/8 | 29.649 | 19.099 | 0.6464 [0.6172, 0.6771] | 5.5% | PASS |
| read_reverse_ba/8 | 39.647 | 27.474 | 0.6876 [0.6626, 0.7135] | 2.8% | PASS |
| read_compact_ab/256 | 742.422 | 736.445 | 0.9981 [0.9645, 1.0330] | 3.4% | PASS |
| read_compact_ba/256 | 1059.828 | 1098.109 | 1.0233 [0.9853, 1.0628] | 2.4% | PASS |
| read_transpose_ab/256 | 736.369 | 738.061 | 0.9973 [0.9752, 1.0198] | 1.9% | PASS |
| read_transpose_ba/256 | 912.407 | 914.972 | 1.0014 [0.9693, 1.0345] | 3.4% | PASS |
| read_reverse_ab/256 | 777.155 | 755.238 | 0.9624 [0.9410, 0.9843] | 2.2% | PASS |
| read_reverse_ba/256 | 1054.201 | 1100.996 | 1.0199 [0.9843, 1.0568] | 2.4% | PASS |
| batch_prepare/8 | 12.822 | 5.733 | 0.4383 [0.4122, 0.4661] | 5.4% | PASS |
| batch_ordinary/8 | 30.180 | 20.562 | 0.6818 [0.6441, 0.7218] | 4.5% | PASS |
| nary_prepare/8 | 24.608 | 23.537 | 0.9804 [0.9426, 1.0196] | 3.7% | PASS |
| nary_ordinary/8 | 56.237 | 56.129 | 1.0002 [0.9651, 1.0366] | 3.0% | PASS |

### attempt2

#### Threads: default
Primary ratio, 95% CI: (0.8427002509445062, 0.7523927657956246, 0.9438470772522456)

| Case | Base us | Candidate us | Ratio [95% CI] | Max process CoV | Non-regression |
|---|---:|---:|---|---:|---|
| prepare_string/2 | 11.708 | 5.370 | 0.4541 [0.4363, 0.4727] | 3.7% | PASS |
| prepare_labels/2 | 10.109 | 4.016 | 0.3981 [0.3772, 0.4202] | 5.0% | PASS |
| ordinary_string/2 | 39.280 | 28.300 | 0.7293 [0.6889, 0.7721] | 5.2% | PASS |
| ordinary_labels/2 | 37.340 | 26.174 | 0.7117 [0.6874, 0.7370] | 6.4% | PASS |
| prepared/2 | 19.352 | 23.049 | 1.1081 [0.9866, 1.2445] | 12.0% | INCONCLUSIVE |
| prepare_string/8 | 11.585 | 5.385 | 0.4560 [0.4326, 0.4808] | 4.3% | PASS |
| prepare_labels/8 | 10.139 | 4.193 | 0.4112 [0.4002, 0.4224] | 3.4% | PASS |
| ordinary_string/8 | 45.617 | 30.064 | 0.6502 [0.6163, 0.6859] | 6.2% | PASS |
| ordinary_labels/8 | 42.889 | 28.001 | 0.6540 [0.5978, 0.7154] | 7.7% | PASS |
| prepared/8 | 25.359 | 21.707 | 0.9347 [0.8159, 1.0708] | 11.6% | PASS |
| prepare_string/32 | 11.558 | 5.385 | 0.4532 [0.4294, 0.4783] | 4.7% | PASS |
| prepare_labels/32 | 9.744 | 4.208 | 0.4254 [0.4026, 0.4495] | 5.0% | PASS |
| ordinary_string/32 | 522.605 | 753.229 | 1.2621 [0.9184, 1.7345] | 36.2% | INCONCLUSIVE |
| ordinary_labels/32 | 719.795 | 701.092 | 1.1138 [0.7085, 1.7512] | 44.0% | INCONCLUSIVE |
| prepared/32 | 620.533 | 579.433 | 0.7655 [0.4922, 1.1904] | 30.1% | INCONCLUSIVE |
| prepare_string/256 | 11.591 | 5.333 | 0.4556 [0.4368, 0.4752] | 4.0% | PASS |
| prepare_labels/256 | 9.898 | 4.303 | 0.4270 [0.4044, 0.4508] | 4.7% | PASS |
| ordinary_string/256 | 815.665 | 605.916 | 0.7722 [0.5921, 1.0070] | 25.8% | PASS |
| ordinary_labels/256 | 744.966 | 879.996 | 1.1773 [0.7072, 1.9597] | 40.7% | INCONCLUSIVE |
| prepared/256 | 643.109 | 1025.172 | 1.4091 [0.9409, 2.1102] | 31.1% | INCONCLUSIVE |
| read_compact_ab/8 | 41.000 | 28.918 | 0.7140 [0.6483, 0.7863] | 8.7% | PASS |
| read_compact_ba/8 | 55.733 | 42.651 | 0.7178 [0.6714, 0.7675] | 7.0% | PASS |
| read_transpose_ab/8 | 43.100 | 32.401 | 0.7335 [0.6753, 0.7966] | 7.1% | PASS |
| read_transpose_ba/8 | 54.988 | 42.431 | 0.7314 [0.6727, 0.7951] | 6.9% | PASS |
| read_reverse_ab/8 | 43.853 | 33.171 | 0.7388 [0.6801, 0.8026] | 7.4% | PASS |
| read_reverse_ba/8 | 55.779 | 42.955 | 0.7362 [0.6800, 0.7970] | 5.7% | PASS |
| read_compact_ab/256 | 910.214 | 1087.093 | 0.9508 [0.6629, 1.3636] | 59.0% | INCONCLUSIVE |
| read_compact_ba/256 | 2733.675 | 3418.267 | 1.2797 [1.1483, 1.4261] | 11.7% | REGRESSION |
| read_transpose_ab/256 | 1011.644 | 960.177 | 0.9570 [0.7094, 1.2909] | 35.5% | INCONCLUSIVE |
| read_transpose_ba/256 | 2812.225 | 3572.707 | 1.2453 [1.1576, 1.3395] | 12.2% | REGRESSION |
| read_reverse_ab/256 | 1033.812 | 999.214 | 0.9666 [0.7203, 1.2972] | 27.9% | INCONCLUSIVE |
| read_reverse_ba/256 | 2697.091 | 3531.430 | 1.2846 [1.0709, 1.5410] | 15.0% | INCONCLUSIVE |
| batch_prepare/8 | 13.267 | 5.993 | 0.4475 [0.4373, 0.4580] | 2.0% | PASS |
| batch_ordinary/8 | 48.786 | 35.403 | 0.7336 [0.6840, 0.7868] | 7.2% | PASS |
| nary_prepare/8 | 25.224 | 25.917 | 1.0129 [0.9918, 1.0345] | 2.0% | PASS |
| nary_ordinary/8 | 80.257 | 76.942 | 0.9643 [0.8699, 1.0690] | 6.7% | PASS |

#### Threads: 1
Primary ratio, 95% CI: (0.6343310780474402, 0.5283242871223386, 0.7616078351583591)

| Case | Base us | Candidate us | Ratio [95% CI] | Max process CoV | Non-regression |
|---|---:|---:|---|---:|---|
| prepare_string/2 | 12.011 | 5.243 | 0.4320 [0.4129, 0.4520] | 3.4% | PASS |
| prepare_labels/2 | 10.584 | 4.019 | 0.3824 [0.3728, 0.3923] | 2.6% | PASS |
| ordinary_string/2 | 31.237 | 19.389 | 0.6523 [0.5810, 0.7324] | 12.0% | PASS |
| ordinary_labels/2 | 30.199 | 17.590 | 0.6286 [0.5284, 0.7477] | 19.0% | PASS |
| prepared/2 | 14.808 | 10.719 | 0.8481 [0.5997, 1.1995] | 32.1% | INCONCLUSIVE |
| prepare_string/8 | 11.816 | 5.074 | 0.4283 [0.4139, 0.4432] | 2.9% | PASS |
| prepare_labels/8 | 10.225 | 4.192 | 0.4030 [0.3779, 0.4297] | 4.3% | PASS |
| ordinary_string/8 | 40.427 | 17.831 | 0.5616 [0.4422, 0.7131] | 18.3% | PASS |
| ordinary_labels/8 | 38.238 | 18.032 | 0.5988 [0.4563, 0.7858] | 23.1% | PASS |
| prepared/8 | 11.841 | 11.142 | 0.8813 [0.6386, 1.2162] | 32.5% | INCONCLUSIVE |
| prepare_string/32 | 11.722 | 5.817 | 0.4942 [0.4770, 0.5119] | 3.0% | PASS |
| prepare_labels/32 | 10.186 | 4.695 | 0.4634 [0.4412, 0.4866] | 4.5% | PASS |
| ordinary_string/32 | 32.771 | 22.450 | 0.6968 [0.5414, 0.8967] | 21.7% | PASS |
| ordinary_labels/32 | 30.295 | 21.454 | 0.7102 [0.5166, 0.9763] | 22.5% | PASS |
| prepared/32 | 13.517 | 14.036 | 1.0050 [0.6426, 1.5717] | 31.0% | INCONCLUSIVE |
| prepare_string/256 | 11.680 | 5.081 | 0.4321 [0.4181, 0.4466] | 3.7% | PASS |
| prepare_labels/256 | 10.453 | 4.014 | 0.3884 [0.3765, 0.4008] | 2.5% | PASS |
| ordinary_string/256 | 743.595 | 730.327 | 0.9917 [0.9594, 1.0250] | 2.6% | PASS |
| ordinary_labels/256 | 743.874 | 731.377 | 0.9840 [0.9683, 0.9999] | 2.0% | PASS |
| prepared/256 | 720.265 | 730.329 | 1.0122 [0.9704, 1.0557] | 3.4% | PASS |
| read_compact_ab/8 | 30.122 | 20.090 | 0.6218 [0.5027, 0.7691] | 21.9% | PASS |
| read_compact_ba/8 | 40.368 | 27.220 | 0.6739 [0.6453, 0.7038] | 6.5% | PASS |
| read_transpose_ab/8 | 28.907 | 19.017 | 0.6389 [0.5888, 0.6933] | 11.4% | PASS |
| read_transpose_ba/8 | 40.466 | 27.272 | 0.6546 [0.6073, 0.7057] | 7.2% | PASS |
| read_reverse_ab/8 | 29.183 | 19.556 | 0.6471 [0.5994, 0.6987] | 9.9% | PASS |
| read_reverse_ba/8 | 40.891 | 27.895 | 0.6602 [0.6231, 0.6996] | 7.5% | PASS |
| read_compact_ab/256 | 737.020 | 740.702 | 1.0035 [0.9594, 1.0495] | 2.9% | PASS |
| read_compact_ba/256 | 1077.332 | 1092.535 | 1.0030 [0.9532, 1.0554] | 4.4% | PASS |
| read_transpose_ab/256 | 738.310 | 729.549 | 0.9964 [0.9711, 1.0223] | 2.2% | PASS |
| read_transpose_ba/256 | 918.651 | 908.636 | 0.9828 [0.9486, 1.0182] | 2.0% | PASS |
| read_reverse_ab/256 | 767.065 | 761.518 | 0.9779 [0.9529, 1.0035] | 2.7% | PASS |
| read_reverse_ba/256 | 1085.537 | 1089.603 | 0.9886 [0.9356, 1.0445] | 3.5% | PASS |
| batch_prepare/8 | 12.296 | 5.741 | 0.4543 [0.4350, 0.4744] | 4.3% | PASS |
| batch_ordinary/8 | 29.424 | 20.603 | 0.6602 [0.5709, 0.7636] | 16.4% | PASS |
| nary_prepare/8 | 24.773 | 24.403 | 1.0073 [0.9569, 1.0603] | 4.6% | PASS |
| nary_ordinary/8 | 56.115 | 55.628 | 0.9942 [0.9122, 1.0835] | 6.6% | PASS |

## Final accepted pinned one-worker comparison

## Threads: 1
Primary ratio, 95% CI: (0.6850946414319503, 0.6660961456100876, 0.7046350152482621)

| Case | Base us | Candidate us | Ratio [95% CI] | Max process CoV | Non-regression |
|---|---:|---:|---|---:|---|
| prepare_string/2 | 11.952 | 5.036 | 0.4218 [0.4057, 0.4386] | 3.4% | PASS |
| prepare_labels/2 | 11.127 | 4.240 | 0.3798 [0.3621, 0.3985] | 4.7% | PASS |
| ordinary_string/2 | 33.163 | 22.026 | 0.6728 [0.6450, 0.7018] | 4.7% | PASS |
| ordinary_labels/2 | 29.968 | 20.370 | 0.6808 [0.6503, 0.7127] | 3.6% | PASS |
| prepared/2 | 13.071 | 12.913 | 0.9862 [0.9553, 1.0180] | 3.8% | PASS |
| prepare_string/8 | 11.821 | 5.103 | 0.4296 [0.4074, 0.4531] | 5.5% | PASS |
| prepare_labels/8 | 11.030 | 4.117 | 0.3715 [0.3547, 0.3892] | 6.1% | PASS |
| ordinary_string/8 | 32.258 | 21.805 | 0.6724 [0.6455, 0.7004] | 3.4% | PASS |
| ordinary_labels/8 | 30.996 | 21.365 | 0.6832 [0.6461, 0.7224] | 4.6% | PASS |
| prepared/8 | 12.673 | 13.130 | 1.0149 [0.9655, 1.0668] | 5.5% | PASS |
| prepare_string/32 | 12.043 | 5.816 | 0.4883 [0.4592, 0.5193] | 4.8% | PASS |
| prepare_labels/32 | 10.768 | 4.740 | 0.4410 [0.4093, 0.4752] | 5.1% | PASS |
| ordinary_string/32 | 36.440 | 26.157 | 0.7108 [0.6722, 0.7516] | 4.7% | PASS |
| ordinary_labels/32 | 33.321 | 23.646 | 0.7130 [0.6893, 0.7375] | 4.5% | PASS |
| prepared/32 | 15.093 | 15.164 | 1.0076 [0.9839, 1.0320] | 4.3% | PASS |
| prepare_string/256 | 11.526 | 4.977 | 0.4424 [0.4088, 0.4789] | 7.9% | PASS |
| prepare_labels/256 | 10.161 | 4.079 | 0.3975 [0.3694, 0.4277] | 7.1% | PASS |
| ordinary_string/256 | 700.470 | 705.822 | 0.9910 [0.9518, 1.0318] | 2.6% | PASS |
| ordinary_labels/256 | 713.614 | 694.943 | 0.9775 [0.9418, 1.0147] | 2.9% | PASS |
| prepared/256 | 697.187 | 699.013 | 1.0055 [0.9703, 1.0420] | 2.8% | PASS |
| read_compact_ab/8 | 33.062 | 22.505 | 0.6650 [0.6270, 0.7054] | 5.1% | PASS |
| read_compact_ba/8 | 45.110 | 31.530 | 0.7191 [0.6824, 0.7578] | 5.0% | PASS |
| read_transpose_ab/8 | 33.352 | 22.308 | 0.6634 [0.6225, 0.7069] | 4.4% | PASS |
| read_transpose_ba/8 | 45.994 | 32.242 | 0.7001 [0.6552, 0.7480] | 4.5% | PASS |
| read_reverse_ab/8 | 33.325 | 22.809 | 0.6824 [0.6399, 0.7278] | 4.9% | PASS |
| read_reverse_ba/8 | 46.241 | 31.920 | 0.6963 [0.6564, 0.7387] | 4.1% | PASS |
| read_compact_ab/256 | 704.061 | 707.030 | 1.0136 [0.9841, 1.0441] | 3.1% | PASS |
| read_compact_ba/256 | 1064.515 | 1017.987 | 0.9626 [0.9160, 1.0115] | 3.2% | PASS |
| read_transpose_ab/256 | 722.098 | 717.099 | 0.9939 [0.9573, 1.0319] | 3.5% | PASS |
| read_transpose_ba/256 | 874.131 | 871.033 | 0.9936 [0.9702, 1.0175] | 3.2% | PASS |
| read_reverse_ab/256 | 728.110 | 736.102 | 1.0041 [0.9665, 1.0433] | 2.8% | PASS |
| read_reverse_ba/256 | 1054.971 | 1003.008 | 0.9520 [0.9361, 0.9682] | 1.9% | PASS |
| batch_prepare/8 | 12.588 | 5.441 | 0.4284 [0.4093, 0.4485] | 5.2% | PASS |
| batch_ordinary/8 | 34.292 | 23.520 | 0.6643 [0.6319, 0.6984] | 5.4% | PASS |
| nary_prepare/8 | 24.239 | 24.182 | 1.0112 [0.9445, 1.0826] | 5.9% | PASS |
| nary_ordinary/8 | 61.657 | 60.819 | 0.9981 [0.9482, 1.0507] | 3.5% | PASS |
