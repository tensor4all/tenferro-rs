# #1762: Linux eager/session attribution and bounded deferral

## Decision and scope

**Defer production changes.** The 2x2 F64 ten-matmul chain has a reproducible
no-AD eager overhead on this Linux host. Repeated session transitions explain
part of the route difference, but the residual cannot honestly be called AD
bookkeeping. The next bounded investigation should target
`tenferro-ad/src/eager_exec.rs::concrete_tensor_read` and the CPU session's
view-materialization route, not leaf construction or a new fast public API.
No fast path, default-thread change, or ownership change is adopted here.

This is Phase 2 of #1771, not the broader historical #1758 program. The
maintainer explicitly selected **2x2, ten operations, one worker** in the
conversation after Phase 1. This overrides the issue's earlier default-thread
measurement instruction for this experiment. Default/multithread, GPU, backward,
capture and functional transforms are **not measured or certified**. Active-AD
forward execution is a separate diagnostic, not an AD correctness claim.

## Reproduction and retained evidence

Library/base: `7dfc01127f4a8752a8bb504641feb396683576c3` (merged #1772).
There is no optimization candidate; only the example and this record are added.
Source: `crates/tenferro-ad/examples/eager_session_chain.rs`.
Archive: `issue-1762-eager-session-data.tar.gz` (source snapshot, Cargo.lock,
pre-timing protocol, environment, all raw samples, host observations, separate
instrumented output and analysis). The archived source checksum identifies the
original untracked experiment; compare it with the tracked example.

Host: Linux `primerose`, AMD EPYC 7713P, Rust 1.97.1. Release/default `cpu-faer`;
no BLAS provider. Both backends explicitly use `CpuBackend::with_threads(1)` and
assert `num_threads() == 1`; stderr records both effective configured counts.
`taskset -c 32` separately restricts CPU affinity. It does not configure workers.
No library instrumentation is added. Initial compile failed because the probe
assumed `Tensor: Clone` and imported the wrong session-host trait; both were
corrected before timing. Those build logs are retained, not performance samples.

```sh
cargo build --release -p tenferro-ad --example eager_session_chain
for i in 1 2 3 4 5 6 7; do
  order=forward
  if [ $((i % 2)) -eq 0 ]; then order=reverse; fi
  taskset -c 32 target/release/examples/eager_session_chain "$order" > "run-$i.csv"
done
# Extract the archive and run its analyze.py to reproduce the summary.
TENFERRO_PROFILE_EAGER_OP_AGG=1 TENFERRO_PROFILE_EAGER_OP_PRINT_EVERY=1000 \
  taskset -c 32 target/release/examples/eager_session_chain > profile.csv 2> profile.log
```

The chain reuses nontrivial compact column-major inputs, which are constructed
outside timing along with the backend/runtime and eager leaves. Concrete chains
borrow the initial tensor; eager clones its initial handle. Every chain times
all ten dependent operations, final owned Tensor retrieval/black-box observation
and result destruction. `to_tensor()` explicitly duplicates eager output; it is
not assumed to be a free handle conversion. All four output values are checked
against an independent 2x2 reference before timing (absolute tolerance 1e-12).
Empty-session diagnostics have no tensor result. The leaf diagnostic includes
construction of its four-element owned Tensor, registration, final retrieval and
destruction; it is not a pure leaf-registration timer.

Seven sequential processes alternate complete case order. Each case warms for
at least 100 ms and retains five samples of at least 100 ms (100-call batches).
Analysis uses each process's median, then paired log ratios against shared
concrete, with Student-t 95% intervals (df=6). The predeclared CV ceiling is 10%
for each chain's seven process medians; all pass, so no retry or exclusions.
Host observations are retained; they do not establish an otherwise idle host.

## Results

Times are microseconds **per complete ten-operation chain**, except leaf (one
construction) and empty_x10 (ten empty sessions). Range/CV use process medians.

| Case | Median us | Range us | CV | Ratio/shared, 95% CI |
|---|---:|---:|---:|---:|
| shared concrete session | 25.182 | 24.720–26.363 | 2.5% | 1 |
| concrete session per op | 96.947 | 94.331–104.908 | 3.6% | 3.842 [3.670, 4.022] |
| ordinary no-AD eager | 191.056 | 186.612–200.349 | 2.4% | 7.572 [7.414, 7.734] |
| active-AD forward | 286.104 | 276.810–304.363 | 3.6% | 11.353 [10.983, 11.735] |
| empty_x10 | 70.960 | 69.514–77.075 | 3.7% | diagnostic only |
| leaf | 20.005 | 19.300–21.248 | 3.5% | diagnostic only |

**PASS for distinguishing route overhead, not a measured optimization speedup.**
The per-op/shared comparison changes session scope around the same concrete
matmuls. It includes effects on resource entry, scheduling and locality; it is
not a measurement of Rayon transfer alone. Empty-session and leaf medians are
not subtracted from eager timing to produce an exact decomposition.

## Attribution and current source contracts

The separately instrumented eager profile contains 30 complete no-AD blocks of
1000 operations. Median per-op sections: total 19.714 us,
`exec_single_output_read` 17.659 us, input-read collection 1.051 us, untracked
result construction 0.543 us, backend-lock acquisition 0.032 us. Sections nest
and instrumentation perturbs timing; they must not be summed. Nonetheless the
execution section, not untracked result bookkeeping or uncontended lock
acquisition, dominates this profile.

The 23 complete active-forward blocks separately show median total 28.363 us,
execution 20.900 us, input materialization 3.179 us, output recording 2.030 us
and tracked result construction 1.197 us. These observations do not identify
backward/capture/functional-transform costs.

Source inspected at the baseline:

- `eager_ops.rs::nary_op` selects the no-AD read route only after checking both
  grad-recording state and capture state. Do not replace that condition with
  `requires_grad == false`.
- `eager_exec.rs::exec_standard_op_on_tensor_reads` opens a backend session per
  ordinary eager operation. Its in-session dispatcher concretizes view inputs,
  promotes dtypes and calls `dot_general_read`.
- `concrete_tensor_read` borrows `TensorRead::Tensor`, but materializes
  `TensorRead::View` through the session. Eager retained records expose reads
  through their allocation-group descriptor. The current execution section
  includes materialization, validation/preparation, provider dispatch,
  allocation and output wrapping; this experiment does not split them exactly.
- `CpuExecSession::to_contiguous_read` has a Tensor-versus-View branch. Its
  comment describing an Arc clone is no longer a sufficient optimization
  contract: `materialize_tensor_read` calls `clone_host_tensor_read`, which
  calls `TypedTensor::duplicate`, rather than cloning an owning Tensor handle.
  Owned `Tensor` does not implement `Clone`. Copy omission must not manufacture
  an aliasing mutable owner from a retained borrowed descriptor.
- `EagerTensor::to_tensor` delegates to `duplicate_value`; successful host
  duplication does not enter a backend session. Therefore final retrieval
  must not simply be counted as another session on this host route.
- `new_leaf` does enter an execution session to produce its semantic retained
  value. This remains #1704's responsibility. Initial leaves are outside the
  chain timer; fixing them cannot explain the timed per-operation residual.

References read: repository/shared performance and numerical rules, #1771 and
#1762, existing `eager_dispatch_baseline` and `dot_general_overhead` benchmarks,
eager operation/record/materialization code, CPU backend/session materialization
and resource-entry paths. Shared rule revision consulted:
`1129d9b6d9ed0d949e161905b59e1cd53e7e00d2`.

## Bounded follow-up, not an authorization to bypass contracts

Investigate whether the existing eager execution owner can pass a validated
compact **borrowed read** to its existing backend read operation without first
creating a new mutable owned tensor. Before implementation, count actual
materializations separately from timing and attribute their share in this same
ordinary chain. Review the historical #1435/#1060 changes against today's
retention contract; do not transplant their old Arc-clone assumptions.

Minimum eligibility: valid retained lifetime, dtype/shape/layout and same-device
checks, no owner synthesis or implicit transfer, unchanged required session
admission/placement, backend locks and unwind/nested-entry behavior. Keep genuine
noncompact and unsupported cases on their existing materialization/error path.
A candidate must test the real public eager chain, invalid shape/dtype/placement,
noncompact inputs and alias/mutation-after-handoff behavior, plus active AD,
capture and functional-transform parity. Remeasure the ordinary chain and a
larger-work control before claiming a speedup. If that is not expressible at the
existing boundary, defer rather than introduce a new public API/storage model.

## Verification

Release build and seven complete runs passed every numerical assertion. The
non-release executable also passed through
`bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo run -p tenferro-ad --example eager_session_chain'`,
including the repository's formatting and CI-parity clippy groups. Debug timings
from that check are not performance evidence. Self-review found no library,
public API, AD-rule or device-contract modifications; the archive retains all
release samples rather than only favorable summaries. Hosted CI is a separate
PR requirement, not implied by these local checks.

This record does not adopt such a candidate. #1704 remains separate; no generic
session-elision, leaf optimization, GPU dispatch change or exhaustive rollout
is bundled into Phase 2.
