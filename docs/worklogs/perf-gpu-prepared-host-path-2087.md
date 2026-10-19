# The prepared, caller-held-session path: what the per-call time is made of (#2087)

## Decisions

- The `dense_batched_matmul_f64_b16_32` trace case is a zero-input prepared
  graph (two bound GPU constants, one `dot_general`) run through
  `Runtime::run_prepared_in` inside one caller-held CUDA session. The committed
  tool for it is `crates/tenferro-gpu/src/cubecl/tests/host_path_phases.rs`
  (`--ignored host_path_phases`): it times that call and several neighbouring
  arms in one process, and it checks every arm against a CPU f64 reference with
  a tolerance. It deliberately reports only whole-arm timings; attributing time
  to internal phases needs a different instrument (below), and asserting timing
  on a shared host would be meaningless.
- Removed work that was redundant per call, not cached:
  - the resident-output and runtime-input predicates in
    `crates/tenferro-gpu/src/cubecl/runtime_adapter.rs` built a `TensorView`
    (a layout, hence a strides allocation) for an *owned* read only to downcast
    its backend storage; an owned read now exposes that storage directly, and a
    view keeps the projection it is defined on;
  - `ScheduledEventDomains::new` seeds the single run of a
    coalescing-eligible schedule directly instead of scanning every location
    into a driver table and a `HashSet` to find one entry. The general path no
    longer carries a second coalescing decision: eligibility has already
    returned by then;
  - `resolve_input_refs` no longer re-validates a prepared graph's *retained*
    inputs when preparation already ran that check on them. The proof travels in
    the prepared handle (`defaults_validated`): preparation with a complete
    replacement input list validated the replacements only, so its handle still
    re-checks the retained values. Caller-supplied inputs are validated on every
    call, and `PreparedCompiledGraph::defaults_validated` is set from how
    preparation resolved its inputs rather than assumed from "is prepared".
- Left in place, with the measured reason:
  - the per-call retirement barrier. `run_prepared_in` documents that it retires
    its own device work before returning, and the harness trace arm times
    exactly that. Removing it is a public-semantics decision, not a host-path
    fix. Its CubeCL `flush` cannot be skipped without knowing whether the run
    queued CubeCL work, which CubeCL 0.11 exposes no way to ask and which
    tracking at tenferro's ~130 launch sites would not cover for extension
    crates.
  - the fresh output's device-address resolution. `Client::empty` is
    asynchronous and a `Handle` carries no pointer, so the address costs one
    blocking device-thread round trip per allocated output. Removing it needs a
    CubeCL allocation API that returns the address, or carrying the address in
    the reuse pool - the latter is a memo, which this issue explicitly rules out.
  - routing the compact-view operands of a prepared graph to cuBLAS instead of
    cuTENSOR: the whole-arm difference between the two session arms measured
    below one microsecond, so not worth a kernel-routing change.

## What the per-call time is made of (separate instrumentation experiment)

The numbers below come from a temporary experiment, not from the committed test:
`Instant` probes were inserted into `run_prepared_in`,
`execute_scheduled_slots`, the CUDA `dot_general` path and the CUDA event
domain, the graph was run 200 times back to back, and the probes were then
removed. That instrumentation and its raw data are not retained, and the
committed test cannot reproduce a per-phase split. Treat the table as the
working hypothesis that motivated the removals, not as a standing measurement.

| phase | us/call |
|---|---:|
| held-session entry | 0.04 |
| prepared preflight (validate, lease, session-support) | 0.30 |
| retained-input reads + metadata check | 2.2 |
| exec-program validation + input count | 0.13 |
| event-domain run setup | 0.7 |
| slot init + input ingress check (2 inputs) | 1.5 |
| `ScheduledEventDomains` staging / node validation / retention / token | 2.4 |
| output collection and materialization | 0.7 |
| `dot_general` launch | 28.6 |
| - output shape + allocation (tensor metadata 1.9 of it) | 2.9 |
| - plan hash / lookup / workspace | 0.65 |
| - `build_layout` | 0.9 |
| - operand resolution (lhs 0.86, rhs 0.47, output 9.2) | 10.5 |
| - `cutensorContract` | 9.2 |
| - validation, context, cross-stream hand-off | 0.6 |
| run retirement (drain) | 6.8-9.5 |
| - CubeCL `flush` (device-thread hand-off) | 3.5-6.4 |
| - `cuStreamSynchronize` | 2.4 |
| **full call** | **~44-48** |

The committed test's whole-arm deltas, measured in one process (a fresh run of
the same code, `--release`, 400 calls per arm) give the same order of
magnitude without claiming a phase split: prepared call 42.6 us median, a
graph-free session contraction 19.1 us, the batched-GEMM enqueue arm 7.2 us,
adding the runtime's synchronize to that arm 17.3 us, a CubeCL `flush` alone
6.4 us, a raw `cuStreamSynchronize` 0.98 us.

## nsys cross-check (harness trace case, 292 prepared calls)

One kernel launch and one stream barrier per call, no per-call events and no
per-call copies: `cuStreamSynchronize` 292, `cuLaunchKernel` 291,
`cuKernelGetFunction` 291, `cuCtxSetCurrent` 667, `cuGetProcAddress_v2` 1236,
`cuEventSynchronize` 2, `cuMemcpy*` 3 (uploads and the verification download),
`cuMemAlloc_v2` 4. It also shows the vendor enqueue resolving its own kernel and
driver entry points every call, which is where the remaining launch cost sits.
These counts were taken on the benchmark's trace case and are recorded in the
issue's report, not in a repository file.

## Verification conclusions and constraints

- The removals are structural: an owned read no longer builds a view to ask
  where its storage is, a coalescing-eligible schedule no longer scans for a
  driver it cannot find twice, and a retained-input check that already ran is
  not repeated. They are covered by existing tests: every CUDA device test
  exercises the owned predicate on instruction outputs, prepared-graph tests
  cover both `resolve_input_refs` branches, and every single-domain schedule
  takes the new event-domain path.
- `prepared_execution_checks_retained_defaults_preparation_did_not_validate`
  (`crates/tenferro-runtime/tests/integration/runtime_public_api.rs`) pins the
  soundness condition: preparing with complete replacement inputs, then running
  with no inputs, still reports `PlaceholderShapeMismatch` for retained values
  that violate the program's declared shape relation.
- The end-to-end effect is **not established** here. The measurement host runs
  at a load average around 65 and switches between two timing states, so
  best-case per-call minima moved from 36.7-37.9 us to 35.7-36.1 us in an
  interleaved A/B of the two revisions while medians moved by several
  microseconds in both directions. The harness reports 0.041-0.046 ms trace /
  0.024-0.025 ms eager for b16_32 and 2.12 ms trace / 2.10 ms eager for
  b1024_256, matching the recorded baseline within that noise. Do not quote a
  microsecond gain from this change; quote the removed work.
- The harness's PyTorch arm is not a like-for-like comparison even so: its
  sampler batches operations and synchronizes once per batch, while the trace
  arm includes one retirement barrier per call by contract.
