# Issue #1525: session-aware prepared extension execution

## Scope

This work addresses the trace-side executor synchronization attributed in
`tenferro-rs#1525`. The public `inv` benchmark showed a small t1 gap and a
large t4 gap between direct and traced execution. The approved scope is one
reviewable tenferro-rs PR: a generic session-aware prepared-extension contract,
runtime dispatch, and the CPU `LuSolvePrepared` proof path. Native CUDA and
wgpu stream/queue mapping remains with the multi-device execution work in
#1471.

## Read and considered

- `REPOSITORY_RULES.md`, `AGENTS.md`, and the execution-session design docs.
- `tenferro-runtime` prepared-operation capability and segmented execution.
- `tenferro-cpu` `CpuOperationEntry`, `CpuExecSession`, buffer-pool, and
  non-reentrant executor contracts.
- `tenferro-linalg` prepared LU solve and its traced extension preparation.
- Issue #1525 attribution and design comments, including the no-fallback and
  no-recursive-session requirements.

## Decisions

- Add optional `supports_session` / `execute_in_session` methods to the
  prepared-operation executor. The default is a typed unsupported result; the
  scheduler never silently falls back after entering a session-capable region.
- Extend segmented execution to admit a single session for each consecutive
  compatible region. A non-session extension remains a boundary, while a
  later compatible prepared extension is still grouped with its neighboring
  native/host work instead of falling back to per-operation admission.
- Keep the extension macro opt-in explicit and backend-type-specific. The
  linalg implementation advertises the capability only for CPU
  `LuSolvePrepared`; CUDA, wgpu, eager wrapper, and other linalg paths retain
  their existing boundaries until their own session mapping is implemented.
- Reuse the already-entered `CpuExecSession` context and buffer pool for the
  prepared solve. The session path does not call `with_backend_session` again.
- Keep the erased CPU capability bridge in the backend leaf. Its raw pointer
  method is an explicitly unsafe contract, checked by the build-local session
  type identity before reconstruction, and the callback cannot return a
  session borrow.

## Alternatives rejected or deferred

- Batching ordinary host/backend nodes was rejected by the issue attribution:
  identity construction was only about 0.33 ms at t4 while prepared solve
  work dominated, and the candidate worsened the trace t4 result.
- A generic CUDA/wgpu implementation was not added. Native stream/queue and
  event mapping belongs to #1471.
- No fused linalg algorithm or cache redesign was added; this PR only removes
  repeated executor/session admission from the proven CPU trace path.

## Verification

Focused public API `inv` measurements were run on the isolated CPU workload
with 15 samples and 3 warmups at both thread counts:

| mode | t1 | t4 |
| --- | ---: | ---: |
| direct/eager | 46.038 ms | 15.160 ms |
| trace | 50.805 ms | 17.443 ms |
| trace/direct delta | +10.4% | +15.1% |

The pre-change attribution on the same public API and machine was 39.828 ms /
47.712 ms at t1 and 15.999 ms / 28.215 ms at t4. The traced t4 time therefore
fell by about 38.2%, with the remaining gap attributed to the linalg work and
not hidden by a threshold change. The final t1 run had normal CPU-frequency
variation relative to the earlier attribution, so the t1 result is reported
as an absolute before/after measurement rather than as a claimed speedup.

Passed checks during implementation:

- runtime prepared-operation source contract and session-dispatch tests;
- tenferro-runtime, tenferro-cpu, tenferro-linalg, and extension-macro checks;
- linalg CPU single-entry tests;
- traced linalg correctness integration tests;
- release public-API benchmark binary and focused `inv` runs at t1/t4.

Remaining verification is the repository fast gate, rules review, and CI on
the final PR. Full CUDA/wgpu session parity is intentionally out of scope.
