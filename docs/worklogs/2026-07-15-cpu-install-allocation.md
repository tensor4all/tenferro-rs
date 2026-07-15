# CPU Mandatory Empty-Install Allocation Removal

## Summary

Every warm `CpuBackend::install` allocated even when its closure was empty. The
execution path cloned an owned CPU vector, inserted an active request into a
tree map, and broadcast execution-owner metadata to every Rayon worker on both
entry and exit. This change moves stable metadata to context construction and
keeps reusable arbiter storage across executions. It removes the CPU-domain,
arbiter-node, and broadcast allocations paid on every backend entry; it does
not claim that the managed Rayon scheduler never allocates.

## Context Read

- `crates/tenferro-cpu/src/{topology,arbiter,context,backend}.rs`
- CPU context, arbiter, backend reentry, placement, and topology tests
- `docs/design/cpu-backend-execution.md`
- Repository-local and shared Rust performance rules

## Decisions

- Store `CpuSet` data in `Arc<[CpuId]>`. CPU domains are immutable structural
  metadata, so execution permits should share them instead of cloning a `Vec`.
- Keep the arbiter's fair waiter queue as `VecDeque`. Store active requests in
  a capacity-retaining `Vec`, because active admission only scans for conflicts
  and removes by request id; ordering is not part of admission fairness.
- Register one shared `Arc<ExecutionScopeState>` in every Rayon worker TLS at
  context construction, and wait for all registrations before returning the
  context. Execution entry changes the shared owner/depth under a mutex and an
  RAII guard clears it after normal return or panic.
- Keep the calling thread's execution-owner TLS. It rejects direct nesting;
  worker TLS plus the shared active state rejects spawned, stolen, sibling, and
  cross-pool reentry without per-execution broadcast.

## Rejected Alternatives

- Retaining Rayon `broadcast` would continue to make empty entry cost depend on
  worker count and allocate scheduler jobs on the warm path.
- An atomic owner without a depth/transition lock would make concurrent context
  misuse and panic cleanup harder to reason about. The mutex is outside tensor
  kernels and preserves the prior serialized owner invariant.
- Replacing `VecDeque` waiters with an unordered retained container would risk
  changing the established older-conflicting-waiter fairness contract.

## Verification

- Before the change, source-path accounting identifies two unconditional
  one-thread allocations per entry: cloning the `CpuSet` vector and inserting
  an active request into `BTreeMap`. Multi-worker entry additionally executes
  two Rayon broadcasts, whose scheduling work grows with the pool. The old
  implementation measured 128 caller-thread allocations over 64 one-worker
  entries (two on every call), so the new minimum-zero gate fails
  deterministically before this change.
- After the change, 64 individually measured warm entries contain at least one
  zero-allocation call for each of one, two, and four workers. Repeating that
  gate five times passed. A diagnostic sustained window still observed one to
  three allocations over 64 two-worker calls, with first sizes of 48 or 1520
  bytes at varying iterations. Crossbeam's injector grows queue blocks
  amortized (its block capacity is 63); warm-up history and other queue traffic
  shift the observed iteration, so this periodic managed-scheduler cost led to
  the narrower contract below.
- A dedicated integration-test allocator measures individual public empty
  installs after warm-up for one, two, and four workers. It requires that a
  warm entry can complete without allocation, proving there is no mandatory
  backend-entry allocation without asserting that Rayon's injector never
  performs periodic maintenance.
- The `cpu_install_overhead` benchmark reports context and backend entry latency
  separately for one, two, and four workers so worker-dependent regressions are
  visible.
- Constructor tests verify every Rayon worker has registered its execution
  scope before construction completes.
- Existing tests cover direct and independent backend nesting, cross-pool
  scheduling, stolen children, parallel siblings, shared contexts, panic
  cleanup, overlapping/disjoint CPU domains, waiter fairness, and provider
  exclusion.
- The cross-pool fixture constructs explicit two-worker `CpuContext` pools so
  it tests managed worker propagation on every platform. A one-thread
  compatibility context has no owned Rayon worker, and ambient global Rayon is
  outside this execution-scope contract.

## Residual Risk

The managed Rayon's injector occasionally allocates scheduler storage after
warm-up. The backend owns this scheduler residual even though Rayon implements
it, and an unbounded zero-allocation promise would therefore be false. The
regression gate therefore targets mandatory per-entry allocation; sustained
allocation rates remain benchmark evidence rather than an exact CI assertion.

The active-request vector uses linear conflict and id scans, as the previous
tree-map path already linearly scanned all active values for every admission.
The representation should be revisited only if workloads demonstrate a large
number of simultaneous disjoint or explicitly reentrant permits.
