# CPU Empty-Install Allocation Removal

## Summary

Warm `CpuBackend::install` allocated even when its closure was empty. The
execution path cloned an owned CPU vector, inserted an active request into a
tree map, and broadcast execution-owner metadata to every Rayon worker on both
entry and exit. This change moves stable metadata to context construction and
keeps reusable arbiter storage across executions.

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

- A dedicated integration-test allocator measures public empty installs after
  warm-up for one, two, and four workers.
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

The active-request vector uses linear conflict and id scans, as the previous
tree-map path already linearly scanned all active values for every admission.
The representation should be revisited only if workloads demonstrate a large
number of simultaneous disjoint or explicitly reentrant permits.
