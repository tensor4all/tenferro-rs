# ResourceArbiter Uncontended Fast-Path Prototype

## Status and scope

This specification defines a measurement-driven prototype for the first eager
CPU fast-path step. It does not add an eager session API, an async runtime, a
thread pool, or another scheduler. The application and tenferro continue to use
the existing `CpuBackend`, `CpuContext`, and `ResourceArbiter` ownership model.

The current pinned one-thread baseline on an AMD EPYC 7713P is approximately:

- empty `CpuBackend::install`: 7.16 us;
- public eager `neg`/`add`/`reduce_sum` on tiny tensors: 9-11 us;
- eager collection of one `TensorRead` into a `Vec`: 0.04-0.05 us.

The prototype therefore targets `ResourceArbiter` admission and release rather
than `SmallVec` conversion or a new public execution API.

## Current cost hypothesis

Every recovering backend entry currently takes the arbiter mutex, allocates a
request ID, appends an uncontended request to the waiter queue, calls
`notify_all`, searches the queue for the newly inserted ID, moves the request
to the active vector, and later takes the mutex again and calls `notify_all` on
permit drop.

The hypothesis is that queueing and unconditional notification account for a
material part of the measured empty-backend entry cost. This prototype tests
that hypothesis without changing the arbiter's synchronization model.

`CpuSet::clone` is not a heap allocation in this path: `CpuSet` already stores
its sorted CPUs in `Arc<[CpuId]>`.

## Design

### Direct admission

`acquire_request` first locks the existing `ArbiterState` and attempts direct
admission before assigning a waiter ID:

1. The waiter queue must be empty. This prevents a new request from bypassing
   an older conflicting waiter.
2. Every active request owned by someone else must be compatible with the new
   request.
3. Reentrant ownership is computed exactly as on the existing slow path.
4. A request ID must be available without overflow.

When these conditions hold, the request is appended directly to `active` and a
normal `ResourcePermit` is returned. There is no waiter insertion, queue
search, or condition-variable notification.

When any condition does not hold, execution uses the existing waiter queue and
condition-variable loop unchanged. Poison recovery and request-ID exhaustion
also continue to use the existing paths.

Conceptually:

```rust
let mut state = inner.state.lock()?;
if state.waiters.is_empty() && state.active_is_compatible(&request, owner) {
    return state.admit_direct(request, owner, Arc::clone(&inner));
}
state.enqueue_and_wait(request, owner, &inner)
```

The implementation should factor ID allocation and active insertion into small
private helpers so the direct and queued paths cannot drift semantically.

### Waiter-aware release

`ResourcePermit::drop` continues to take the same arbiter mutex and remove its
request from `active`. It calls `notify_all` only when the waiter queue is
non-empty after removal.

This is race-safe under the same mutex. A waiter arriving after an empty-queue
check observes the already-updated active set and either admits immediately or
waits for another conflicting active permit. No waiter depends on a
notification for a state change that occurred before it joined the queue.

### What this prototype does not do

- It does not introduce lock-free or atomic admission.
- It does not retain a permit across eager operations.
- It does not change `CpuContext` or Rayon pool ownership.
- It does not add Tokio, futures, blocking-task helpers, or executor adapters.
- It does not change public APIs or error types.
- It does not change NUMA overlap, provider-exclusive, reentrant-owner,
  fairness, poison-recovery, or request-ID-exhaustion semantics.
- It does not mix application job scheduling with tenferro resource
  arbitration.

An atomic admission protocol requires a separate design only if this minimal
prototype proves that queue bypass is insufficient. Such a protocol must still
support overlapping CPU-set conflicts and older-waiter fairness; a second
thread-management layer or a global exclusive bit is not acceptable.

## Correctness verification

Existing arbiter tests must continue to cover:

- disjoint CPU sets executing concurrently;
- overlapping CPU sets excluding each other;
- provider-exclusive requests excluding CPU-set requests;
- older conflicting waiters not being bypassed;
- compatible requests progressing when fairness permits;
- reentrant-owner semantics;
- poisoned mutex recovery;
- request-ID exhaustion recovery;
- permit drop after normal execution and unwinding.

Focused tests will additionally distinguish direct admission from queued
admission through test-only counters or state inspection. Production behavior
must not expose those counters or add profiling work when disabled.

## Performance experiment

The comparison reuses the same host and pins the benchmark process to CPU 0.
The primary command is:

```console
RAYON_NUM_THREADS=1 taskset -c 0 \
  cargo bench -p tenferro-cpu --bench cpu_install_overhead
```

The public eager baseline is then repeated with:

```console
RAYON_NUM_THREADS=1 taskset -c 0 \
  cargo bench -p tenferro-ad --bench eager_dispatch_baseline
```

Criterion configuration, compiler profile, input sizes, backend thread count,
and CPU pinning must match the recorded baseline. The report must include the
median and 95% confidence interval for at least:

- empty one-thread `CpuBackend::install`;
- eager `neg_f64/1`;
- eager `add_f64/1`;
- eager `reduce_sum_f64/1`;
- eager `dot_general_f64/1`.

The prototype is considered promising when empty backend entry improves by at
least 20% without a statistically significant regression in any listed eager
case. Reaching 2 us or less for empty backend entry is a strong result. A
smaller improvement falsifies queue bookkeeping as the dominant explanation
and triggers a separate atomic-admission design rather than further unrelated
micro-optimizations.

This threshold evaluates whether to continue the prototype; it is not the
final eager non-inferiority threshold for the execution-engine architecture.

## Deliverables

The prototype branch contains:

1. the minimal arbiter implementation change;
2. focused correctness tests for direct and queued admission;
3. unchanged or minimally extended Criterion benchmarks;
4. a worklog containing before/after results and residual risks.

No pull request is created from the prototype unless the measurements and
correctness results justify promoting it to an implementation change.
