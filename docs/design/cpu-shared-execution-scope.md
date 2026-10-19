# CPU shared execution scope

Accepted feature: [#1801](https://github.com/tensor4all/tenferro-rs/issues/1801).

## Contract

`CpuBackend::with_execution_scope` enters one Tenferro-managed CPU executor and
holds its existing resource permit for the synchronous callback. Ordinary eager
operations, AD execution and prepared trace execution may use clones of that
same immutable backend witness sequentially inside the callback. The operation
and AD APIs are unchanged. External/fallible executor domains are not supported
by this initial scope API; their operation-level admission remains unchanged.

A scope is not a borrowed `Engine`: it holds no mutable buffer/cache
borrow while calling user code. Each operation/session still borrows the owning
engine's resources exclusively and releases that borrow before the next call.
Output allocation semantics, provider dispatch, AD recording, runtime epoch
checks and runtime-owned caches remain unchanged.

## Implementation boundary

The CPU backend owns a thread-local, callback-lifetime admission record containing
the existing immutable backend identity, engine and resource permit. This is not
a cache and stores no borrowed pointers or tensor data. It exists only on the
executor thread running the callback, and RAII restores it on return or unwind.
A separate operation loan prevents nested backend entry while an operation or
borrowed session is active. Other Rayon worker threads do not inherit this
admission; the existing execution-owner guard continues to reject their re-entry.

A matching operation borrows the scope permit rather than acquiring another one.
The provider entry helper reuses the entered domain only while this operation
loan is active and both the domain and permit are the scope's exact objects.
Provider-specific parallel mode selection remains per operation. Ordinary entry
outside a scope uses the original executor/admission path without an additional
heap allocation.

Explicit nested scopes and unsupported domains return errors. Calls through the
existing infallible `BackendSessionHost` entry retain their established panic
contract for invalid nested/cross-witness entry; fallible operation admission can
report a typed runtime-state error. No guard is disabled, no raw reference is
published through TLS, and no execution resource outlives the callback.

## Verification and measurement

Tests must prove one executor installation across repeated operations/sessions,
retain nested-entry rejection and provider exclusion, and verify recovery after
errors and unwinding. High-level eager/trace tests must cover primal and AD
values, including 2x2 batch16 matmul. One-thread and four-thread configurations
are explicit; numerical agreement is independent of any claimed timing benefit.

Downstream benchmarks enter the scope, prepare inputs/plans, and warm up outside
the clock. Output allocation belongs to allocation-returning operations; retained
outputs and per-sample gradient reset are handled outside timing. After the Rust
PR merges, measure the affected CPU cases at its merged SHA on idle, recorded
CPU cores in the MKL devcontainer and publish complete 1T/4T results. This API
makes amortized entry possible; it does not promise a four-thread speedup.
