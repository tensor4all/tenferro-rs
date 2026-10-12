# Consumer read handoff (U4-2b): scope decision (#1945 U4 item 1)

## Decision

**U4-2b is scoped to the cross-domain route inside one owning runtime and one provider runtime.** The consumer
reads a published value from another event domain of the same `tenferro_runtime::Runtime`, backed by the same
`CudaRuntime`, authorized by a registry-bound admission. **Cross-runtime and cross-device reads are explicitly
unsupported** and must not be advertised: they are the source of the retention problems below, and the
user-visible value of item 1 (reading a published value without a host round-trip) is already delivered within
one runtime by the merged U4-2a copy (`docs/design/consumer-copy-1945-u4-2a.md`).

## Why this scope

Item 1's wording ("authorized read access plus a domain-scoped dependency token obtained from the runtime
registry") is about **domains**, not about runtimes: domains are minted by the owning runtime's freeze, and one
`CudaBackend` can back several engine registrations, i.e. several domains. Four pre-implementation reviews of the
earlier cross-runtime designs rejected them for the same class of reasons, which the scope above removes:

- the publication's frontier is a native `CUevent`, not an `EventToken` (`crates/tenferro-gpu/src/cubecl/transfer.rs`);
- a driver-level admission hook cannot identify the destination run, and a driver-run route bypasses the
  runtime's provenance check, which lives in `ScheduledEventDomains::classify_dependencies`
  (`crates/tenferro-runtime/src/runtime/execution.rs`);
- a held session carries no frozen witness (`crates/tenferro-gpu/src/cubecl/exec_session.rs`);
- a borrowed read view cannot protect an asynchronous read (drop order, `download_into`, `mem::forget`,
  `Unproven`);
- the run's returned completion owns a strong `CudaRuntime` (`crates/tenferro-gpu/src/cubecl/event_domain.rs`),
  so a retirement record that holds it in that runtime's own queue is a cycle;
- a strong pin on a *different* runtime avoids only a direct self-cycle: two reciprocal copies (A -> B, B -> A)
  can still retain each other and stall teardown.

## Ownership invariant

An in-flight cross-domain read is owned by the **one** runtime that owns both ends: the retirement record lives
in that runtime's queue, holds the source value (which keeps the published allocation alive) and the destination
tensor, and refers to the runtime **weakly** so the record cannot pin the runtime that stores it. Because both
ends share one runtime, no cross-runtime retention exists and the reciprocal case cannot arise. The completion
owner must satisfy the same rule on **both** sides - the publication's frontier and the completion returned by
the run - since an owning `CudaEventHandle` would reintroduce the cycle.

## What remains to implement

1. **External-completion registration on the owning runtime (the fifth contract).** A publication's frontier
   must become a *scheduled* dependency so `classify_dependencies` performs the provenance check against the
   token's `origin()`; today no such surface exists and the scheduler host-waits cross-domain transfer
   dependencies. The smallest seam is the one callers already use to move a value between domains - the
   transfer request (`TransferRequest` is currently `pub(crate)` and constructed inside the scheduler), which
   would accept a published source.
2. **A weak-runtime completion owner for the returned run completion**, so the record described above is
   cycle-free (the existing `CudaEventHandle` holds a strong runtime).

## Status

Not implemented and not advertised. The capability matrix records U4-2a as implemented and measured, and U4-2b
as scoped-but-not-implemented with cross-runtime/cross-device unsupported. This record needs its own
pre-implementation review before any code is written.
