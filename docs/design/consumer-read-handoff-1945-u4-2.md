# Consumer read handoff (U4-2b): out of scope under the current domain model (#1945 U4 item 1)

## Decision

**U4-2b is not implemented, and it is out of scope under the current runtime domain model.** The part of
#1945 item 1 that asks for "authorized read access plus a domain-scoped dependency token obtained from the
runtime registry" needs two *distinct* event domains to exist in one running configuration, and the ordinary
CUDA path cannot produce them: the native adapter assigns one fixed provider target per device and the runtime
rejects a second registration with the same target. Cross-runtime and cross-device reads are likewise
unsupported. The user-visible part of item 1 - reading a published value without a host round-trip - is
delivered within one runtime by U4-2a (`docs/design/consumer-copy-1945-u4-2a.md`, implemented and measured).

## Why the cross-domain pair is unreachable

An `EventDomainId` is a runtime id, an epoch and a **registration identity**
(`crates/tenferro-runtime/src/runtime/schedule.rs:38-55`), minted at freeze
(`crates/tenferro-runtime/src/runtime/snapshot.rs:2476-2491`). Generic assembly can share a backend between
engines, but the CUDA adapter assigns the fixed provider target `tenferro.cuda / device:<ordinal>`
(`crates/tenferro-gpu/src/cubecl/runtime_adapter.rs:138-145`), and registering a second engine with that same
provider/device identity is rejected by the duplicate-target check
(`crates/tenferro-runtime/src/runtime/snapshot.rs:2112-2123`, `:2282-2296`, `:2306-2317`; regression
`crates/tenferro-runtime/src/runtime/tests/snapshot.rs:157-185`). Two independently constructed CUDA runtimes
on the same device are rejected by the same rule. A cross-epoch reading is conceivable in principle, but it
needs explicit old-frontier admission semantics and is not an existing scheduled path.

## What was tried, and why each attempt failed

Four designs were written and each was rejected by an independent pre-implementation review, every rejection
verified against the source. Recorded here once, so the next attempt does not repeat them:

- the publication's frontier is a native `CUevent`, not an `EventToken`
  (`crates/tenferro-gpu/src/cubecl/transfer.rs`);
- a driver-level admission hook cannot identify the destination run, and a driver-run route bypasses the
  runtime's provenance check, which lives in `ScheduledEventDomains::classify_dependencies`
  (`crates/tenferro-runtime/src/runtime/execution.rs`);
- a held session carries no frozen witness (`crates/tenferro-gpu/src/cubecl/exec_session.rs`);
- a borrowed read view cannot protect an asynchronous read (drop order, `download_into`, `mem::forget`,
  `Unproven`);
- a retirement record that holds a run's returned completion owns a strong `CudaRuntime`
  (`crates/tenferro-gpu/src/cubecl/event_domain.rs`), so queueing it in that runtime is a cycle;
- a strong pin on a *different* runtime avoids only a direct self-cycle: two reciprocal copies can retain each
  other and stall teardown;
- and, decisively for this decision, an external completion cannot simply be "registered": it must enter
  planning and input ownership (`EventDependency` names a domain, slot and generation; schedule validation
  accepts only earlier node completions; `ExecutionInputs` has no completion binding), which would require a new
  public submission/input-binding contract
  (`crates/tenferro-runtime/src/runtime/{schedule.rs,execution.rs,transfer.rs}`).

## What would have to change to revisit it

Either (a) authorize cross-epoch publications with explicit old-frontier admission semantics, or (b) change the
native registration/domain model so one configuration can hold several CUDA domains on one device - both are
maintainer-level decisions about the domain model, not implementation choices inside a consumer slice. If either
is taken, the ownership invariant to start from is: one runtime owns both ends, the in-flight record refers to
it weakly and holds the source value and destination tensor, and the completion owner is weak on **both** sides.

## Status

Not implemented and not advertised. The capability matrix records U4-2a as implemented and measured, and U4-2b
as out of scope under the current domain model.
