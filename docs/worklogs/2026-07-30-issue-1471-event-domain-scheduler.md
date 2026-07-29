# Worklog: #1471 Event-Domain Contract

## Scope

This PR defines the backend-neutral event-domain contract required by the
production scheduler. It also attaches deterministic dependency and completion
metadata to scheduled nodes and retains explicitly registered drivers in
immutable runtime snapshots.

This PR does not execute scheduled nodes through event domains. Production
activation, native CUDA/WebGPU adapters, fail-fast draining, and asynchronous
buffer retirement remain follow-up PRs under #1471.

## Implemented Contract

`EventDomainDriver` creates per-execution `EventDomainRun` state. A run admits a
submission closure after its dependency tokens and returns an opaque
`EventToken`. Tokens support repeatable, concurrent host waits because one
completion may fan out to multiple foreign event domains.

The blocking `ImmediateEventDomainDriver` waits dependencies before invoking
the closure and returns an already-ready token. CPU registration opts into this
driver explicitly. `EngineRegistration::new` leaves the driver absent so CUDA
or WebGPU registration cannot silently inherit synchronous completion
semantics.

Drivers are retained by `EngineRegistration`, then by the immutable runtime
snapshot. Driver lookup remains runtime-internal; external callers cannot start
runs outside scheduler admission.

`ScheduledGraph::validate` rejects dependencies that do not name prior
completions and rejects duplicate completion identities. Operation and transfer
nodes carry deterministic, deduplicated dependency metadata.

## Activation Requirements

The production activation stack must:

- reject a scheduled engine that has no explicit event-domain driver;
- install native CUDA and WebGPU drivers before enabling their scheduled
  execution paths;
- keep same-backend dependencies as native stream or queue waits and use
  `EventToken::wait` only for foreign-domain fallback;
- stop admitting nodes after the first enqueue or launch error;
- drain every started domain in canonical event-domain order without holding a
  runtime, driver, backend, or provider lock;
- drain during panic unwinding before releasing retired values;
- retain logically dead buffers until all started domains have drained;
- exercise transfer ordering, failure cleanup, fanout waits, and lock ordering
  with two logical CPU-backed devices.

## TDD Record

RED:

- an engine registration inherited an immediate driver without opting in;
- schedule validation accepted duplicate completion identities;
- the immediate-domain contract did not test repeated waits or dependency
  failure before launch.

GREEN:

- registrations represent driver absence explicitly and CPU registration adds
  the immediate driver;
- snapshot tests retain an explicitly registered driver;
- repeated/concurrent waits and dependency-failure launch suppression are
  covered;
- duplicate completion identities are rejected.

## Explicit Non-Goals

- Production dispatch through `EventDomainRun`.
- CUDA events or streams.
- WebGPU submission-index or queue-future integration.
- Collectives, distributed tensors, real two-card CI, event-slot recycling, or
  cancellation.

## Verification

- `cargo test -p tenferro-runtime --lib`: 349 passed.
- `cargo test -p tenferro-runtime --test integration runtime_event_domains`:
  4 passed.
- `cargo test -p tenferro-runtime --doc`: 382 passed.
- `cargo check -p tenferro-runtime`: passed.
- `cargo check -p tenferro-cpu`: passed.
- `scripts/check-pr-fast.sh --coverage-reviewed` with the focused runtime
  commands: passed, including workspace and extension clippy checks.
- `scripts/repository-rules-review.py`: passed with no findings.
- Independent contract review: architecture and correctness findings resolved;
  public per-item doctests added in response to the final documentation review.
