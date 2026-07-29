# Worklog: #1471 Event-Domain Scheduler

## Scope

Implement the backend-neutral event-domain scheduler after the production
transfer and submission-admission PRs. This PR covers runtime scheduling and
deterministic fake domains only. CUDA and WebGPU adapters remain separate.

## Design

Each frozen engine owns an `EventDomainDriver`. A driver creates a per-run
`EventDomainRun` that accepts dependency tokens and a one-shot launch closure,
enqueues the launch in its native domain, and returns an opaque completion
token. The immediate driver invokes the closure synchronously and returns a
ready token, preserving the current CPU behavior.

The scheduler stores completion tokens by the schedule's event point. Before
enqueuing a node it resolves all declared dependencies and passes their tokens
to the destination domain run. Independent nodes therefore enqueue without a
host wait. The domain adapter, rather than the scheduler, decides whether a
dependency becomes a native stream wait, queue dependency, or immediate host
completion.

On the first enqueue or launch error, the scheduler stops admitting new nodes.
It drains every started domain in ascending event-domain ID order and returns
the first error. Drain errors after an earlier failure are retained as
secondary diagnostics rather than replacing the original cause. No scheduler
mutex is held while calling an engine, transfer provider, or domain driver.

Logical last-use remains unchanged, but slots removed at last use move into a
per-run retired-value owner. They are dropped only after every started domain
has drained. This separates logical reuse from physical buffer lifetime and
prevents asynchronous work from observing freed storage.

## TDD Order

1. Schedule operations carry deterministic, deduplicated producer
   dependencies and reject unknown or forward event references.
2. Immediate domains preserve existing execution behavior.
3. A delayed fake domain proves transfer completion precedes its consumer
   while independent nodes enqueue before the final drain.
4. A failing fake domain proves first-error fail-fast and draining of
   previously started work.
5. Retired-value probes prove buffers outlive the completion that consumes
   them.
6. Opposing engine order proves canonical drain order and callback execution
   without runtime lock inversion.

## Explicit Non-Goals

- CUDA events, streams, peer transfer, or host-staging policy.
- WebGPU submission-index or queue-future integration.
- Collectives, distributed tensors, real two-card CI, event-slot recycling, or
  cancellation.

## Verification

Pending implementation. Required before PR:

- focused schedule tests with recorded RED and GREEN results;
- runtime event-domain integration tests;
- complete `tenferro-runtime` tests and doctests;
- CUDA/WebGPU feature checks to preserve adapter compilation;
- repository fast check and repository-rules review.
