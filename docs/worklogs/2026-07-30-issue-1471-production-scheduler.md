# Worklog: #1471 Production Event-Domain Scheduler

## Scope

This PR activates the previously merged event-domain contract in the
production `Runtime::run_compiled*` scheduler. Scheduled operations, transfers,
and explicit barriers now execute through their registered
`EventDomainDriver`. Collectives remain explicitly unsupported.

Native CUDA and WebGPU event-domain adapters landed before this activation.
This PR does not add cancellation, admission control, collectives, distributed
tensors, real two-card CI, or a non-blocking public submission API.

## Execution Contract

The scheduler preflights and begins one run for each event domain used by a
scheduled execution before input ingress or the first launch. Runs are retained
in deterministic first-use order. Every scheduled dependency is resolved to
its prior opaque completion token and passed to the completion domain before
the node's launch closure runs.

The first enqueue or launch error stops later admission. All started domains
are drained in first-use order on success and failure without holding runtime,
driver, backend, or provider locks. Tensor values remain retained until drain
finishes. Output collection occurs only after successful drain. The public
driver contract requires `drain` to retire resource access before returning on
both success and error and requires best-effort retirement from `Drop` during
unwinding. If execution and drain both fail, the primary execution error
remains the typed source and the cleanup error is retained in the diagnostic.

The event-domain run store is declared after the execution value stores. Rust's
reverse drop order therefore invokes native run `Drop` cleanup before releasing
tensor storage during panic unwinding.

CUDA retirement falls back from unavailable stream lookup to a context-wide
barrier and attempts the barrier even when context selection reports an error.
WebGPU retirement falls back from exact completion-submission failure to a
whole-client synchronization. Both adapters preserve primary and cleanup
diagnostics while satisfying the retirement boundary on `drain` and `Drop`.

An engine used by a schedule must register a driver. Missing drivers fail with
a typed runtime execution error during whole-schedule preflight, before input
ingress or any node launch, rather than silently selecting immediate execution.

## TDD Record

RED:

- the two-logical-device transfer integration test observed no event-domain
  activity because production execution bypassed `EventDomainRun`;
- transfer failure cleanup could not prove that every started domain drained.

GREEN:

- operation and transfer nodes produce and consume recorded completion tokens;
- the transfer path crosses source and destination domains end to end;
- a failed reverse transfer is admitted once, later execution is suppressed,
  and both started domains drain;
- a later scheduled domain without a driver fails whole-schedule preflight
  before the first domain begins;
- drain failure suppresses outputs, combined execution and drain failures
  preserve both diagnostics, and a failure in one domain does not skip later
  drains;
- injected launch panic drops the event-domain run before tracked intermediate
  tensor storage;
- all integration fixtures opt into the immediate driver explicitly.

## Correctness And Lifetime Audit

- schedule validation guarantees every dependency names an earlier completion;
- completion tokens remain retained for the whole scheduled execution;
- outputs are collected only after drain succeeds;
- success, execution failure, drain failure, and combined failure all release
  value stores only after explicit drain;
- panic cleanup is delegated to the already tested native run `Drop`
  implementations, with scheduler declaration order preserving storage
  lifetime;
- transfer provider execution remains on the existing typed, validated path.

## Verification

- `cargo test -p tenferro-runtime --lib`: 349 passed.
- `cargo test -p tenferro-runtime --test integration`: 109 passed.
- Focused two-engine transfer success and failure tests: passed.
- `cargo check -p tenferro-gpu --no-default-features --features
  cpu-faer,cuda,webgpu`: passed.
- CUDA event-domain ignored hardware test: passed on local NVIDIA A100 80 GB
  PCIe.
- WebGPU event-domain ignored hardware test: passed on the local Vulkan
  adapter.
- `scripts/check-pr-fast.sh --coverage-reviewed` with the focused runtime and
  CUDA/WebGPU feature commands: passed.
- `scripts/repository-rules-review.py`: passed with no findings.
- Independent correctness review: drain retirement, panic lifetime,
  missing-driver preflight, and native fallback findings resolved; final
  review reported no blocker.
