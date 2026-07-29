# Worklog: #1471 CUDA Event Domain

## Scope

This PR installs a native CUDA event-domain driver in the CUDA engine
registration. It is the CUDA adapter required by the backend-neutral
event-domain contract; production scheduler activation remains a separate PR
under #1471.

## Design

Each run captures the current CubeCL `StreamId` once. Every enqueue restores
that identity around dependency admission, kernel submission, and completion
recording, so moving a run between scheduler workers cannot silently split it
across thread-local CubeCL streams.

Same-device CUDA dependencies become native `cuStreamWaitEvent` operations.
Foreign event tokens use the contract's repeatable host-wait fallback. A
timing-disabled CUDA event recorded after each successful launch is both the
returned completion token and the run's drain boundary.

CUDA driver failures retain their backend source through
`tenferro_tensor::Error::BackendSource` and the runtime `TensorRuntime`
boundary. If a launch partially submits work and then fails, or completion
recording fails, the driver synchronizes the captured stream before returning
the error. An armed submission guard performs the same retirement barrier
during panic unwinding. Run drain and drop also synchronize the captured stream
directly rather than relying only on the last successfully recorded event.
This prevents buffer ownership from unwinding while untracked CUDA work is
still in flight.

The raw event wrapper has explicit `Send` and `Sync` safety invariants. It
retains the CUDA runtime and primary context, selects that context before every
driver call, and destroys the event only under unique ownership.

## TDD Record

RED:

- CUDA registration had no native event-domain driver.
- A CUDA run had no completion token for native dependency ordering.

GREEN:

- registration attaches `CudaEventDomainDriver`;
- the A100 test launches two dependent kernels, moves the run across a worker
  thread and back, waits one token twice, drains the run, and checks `[3, 6]`;
- a failing foreign dependency suppresses the launch closure;
- a post-launch panic retires the queued work during unwind and preserves the
  externally retained output;
- launch or event-record failure synchronizes the captured stream before error
  return.

## Explicit Non-Goals

- WebGPU queue completion.
- Production scheduled-node activation.
- CUDA graph capture, event pooling, multiple streams per event domain, or
  real two-card CI.

## Verification

- `cargo test -p tenferro-gpu --features cuda cubecl::tests::runtime_adapter
  --no-run`: passed.
- Local NVIDIA A100 80GB PCIe:
  `cuda_event_domain_tokens_are_repeatable_and_order_native_dependencies`
  passed.
- `cargo test -p tenferro-gpu --features cuda --lib`: 62 passed, 113
  hardware-gated tests ignored.
- `scripts/check-pr-fast.sh --coverage-reviewed` with the CUDA library and A100
  event-domain tests: passed.
- `scripts/repository-rules-review.py`: passed with no findings.
- Independent CUDA lifetime and failure-path review: no remaining merge
  blockers.
