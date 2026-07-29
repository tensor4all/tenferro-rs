# Worklog: #1471 WebGPU Event Domain

## Scope

This PR installs a native WebGPU event-domain driver in the WebGPU engine
registration. It depends on CubeCL #15 for exact native WGPU submission
completions and cubek #11 for a single aligned CubeCL dependency graph.
Production scheduler activation remains a separate PR under #1471.

## Design

Each run captures one CubeCL `StreamId` and restores it for every enqueue,
completion submission, drain, and drop. Successful launches are flushed
through `WgpuServer::submit_stream_completion`, which returns the exact
`wgpu::SubmissionIndex` represented by the event token.

Tokens from clones of the same driver rely on WGPU's ordered queue submissions
and do not block the host. Foreign tokens use the event-domain contract's
repeatable host-wait fallback before the launch closure runs.

The launch closure runs exactly once. An armed cleanup guard submits and waits
for the captured stream when the closure returns an error or unwinds after
submitting work. Run drain and drop use the same exact submission boundary
rather than a process-wide or device-wide synchronization.

The native-only driver is not compiled for wasm, where CubeCL does not expose
native submission handles.

## TDD Record

RED:

- WebGPU engine registration had no native event-domain driver.
- No WebGPU event token could represent the exact queue submission for a run.

GREEN:

- registration attaches `WebGpuEventDomainDriver`;
- the A100 Vulkan test moves a run across a worker thread, launches two
  dependent transposes, waits one token twice, and drains the run;
- a failing foreign dependency suppresses the launch closure;
- a post-launch panic retires queued work before the retained output is read.

## Explicit Non-Goals

- Production scheduled-node activation.
- Browser/wasm queue completion handles.
- Multiple physical GPU CI, collectives, or distributed tensors.
- WebGPU elementwise coverage.

## Verification

- `cargo test -p tenferro-gpu --features webgpu
  webgpu_registration_installs_native_event_domain_driver`: passed.
- Local NVIDIA A100 80GB PCIe through Vulkan:
  `webgpu_event_domain_tokens_are_repeatable_and_order_native_dependencies`
  passed.
- `python3 scripts/ci/run_profile.py ci-config`: passed after aligning the
  checked CubeCL revision with the workspace pin.
