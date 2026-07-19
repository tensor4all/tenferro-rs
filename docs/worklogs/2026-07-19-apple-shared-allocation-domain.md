# Apple shared allocation domain substrate

## Summary

This change establishes the backend-neutral storage contract used by the Apple
CPU/Metal integration. It does not add FFT or linalg execution. A fresh
`AppleContext` owns one host-visible Metal CubeCL client, one allocation domain,
and paired CPU/WebGPU backend handles.

## Context reviewed

- `AGENTS.md`, `REPOSITORY_RULES.md`, and shared tensor4all Rust, performance,
  numerical, documentation, and test rules
- `docs/design/gpu-backend-design.md`
- CubeCL host-visible primary allocation commit
  `11b52669f13e27bbe188f988fd696df6d989a562`
- CubeK compatibility commit `fb5dbc8e994bb3023bfefe38abe0140f29cbb15e`

## Decisions

- Host reads use an RAII guard that dereferences to a typed immutable slice.
- Host writes are deliberately write-only: callers can inspect the logical
  length and replace the full contents, but cannot obtain `&mut [T]`. This
  matches WGPU's `WriteOnly<[u8]>` contract.
- `CpuBackend` retains an object-safe shared allocation owner, not only a domain
  ID. CPU extension crates can therefore allocate matching-domain outputs
  without depending on `tenferro-gpu`.
- Apple `WebGpuBuffer` values retain CubeCL's resolved managed resource lease.
  Domain and physical allocation IDs remain available through backend-neutral
  tensor metadata.
- Explicit unified creation increments upload bytes once. Guarded mappings and
  GPU launches do not increment transfer counters; explicit download does.
- Ordinary device-local WebGPU behavior remains unchanged.

## Rejected alternatives

- Exposing `&mut [T]` from the portable write guard was rejected because WGPU
  intentionally exposes write-only mappings.
- Storing only `AllocationDomainId` in `CpuBackend` was rejected because it
  cannot create same-domain FFT/linalg outputs.
- Reusing CubeCL's default WGPU client was rejected because independently
  created Apple contexts require distinct runtime clients and domains.

## Verification

- Full `tenferro-tensor` unit tests and doctests
- Full `tenferro-cpu` library tests
- Full `tenferro-gpu` WebGPU integration suite on local Metal, including four
  Apple context tests
- `tenferro-gpu` WebGPU doctests
- clippy with warnings denied for tensor, CPU, and WebGPU GPU targets
- public `Result` error documentation audit

## Remaining scope

RustFFT guarded execution, CubeK FFT launches, and the initial mapped CPU linalg
operation are separate commits built on this substrate. Publishing and release
work remain out of scope.
