# API Reference

## Rustdoc

The public API is split by responsibility. Start with
[`tenferro-runtime`](./tenferro_runtime/index.html) for concrete tensors,
traced graphs, compilation, execution, and extension runtime registration; add
[`tenferro-ad`](./tenferro_ad/index.html) when you need eager execution or
automatic differentiation.

For contributors, internal crate APIs are also available in the
[full workspace Rustdoc index](./index.html).

## Workspace Crates

- [tenferro-runtime](./tenferro_runtime/index.html): concrete tensor helpers,
  traced tensors, graph compilation/execution, extension runtime registration,
  and extension cache storage
- [tenferro-ad](./tenferro_ad/index.html): eager runtime, eager tensors, and
  traced AD extension traits
- [tenferro-xla](./tenferro_xla/index.html): experimental StableHLO lowering
  and runtime-loaded PJRT plugin support for static-shaped traced programs
- [tenferro-tensor](./tenferro_tensor/index.html): dense runtime tensors,
  typed views, backend traits, and backend-independent contracts
- [tenferro-cpu](./tenferro_cpu/index.html): public CPU backend, CPU execution
  sessions, execution context, provider selection, thread policy, and
  resource-pool controls
- [tenferro-gpu](./tenferro_gpu/index.html): CubeCL/CUDA backend and GPU
  transfer helpers
- [tenferro-einsum](./tenferro_einsum/index.html): subscripts,
  contraction planning, concrete/traced/eager einsum APIs, extension runtime,
  and AD rule
- [tenferro-linalg](./tenferro_linalg/index.html): linear algebra traced APIs,
  eager helpers, extension runtime, and optional linalg AD rules
- [tenferro-fft](./tenferro_fft/index.html): FFT extension runtime and
  public concrete/traced FFT APIs
- [cubecl-kernel-sample](./cubecl_kernel_sample/index.html): sample CUDA
  CubeCL kernel exercising the public `raw`/`cubecl` session seams

## Extension and Consumer Crates

These crates are not published. They show how an application adds a scalar
tenferro does not declare and how it composes that support with the canonical
one, and their tests are the executable evidence for those boundaries. The
[adding an external scalar](../guides/external-scalars.md) guide runs the path
end to end and names the test behind every claim.

- [tenferro-df64-proof](./tenferro_df64_proof/index.html): an externally defined
  scalar with its own arithmetic, QR factorisation, extension operations, and
  first-order AD rules
- [tenferro-scalar-consumer-algorithm](./tenferro_scalar_consumer_algorithm/index.html):
  the algorithm role, which states the capabilities it needs and never names a
  scalar, a provider, or a dtype
- [tenferro-scalar-consumer-application](./tenferro_scalar_consumer_application/index.html):
  the application role, which binds those capabilities to canonical support and
  to the external scalar

## Internal Implementation Crates

These crates are documented for contributors and crate-boundary review. They
are not the recommended application-facing API surface.

- [tenferro-tensor-core](./tenferro_tensor_core/index.html): host-only tensor
  data model, dtype tags, scalar traits, rank metadata, and metadata-only views
- [tenferro-core-ops](./tenferro_core_ops/index.html): internal primitive
  operation catalog metadata
- [tenferro-cpu-basic](./tenferro_cpu_basic/index.html): shared CPU buffer
  pool, full-overwrite guard, and host strided adapters
- [tenferro-internal-cpu-kernels](./tenferro_internal_cpu_kernels/index.html):
  ordinary CPU dtype-dispatch kernels and pool-aware read-into replay
- [tenferro-cpu-fused](./tenferro_cpu_fused/index.html): runtime-DAG fused
  elementwise CPU adapter
- [tenferro-internal-ops](./tenferro_ops/index.html): graph operation
  vocabulary and AD rule implementations
- [tenferro-internal-extension-macros](./tenferro_extension_macros/index.html):
  internal extension-op registration macros
