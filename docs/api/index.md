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

## Internal Implementation Crates

These crates are documented for contributors and crate-boundary review. They
are not the recommended application-facing API surface.

- [tenferro-tensor-core](./tenferro_tensor_core/index.html): host-only tensor
  data model, dtype tags, scalar traits, rank metadata, and metadata-only views
- [tenferro-core-ops](./tenferro_core_ops/index.html): internal primitive
  operation catalog metadata
- [tenferro-internal-cpu-kernels](./tenferro_internal_cpu_kernels/index.html):
  internal CPU elementwise kernels and typed buffer-pool implementation
- [tenferro-internal-ops](./tenferro_ops/index.html): graph operation
  vocabulary and AD rule implementations
- [tenferro-internal-extension-macros](./tenferro_extension_macros/index.html):
  internal extension-op registration macros
