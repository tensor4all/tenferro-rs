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
- [tenferro-tensor](./tenferro_tensor/index.html): dense runtime tensors,
  typed views, backend traits, and backend-independent contracts
- [tenferro-cpu](./tenferro_cpu/index.html): CPU backend, CPU execution
  sessions, CPU kernels, buffer pools, and CPU provider selection
- [tenferro-gpu](./tenferro_gpu/index.html): CubeCL/CUDA backend and GPU
  transfer helpers
- [tenferro-einsum](./tenferro_einsum/index.html): subscripts,
  contraction planning, traced/eager einsum APIs, extension runtime, and AD rule
- [tenferro-linalg](./tenferro_linalg/index.html): linear algebra traced APIs,
  eager helpers, extension runtime, and optional linalg AD rules
- [tenferro-fft](./tenferro_fft/index.html): FFT extension runtime and
  public FFT APIs
