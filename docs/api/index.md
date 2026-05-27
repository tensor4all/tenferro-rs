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
  backend traits, CPU backend, and core execution kernels
- [tenferro-gpu](./tenferro_gpu/index.html): CubeCL/CUDA backend and GPU
  transfer helpers
- [tenferro-einsum](./tenferro_einsum/index.html): subscripts,
  contraction planning, traced/eager einsum APIs, extension runtime, and AD rule
- [tenferro-linalg](./tenferro_linalg/index.html): linear algebra traced APIs,
  eager helpers, extension runtime, and optional linalg AD rules
- [tenferro-fft](./tenferro_fft/index.html): FFT extension runtime and
  public FFT APIs
- [tenferro-core-ops](./tenferro_core_ops/index.html): internal core primitive
  operation catalog used by graph, runtime, and backend dispatch
- [tenferro-internal-ops](./tenferro_ops/index.html): graph op vocabulary and
  AD rule implementations
- [tenferro-internal-device](./tenferro_device/index.html): shared device and
  error infrastructure
- [tenferro-internal-extension-macros](./tenferro_extension_macros/index.html):
  procedural macros for extension-op registration

## Architecture Summary

```text
tenferro-tensor          -> tenferro-internal-device
tenferro-gpu             -> tenferro-tensor
tenferro-internal-ops    -> tenferro-core-ops, tenferro-tensor
tenferro-runtime         -> tenferro-internal-ops, tenferro-tensor
tenferro-ad              -> tenferro-runtime, tenferro-internal-ops

tenferro-einsum          -> tenferro-runtime
tenferro-linalg          -> tenferro-runtime, tenferro-tensor
tenferro-linalg/cuda     -> tenferro-gpu
tenferro-fft             -> tenferro-runtime
tenferro-internal-extension-macros    -- extension-op registration macros
```
