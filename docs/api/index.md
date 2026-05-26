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
- [tenferro-linalg](./tenferro_linalg/index.html): linear algebra traced APIs
  and extension runtime
- [tenferro-linalg-ad](./tenferro_linalg_ad/index.html): linalg eager APIs and
  linalg AD registration
- [tenferro-fft](./tenferro_fft/index.html): FFT extension runtime and
  public FFT APIs
- [tenferro-internal-ops](./tenferro_ops/index.html): graph op vocabulary and
  AD rule implementations
- [tenferro-internal-device](./tenferro_device/index.html): shared device and
  error infrastructure
- [tenferro-internal-extension-macros](./tenferro_extension_macros/index.html):
  procedural macros for extension-op registration

## Architecture Summary

```text
tenferro-internal-device
    |
tenferro-tensor            -- dense runtime tensors, TensorBackend, CpuBackend
    |\
    | \-- tenferro-gpu    -- CubeCL/CUDA backend and GPU transfers
    |
tenferro-internal-ops      -- StdTensorOp, ExtensionOp boundary, PrimitiveOp rules
    |
tenferro-runtime           -- TracedTensor, GraphCompiler, GraphExecutor
    |\
    | \-- tenferro-ad     -- EagerRuntime, EagerTensor, traced AD traits

tenferro-einsum   -- standard einsum extension
tenferro-linalg   -- standard linalg extension
tenferro-linalg-ad -- linalg AD extension
tenferro-fft      -- standard FFT extension
tenferro-internal-extension-macros    -- extension-op registration macros
```
