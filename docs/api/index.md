# API Reference

## Rustdoc

`tenferro` is the main user-facing crate. See the
[Rustdoc API documentation](./tenferro/index.html).

For contributors, internal crate APIs are also available in the
[full workspace Rustdoc index](./index.html).

## Workspace Crates

- [tenferro](./tenferro/index.html): user-facing facade for eager
  tensors, traced tensors, AD, graph compilation/execution, extension
  registration, and backend selection
- [tenferro-internal-tensor](./tenferro_tensor/index.html): dense runtime
  tensors, backend traits, CPU backend, and internal CUDA backend integration
- [tenferro-internal-runtime](./tenferro_runtime/index.html): extension runtime
  registration and extension cache storage
- [tenferro-einsum](./tenferro_einsum/index.html): subscripts,
  contraction planning, traced/eager einsum APIs, extension runtime, and AD rule
- [tenferro-linalg](./tenferro_linalg/index.html): traced/eager linear
  algebra APIs, extension runtime, and AD rules where implemented
- [tenferro-fft](./tenferro_fft/index.html): FFT extension runtime and
  public FFT APIs
- [tenferro-internal-ops](./tenferro_ops/index.html): graph op vocabulary and
  AD rule implementations
- [tenferro-internal-device](./tenferro_device/index.html): shared device and
  error infrastructure
- [tenferro-internal-gpubackend](./tenferro_gpubackend/index.html): internal CubeCL
  kernel crate used by the CUDA backend
- [tenferro-internal-extension-macros](./tenferro_extension_macros/index.html):
  procedural macros for extension-op registration

## Architecture Summary

```text
tenferro-internal-device
    |
tenferro-internal-tensor   -- dense runtime tensors, TensorBackend, CpuBackend
    |
tenferro-internal-runtime  -- extension runtime registration + extension caches
    |
tenferro          -- TracedTensor, GraphCompiler, GraphExecutor, public APIs

tenferro-internal-ops      -- StdTensorOp, ExtensionOp boundary, PrimitiveOp rules
tenferro-einsum   -- standard einsum extension
tenferro-linalg   -- standard linalg extension
tenferro-fft      -- standard FFT extension
tenferro-internal-gpubackend              -- internal CUDA kernel definitions
tenferro-internal-extension-macros    -- extension-op registration macros
```
