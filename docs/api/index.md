# API Reference

## Rustdoc

`tenferro` is the main user-facing crate. See the
[Rustdoc API documentation](../rustdoc/tenferro/index.html).

For contributors, internal crate APIs are also available in the
[full workspace Rustdoc index](../rustdoc/index.html).

## Workspace Crates

- [tenferro](../rustdoc/tenferro/index.html): user-facing facade for eager
  tensors, traced tensors, AD, graph compilation/execution, extension
  registration, and backend selection
- [tenferro-tensor](../rustdoc/tenferro_tensor/index.html): dense runtime
  tensors, backend traits, CPU backend, and internal CUDA backend integration
- [tenferro-runtime](../rustdoc/tenferro_runtime/index.html): extension runtime
  registration and extension cache storage
- [tenferro-einsum](../rustdoc/tenferro_einsum/index.html): subscripts,
  contraction planning, traced/eager einsum APIs, extension runtime, and AD rule
- [tenferro-linalg](../rustdoc/tenferro_linalg/index.html): traced/eager linear
  algebra APIs, extension runtime, and AD rules where implemented
- [tenferro-fft](../rustdoc/tenferro_fft/index.html): FFT extension runtime and
  public FFT APIs
- [tenferro-ops](../rustdoc/tenferro_ops/index.html): graph op vocabulary and
  AD rule implementations
- [tenferro-device](../rustdoc/tenferro_device/index.html): shared device and
  error infrastructure
- [tenferro-gpubackend](../rustdoc/tenferro_gpubackend/index.html): internal CubeCL
  kernel crate used by the CUDA backend
- [tenferro-extension-macros](../rustdoc/tenferro_extension_macros/index.html):
  procedural macros for extension-op registration

## Architecture Summary

```text
tenferro-device
    |
tenferro-tensor   -- dense runtime tensors, TensorBackend, CpuBackend
    |
tenferro-runtime  -- extension runtime registration + extension caches
    |
tenferro          -- TracedTensor, GraphCompiler, GraphExecutor, public APIs

tenferro-ops      -- StdTensorOp, ExtensionOp boundary, PrimitiveOp rules
tenferro-einsum   -- standard einsum extension
tenferro-linalg   -- standard linalg extension
tenferro-fft      -- standard FFT extension
tenferro-gpubackend              -- internal CUDA kernel definitions
tenferro-extension-macros    -- extension-op registration macros
```
