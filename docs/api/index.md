# API Reference

## Rustdoc

`tenferro` is the main user-facing crate. See the
[Rustdoc API documentation](../rustdoc/tenferro/index.html).

For contributors, internal crate APIs are also available in the
[full workspace Rustdoc index](../rustdoc/index.html).

## Workspace Crates

- [tenferro](../rustdoc/tenferro/index.html): user-facing facade for eager
  tensors, traced tensors, einsum, linalg, AD, and backend selection
- [tenferro-tensor](../rustdoc/tenferro_tensor/index.html): dense runtime
  tensors, backend traits, CPU backend, and internal CUDA backend integration
- [tenferro-einsum](../rustdoc/tenferro_einsum/index.html): subscripts,
  contraction planning, and lowering helpers
- [tenferro-ops](../rustdoc/tenferro_ops/index.html): graph op vocabulary and
  AD rule implementations
- [tenferro-device](../rustdoc/tenferro_device/index.html): shared device and
  error infrastructure
- [tenferro-cubecl](../rustdoc/tenferro_cubecl/index.html): internal CubeCL
  kernel crate used by the CUDA backend
- [tenferro-extension-macros](../rustdoc/tenferro_extension_macros/index.html):
  procedural macros for extension-op registration

## Architecture Summary

```text
tenferro-device
    |
tenferro-tensor   -- dense runtime tensors, TensorBackend, CpuBackend
    |
tenferro-ops      -- StdTensorOp, ExtensionOp boundary, PrimitiveOp rules
    |
tenferro-einsum   -- subscripts + contraction planning + fragment builder
    |
tenferro          -- TracedTensor, GraphCompiler, GraphExecutor, public APIs

tenferro-cubecl              -- internal CUDA kernel definitions
tenferro-extension-macros    -- extension-op registration macros
```
