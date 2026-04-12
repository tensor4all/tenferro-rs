# API Reference

## Rustdoc

`tenferro` is the main user-facing crate. See the
[Rustdoc API documentation](../../api/tenferro/index.html).

For contributors, internal crate APIs are also available in the
[full workspace API index](../../api/index.html).

## Workspace Crates

- [tenferro](../../api/tenferro/index.html): traced frontend, execution engine, public
  einsum API, public linalg API, and first-order AD entry points
- [tenferro-tensor](../../api/tenferro_tensor/index.html): concrete dense `Tensor` /
  `TypedTensor` values, backend traits, CPU backend, and GPU backend stubs
- [tenferro-einsum](../../api/tenferro_einsum/index.html): subscripts, contraction trees,
  contraction optimization, and fragment-building helpers
- [tenferro-ops](../../api/tenferro_ops/index.html): graph op vocabulary and AD rule
  implementations for `StdTensorOp`
- [tenferro-algebra](../../api/tenferro_algebra/index.html): semiring and algebra traits
- [tenferro-device](../../api/tenferro_device/index.html): shared device and error types

## Architecture Summary

```text
tenferro-device
    |
tenferro-algebra
    |
tenferro-tensor   -- dense runtime tensors, TensorBackend, CpuBackend
    |
tenferro-ops      -- StdTensorOp / SemiringOp, PrimitiveOp rules
    |
tenferro-einsum   -- subscripts + contraction planning + fragment builder
    |
tenferro          -- Engine, TracedTensor, lowering, execution, public APIs
```
