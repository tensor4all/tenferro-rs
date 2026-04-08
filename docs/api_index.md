# tenferro-rs API Index

This site documents the current workspace: a 6-crate core for dense,
graph-based tensor computation.

## Current Workspace

- [tenferro](tenferro/index.html): traced frontend, execution engine, public
  einsum API, public linalg API, and first-order AD entry points
- [tenferro-tensor](tenferro_tensor/index.html): concrete dense `Tensor` /
  `TypedTensor` values, backend traits, CPU backend, and GPU backend stubs
- [tenferro-einsum](tenferro_einsum/index.html): subscripts, contraction trees,
  contraction optimization, and fragment-building helpers
- [tenferro-ops](tenferro_ops/index.html): graph op vocabulary and AD rule
  implementations for `StdTensorOp`
- [tenferro-algebra](tenferro_algebra/index.html): semiring and algebra traits
- [tenferro-device](tenferro_device/index.html): shared device and error types

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

The repository no longer ships the older facade, internal, FFI, or extension
crate families in the active tree. Historical references to those crates may
still appear in archived plan documents under `docs/plans/`.
