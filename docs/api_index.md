# tenferro-rs

General-purpose tensor computation library in Rust.
Provides dense tensor types with CPU/GPU support, a cuTENSOR/hipTensor-compatible
operation protocol, high-level einsum with N-ary contraction tree optimization,
and automatic differentiation.

**Current phase**: API skeleton (POC). Public signatures and documentation are
in place; most function bodies use `todo!()`. The purpose of this phase is to
validate the API design before writing implementations.

See the [ecosystem overview](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/tenferro_unified_tensor_backend.md)
in tensor4all-meta for high-level architecture and future phase plans.

## Workspace Architecture

```
Layer 4: tenferro-einsum     High-level einsum on Tensor<T>, N-ary contraction
                             tree, algebra dispatch, einsum AD rules
Layer 3: tenferro-tensor     Tensor<T> = DataBuffer + shape + strides,
                             zero-copy view ops, impl Differentiable
Layer 2: tenferro-prims      "Tensor BLAS": TensorPrims<A> trait
                             (algebra-parameterized), plan-based execution
Shared:  chainrules-core   Core AD traits: Differentiable, ReverseRule<V>,
                             ForwardRule<V> (no tensor deps)
         chainrules          AD engine: Tape<V>, TrackedTensor<V>,
                             DualTensor<V> (← chainrules-core)
         tenferro-algebra    HasAlgebra trait, Semiring trait, Standard type
         tenferro-device     Device enum, Error/Result types

Foundation: strided-rs       Independent workspace
                             (strided-traits -> strided-view -> strided-kernel)
```

### tenferro-device

Shared infrastructure: `LogicalMemorySpace` (MainMemory, GpuMemory),
`ComputeDevice` (Cpu, Cuda, Hip), workspace-wide `Error`/`Result` types.

### tenferro-algebra

Minimal algebra foundation. `HasAlgebra` trait maps scalar types to their
algebra (e.g., `f64 -> Standard`), enabling automatic backend inference.
`Semiring` trait for algebra-generic operations.

### tenferro-prims

Low-level "Tensor BLAS" protocol. `TensorPrims<A>` trait parameterized by
algebra `A` with a cuTENSOR-compatible plan-based execution model
(`PrimDescriptor -> plan -> execute`).

Core ops (universal set): `batched_gemm`, `reduce`, `trace`, `permute`,
`anti_trace`, `anti_diag`. Extended ops (dynamically queried):
`contract`, `elementwise_mul`.

### tenferro-tensor

`Tensor<T>` type with `DataBuffer` (CPU/GPU), shape/strides metadata,
and zero-copy view operations (`permute`, `broadcast`, `diagonal`, `reshape`).
`TensorView<'a, T>` for borrowed views.

### chainrules-core

Core AD trait definitions (like Julia's ChainRulesCore.jl), independent of any
tensor type. `Differentiable` trait defines the tangent space; concrete types
(e.g., `Tensor<T>`) implement it in their own crates. Rule extension traits
(`ReverseRule<V>`, `ForwardRule<V>`) for per-operation AD rules.

### chainrules

AD engine (like Zygote.jl in Julia's ecosystem). Provides `Tape<V>`
(explicit tape, TensorFlow GradientTape style), `TrackedTensor<V>`
(reverse-mode wrapper), `DualTensor<V>` (forward-mode wrapper).
Gradient computation via `tape.pullback()`, HVP via `tape.hvp()`.
Depends only on `chainrules-core`. Re-exports all of `chainrules-core`
so downstream crates can depend on just `chainrules` for both traits and engine.

Operation-specific AD rules live with their operations, not here.

### tenferro-einsum

High-level einsum API with three levels: string notation (`einsum`),
pre-built subscripts (`einsum_with_subscripts`), and pre-optimized tree
(`einsum_with_plan`). Each has allocating, accumulating (`_into`), and
consuming (`_owned`) variants.

Einsum AD rules: `tracked_einsum`, `dual_einsum`, `einsum_rrule`,
`einsum_frule`, `einsum_hvp`.
