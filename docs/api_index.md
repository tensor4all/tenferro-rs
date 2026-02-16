# tenferro-rs

General-purpose tensor computation library in Rust.
Provides dense tensor types with CPU/GPU support, a cuTENSOR/hipTensor-compatible
operation protocol, high-level einsum with N-ary contraction tree optimization,
and automatic differentiation.

**Current phase**: API skeleton (POC). Public signatures and documentation are
in place; most function bodies use `todo!()`. The purpose of this phase is to
validate the API design before writing implementations.

See the [design document](https://github.com/tensor4all/tenferro-rs/blob/main/docs/design/tenferro_design.md)
for architecture, API design, and future phase plans.

## Workspace Architecture

```
Layer 5: tenferro-capi       C-API (FFI) for Julia/Python: exposes einsum + SVD
                             with stateless rrule/frule (f64 only),
                             DLPack v1.0 zero-copy tensor exchange
Layer 4: tenferro-einsum     High-level einsum on Tensor<T>, N-ary contraction
                             tree, algebra dispatch, einsum AD rules
         tenferro-linalg     Tensor-level SVD/QR/LU/eigen, linalg AD rules
Layer 3: tenferro-tensor     Tensor<T> = DataBuffer + shape + strides,
                             zero-copy view ops, impl Differentiable
Layer 2: tenferro-prims      "Tensor BLAS": TensorPrims<A> trait
                             (algebra-parameterized), plan-based execution
Shared:  tenferro-algebra    HasAlgebra trait, Semiring trait, Standard type,
                             Scalar trait, Conjugate trait
         tenferro-device     Device enum, Error/Result types

Extern:  chainrules-core     Core AD traits: Differentiable, ReverseRule<V>,
                             ForwardRule<V> (no tensor deps)
         chainrules          AD engine: Tape<V>, TrackedTensor<V>,
                             DualTensor<V> (← chainrules-core)

Foundation: strided-rs       Independent workspace (used only by tenferro-prims)
                             (strided-traits -> strided-view -> strided-kernel)
```

### Dependency Graph

Click a node to open the crate documentation.

<div class="dep-graph"><object data="dep_graph.svg" type="image/svg+xml">Dependency graph</object></div>

## Crates

### [tenferro-capi](tenferro_capi/index.html) <small>(Layer 5)</small>

C-API (FFI) for Julia, Python (JAX, PyTorch), and other languages.
Exposes tensor lifecycle, einsum, and SVD (including AD rules) via
opaque pointers and status codes. f64 only in this POC phase.

Design principles: opaque `TfeTensorF64` handles, `tfe_status_t`
error codes, `catch_unwind` for panic safety, DLPack v1.0 for zero-copy
tensor exchange across language boundaries (NumPy, PyTorch, JAX, DLPack.jl).

AD approach: stateless `rrule`/`frule` only — host languages manage
their own AD tapes (ChainRules.jl, PyTorch autograd, JAX custom_vjp).

### [tenferro-einsum](tenferro_einsum/index.html) <small>(Layer 4)</small>

High-level einsum API with three levels: string notation (`einsum`),
pre-built subscripts (`einsum_with_subscripts`), and pre-optimized tree
(`einsum_with_plan`). Each has allocating, accumulating (`_into`), and
consuming (`_owned`) variants.

Einsum AD rules: `tracked_einsum`, `dual_einsum`, `einsum_rrule`,
`einsum_frule`, `einsum_hvp`.

### [tenferro-linalg](tenferro_linalg/index.html) <small>(Layer 4)</small>

Tensor-level linear algebra decompositions: SVD, QR, LU, eigendecomposition.
Users specify left/right dimension indices; the crate handles
matricize -> decompose -> unmatricize internally via external backends
(faer for CPU, cuSOLVER for GPU).

Primary functions: `svd`, `qr`, `lu`, `eigen`.
Result types: `SvdResult`, `QrResult`, `LuResult`, `EigenResult`.
SVD truncation: `SvdOptions` (max_rank, cutoff).

Linalg AD rules: `svd_rrule`, `svd_frule`,
`qr_rrule`, `qr_frule`, `lu_rrule`, `lu_frule`,
`eigen_rrule`, `eigen_frule`.

### [tenferro-tensor](tenferro_tensor/index.html) <small>(Layer 3)</small>

`Tensor<T>` type with `DataBuffer` (Rust-owned or externally-owned via DLPack),
shape/strides metadata, and zero-copy view operations (`permute`, `broadcast`,
`diagonal`, `reshape`, `select`, `narrow`). `TensorView<'a, T>` for borrowed
views. Factory functions: `zeros`, `ones`, `eye`. Triangular extraction:
`tril`, `triu`.

### [tenferro-prims](tenferro_prims/index.html) <small>(Layer 2)</small>

Low-level "Tensor BLAS" protocol. `TensorPrims<A>` trait parameterized by
algebra `A` with a cuTENSOR-compatible plan-based execution model
(`PrimDescriptor -> plan -> execute`).

Core ops (universal set): `batched_gemm`, `reduce`, `trace`, `permute`,
`anti_trace`, `anti_diag`, `elementwise_unary`. Extended ops (dynamically queried):
`contract`, `elementwise_mul`.

### [tenferro-algebra](tenferro_algebra/index.html) <small>(Shared)</small>

Minimal algebra foundation. `HasAlgebra` trait maps scalar types to their
algebra (e.g., `f64 -> Standard`), enabling automatic backend inference.
`Semiring` trait for algebra-generic operations. `Scalar` trait (blanket impl
for `Copy + Send + Sync + Add + Mul + Zero + One + PartialEq`) defines
minimum element type requirements. `Conjugate` trait for complex conjugation
(identity for real types).

### [tenferro-device](tenferro_device/index.html) <small>(Shared)</small>

Shared infrastructure: `LogicalMemorySpace` (MainMemory, GpuMemory),
`ComputeDevice` (Cpu, Cuda, Hip), workspace-wide `Error`/`Result` types.

## External Crates (extern/)

### [chainrules-core](chainrules_core/index.html) <small>(Extern)</small>

Core AD trait definitions (like Julia's ChainRulesCore.jl), independent of any
tensor type. `Differentiable` trait defines the tangent space; concrete types
(e.g., `Tensor<T>`) implement it in their own crates. Rule extension traits
(`ReverseRule<V>`, `ForwardRule<V>`) for per-operation AD rules.

### [chainrules](chainrules/index.html) <small>(Extern)</small>

AD engine (like Zygote.jl in Julia's ecosystem). Provides `Tape<V>`
(explicit tape, TensorFlow GradientTape style), `TrackedTensor<V>`
(reverse-mode wrapper), `DualTensor<V>` (forward-mode wrapper).
Gradient computation via `tape.pullback()`, HVP via `tape.hvp()`.
Depends only on `chainrules-core`. Re-exports all of `chainrules-core`
so downstream crates can depend on just `chainrules` for both traits and engine.

Operation-specific AD rules live with their operations, not here.
