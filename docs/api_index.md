# tenferro-rs

General-purpose tensor computation library in Rust.
Provides dense tensor types with CPU/GPU support, a cuTENSOR/hipTensor-compatible
operation protocol, high-level einsum with N-ary contraction tree optimization,
and automatic differentiation.

**Current phase**: API skeleton (POC). Public signatures and documentation are
in place; most function bodies use `todo!()`. The purpose of this phase is to
validate the API design before writing implementations.

See the [design documents](https://github.com/tensor4all/tenferro-rs/blob/main/docs/design/index.md)
for architecture, API design, and future phase plans.

## Workspace Architecture

```
Layer 5: tenferro-capi       C-API (FFI) for Julia/Python: exposes einsum + SVD
                             with stateless rrule/frule (f64 only),
                             DLPack v1.0 zero-copy tensor exchange
Layer 4: tenferro-einsum     High-level einsum on Tensor<T>, N-ary contraction
                             tree, semiring-core/fast-path dispatch, einsum AD rules
         tenferro-linalg     Public tensor linalg APIs, composite lowering, AD rules
Layer 3: tenferro-prims      Semiring/scalar/analytic execution families
         tenferro-linalg-prims Backend-facing linalg factorization/solve contracts
Layer 2: tenferro-tensor     Tensor<T> = DataBuffer + shape + strides,
                             zero-copy view ops, impl Differentiable
Shared:  tenferro-algebra    HasAlgebra trait, Semiring trait, Standard type,
                             Scalar trait, Conjugate trait
         tenferro-device     Device enum, Error/Result types

Extern:  chainrules-core     Core AD traits: Differentiable, ReverseRule<V>,
                             ForwardRule<V> (no tensor deps)
         chainrules          AD engine: Tape<V>, TrackedTensor<V>,
                             DualTensor<V> (← chainrules-core)
         chainrules-scalarops Scalar AD rules for primitive real/complex operations

Foundation: strided-rs       Independent workspace (used only by tenferro-prims)
                             (strided-traits -> strided-view -> strided-kernel)

Extension:  tenferro-tropical       Tropical semiring operations (MaxPlus, MinPlus, MaxMul)
            tenferro-tropical-capi  C-API for tropical einsum
            tenferro-burn           Burn deep learning framework bridge
            tenferro-mdarray        mdarray multidimensional array bridge
            tenferro-dyadtensor     Dynamic dyadic tensor API and AD runtime bridge
```

### Dependency Graph

Click a node to jump to its description below.

<div class="dep-graph"><object data="dep_graph.svg" type="image/svg+xml">Dependency graph</object></div>

## Crates

<a id="tenferro-capi"></a>
### [tenferro-capi](tenferro_capi/index.html) <small>(Layer 5)</small>

C-API (FFI) for Julia, Python (JAX, PyTorch), and other languages.
Exposes tensor lifecycle, einsum, and SVD (including AD rules) via
opaque pointers and status codes. f64 only in this POC phase.

Design principles: opaque `TfeTensorF64` handles, `tfe_status_t`
error codes, `catch_unwind` for panic safety, DLPack v1.0 for zero-copy
tensor exchange across language boundaries (NumPy, PyTorch, JAX, DLPack.jl).

AD approach: stateless `rrule`/`frule` only — host languages manage
their own AD tapes (ChainRules.jl, PyTorch autograd, JAX custom_vjp).

<a id="tenferro-einsum"></a>
### [tenferro-einsum](tenferro_einsum/index.html) <small>(Layer 4)</small>

High-level einsum API with three levels: string notation (`einsum`),
pre-built subscripts (`einsum_with_subscripts`), and pre-optimized tree
(`einsum_with_plan`). Each has allocating, accumulating (`_into`), and
consuming (`_owned`) variants.

Einsum AD rules: `tracked_einsum`, `dual_einsum`, `einsum_rrule`,
`einsum_frule`, `einsum_hvp`.

<a id="tenferro-linalg"></a>
### [tenferro-linalg](tenferro_linalg/index.html) <small>(Layer 4)</small>

Tensor-level linear algebra: decompositions (SVD, QR, LU, Cholesky, eigen),
solvers (solve, lstsq, solve_triangular), and utilities (inv, det, slogdet,
pinv, matrix_exp, norm). External backends: faer (CPU), cuSOLVER (GPU).

Decompositions: `svd`, `qr`, `lu`, `cholesky`, `eigen` (symmetric), `eig` (general).
Solvers: `solve`, `lstsq`, `solve_triangular`.
Utilities: `inv`, `det`, `slogdet`, `pinv`, `matrix_exp`, `norm`.

All operations have stateless AD rules (`_rrule`, `_frule`).

<a id="tenferro-linalg-prims"></a>
### [tenferro-linalg-prims](tenferro_linalg_prims/index.html) <small>(Layer 3)</small>

Backend-facing structured linalg kernel contracts used by `tenferro-linalg`.
This crate holds tensor-level solve/factorization/eigensolver traits and
structured result types. It is intentionally smaller than the public linalg API
surface and exists to keep `tenferro-prims` focused on semiring/scalar
execution.

<a id="tenferro-tensor"></a>
### [tenferro-tensor](tenferro_tensor/index.html) <small>(Layer 2)</small>

`Tensor<T>` type with `DataBuffer` (Rust-owned or externally-owned via DLPack),
shape/strides metadata, and zero-copy view operations (`permute`, `broadcast`,
`diagonal`, `reshape`, `select`, `narrow`). `TensorView<'a, T>` for borrowed
views. Factory functions: `zeros`, `ones`, `eye`. Triangular extraction:
`tril`, `triu`.

<a id="tenferro-prims"></a>
### [tenferro-prims](tenferro_prims/index.html) <small>(Layer 3)</small>

Low-level tensor execution substrate. The current public direction is the split
protocol family:

- `TensorSemiringCore`
- `TensorSemiringFastPath`
- `TensorScalarPrims`
- `TensorAnalyticPrims`

These family traits are currently backed by a migration layer over the legacy
`TensorPrims<A>` surface.

<a id="tenferro-algebra"></a>
### [tenferro-algebra](tenferro_algebra/index.html) <small>(Shared)</small>

Minimal algebra foundation. `HasAlgebra` trait maps scalar types to their
algebra (e.g., `f64 -> Standard`), enabling automatic backend inference.
`Semiring` trait for algebra-generic operations. `Scalar` trait (blanket impl
for `Copy + Send + Sync + Add + Mul + Zero + One + PartialEq`) defines
minimum element type requirements. `Conjugate` trait for complex conjugation
(identity for real types).

<a id="tenferro-device"></a>
### [tenferro-device](tenferro_device/index.html) <small>(Shared)</small>

Shared infrastructure: `LogicalMemorySpace` (MainMemory, GpuMemory),
`ComputeDevice` (Cpu, Cuda, Rocm), workspace-wide `Error`/`Result` types.

## External Crates (extern/)

<a id="chainrules-core"></a>
### [chainrules-core](chainrules_core/index.html) <small>(Extern)</small>

Core AD trait definitions (like Julia's ChainRulesCore.jl), independent of any
tensor type. `Differentiable` trait defines the tangent space; concrete types
(e.g., `Tensor<T>`) implement it in their own crates. Rule extension traits
(`ReverseRule<V>`, `ForwardRule<V>`) for per-operation AD rules.

<a id="chainrules"></a>
### [chainrules](chainrules/index.html) <small>(Extern)</small>

AD engine (like Zygote.jl in Julia's ecosystem). Provides homogeneous
`Tape<V>` graphs (explicit tape, TensorFlow GradientTape style),
`TrackedTensor<V>` (reverse-mode wrapper), `DualTensor<V>` (forward-mode
wrapper), and `Variable<V>` for torch-like gradient queries and accumulation.
Gradient computation uses `tape.pullback()` / `tape.hvp()` or the
`Variable<V>` / `autograd::grad_*` surface. Tensor scalar semantics follow
PyTorch: scalar tensors are rank-0, while implicit reverse seeds still depend
on `num_elements() == 1`. Depends only on `chainrules-core` and re-exports it
so downstream crates can depend on just `chainrules` for both traits and engine.

Operation-specific AD rules live with their operations, not here.

<a id="chainrules-scalarops"></a>
### [chainrules-scalarops](chainrules_scalarops/index.html) <small>(Extern)</small>

Scalar-focused AD rules layered on top of `chainrules` and `chainrules-core`.
Provides `rrule` and `frule` implementations for primitive scalar arithmetic,
projection, conjugation, powers, and related real/complex helper operations so
tensor crates can reuse the same scalar differentiation behavior.

## Extension Crates (extension/)

<a id="tenferro-tropical"></a>
### [tenferro-tropical](tenferro_tropical/index.html) <small>(Extension)</small>

Tropical semiring tensor operations. Extends the tenferro algebra-parameterized
architecture with three tropical semirings: MaxPlus (⊕=max, ⊗=+),
MinPlus (⊕=min, ⊗=+), and MaxMul (⊕=max, ⊗=×).

Provides scalar wrappers (`MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>`), algebra
markers (`MaxPlusAlgebra`, etc.), `TensorPrims` implementations for each algebra,
and `ArgmaxTracker` for recording winner indices during tropical forward passes.

<a id="tenferro-tropical-capi"></a>
### [tenferro-tropical-capi](tenferro_tropical_capi/index.html) <small>(Extension)</small>

C-API (FFI) for tropical semiring tensor operations. Extends `tenferro-capi`
with tropical einsum functions (`tfe_tropical_einsum_<algebra>_f64`) and
their AD rules (rrule/frule) for MaxPlus, MinPlus, and MaxMul semirings.
Reuses `TfeTensorF64` handles from `tenferro-capi`.

<a id="tenferro-burn"></a>
### [tenferro-burn](tenferro_burn/index.html) <small>(Extension)</small>

Bridge between the [Burn](https://burn.dev) deep learning framework and tenferro
tensor network operations. Defines `TensorNetworkOps` backend extension trait
with `tn_einsum`, implements forward pass for `NdArray<f64>` and backward pass
for `Autodiff<B, C>`, and provides Burn tensor / tenferro tensor conversion
utilities.

<a id="tenferro-mdarray"></a>
### [tenferro-mdarray](tenferro_mdarray/index.html) <small>(Extension)</small>

Bridge between [mdarray](https://crates.io/crates/mdarray) multidimensional
arrays and tenferro tensors. Provides `mdarray_to_tensor` and `tensor_to_mdarray`
conversion functions for bidirectional data exchange between
`Array<T, DynRank>` and `Tensor<T>`.

<a id="tenferro-dyadtensor"></a>
### [tenferro-dyadtensor](tenferro_dyadtensor/index.html) <small>(Extension)</small>

Dynamic dyadic tensor API that layers runtime-selected scalar/tensor types,
structured tensor layouts, and AD-aware operation builders on top of the core
tenferro crates. It provides the higher-level builder surface used by eager
einsum and linalg flows, plus dynamic wrappers for forward- and reverse-mode
execution.
