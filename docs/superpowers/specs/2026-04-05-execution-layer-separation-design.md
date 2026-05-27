# v2 Design: Execution Layer Separation + Contiguous Memory

Supersedes v1 architecture. Implements the v2.1 proposal from issue #621.

## Goals

1. Single execution owner per operation (no scattered `eval` across crate layers)
2. `Operand` trait removed from computegraph; `GraphOp`/`EvalGraphOp` split
3. Contiguous column-major memory only (strides removed from `TypedTensor<T>`)
4. `TensorBackend` + `SemiringBackend<Alg>` replace `SemiringCore` as the backend abstractions
5. All CPU kernels use optimized implementations (strided-kernel, faer, or blas/lapack)
6. Config types (`DotGeneralConfig`, etc.) move from tenferro-ops to tenferro-tensor
7. Tropical algebra end-to-end test (GEMM は naive loop で可、設計検証が目的)
8. GPU backend stubs (CUDA/ROCm skeleton impl) で trait 設計の CPU/GPU 汎用性を検証
9. Standard/custom algebra + CPU/GPU の4象限が全てコンパイルを通ることを保証

## Crate Responsibilities

| Crate | Role | Changes |
|---|---|---|
| **computegraph** | Graph IR | Already done: `Operand` removed, `GraphOp`/`EvalGraphOp` split |
| **tenferro-device** | Device/error types | No change |
| **tenferro-algebra** | Algebra traits | No change |
| **tenferro-tensor** | Tensor runtime: types + kernels + backends | Major restructure (see below) |
| **tenferro-ops** | Op metadata: `GraphOp`, `PrimitiveOp`, `SemiringOps` | Remove `eval`, drop `Operand` dep |
| **tenferro-einsum** | Einsum graph construction | Minimal change |
| **tenferro** | Pipeline orchestration: Engine, compiler, TracedTensor | Remove execution code (moved to tenferro-tensor) |

## TypedTensor<T> -- Contiguous Only

```rust
pub struct TypedTensor<T> {
    pub buffer: Buffer<T>,    // Host(Vec<T>) | Backend(BufferHandle<T>)
    pub shape: Vec<usize>,    // strides derived from shape (col-major)
    pub placement: Placement, // GPU/CPU placement
}
```

Removed fields:
- `strides` -- always col-major, derived from `shape`
- `preferred_compute_device`

Retained types: `Buffer<T>`, `BufferHandle<T>`, `Placement`, `ComputeDevice`, `MemoryKind`, `DType`, `ConjElem`.

`Tensor` enum inherent methods: metadata only (`shape()`, `dtype()`). All compute methods removed.

## tenferro-tensor Internal Structure

```
tenferro-tensor/src/
  lib.rs
  types.rs                  TypedTensor<T>, Tensor, Buffer<T>, DType, Placement,
                            MemoryKind, ComputeDevice, BufferHandle<T>, ConjElem
  config.rs                 DotGeneralConfig, CompareDir, GatherConfig, ScatterConfig,
                            SliceConfig, PadConfig (moved from tenferro-ops)
  backend.rs                TensorBackend + SemiringBackend<Alg> trait definitions
  cpu/
    mod.rs
    backend.rs              CpuBackend: impl TensorBackend
    elementwise.rs          strided-kernel: add, mul, neg, conj, div, abs, sign,
                              max, min, compare, select, clamp,
                              exp, log, sin, cos, tanh, sqrt, rsqrt, pow, expm1, log1p
    reduction.rs            strided-kernel: reduce_sum, reduce_prod, reduce_max, reduce_min
    structural.rs           strided-kernel: transpose, broadcast_in_dim, extract_diagonal
                            dedicated impl: reshape (metadata only), embed_diagonal
    indexing.rs             dedicated impl: gather, scatter, slice, dynamic_slice,
                              pad, concatenate, reverse
    gemm/
      mod.rs                feature-gate dispatch
      faer_gemm.rs          cpu-faer: FaerGemm trait
      blas_gemm.rs          cpu-blas: BlasGemm trait
    linalg/
      mod.rs                feature-gate dispatch
      faer_linalg.rs        cpu-faer: svd, qr, eigh, cholesky, solve
      lapack_linalg.rs      cpu-blas: svd, qr, eigh, cholesky, solve
  cuda/                     feature: cuda (future)
    mod.rs
    backend.rs              CudaBackend: impl TensorBackend
    kernels.rs
  rocm/                     feature: rocm (future)
    mod.rs
    backend.rs              RocmBackend: impl TensorBackend
    kernels.rs
```

### Kernel Implementation Strategy

| Category | Implementation | Notes |
|---|---|---|
| Elementwise (add, mul, neg, exp, ...) | strided-kernel (`map_into`, `zip_map2_into`, `zip_map3_into`) | Cache-optimized, SIMD fast path |
| Reduction (reduce_sum, ...) | strided-kernel (`reduce`, `reduce_axis`) | |
| Structural (transpose, broadcast, extract_diag) | strided-kernel (`permute`+`copy_into`, `broadcast`, `diagonal_view`) | |
| reshape | Metadata only (shape swap, no data copy) | Contiguous-only makes this trivial |
| embed_diagonal | Dedicated implementation | No strided-kernel API for this |
| Indexing (gather, scatter, ...) | Dedicated implementation | Index-driven, not element-wise |
| GEMM | faer (`strided_gemm`) or BLAS (`cblas_dgemm` etc.) | Feature-gated |
| Linalg (svd, qr, ...) | faer or LAPACK | Feature-gated |

**Rule: No naive CPU loop fallbacks.** All CPU kernels must use strided-kernel, faer, or blas/lapack.

## TensorBackend Trait

```rust
pub trait TensorBackend {
    // Elementwise
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn neg(&mut self, input: &Tensor) -> Tensor;
    fn conj(&mut self, input: &Tensor) -> Tensor;
    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn abs(&mut self, input: &Tensor) -> Tensor;
    fn sign(&mut self, input: &Tensor) -> Tensor;
    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> Tensor;
    fn select(&mut self, pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Tensor;
    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> Tensor;

    // Analytic
    fn exp(&mut self, input: &Tensor) -> Tensor;
    fn log(&mut self, input: &Tensor) -> Tensor;
    fn sin(&mut self, input: &Tensor) -> Tensor;
    fn cos(&mut self, input: &Tensor) -> Tensor;
    fn tanh(&mut self, input: &Tensor) -> Tensor;
    fn sqrt(&mut self, input: &Tensor) -> Tensor;
    fn rsqrt(&mut self, input: &Tensor) -> Tensor;
    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn expm1(&mut self, input: &Tensor) -> Tensor;
    fn log1p(&mut self, input: &Tensor) -> Tensor;

    // Structural
    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> Tensor;
    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> Tensor;
    fn broadcast_in_dim(&mut self, input: &Tensor, shape: &[usize], dims: &[usize]) -> Tensor;
    fn extract_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor;
    fn embed_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor;

    // Reduction
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;
    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;
    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;
    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;

    // GEMM
    fn dot_general(&mut self, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor;

    // Indexing
    fn gather(&mut self, input: &Tensor, config: &GatherConfig) -> Tensor;
    fn scatter(&mut self, input: &Tensor, updates: &Tensor, config: &ScatterConfig) -> Tensor;
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> Tensor;
    fn dynamic_slice(&mut self, input: &Tensor, starts: &Tensor) -> Tensor;
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> Tensor;
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> Tensor;
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> Tensor;

    // Linalg
    fn cholesky(&mut self, input: &Tensor) -> Tensor;
    fn svd(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn qr(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn eigh(&mut self, input: &Tensor) -> Vec<Tensor>;
    fn solve(&mut self, a: &Tensor, b: &Tensor) -> Tensor;
}
```

## SemiringBackend\<Alg\> Trait

Custom algebra backend. Operates on `TypedTensor<Alg::Scalar>` (typed).
Independent of `TensorBackend` (no supertrait relationship).
Context is `&mut self` — the backend struct holds thread pool, CUDA stream, scratch buffers, etc.

```rust
pub trait SemiringBackend<Alg: Semiring> {
    /// Required: batched GEMM on TypedTensor (device-agnostic).
    /// The backend struct (&mut self) carries execution context
    /// (thread pool, CUDA stream, etc.).
    fn batched_gemm(
        &mut self,
        lhs: &TypedTensor<Alg::Scalar>,
        rhs: &TypedTensor<Alg::Scalar>,
        config: &DotGeneralConfig,
    ) -> TypedTensor<Alg::Scalar>;

    /// Default: strided-kernel zip_map2_into with Alg::add
    fn add(&mut self, lhs: &TypedTensor<Alg::Scalar>, rhs: &TypedTensor<Alg::Scalar>)
        -> TypedTensor<Alg::Scalar> { /* ... */ }

    /// Default: strided-kernel zip_map2_into with Alg::mul
    fn mul(&mut self, lhs: &TypedTensor<Alg::Scalar>, rhs: &TypedTensor<Alg::Scalar>)
        -> TypedTensor<Alg::Scalar> { /* ... */ }

    /// Default: strided-kernel reduce with Alg::add
    fn reduce_sum(&mut self, input: &TypedTensor<Alg::Scalar>, axes: &[usize])
        -> TypedTensor<Alg::Scalar> { /* ... */ }
}
```

CPU helper for users who provide a raw-slice GEMM function:

```rust
/// Wraps a raw-slice GEMM fn into a batched_gemm on TypedTensor.
/// Handles dimension analysis + batch loop.
pub fn batched_gemm_from_slice_gemm<Alg: Semiring>(
    lhs: &TypedTensor<Alg::Scalar>,
    rhs: &TypedTensor<Alg::Scalar>,
    config: &DotGeneralConfig,
    gemm_fn: impl Fn(usize, usize, usize, &[Alg::Scalar], usize, bool,
                     &[Alg::Scalar], usize, bool, &mut [Alg::Scalar], usize),
) -> TypedTensor<Alg::Scalar> { /* ... */ }
```

Structural ops (transpose, reshape, broadcast\_in\_dim, extract\_diagonal,
embed\_diagonal) are algebra-independent free functions, not on this trait.

### Execution Paths

Both paths share the same compilation pipeline (TransposeFolding, DotDecomposer, etc.).
Context flows through `&mut backend`:

```
Standard:  Fragment<StdTensorOp> → StableHLO → ExecIR → eval_exec_ir(&mut B: TensorBackend)
Custom:    Fragment<SemiringOp<Alg>> → StableHLO → ExecIR → eval_semiring_ir(&mut B: SemiringBackend<Alg>)
```

`eval_semiring_ir` dispatches algebra-dependent ops through `SemiringBackend<Alg>`
and structural ops through shared free functions. Non-semiring ops (analytic,
indexing, linalg) in the ExecIR trigger an error.

### User Extension (Custom Algebra)

Minimal: implement `Semiring` (4 methods) + `SemiringBackend::batched_gemm` (1 method):

```rust
impl Semiring for TropicalAlgebra {
    fn zero() -> f64 { f64::NEG_INFINITY }
    fn one() -> f64 { 0.0 }
    fn add(a: f64, b: f64) -> f64 { a.max(b) }
    fn mul(a: f64, b: f64) -> f64 { a + b }
}

// CPU: use helper to wrap raw-slice GEMM
impl SemiringBackend<TropicalAlgebra> for CpuBackend {
    fn batched_gemm(&mut self, lhs: &TypedTensor<f64>, rhs: &TypedTensor<f64>,
                    config: &DotGeneralConfig) -> TypedTensor<f64> {
        batched_gemm_from_slice_gemm::<TropicalAlgebra>(lhs, rhs, config,
            |m,n,k,a,lda,ta,b,ldb,tb,c,ldc| { /* tropical_gemm call */ })
    }
}

// GPU stub: device-buffer GEMM (future)
impl SemiringBackend<TropicalAlgebra> for CudaBackend {
    fn batched_gemm(&mut self, lhs: &TypedTensor<f64>, rhs: &TypedTensor<f64>,
                    config: &DotGeneralConfig) -> TypedTensor<f64> {
        todo!("CUDA tropical batched GEMM")
    }
}
```

## tenferro-ops Changes

```rust
impl GraphOp for StdTensorOp {
    type Operand = Tensor;
    type Context = ();
    type InputKey = TensorInputKey;

    fn n_inputs(&self) -> usize { /* unchanged */ }
    fn n_outputs(&self) -> usize { /* unchanged */ }
    // eval REMOVED -- EvalGraphOp is NOT implemented for StdTensorOp
}
```

Deleted: `eval` method, `use computegraph::Operand`.
Retained: `PrimitiveOp`, `SemiringOps`. Config types move to tenferro-tensor.

### SemiringOp\<Alg\> Changes

Type parameter changed from scalar T to algebra Alg. Same eval removal as StdTensorOp:

```rust
pub struct SemiringOp<Alg: Algebra> {
    pub kind: SemiringOpKind,
    _marker: PhantomData<Alg>,
}

impl<Alg: Algebra> GraphOp for SemiringOp<Alg> {
    type Operand = TypedTensor<Alg::Scalar>;
    type Context = ();
    type InputKey = SemiringInputKey;

    fn n_inputs(&self) -> usize { /* from SemiringOpKind */ }
    fn n_outputs(&self) -> usize { 1 }
    // eval REMOVED -- EvalGraphOp is NOT implemented
}
```

Deleted: `eval` (was `todo!()`), `use computegraph::Operand`, `T: Operand` bound.
Changed: `SemiringOp<T>` → `SemiringOp<Alg: Algebra>`, `type Operand` = `TypedTensor<Alg::Scalar>`.
Retained: `SemiringOps` impl, `SemiringOpKind`.

## tenferro Changes

```
tenferro/src/
  lib.rs
  engine.rs             Engine<B: TensorBackend> (cache + buffer pool)
  traced.rs             TracedTensor
  einsum.rs             einsum API
  compiler.rs           StableHLO -> ExecIR (optimization passes)
  stablehlo.rs          StableHLO IR definition
  exec.rs               ExecOp, ExecProgram, eval_exec_ir<B: TensorBackend>
  buffer_pool.rs        BufferPool
  error.rs              Error types
```

Deleted (moved to tenferro-tensor):
- `backend.rs` (SemiringCore trait)
- `cpu_backend.rs` (CpuBackend)
- `structural.rs`
- `standard.rs`
- `gemm/`
- `indexing.rs`
- `reduction.rs`
- `linalg.rs`

`eval_exec_ir` dispatches all `ExecOp` variants directly through `TensorBackend` methods:

```rust
pub fn eval_exec_ir<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Vec<Tensor> {
    for inst in &program.instructions {
        let result = match &inst.op {
            ExecOp::Add => backend.add(lhs, rhs),
            ExecOp::Multiply => backend.mul(lhs, rhs),
            ExecOp::Negate => backend.neg(input),
            ExecOp::Conj => backend.conj(input),
            ExecOp::Permute { perm } => backend.transpose(input, perm),
            ExecOp::Reshape { shape } => backend.reshape(input, shape),
            ExecOp::BroadcastInDim { shape, dims } => backend.broadcast_in_dim(input, shape, dims),
            ExecOp::BatchedGemm(config) => backend.dot_general(lhs, rhs, config),
            ExecOp::ReduceSum { axes } => backend.reduce_sum(input, axes),
            // ... all other ops
        };
        slots[inst.output_slots[0]] = Some(result);
    }
}
```

## Feature Flags (tenferro-tensor/Cargo.toml)

```toml
[features]
default = ["cpu-faer"]

# Exactly one required (compile_error! enforced)
cpu-faer = ["dep:faer"]
cpu-blas = ["dep:cblas-sys", "dep:lapack"]

# BLAS provider selection (cpu-blas only)
provider-src = ["cpu-blas", "dep:blas-src", "dep:cblas-src", "dep:lapack-src"]
src-openblas = ["provider-src", "blas-src/openblas"]
src-accelerate = ["provider-src", "blas-src/accelerate"]
src-intel-mkl-dynamic-parallel = ["provider-src", "blas-src/intel-mkl-dynamic-parallel"]

# GPU backends (future)
cuda = ["dep:cudarc"]
```

Build-time enforcement:

```rust
#[cfg(not(any(feature = "cpu-faer", feature = "cpu-blas")))]
compile_error!("enable at least one CPU backend: cpu-faer or cpu-blas");
```

## Dependency Graph

```
strided-rs (external)
    |
tenferro-tensor
  <- num-traits, num-complex, strided-kernel, strided-view, strided-traits
  <- faer (cpu-faer) / cblas-sys + lapack (cpu-blas)
  <- cudarc (cuda)
    |
tenferro-ops
  <- computegraph (GraphOp only), chainrules-core
    |
tenferro-einsum
  <- omeco
    |
tenferro
  <- thiserror
```

## Migration Summary

| Current location | Destination |
|---|---|
| `tenferro-tensor/src/operand.rs` | Delete; kernels to `tenferro-tensor/src/cpu/` |
| `tenferro-tensor/src/tensor_data.rs` | Delete |
| `tenferro/src/backend.rs` (SemiringCore) | `tenferro-tensor/src/backend.rs` (TensorBackend) |
| `tenferro/src/cpu_backend.rs` | `tenferro-tensor/src/cpu/backend.rs` |
| `tenferro/src/gemm/` | `tenferro-tensor/src/cpu/gemm/` |
| `tenferro/src/structural.rs` | `tenferro-tensor/src/cpu/structural.rs` |
| `tenferro/src/standard.rs` | `tenferro-tensor/src/cpu/elementwise.rs` |
| `tenferro/src/reduction.rs` | `tenferro-tensor/src/cpu/reduction.rs` |
| `tenferro/src/indexing.rs` | `tenferro-tensor/src/cpu/indexing.rs` |
| `tenferro/src/linalg.rs` | `tenferro-tensor/src/cpu/linalg/` |
| `tenferro-ops/src/std_tensor_op.rs` eval block | Delete |
