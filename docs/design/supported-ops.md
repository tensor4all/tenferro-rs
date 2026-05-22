# Supported Operations By Crate

This page is the implementation-facing inventory for the current workspace. It
is operational, not aspirational: unsupported families are called out
explicitly. The public user docs still focus on the `tenferro` facade crate.

---

## `tenferro-tensor`

`tenferro-tensor` owns dense tensor storage, dtype dispatch, backend traits, CPU
execution, and the internal CUDA backend integration. Public user docs should
call this the CUDA backend through the `tenferro::cuda` facade; this
implementation-facing page also names the CubeCL layer that backs it.

### Tensor Values

- `Tensor` dynamic dtype wrapper for `F32`, `F64`, `I64`, `C32`, and `C64`.
- `TypedTensor<T>` typed dense tensor payload.
- Host buffers by default; CubeCL device buffers behind the `cubecl` feature.

### Backend Surface

`TensorBackend` / `TensorExec` currently cover:

- Elementwise: add, multiply, negate, conjugate, divide, abs, sign, maximum,
  minimum, compare, select, clamp.
- Analytic: exp, log, sin, cos, tanh, sqrt, rsqrt, pow, expm1, log1p.
- Structural: transpose, reshape, broadcast, convert, diagonal
  extraction/embedding, triangular masks.
- Reductions: sum, product, max, min.
- Contraction: `dot_general`.
- Indexing: gather, scatter, slice, dynamic slice, pad, concatenate, reverse.
- Shape packing helpers: `Tensor::stack` and `Tensor::index_select` compose
  reshape/concatenate/gather for host-known positions.
- Linalg: Cholesky, triangular solve, LU, full-pivot LU, full-pivot LU solve,
  SVD, QR, symmetric/Hermitian eigendecomposition, and general eigendecomposition
  where the backend supports it.
- Placement: explicit host/device upload and download hooks.
- Optional backend elementwise fusion.

### CPU Status

The CPU backend is the main complete backend. Exactly one CPU feature must be
enabled:

- `cpu-faer` for faer-backed GEMM/linalg,
- `cpu-blas` for BLAS/LAPACK-backed GEMM/linalg.

Elementwise, reductions, structural operations, indexing, `dot_general`, and
dense linalg are implemented on CPU for the supported dtype subset of each op.

### CUDA/CubeCL Status

The public facade exposes this backend as `tenferro::cuda::CudaBackend` behind
the `cuda` feature. Internally, the `cubecl` feature enables
`cubecl::CubeclBackend`, backed by CubeCL/CubeCL-CUDA and runtime-loaded
cuTENSOR, cuSOLVER, and cuBLAS. Static kernels live in the internal
`tenferro-cubecl` crate.

Implemented GPU coverage is broad. The user-facing
[`Devices and GPU`](../guides/devices-and-gpu.md) guide contains the current
CUDA operation and dtype matrix. The high-level categories are:

- explicit upload/download and device pointer bridge,
- `F32`/`F64` elementwise arithmetic, comparison, selection, clamp, and
  analytic unary operations, plus `C32`/`C64` add/mul/div/neg/conj,
- reductions including sum/product for all tensor dtypes and min/max for
  `F32`/`F64`,
- structural operations including transpose, reshape, broadcast, reverse,
  concatenate, diagonal extraction/embedding, and triangular masks for all
  tensor dtypes,
- slice/pad/concatenate/reverse for all tensor dtypes and
  gather/scatter/dynamic_slice for floating and complex data with numeric
  index tensors,
- cuTENSOR-backed contraction paths for real and complex floating dtypes,
- cuSOLVER/cuBLAS linalg paths for real and complex floating dtypes.

Unsupported GPU operations and unsupported dtypes return `BackendFailure`.
Known CUDA backend limitations are operation-specific: `eig`,
`full_piv_lu`, `full_piv_lu_solve`, `dynamic_update_slice`, `I64`
numeric/linalg gaps, and selected complex analytic or ordering operations.
`eig` is not provided by cuSOLVER and permanently returns `BackendFailure` on
CubeCL. ROCm is only a feature stub.

## `tenferro-ops`

`tenferro-ops` owns the graph operation vocabulary and graph-level AD rules.

- `StdTensorOp` is the mainline operation vocabulary.
- `PrimitiveOp::linearize` and `PrimitiveOp::transpose_rule` are the semantic
  source of truth for AD rules.
- The `ExtensionOp` boundary exists for registered extension operations.
- Non-mainline semiring/algebra graph surfaces remain transitional and should
  not be extended by new work.

## `tenferro-einsum`

`tenferro-einsum` owns subscript parsing, contraction planning, graph-fragment
lowering, and eager concrete execution.

Implemented:

- `Subscripts::parse` and integer-label `Subscripts::new`.
- `NestedEinsum::parse` for parenthesized contraction order.
- `ContractionTree::optimize`, `optimize_with_options`, and `from_pairs`.
- `build_einsum_fragment` for traced graph lowering.
- `eager_einsum` and `eager_einsum_owned` for concrete `Tensor` execution.
- Repeated-label semantics:
  - `ii->` trace,
  - `ii->i` diagonal extraction,
  - `iij->ij` higher-rank diagonal extraction,
  - `i->ii` diagonal embedding.

Strict binary lowering is an optimization only. It rejects repeated-label
patterns and lets the general path handle diagonalization.

## `tenferro`

`tenferro` is the user-facing facade.

Implemented public surfaces include:

- `TracedTensor` graph construction and evaluation through `Engine`.
- `EagerTensor` / `EagerRuntime` for eager scalar-loss reverse-mode workflows.
- Lazy traced `einsum` and `einsum_with`.
- Public linalg free functions such as `svd`, `qr`, `cholesky`, `solve`,
  `triangular_solve`, `lu`, `full_piv_lu`, `eig`, `eigh`, `pinv`, `det`,
  `slogdet`, and `norm`.
- Public AD transforms such as VJP/JVP/HVP over supported traced dense numeric
  paths.
- Compiled execution through `ExecProgram` / `eval_exec_ir`.
- Shape packing on concrete, eager, and traced tensors: use `stack(..., -1)`
  to create a trailing batch axis and `index_select(-1, positions)` to align
  entries along that axis.

The facade can evaluate through `CpuBackend` or the CUDA backend when the
program uses operations supported by that backend and tensors are placed
explicitly by the execution pipeline or caller. Unsupported GPU ops return
errors rather than silently falling back to CPU.

## `tenferro-device`

`tenferro-device` owns shared device and error infrastructure:

- logical memory spaces,
- compute device metadata,
- common `Error` and `Result` types,
- conversions from lower-level strided errors.

## AD Support Notes

Current mainline AD coverage is intentionally narrower than primal execution.
Rules must live in `tenferro-ops/src/ad/` and must have corresponding
oracle/finite-difference coverage before being treated as supported mainline AD.

Linalg AD for new matrix rules is separate work from the structural/einsum and
CubeCL documentation updates covered here.
