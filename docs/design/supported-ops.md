# Supported Operations By Crate

This page is the implementation-facing inventory for the current workspace. It
is operational, not aspirational: unsupported families are called out
explicitly. Public user docs import the direct public crates instead of a broad
facade.

---

## `tenferro-tensor`

`tenferro-tensor` owns dense tensor storage, dtype dispatch, backend traits,
CPU execution, and backend-parametric concrete tensor kernels.

### Tensor Values

- `Tensor` dynamic dtype wrapper for `F32`, `F64`, `I32`, `I64`, `Bool`,
  `C32`, and `C64`. It remains dynamic-rank.
- `TypedTensor<T, R = DynRank>` typed runtime tensor payload with optional
  compile-time rank metadata.
- `TypedTensorView` and `TypedTensorViewMut` carry arbitrary strides, offsets,
  and metadata-only layout transforms.
- Owned tensors are compact column-major and may hold host buffers or
  backend-owned buffers. Compact-only boundaries may canonicalize views within
  the same placement, but they do not silently transfer between CPU and GPU.

### Backend Surface

`TensorBackend` / `BackendSession` currently cover:

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
- Placement: explicit host/device upload and download hooks.
- Optional backend elementwise fusion.

### CPU Status

The CPU backend is the main complete backend. Exactly one CPU feature must be
enabled:

- `cpu-faer` for faer-backed GEMM,
- `cpu-blas` for BLAS-backed GEMM.

Elementwise, reductions, structural operations, indexing, `dot_general`, and
the standard linalg extension are implemented on CPU for the supported dtype
subset of each op.

### CUDA/CubeCL Status

The public GPU crate exposes this backend as
`tenferro_gpu::cubecl::CubeclBackend` behind the `cuda` feature. It is backed
by CubeCL/CubeCL-CUDA and runtime-loaded cuTENSOR, cuSOLVER, and cuBLAS.
Static kernels live in `crates/tenferro-gpu/src/kernels`.

Implemented GPU coverage is broad. The user-facing
[`Devices and GPU`](../guides/devices-and-gpu.md) guide contains the current
CUDA operation and dtype matrix. The high-level categories are:

- explicit upload/download and device pointer bridge,
- `F32`/`F64` elementwise arithmetic, comparison, selection, clamp, and
  analytic unary operations, plus `C32`/`C64` add/mul/div/neg/conj,
- reductions including sum/product for `F32`, `F64`, `I32`, `I64`, `C32`, and
  `C64`, and min/max for `F32`/`F64`,
- reshape for all public tensor dtypes, and other structural operations
  including transpose, broadcast, reverse, concatenate, diagonal
  extraction/embedding, and triangular masks for `F32`, `F64`, `I32`, `I64`,
  `C32`, and `C64`,
- slice/pad/concatenate/reverse for `F32`, `F64`, `I32`, `I64`, `C32`, and
  `C64`, and
  gather/scatter/dynamic_slice for floating and complex data with numeric
  index tensors,
- cuTENSOR-backed contraction paths for real and complex floating dtypes,
- cuSOLVER/cuBLAS linalg extension paths for real and complex floating dtypes.

Unsupported GPU operations and unsupported dtypes return `BackendFailure`.
Known CUDA backend limitations are operation-specific: `eig`,
`full_piv_lu`, `full_piv_lu_solve`, `dynamic_update_slice`, `I64`
numeric/linalg gaps, and selected complex analytic or ordering operations.
`eig` is not provided by cuSOLVER and permanently returns `BackendFailure` on
CubeCL. ROCm is only a feature stub.

## `tenferro-internal-ops`

`tenferro-internal-ops` owns the graph operation vocabulary and graph-level AD rules.

- `StdTensorOp` is the mainline operation vocabulary.
- `PrimitiveOp::linearize` and `PrimitiveOp::transpose_rule` are the semantic
  source of truth for AD rules.
- The `ExtensionOp` boundary exists for registered extension operations.
- With `default-features = false`, AD-specific rule code is not compiled.
- Non-mainline semiring/algebra graph surfaces remain transitional and should
  not be extended by new work.

## `tenferro-runtime`

`tenferro-runtime` owns operation-agnostic runtime infrastructure:

- `ExtensionRegistry` and `ExtensionExecutor` for backend-parametric extension
  runtime registration,
- `ExtensionExecutionContext` for passing backend and extension cache state to
  one runtime call,
- `ExtensionCacheStore`, `ExtensionCacheKey`, and cache selectors/limits.

Applications import these runtime types directly from `tenferro-runtime`.

## `tenferro-einsum`

`tenferro-einsum` is the standard einsum extension. It owns subscript parsing,
contraction planning, graph-fragment lowering, eager concrete execution,
runtime registration, extension-owned caches, and the einsum AD rule when the
`autodiff` feature is enabled.

Implemented:

- `Subscripts::parse` and integer-label `Subscripts::new`.
- `NestedEinsum::parse` for parenthesized contraction order.
- `ContractionTree::optimize`, `optimize_with_options`, and `from_pairs`.
- `build_einsum_fragment` for traced graph lowering.
- `eager_einsum` and `eager_einsum_owned` for concrete `Tensor` execution.
- `tenferro_einsum::traced_tensor::einsum` and `tenferro_einsum::register_runtime` for traced
  extension use.
- Repeated-label semantics:
  - `ii->` trace,
  - `ii->i` diagonal extraction,
  - `iij->ij` higher-rank diagonal extraction,
  - `i->ii` diagonal embedding.

Strict binary lowering is an optimization only. It rejects repeated-label
patterns and lets the general path handle diagonalization.

## `tenferro-linalg`

`tenferro-linalg` is the standard linalg extension. It exposes traced linalg
functions such as `svd`, `qr`, `cholesky`, `solve`, `triangular_solve`, `lu`,
`full_piv_lu`, `eig`, `eigh`, `pinv`, `det`, `slogdet`, and `norm`, plus an
eager `EagerTensor` surface when `autodiff` is enabled.

The crate owns the linalg extension payload, direct `LinalgBackend` trait,
runtime registration, CPU linalg kernels, CUDA linalg bridge code, and linalg
AD rules where implemented. `tenferro-gpu` remains the CUDA backend and resource
owner; `tenferro-linalg` optionally depends on it for CUDA linalg execution.

## `tenferro-fft`

`tenferro-fft` is the standard FFT extension. It follows the same explicit
runtime registration model as einsum and linalg.

## Public Crates

The workspace intentionally has no root `tenferro` facade crate. Applications
import runtime APIs from `tenferro-runtime`, eager and transform AD APIs from
`tenferro-ad`, and operation families from explicit crates such as
`tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft`.

Runtime surfaces can evaluate through `CpuBackend` or the CUDA backend when the
program uses operations supported by that backend and tensors are placed
explicitly by the execution pipeline or caller. Unsupported GPU ops return
errors rather than silently falling back to CPU.

## AD Support Notes

Current mainline AD coverage is intentionally narrower than primal execution.
Core primitive rules live in `crates/tenferro-internal-ops/src/ad/`; extension-specific rules
live in the owning extension crate. Rules must have corresponding
oracle/finite-difference coverage before being treated as supported mainline AD.

The default feature set enables AD. Builds without AD use
`default-features = false` plus an explicit backend feature such as `cpu-faer`;
AD/eager-AD tests and AD rule modules are excluded in that configuration.
