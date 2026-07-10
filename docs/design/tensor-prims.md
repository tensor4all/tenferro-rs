# Tensor Backend Protocol

This file keeps the historical `tensor-prims.md` name, but the current
implementation no longer has a separate `tenferro-prims` crate or the old
`TensorPrims` protocol families. Dense tensor execution is owned by
`tenferro-tensor` for backend traits and value types, with concrete CPU
execution in `tenferro-cpu`.

The current backend protocol is:

- `TensorBackend`: standalone dense tensor operations plus long-lived backend
  state,
- `BackendSession`: execution-session surface used by compiled-program evaluation,
- `ElementwiseFusionPlan`: optional fused elementwise execution plan shared by
  segmentation and GPU fusion.

---

## Layering

```text
tenferro-runtime / tenferro-internal-ops
    traced graph, shape inference, AD rules, ExecProgram
        │
        ▼
tenferro-einsum
    Subscripts, ContractionTree, build_einsum_fragment, eager_einsum
        │
        ▼
tenferro-tensor
    Tensor, TypedTensor<T, R>, TypedTensorView, TensorBackend, BackendSession
        │
        ├── tenferro_cpu::CpuBackend
        └── tenferro_gpu::CudaBackend (feature = "cuda")
        ▲
        │
tenferro-tensor-core
    Rank/layout metadata and host-only adapters
```

There is exactly one dense runtime tensor type family (`Tensor` /
`TypedTensor<T, R>`) and one backend trait surface. Higher layers should depend
on `TensorBackend`/`BackendSession`, not on a backend-specific context or
deleted primitive-family traits. `tenferro-tensor-core` owns rank/layout
metadata and host-only adapters; backend-capable `TypedTensor<T, R>` lives in
`tenferro-tensor`.

## Backend Operation Surface

`TensorBackend` and `BackendSession` expose the operations needed by the compiled
execution pipeline:

| Category | Examples |
| --- | --- |
| Elementwise | `add`, `mul`, `neg`, `div`, `abs`, `maximum`, `minimum`, `compare`, `select`, `clamp` |
| Analytic | `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, `pow`, `expm1`, `log1p` |
| Structural | `transpose`, `reshape`, `broadcast_in_dim`, `extract_diagonal`, `embed_diagonal`, `tril`, `triu` |
| DType / value conversion | checked `convert`, explicit lossy `cast` |
| Reductions | `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min` |
| Contraction | `dot_general` |
| Indexing | `gather`, `scatter`, `slice`, `dynamic_slice`, `pad`, `concatenate`, `reverse` |
| Linalg | `cholesky`, `triangular_solve`, `lu`, `full_piv_lu`, `full_piv_lu_solve`, `svd`, `qr`, `eigh`, `eig` |
| Placement | `upload_host_tensor`, `download_to_host` |
| Memory reuse | `reclaim_buffer` |

Unsuffixed operation methods consume owned compact tensors. Methods with a
`_read` suffix accept borrowed views or `TensorRead` inputs. Metadata-only
operations that return views use the `_view` suffix.

Each backend may return a typed unsupported error for operations or dtypes it
does not support. Higher layers must not silently fall back across devices.

## CPU Backend

`tenferro_cpu::CpuBackend` is present when at least one CPU provider feature is enabled.
CPU provider features are additive:

- `cpu-faer` for faer-backed GEMM/linalg,
- `cpu-blas` for BLAS/LAPACK-backed GEMM/linalg,
- `cpu-tblis` for supported TBLIS-backed `dot_general` contractions.

CPU execution uses strided-kernel for elementwise/reduction/structural work and
faer or BLAS/LAPACK for GEMM and linalg. At least one of `cpu-faer` or
`cpu-blas` remains required as the fallback/linalg provider when `cpu-tblis` is
enabled. `CpuBackend` stores the runtime provider selection for an individual
backend instance. `CpuBackend::new()` chooses the compiled default provider:
BLAS if `cpu-blas` is compiled, otherwise faer. Explicit constructors such as
`CpuBackend::with_kind` and `CpuBackend::with_threads_and_kind` can select any
provider compiled into the binary. `CpuBackendKind::Tblis` attempts supported
TBLIS contractions first and falls back to the compiled faer/BLAS provider for
other valid `dot_general` shapes. `CpuContext` stores the CPU thread count as
the single source of truth for tenferro-owned CPU parallelism and owns the Rayon
thread pool used by multi-thread contexts.

`CpuBackend::with_backend_session` runs the whole compiled program through
`CpuExecSession`, reusing the backend buffer pool and avoiding per-op session
setup. Session execution enters `CpuContext::install`, so strided CPU kernels
use the backend-owned Rayon pool when `strided-kernel/parallel` is enabled.
faer-backed kernels receive `Par::Seq` for one thread or `Par::rayon(0)` for
multi-threaded execution inside that pool. BLAS/LAPACK and TBLIS provider
threading remain provider-owned.

## CubeCL Backend

`tenferro_gpu::CudaBackend` is the current GPU backend under the `cuda`
feature. It targets NVIDIA CUDA through CubeCL/CubeCL-CUDA and uses runtime-loaded
cuTENSOR, cuSOLVER, and cuBLAS where needed.

Important design points:

- GPU tensors must be explicitly uploaded with `upload_tensor`.
- Host access must explicitly download with `download_tensor`.
- `upload_host_tensor` and `download_to_host` are used by the compiled
  execution pipeline for placement-aware execution.
- GPU operations receiving CPU tensors return `BackendFailure` with an upload
  hint.
- Result-returning CPU operations receiving GPU tensors return `BackendFailure`
  with a download hint where the buffer is detected at the CPU boundary.
- ROCm is only a feature stub today.

See [gpu-backend-design.md](./gpu-backend-design.md) for the CubeCL-specific
module layout and test command.

## Elementwise Fusion

`ElementwiseFusionPlan` is the canonical backend-neutral description of fused
elementwise work. Backends opt in by overriding
`execute_elementwise_fusion`; the default implementation returns `Ok(None)`,
allowing the segmented execution pipeline to run individual ops.

CubeCL implements this hook through `crates/tenferro-gpu/src/cubecl/fusion/` for
eligible GPU-resident plans. CPU currently relies on the ordinary op sequence.

## Einsum And DotGeneral

Einsum lowering produces graph operations and ultimately `dot_general`,
reductions, broadcasts, transposes, and diagonal operations over the same
`TensorBackend` surface. The old semiring fast-path/core split is not part of
the current mainline implementation.

For column-major layout, contraction lowering keeps compute dimensions on the
left and batch dimensions on the right:

```text
lhs: [lhs_free..., contract..., batch...]
rhs: [contract..., rhs_free..., batch...]
out: [lhs_free..., rhs_free..., batch...]
```

This convention preserves contiguous per-batch slices for the GEMM-like backend
path.

## Linalg

Linalg is not a separate backend crate today. Dense linalg operations are part
of the linalg extension backend surface, with CPU implementations under
`crates/tenferro-linalg/src/cpu` and CubeCL/CUDA implementations under
`crates/tenferro-linalg/src/gpu`.

General eigendecomposition is a permanent CUDA limitation for cuSOLVER:
`CudaBackend::eig` returns `BackendFailure`, and callers must explicitly
download to CPU if they want CPU `eig`.
