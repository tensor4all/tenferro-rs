# GPU Backend Design

This document is developer-facing. Public user docs describe the GPU surface as
the CUDA backend exposed from `tenferro_gpu::cubecl::{CubeclBackend,
upload_tensor, download_tensor}`. The active implementation behind that public
surface lives in `tenferro-gpu/src/cubecl/`, gated by the `cuda` feature. It
targets NVIDIA CUDA devices through CubeCL and CubeCL-CUDA, with CUDA library
support for cuTENSOR, cuSOLVER, and cuBLAS.

CUDA GPU support is implemented through the feature-gated CubeCL backend across
the concrete tensor, eager, and traced execution surfaces. Coverage includes
allocation, explicit CPU/GPU transfer, broad structural/elementwise/reduction
kernels, cuTENSOR contractions, and cuSOLVER/cuBLAS linear algebra paths.
Performance optimization is still active work. The remaining unsupported CUDA
cases are operation-specific: `eig`, `full_piv_lu`, `full_piv_lu_solve`,
`dynamic_update_slice`, integer numeric/linalg gaps, `Bool` kernel gaps beyond
transfer and reshape, and selected complex analytic or ordering operations.
HIP/ROCm is still a feature stub rather than a supported execution path.

See also:

- `tenferro-gpu/src/cubecl/` for the implementation,
- `tenferro-gpu/src/kernels/` for static CubeCL kernel definitions and kernel-level
  validation,
- `AGENTS.md` for the current GPU status and local test command,
- [backend-contract.md](../spec/backend-contract.md) for placement rules,
- [tensor-prims.md](./tensor-prims.md) for tensor operation families.

---

## Current Module Structure

```text
tenferro-gpu/src/kernels/
    elementwise.rs         static elementwise CubeCL kernels
    structural.rs          static structural and conversion CubeCL kernels
    indexing.rs            static slice/gather/scatter/pad CubeCL kernels
    diagonal.rs            static diagonal and triangular-mask CubeCL kernels
    reduce/                reduction validation, launch helpers, and kernels

tenferro-gpu/src/cubecl/
    mod.rs                 CubeclBackend and TensorBackend implementation
    runtime.rs             CubeCL/CUDA runtime initialization and stream access
    memory.rs              upload_tensor, download_tensor, device pointer bridge
    dispatch.rs            shared launch helpers and dtype dispatch
    fusion/                fused elementwise classification and code generation
    gemm.rs                cuTENSOR/cuBLAS-backed contraction support
    linalg.rs              cuSOLVER/cuBLAS-backed linalg support
    ffi/                   runtime-loaded CUDA library bindings
    tests/                 ignored GPU tests
```

The public backend type is `tenferro_gpu::cubecl::CubeclBackend`. There are no
separate in-tree `CudaBackend` and `RocmBackend` implementations. CUDA is
selected by enabling the `cuda` feature, which depends on the workspace-pinned
CubeCL fork and the CubeCL CUDA runtime.

## Kernel Ownership

Static CubeCL kernel definitions live under `tenferro-gpu/src/kernels`. The
tensor backend crate must not keep duplicate static kernels once they have been
moved. This keeps copied/adapted CubeK-derived code, tenferro-specific kernel
definitions, and third-party notices in one crate.

`tenferro-gpu/src/cubecl/` still owns tensor values, device placement,
allocation, upload/download, CUDA library FFI, TensorBackend dispatch, and
runtime-generated fused elementwise code. Those are backend integration
concerns rather than reusable static kernels.

## Dependency Source

The workspace intentionally depends on the `tensor4all/cubecl` fork:

```toml
cubecl = { git = "https://github.com/tensor4all/cubecl.git", rev = "f5e5ec178f9aebca9362b829ffef708f720ff692", features = ["cuda"] }
cubecl-cuda = { git = "https://github.com/tensor4all/cubecl.git", rev = "f5e5ec178f9aebca9362b829ffef708f720ff692" }
cubecl-runtime = { git = "https://github.com/tensor4all/cubecl.git", rev = "f5e5ec178f9aebca9362b829ffef708f720ff692" }
```

Keep this fork dependency until upstream CubeCL has the required support and the
workspace is deliberately migrated. Do not replace it with crates.io CubeCL as
part of unrelated GPU or documentation work.

## Runtime And Library Loading

`CubeclRuntime::new(device_ordinal)` initializes CUDA and creates the CubeCL
CUDA client for one device. GPU kernels are JIT-compiled by CubeCL, so local
CUDA toolkit configuration matters.

`CubeclRuntime::synchronize()` is the explicit host-side barrier for direct GPU
backend code. It synchronizes the current CubeCL CUDA stream and does not
download tensor data. Higher-level eager CUDA execution exposes the same barrier
through `EagerRuntime::synchronize()`, with CPU eager runtimes treating the call
as a no-op.

cuTENSOR, cuSOLVER, and cuBLAS are loaded lazily through the FFI layer. The
backend first uses default soname/path candidates and allows explicit override
with these variables:

| Variable | Library |
| --- | --- |
| `TENFERRO_CUTENSOR_PATH` | cuTENSOR |
| `TENFERRO_CUSOLVER_PATH` | cuSOLVER |
| `TENFERRO_CUBLAS_PATH` | cuBLAS |

Local GPU test runs should also set:

| Variable | Purpose |
| --- | --- |
| `CUDA_PATH` | CUDA toolkit root used by CubeCL/NVRTC |
| `LD_LIBRARY_PATH` | CUDA, cuTENSOR, cuSOLVER, and cuBLAS library lookup |
| `CUBECL_DEBUG_LOG=0` | Suppress generated-kernel log spam |

## Kernel Metadata Contract

Owned runtime tensors are compact column-major tensors. The shape determines
the logical layout; dense column-major strides are `[1, d_0, d_0 * d_1, ...]`.
Arbitrary strides live on `TypedTensorView`/`TypedTensorViewMut` or
`TensorLayout` metadata until an explicit same-placement canonicalization
boundary. See
[backend-contract.md](../spec/backend-contract.md#vii-layout-and-device-contract)
for the runtime layout contract.

Host row-major import helpers canonicalize input into owned column-major host
tensors before transfer. Device tensors themselves remain column-major. This
keeps existing CubeCL kernels correct, including raw linear buffer kernels that
do not consume tensor stride metadata.

CubeCL kernels that perform logical tensor indexing must receive tensor
metadata through CubeCL tensor metadata. There is no hidden row-major fallback
and no implicit global shape state.

- Tensor shape extents and strides are runtime tensor metadata. Logical kernels
  must receive them through `TensorBinding` and access them inside kernels
  through CubeCL `Tensor` methods such as `shape(axis)`, `stride(axis)`, and
  `coordinate(index, axis)`.
- Rank may be passed as a `#[comptime]` loop bound when CubeCL needs fixed-size
  local index buffers or unrolled axis loops. This rank must be derived from
  the validated tensor metadata at the launch boundary and must not carry shape
  extents or strides.
- `#[comptime]` is reserved for operation attributes and algorithm
  configuration. This includes attributes such as transpose `perm`,
  broadcast/gather/scatter dimension-number mappings, static slice strides,
  axis sets, reduce strategy, and kernel blueprints. Different attribute values
  may compile as different CubeCL specializations.
- Permute-like operations should canonicalize their launch attributes where the
  transformation is mathematically identical. In particular, adjacent axes that
  stay contiguous in column-major layout should be fused before choosing the
  effective `perm` and rank when doing so preserves observable shape semantics.
  This reduces CubeCL JIT specialization patterns without changing the public
  tensor contract.
- Raw `ArrayArg` is allowed only for linear-buffer kernels that do not perform
  logical tensor indexing, such as elementwise kernels and raw dtype conversion
  helpers. A logical indexing kernel may use raw arrays only with a local comment
  explaining why `TensorBinding` cannot express the access pattern.
- View canonicalization and copy-back kernels are the current exception for
  logical indexing over raw arrays: `TensorBinding` metadata cannot represent
  signed arbitrary view strides, so these kernels receive the validated
  `TensorLayout` shape, signed stride, and offset metadata from the launch
  boundary and index the CubeCL allocation directly. They must still launch over
  the full logical output or update domain and must not download to host.
- Kernel crates must not invent or cache host-side tensor shape snapshots that
  can drift from the `TypedTensor` or `ExecInstruction` metadata. Shape
  validation belongs at the launch/backend boundary before unsafe launch.

The caller owns validation that the buffer length matches the dense shape
product before creating `TensorBinding` or raw array arguments. Existing helper
functions in `tenferro-gpu/src/cubecl/dispatch.rs` are the current source of
truth for this boundary.

## Launch Configuration Contract

Elementwise, structural, indexing, and reduction kernels should launch enough
parallel work items to cover the output or update domain. Single-thread launch
is not an acceptable correctness fallback for new or modified kernels.

Reduction `Auto` strategy may use one unit per keepdims output element only
when the reduce-axis length is bounded by the hardware plane width. Larger
reduce axes must use a parallel plane/subgroup reduction strategy, or return an
unsupported-strategy error when the runtime cannot provide plane operations.
The explicit `Unit` strategy remains available as a requested serial strategy,
but `Auto` must not silently route unbounded reduce-axis work to one worker.

Scatter uses a two-phase launch: first a parallel copy initializes `out` from
`operand`, then a parallel update kernel covers the scatter update domain.
Overlapping add-scatter updates use CubeCL atomic add for supported real scalar
parts. Complex scatter is represented as atomic adds to the real and imaginary
parts, following the same decomposition used by JAX GPU lowering for complex
scatter-add. Because floating-point atomic addition does not define a stable
inter-thread accumulation order, overlapping floating-point scatter updates are
numerically nondeterministic within normal floating-point roundoff.

## Device Transfer Policy

tenferro follows the PyTorch convention: no implicit CPU/GPU transfer at tensor
API boundaries. Callers upload tensors before GPU backend operations and
download results explicitly when host access is needed.

Same-placement canonicalization is allowed: host views may be copied into host
compact tensors, and CUDA views may be copied into CUDA compact tensors. It is
not a transfer mechanism.

```rust,ignore
use tenferro_gpu::cubecl::{download_tensor, upload_tensor, CubeclBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let mut backend = CubeclBackend::new(0)?;
let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);

let gpu_a = upload_tensor(backend.runtime(), &a)?;
let gpu_b = upload_tensor(backend.runtime(), &b)?;
let gpu_c = backend.add(&gpu_a, &gpu_b)?;
let cpu_c = download_tensor(backend.runtime(), &gpu_c)?;
```

The execution pipeline handles placement internally for compiled programs:
constants are uploaded through `upload_host_tensor()`, metadata-only operations
read metadata without bulk host transfer, and host-dependent scalar cases
download only the required scalar values.

Error behavior:

| Case | Behavior |
| --- | --- |
| GPU op receives a CPU tensor | `Error::BackendFailure` with an upload hint |
| CPU op receives a GPU tensor | `Error::BackendFailure` with a download hint for `Result` APIs |
| `TypedTensor::host_data()` on a GPU buffer | panic with a diagnostic |

## Implemented Coverage

The public CUDA backend implements `TensorBackend` for the main dense CUDA
execution surface. Internally, that coverage is provided by CubeCL kernels and
CUDA library calls:

| Category | Current status |
| --- | --- |
| Allocation/transfer | CUDA allocation, upload, download, raw pointer bridge for all public tensor dtypes |
| Elementwise | `F32`/`F64` arithmetic, comparison, selection, clamp, and analytic unary ops; `C32`/`C64` add/mul/div/neg/conj |
| Reductions | sum/prod for `F32`, `F64`, `I32`, `I64`, `C32`, and `C64`; min/max for `F32`/`F64` |
| Structural | reshape for all public tensor dtypes; transpose, broadcast, reverse, concatenate, diagonal extraction/embedding, and triangular masks for non-`Bool` dtypes with CubeCL element storage |
| Indexing | slice/pad/concatenate/reverse for `F32`, `F64`, `I32`, `I64`, `C32`, and `C64`; gather/dynamic_slice for `F32`, `F64`, `I32`, `C32`, and `C64` data with `F32`, `F64`, `I32`, or `I64` start/index tensors; scatter for floating and complex data with those numeric index tensors |
| Contraction | cuTENSOR-backed paths for supported real and complex floating dtypes |
| Linalg | cuSOLVER/cuBLAS-backed SVD, QR, Cholesky, LU, Eigh, LU solve, and triangular solve for supported real and complex floating dtypes |

CUDA SVD follows JAX-compatible default driver selection as an internal backend
policy: use cuSOLVER Jacobi `gesvdj` when both matrix dimensions are at most
1024, otherwise use QR-based `gesvd`. `gesvdj` returns V, so the backend
materializes the public `vt` output by copying V to V^H on the device. The
singular-values-only path still passes scratch U/V buffers to `gesvdj` because
cuSOLVER rejects null U/V pointers on that path.

The published [`Devices and GPU`](../guides/devices-and-gpu.md) guide contains
the current CUDA operation and dtype matrix. Keep that matrix synchronized with
the `CubeclBackend` `TensorBackend` implementation when adding or removing CUDA
dispatch arms.

General eigendecomposition (`eig`, LAPACK `dgeev` style) is not provided by
cuSOLVER. The CUDA backend returns `BackendFailure`; users must explicitly
download to CPU and call the CPU backend.

## Unsupported And Deferred Work

The following are intentionally outside the current batch:

- GPU benchmark work,
- HIP/ROCm implementation,
- replacing the CubeCL fork,
- selected complex analytic kernels and ordering operations,
- CUDA implementations for `full_piv_lu`, `full_piv_lu_solve`, and
  `dynamic_update_slice`,
- integer numeric/linalg CUDA kernels beyond structural and reduction paths,
- `Bool` CUDA kernels beyond allocation, upload/download, and metadata-only
  reshape,
- changing the public placement contract.

## Tests

GPU tests are ignored so regular CPU-only test runs remain portable. Run them on
a CUDA machine with:

```sh
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-gpu --features cuda -- --ignored
```

These tests are correctness tests, not benchmarks.
