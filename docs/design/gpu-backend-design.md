# GPU Backend Design

This document is developer-facing. Public user docs describe GPU providers as
explicit backend choices. CUDA is exposed as `tenferro_gpu::CudaBackend`.
WebGPU is exposed as `tenferro_gpu::WebGpuBackend`. Backend
features are additive provider choices: `cuda` and `webgpu` are both explicit,
neither is enabled by default as a GPU provider, and downstream crates may
enable either or both. Eager and operation crates should propagate the same
concrete feature names instead of introducing default GPU providers or a vague
public `gpu` feature.

The active CUDA implementation lives in `crates/tenferro-gpu/src/cubecl/`,
gated by the `cuda` feature. It targets NVIDIA CUDA devices through CubeCL and
CubeCL-CUDA, with CUDA library support for cuTENSOR, cuSOLVER, and cuBLAS. The
WebGPU implementation lives in a separate provider module, gated by the
`webgpu` feature, and targets CubeCL-WGPU without depending on CUDA runtime or
CUDA library bindings.

CUDA GPU support is implemented through the feature-gated CubeCL backend across
the concrete tensor, eager, and traced execution surfaces. Coverage includes
allocation, explicit CPU/GPU transfer, broad structural/elementwise/reduction
kernels, cuTENSOR contractions, and cuSOLVER/cuBLAS linear algebra paths.
Performance optimization is still active work. The remaining unsupported CUDA
cases are operation-specific: `eig`, `full_piv_lu`, `full_piv_lu_solve`,
`dynamic_update_slice`, integer numeric/linalg gaps beyond add/mul/compare/select
and sum/prod reductions, `Bool` arithmetic/reduction/linalg gaps, and
selected complex analytic or ordering operations.
WebGPU is being introduced incrementally. The implemented path covers explicit
transfer plus `F32` `dot_general` through a CubeK BGEMM planner. `C32` GEMM is
implemented through a CubeK-owned complex GEMM launch API that lowers to real
`F32` matmuls and handles conjugation flags. `F64`, `C64`,
zero-contracting-size matmul, and non-matmul tensor ops remain explicit
unsupported paths rather than CPU fallbacks. HIP/ROCm is still a reserved
feature stub rather than a supported execution path.

See also:

- `crates/tenferro-gpu/src/cubecl/` for the implementation,
- `crates/tenferro-gpu/src/kernels/` for static CubeCL kernel definitions and kernel-level
  validation,
- `AGENTS.md` for the current GPU status and local test command,
- [backend-contract.md](../spec/backend-contract.md) for placement rules,
- [tensor-prims.md](./tensor-prims.md) for tensor operation families.

---

## Current Module Structure

```text
crates/tenferro-gpu/src/kernels/
    elementwise.rs         static elementwise CubeCL kernels
    structural.rs          static structural and conversion CubeCL kernels
    indexing.rs            static slice/gather/scatter/pad CubeCL kernels
    diagonal.rs            static diagonal and triangular-mask CubeCL kernels
    reduce/                reduction validation, launch helpers, and kernels

crates/tenferro-gpu/src/cubecl/
    mod.rs                 CUDA backend and TensorBackend implementation
    runtime.rs             CubeCL/CUDA runtime initialization and synchronization
    memory.rs              upload_tensor, download_tensor, device pointer bridge
    dispatch.rs            private shared launch helpers and dtype dispatch
    interop.rs             owner-scoped launch/allocation bridge for operation crates
    fusion/                fused elementwise classification and code generation
    gemm.rs                cuTENSOR/cuBLAS-backed contraction support
    linalg.rs              cuSOLVER/cuBLAS-backed linalg support
    ffi/                   runtime-loaded CUDA library bindings
    tests/                 ignored GPU tests

crates/tenferro-gpu/src/webgpu/
    mod.rs                 WebGpuBackend provider facade and shared buffer helpers
    runtime.rs             CubeCL-WGPU runtime initialization and synchronization
    memory.rs              upload_webgpu_tensor and download_webgpu_tensor
    gemm.rs                CubeK-backed F32/C32 dot_general planner and launch support
    kernels.rs             WebGPU-private dot_general pack kernels
```

The provider-specific public backend types are `tenferro_gpu::CudaBackend` and
`tenferro_gpu::WebGpuBackend`; CubeCL naming is an implementation detail. CUDA is selected by
enabling the `cuda` feature, which depends on the workspace-pinned CubeCL fork
and the CubeCL CUDA runtime. WebGPU is selected by enabling the `webgpu`
feature, which depends on CubeCL-WGPU and the CubeK matmul provider. Enabling
both features must compile and must not merge the two runtime types.

## Kernel Ownership

Static CubeCL kernel definitions live under `crates/tenferro-gpu/src/kernels`. The
tensor backend crate must not keep duplicate static kernels once they have been
moved. This keeps copied/adapted CubeK-derived code, tenferro-specific kernel
definitions, and third-party notices in one crate.

`crates/tenferro-gpu/src/cubecl/` still owns tensor values, device placement,
allocation, upload/download, CUDA library FFI, TensorBackend dispatch, and
runtime-generated fused elementwise code. Those are backend integration
concerns rather than reusable static kernels.

## Dependency Source

The workspace intentionally depends on temporary `tensor4all` fork packages for
CubeCL and CubeK while the required patches are pending upstream. CUDA and
WebGPU runtime dependencies are feature-owned by `tenferro-gpu`; the workspace
dependency declaration must not force CUDA for WebGPU-only builds:

```toml
cubecl = { package = "t4a-cubecl", version = "=0.10.0", default-features = false }
cubecl-cuda = { package = "t4a-cubecl-cuda", version = "=0.10.0" }
cubecl-common = { package = "t4a-cubecl-common", version = "=0.10.0" }
cubecl-wgpu = { package = "t4a-cubecl-wgpu", version = "=0.10.0" }
cubecl-runtime = { package = "t4a-cubecl-runtime", version = "=0.10.0" }
cubek-matmul = { package = "t4a-cubek-matmul", version = "=0.2.0", default-features = false }
cubek-std = { package = "t4a-cubek-std", version = "=0.2.0", default-features = false }
```

Keep these fork package dependencies until upstream CubeCL/CubeK have the
required support and the workspace is deliberately migrated. Do not replace them
with upstream crates.io CubeCL/CubeK as part of unrelated GPU or documentation
work.

CubeK matmul integration should branch from the CubeK release paired to CubeCL
0.10.0: start from CubeK `v0.2.0` / `cubek-matmul 0.2.0`. If complex GEMM or
WebGPU fixes require a tensor4all fork, publish tensor4all-owned CubeK crates
from that branch rather than vendoring CubeK into this repository. Local
development may use sibling checkouts with Cargo `path` dependencies, but
committed tenferro manifests should use the published `t4a-*` package aliases
unless the PR deliberately targets a pre-publish staging branch. In that staging
case, use Cargo's multiple-location form with `git`, `rev`, and `version` so CI
uses the fork commit while the packaged manifest retains the registry version.
After the `t4a-*` packages are on crates.io, remove the `git` and `rev` keys and
keep the exact version requirements.

## Runtime And Library Loading

`CudaRuntime::new(device_ordinal)` initializes CUDA and creates the CubeCL CUDA
client for one device. GPU kernels are JIT-compiled by CubeCL, so local CUDA
toolkit configuration matters.

`WebGpuRuntime::new(device_ordinal)` creates a CubeCL-WGPU client for a WebGPU
adapter. WebGPU runtime initialization must not call CUDA driver/runtime APIs,
load CUDA libraries, or require CUDA environment variables.

Future ROCm support should follow the same explicit-provider model, but it must
not require ROCm libraries to be present for a binary that merely includes the
reserved `rocm` feature. The intended substrate is a CubeCL HIP fork or patch
that runtime-loads HIP libraries with the same discipline as the CUDA FFI layer.
Until that loader-backed substrate is implemented and tested on ROCm hardware,
tenferro must keep ROCm unavailable as an execution backend and must not publish
a ROCm quickstart.

`CudaRuntime::synchronize()` is the explicit host-side barrier for direct CUDA
backend code. It synchronizes the current CubeCL CUDA stream and does not
download tensor data. `WebGpuRuntime::synchronize()` is the corresponding
WebGPU queue/device barrier. Higher-level eager GPU execution exposes the same
barrier through `EagerRuntime::synchronize()`, with CPU eager runtimes treating
the call as a no-op.

cuTENSOR, cuSOLVER, and cuBLAS are CUDA-only and are loaded lazily through the
CUDA FFI layer. The CUDA backend first uses default soname/path candidates and
allows explicit override with these variables:

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

## Runtime Cache Ownership

`CudaBackend` owns CUDA extension backend-state caches. Extension crates may
store type-indexed CUDA handles or plans through
`CudaBackend::cuda_extension_cache()`, but the backend remains the lifetime
and resource owner.

The CUDA extension cache has a bounded default capacity of 16 type entries.
Applications can configure it with
`CudaBackend::set_cuda_extension_cache_max_entries`, clear it with
`CudaBackend::clear_cuda_extension_cache`, and inspect retained entries and
logical retained bytes with `CudaBackend::cuda_extension_cache_stats`.
Retained bytes are estimates of cache-owned payloads, not process RSS or
allocator arena usage.

WebGPU provider caches must be owned by `WebGpuBackend` or a WebGPU runtime
cache object with the same bounded-default, clear, configure, and stats
requirements before they become long-lived.

GPU scratch-buffer pools should eventually expose a provider-independent stats
shape across CUDA, WebGPU, and future ROCm:

| Field | Meaning |
| --- | --- |
| `retained_buffers` | Number of buffers currently retained by the pool |
| `retained_bytes` | Logical bytes retained by the pool |
| `acquire_calls` | Total acquire requests |
| `release_calls` | Total release requests |
| `reuse_hits` | Acquires served from retained buffers |
| `allocation_misses` | Acquires requiring a new allocation |
| `evictions` | Retained buffers dropped because of pool limits |
| `high_water_retained_bytes` | Peak logical retained bytes |

This common stats design is future direction only. It must not be introduced by
rewriting existing CUDA contraction allocation behavior. CUDA `dot_general`
continues to allocate cuTENSOR workspace through the existing runtime client
path, and this WebGPU work does not alter CUDA buffer pools, CUDA scratch
reuse, or CUDA library-call algorithms.

## Operation-Crate Interop Boundary

`crates/tenferro-gpu/src/cubecl/dispatch.rs` is private backend glue. It owns
shape/buffer validation before unsafe CubeCL launch arguments are constructed.
Sibling operation crates must not import it directly.

Operation crates that need to launch their own CubeCL kernels, such as
`tenferro-linalg`, use the owner-scoped `tenferro_gpu::cuda_interop` module
instead. That module intentionally exposes only the bridges that cannot live in
`tenferro-gpu` without creating an operation-crate dependency cycle:

- one-dimensional launch configuration helpers,
- checked `TensorBinding` / `ArrayArg` construction,
- typed output allocation, typed upload/download, and typed device-pointer
  extraction,
- byte workspaces kept alive for CUDA library calls,
- scoped access to the CubeCL client for operation-owned kernel launches.

`CudaRuntime::client`, `CudaRuntime::raw_cuda_stream`, `CubeclBuffer`
fields, and raw `CubeclBuffer` constructors are not public API. Public tensor
users should use `CudaBackend`, `upload_tensor`, `download_tensor`,
`device_ptr`, and `CudaRuntime::synchronize`.

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
  broadcast/gather/scatter dimension-number mappings, static slice step
  attributes, axis sets, reduce strategy, and kernel blueprints. Different
  attribute values may compile as different CubeCL specializations.
- Do not pass tensor shape extents, strides, buffer lengths, flattened products,
  or other runtime tensor sizes as `#[comptime]` parameters. The WebGPU
  `dot_general` pack kernels pass only axis-role lists and rank as compile-time
  launch attributes; shape and stride values are read from `TensorBinding`
  metadata inside the kernel.
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
functions in `crates/tenferro-gpu/src/cubecl/dispatch.rs` are the current source of
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
compact tensors, and GPU views may be copied into compact tensors on the same
GPU provider. It is not a transfer mechanism.

```text
use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let mut backend = CudaBackend::new(0)?;
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
| CUDA op receives a WebGPU tensor, or WebGPU op receives a CUDA tensor | `Error::BackendFailure` naming the expected provider |

## Implemented Coverage

The public CUDA backend implements `TensorBackend` for the main dense CUDA
execution surface. Internally, that coverage is provided by CubeCL kernels and
CUDA library calls:

| Category | Current status |
| --- | --- |
| Allocation/transfer | CUDA allocation, upload, download, raw pointer bridge for all public tensor dtypes |
| Elementwise | `F32`/`F64` arithmetic, comparison, selection, clamp, and analytic unary ops; `I32`/`I64` add/mul/compare/select; `C32`/`C64` add/mul/div/neg/conj |
| Reductions | sum/prod for `F32`, `F64`, `I32`, `I64`, `C32`, and `C64`; min/max for `F32`/`F64` |
| Structural | reshape, transpose, broadcast, reverse, concatenate, diagonal extraction/embedding, triangular masks, slice, and pad support all public tensor dtypes; Bool data movement uses its one-byte device representation |
| DType conversion | checked `convert` and explicit `cast` cover every CPU-supported pair among the seven public dtypes; explicit real/complex-to-integer validation uses a small device flag and never downloads the input tensor |
| Indexing | gather supports `F32`, `F64`, `I32`, `Bool`, `C32`, and `C64` data with CPU-supported `F32`, `F64`, `I32`, or `I64` index tensors; dynamic_slice supports those numeric/complex data dtypes with numeric starts, but Bool data only with `I32`/`I64` starts; additive scatter remains limited to floating and complex data and explicitly excludes Bool data |
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
the `CudaBackend` `TensorBackend` implementation when adding or removing CUDA
dispatch arms.

General eigendecomposition (`eig`, LAPACK `dgeev` style) is not provided by
cuSOLVER. The CUDA backend returns `BackendFailure`; users must explicitly
download to CPU and call the CPU backend.

The WebGPU backend currently has narrower coverage:

| Category | WebGPU status |
| --- | --- |
| Allocation/transfer | WebGPU allocation, upload, and download for `F64`, `F32`, `I32`, `I64`, `Bool`, `C64`, and `C32` tensors |
| Real contraction | CubeK/CubeCL-backed `F32` `dot_general` through a BGEMM planner, including batched and same-device packed operand layouts covered by tests |
| Complex contraction | `C32` `dot_general` and `dot_general_with_conj` through a CubeK-owned complex GEMM API. tenferro normalizes `DotGeneralConfig` into CubeK-compatible batched matmul bindings; CubeK owns temporary real buffers, split/compose kernels, conjugation signs, and future native complex-kernel replacement |
| Deferred contraction coverage | `F64`, `C64`, zero-contracting-size matmul, and broader planner stress coverage |
| Other tensor ops | Explicit unsupported `BackendFailure`; no CPU fallback and no hidden provider transfer |

## Unsupported And Deferred Work

The following are intentionally outside the current batch:

- GPU benchmark work,
- HIP/ROCm execution backend implementation,
- replacing the CubeCL fork,
- selected complex analytic kernels and ordering operations,
- CUDA implementations for `full_piv_lu`, `full_piv_lu_solve`, and
  `dynamic_update_slice`,
- integer numeric/linalg CUDA kernels beyond add/mul/compare/select,
  structural paths, and sum/prod reductions,
- `Bool` CUDA arithmetic, reductions, linalg, and additive scatter beyond allocation, upload/download, metadata-only
  reshape,
- changing the public placement contract,
- WebGPU elementwise, reduction, indexing, and linalg kernels beyond explicit
  transfer and CubeK-backed `F32`/`C32` contraction.

## Tests

CUDA GPU tests are ignored so regular CPU-only test runs remain portable. Run
them on a CUDA machine with:

```sh
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-gpu --features cuda -- --ignored
```

These tests are correctness tests, not benchmarks.

WebGPU provider tests should have two layers: portable source/feature contract
tests that run in ordinary CI, and adapter-optional runtime tests that return
early when no WebGPU adapter is available. Runtime tests must compare meaningful
tensor values or residuals, not only shapes.
