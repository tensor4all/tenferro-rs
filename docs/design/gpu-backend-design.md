# GPU Backend Design

This document describes the current GPU backend in tenferro-rs. The active
implementation is the CubeCL backend in `tenferro-tensor/src/cubecl/`, gated by
the `cubecl` feature. It targets NVIDIA CUDA devices through CubeCL and
CubeCL-CUDA, with CUDA library support for cuTENSOR, cuSOLVER, and cuBLAS.

GPU support is partial and experimental. CUDA allocation, explicit CPU/GPU
transfer, many structural/elementwise/reduction kernels, selected cuTENSOR
contractions, and selected cuSOLVER/cuBLAS linear algebra paths exist. HIP/ROCm
is still a feature stub and is not a supported execution path.

See also:

- `tenferro-tensor/src/cubecl/` for the implementation,
- `AGENTS.md` for the current GPU status and local test command,
- [backend-contract.md](../spec/backend-contract.md) for placement rules,
- [tensor-prims.md](./tensor-prims.md) for tensor operation families.

---

## Current Module Structure

```text
tenferro-tensor/src/cubecl/
    mod.rs                 CubeclBackend and TensorBackend implementation
    runtime.rs             CubeCL/CUDA runtime initialization and stream access
    memory.rs              upload_tensor, download_tensor, device pointer bridge
    dispatch.rs            shared launch helpers and dtype dispatch
    kernels/               CubeCL kernels for elementwise, reductions, indexing,
                            diagonal, and structural ops
    fusion/                fused elementwise classification and code generation
    gemm.rs                cuTENSOR/cuBLAS-backed contraction support
    linalg.rs              cuSOLVER/cuBLAS-backed linalg support
    ffi/                   runtime-loaded CUDA library bindings
    tests/                 ignored GPU tests
```

The backend type is `CubeclBackend`. There are no separate in-tree
`CudaBackend` and `RocmBackend` implementations. CUDA is selected by enabling
the `cubecl` feature, which depends on the workspace-pinned CubeCL fork and the
CubeCL CUDA runtime.

## Dependency Source

The workspace intentionally depends on the `shinaoka/cubecl` fork:

```toml
cubecl = { git = "https://github.com/shinaoka/cubecl.git", rev = "929c8a96", features = ["cuda"] }
cubecl-cuda = { git = "https://github.com/shinaoka/cubecl.git", rev = "929c8a96" }
cubecl-runtime = { git = "https://github.com/shinaoka/cubecl.git", rev = "929c8a96" }
```

Keep this fork dependency until upstream CubeCL has the required support and the
workspace is deliberately migrated. Do not replace it with crates.io CubeCL as
part of unrelated GPU or documentation work.

## Runtime And Library Loading

`CubeclRuntime::new(device_ordinal)` initializes CUDA and creates the CubeCL
CUDA client for one device. GPU kernels are JIT-compiled by CubeCL, so local
CUDA toolkit configuration matters.

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

## Device Transfer Policy

tenferro follows the PyTorch convention: no implicit CPU/GPU transfer at tensor
API boundaries. Callers upload tensors before GPU backend operations and
download results explicitly when host access is needed.

```rust,ignore
use tenferro_tensor::cubecl::{download_tensor, upload_tensor, CubeclBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let mut backend = CubeclBackend::new(0)?;
let a = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
let b = Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]);

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
| CPU op receives a GPU tensor | panic, treated as a programming error |
| `TypedTensor::host_data()` on a GPU buffer | panic with a diagnostic |

## Implemented Coverage

The CubeCL backend implements enough of `TensorBackend` to run a growing subset
of compiled tensor programs on CUDA. Coverage includes:

| Category | Current status |
| --- | --- |
| Allocation/transfer | CUDA allocation, upload, download, raw pointer bridge |
| Elementwise | many float ops; selected complex ops; unsupported complex analytic ops return `BackendFailure` |
| Reductions | sum/prod for real and complex; min/max for real |
| Structural | transpose, reshape, broadcast, reverse, concatenate, diagonal extraction/embedding, triangular masks |
| Indexing | selected gather/scatter/slice/pad/reverse paths through CubeCL kernels |
| Contraction | cuTENSOR/cuBLAS-backed paths for supported real dtypes |
| Linalg | cuSOLVER/cuBLAS-backed SVD, QR, Cholesky, LU, Eigh, and triangular solve where supported |

General eigendecomposition (`eig`, LAPACK `dgeev` style) is not provided by
cuSOLVER. `CubeclBackend::eig` returns `BackendFailure`; users must explicitly
download to CPU and call the CPU backend.

Complex CubeCL support is intentionally not expanded in this batch because the
required CubeCL support is being handled upstream.

## Unsupported And Deferred Work

The following are intentionally outside the current batch:

- GPU benchmark work,
- HIP/ROCm implementation,
- replacing the CubeCL fork,
- broad complex kernel expansion before upstream CubeCL support lands,
- making every CPU tensor op have a GPU implementation,
- changing the public placement contract.

## Tests

GPU tests are ignored so regular CPU-only test runs remain portable. Run them on
a CUDA machine with:

```sh
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-tensor --features cubecl -- --ignored
```

These tests are correctness tests, not benchmarks.
