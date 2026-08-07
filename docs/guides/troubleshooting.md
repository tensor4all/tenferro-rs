# Troubleshooting

## CUDA Library Load Failures

If a CUDA run fails while loading cuTENSOR, cuSOLVER, or cuBLAS, first check
that the CUDA runtime libraries are on the dynamic-linker path:

```bash
CUDA_PATH=/usr/local/cuda-12.4
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

CUDA 12.4 is the minimum runtime. Use a CUDA 12.8-or-newer driver and NVRTC to
enable the complete CubeCL capability set.

For non-standard installs, set the exact library paths:

```bash
TENFERRO_CUTENSOR_PATH=/opt/cuda/lib64/libcutensor.so.2
TENFERRO_CUSOLVER_PATH=/opt/cuda/lib64/libcusolver.so.12
TENFERRO_CUBLAS_PATH=/opt/cuda/lib64/libcublas.so.12
```

## Unsupported PTX Version

`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` usually means the loaded NVRTC is newer
than the host driver API. Check both versions with `nvidia-smi` and the NVRTC
library selected by `LD_LIBRARY_PATH`. On a driver reporting CUDA 12.4, load
NVRTC 12.4; on a driver reporting CUDA 12.8 or newer, load NVRTC 12.8 to retain
the full CubeCL feature set.

## Expected GPU Tensor

An error like `expected GPU tensor ... use upload_tensor()` means a CUDA
backend operation received CPU data. Upload first:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#troubleshooting_9 -->
```rust
use tenferro_gpu::cuda::{cuda_devices, gpu_available, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

if !gpu_available() {
    return Ok(());
}
let devices = cuda_devices()?;
let device = devices.first().ok_or("no CUDA device is visible")?;
let backend = CudaBackend::new(device.id())?;
let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
let gpu_x = upload_tensor(backend.runtime(), &x)?;
assert_eq!(gpu_x.shape(), &[2]);
```
<!-- end-snippet-source -->

## Host Access to GPU Tensors

Host access methods read CPU memory. If a tensor lives on CUDA memory, download
it before inspecting values:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#troubleshooting_10 -->
```rust
use tenferro_gpu::cuda::{cuda_devices, download_tensor, gpu_available, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

if !gpu_available() {
    return Ok(());
}
let devices = cuda_devices()?;
let device = devices.first().ok_or("no CUDA device is visible")?;
let backend = CudaBackend::new(device.id())?;
let x = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
let gpu_x = upload_tensor(backend.runtime(), &x)?;
let cpu_x = download_tensor(backend.runtime(), &gpu_x)?;
assert_eq!(cpu_x.as_slice::<f64>()?, &[3.0]);
```
<!-- end-snippet-source -->

Compacting a view does not change that transfer rule. Host views compact to
host tensors; CUDA views compact on CUDA when the backend supports that path.
Neither path silently moves tensor data between CPU and GPU.

## Dtype Mismatch

Typed accessors must match the tensor dtype. If `as_slice::<f64>()` fails,
check whether the tensor was created from `f32`, complex values, or another
supported scalar type.

## view is not slice-contiguous

A view cannot be exposed as a borrowed slice unless its layout is contiguous.
Materialize a compact owner with `to_contiguous` before requesting the slice:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#troubleshooting_11 -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::TensorViewCanonicalization;
use tenferro_tensor::TypedTensor;

let source = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let transposed = source.as_view().transpose_view([1, 0])?;
let error = transposed.as_slice().unwrap_err();
assert!(error
    .to_string()
    .contains("view is not contiguous column-major"));

let mut backend = CpuBackend::new();
let compact = backend.to_contiguous(&transposed)?;
assert_eq!(compact.as_slice()?, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```
<!-- end-snippet-source -->

## mutable tensor layout may overlap physical elements

Mutable views must not alias one physical element from multiple logical
positions. Materialize or construct a contiguous owner before requesting
mutable access:

<!-- snippet-source: docs/tutorial-code/src/bin/execution_snippets.rs#troubleshooting_12 -->
```rust
use tenferro_tensor::{TypedTensor, TypedTensorViewMut};

let mut data = [1.0_f64, 2.0];
let error = TypedTensorViewMut::from_slice(vec![2], vec![0], 0, &mut data).unwrap_err();
assert!(error
    .to_string()
    .contains("mutable tensor layout may overlap physical elements"));

let mut owner = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
{
    let mut view = owner.as_view_mut();
    *view.get_mut(&[0]).ok_or("index in contiguous owner")? = 3.0;
}
assert_eq!(owner.as_slice()?, &[3.0, 2.0]);
```
<!-- end-snippet-source -->

## Column-Major and Row-Major Confusion

`Tensor::from_vec_col_major` expects tenferro's physical column-major order.
When porting PyTorch, NumPy, or JAX examples that use row-major flat data,
explicitly reorder the buffer at the import boundary. Export with
`into_vec_col_major::<T>()`; consumers that require another order should convert
outside tenferro.
See [Memory Order](memory-order.md).

## CPU Backend Feature Selection

At least one CPU fallback/linalg backend feature must be enabled. `cpu-faer`
is the default, and `cpu-blas` can be enabled by itself or together with
`cpu-faer`:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", features = ["cpu-blas"] }
```

`cpu-blas` is the generic CBLAS/LAPACK backend. If the build should select a
concrete provider from Cargo features, enable exactly one of `blas-openblas`,
`blas-accelerate`, or `blas-mkl` on the CPU-using tenferro crates:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/crates/tenferro-runtime", default-features = false, features = ["blas-openblas"] }
tenferro-cpu = { path = "/path/to/tenferro-rs/crates/tenferro-cpu", default-features = false, features = ["blas-openblas"] }
```

Cargo features are additive. If two explicit provider features are enabled by
different dependencies, tenferro stops at compile time instead of linking an
ambiguous BLAS/LAPACK provider set. Use `OPENBLAS_LIB_DIR` for non-standard
OpenBLAS installs, and `MKLROOT` or `MKL_LIB_DIR` for non-standard MKL installs
when the provider build scripts need a library path.

TBLIS is not a `tenferro-cpu` feature. The in-repository
`ext/tenferro-cpu-tblis` crate is an unpublished external-provider example for
overriding only the general `dot_general` provider slot. Keep `cpu-faer` or
`cpu-blas` enabled for the complete CPU backend and install the TBLIS provider
through `CpuProviderBundleBuilder` when experimenting with that route.

`CpuBackend::new()` selects the compiled default provider: BLAS when `cpu-blas`
is compiled, otherwise faer. Use `CpuBackend::with_kind(CpuBackendKind::Faer)`
when faer should handle provider-backed kernels in a build that includes faer.
See
[Parallelism and Caching](parallelism-and-caching.md) for thread-count and
cache-retention controls.
