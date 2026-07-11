# Troubleshooting

## CUDA Library Load Failures

If a CUDA run fails while loading cuTENSOR, cuSOLVER, or cuBLAS, first check
that the CUDA runtime libraries are on the dynamic-linker path:

```bash
CUDA_PATH=/usr/local/cuda-12.8
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

For non-standard installs, set the exact library paths:

```bash
TENFERRO_CUTENSOR_PATH=/opt/cuda/lib64/libcutensor.so.2
TENFERRO_CUSOLVER_PATH=/opt/cuda/lib64/libcusolver.so.12
TENFERRO_CUBLAS_PATH=/opt/cuda/lib64/libcublas.so.12
```

## Expected GPU Tensor

An error like `expected GPU tensor ... use upload_tensor()` means a CUDA
backend operation received CPU data. Upload first:

```rust
use tenferro_gpu::{upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let backend = CudaBackend::new(0).unwrap();
let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
let gpu_x = upload_tensor(backend.runtime(), &x).unwrap();
assert_eq!(gpu_x.shape(), &[2]);
```

## Host Access to GPU Tensors

Host access methods read CPU memory. If a tensor lives on CUDA memory, download
it before inspecting values:

```rust
use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let backend = CudaBackend::new(0).unwrap();
let x = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]);
let gpu_x = upload_tensor(backend.runtime(), &x).unwrap();
let cpu_x = download_tensor(backend.runtime(), &gpu_x).unwrap();
assert_eq!(cpu_x.as_slice::<f64>().unwrap(), &[3.0]);
```

Compacting a view does not change that transfer rule. Host views compact to
host tensors; CUDA views compact on CUDA when the backend supports that path.
Neither path silently moves tensor data between CPU and GPU.

## Dtype Mismatch

Typed accessors must match the tensor dtype. If `as_slice::<f64>()` fails,
check whether the tensor was created from `f32`, complex values, or another
supported scalar type.

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

`cpu-tblis` is an optional runtime-loaded `dot_general` contraction provider.
It is additive to `cpu-faer`/`cpu-blas`; select the base provider with
`CpuBackendKind` and opt into TBLIS attempts with `DotGeneralProvider`.
Use `cpu-tblis-linked` when the build should link the bundled/source TBLIS path.

`CpuBackend::new()` selects the compiled default provider: BLAS when `cpu-blas`
is compiled, otherwise faer. Use `CpuBackend::with_kind(CpuBackendKind::Faer)`
when faer should handle provider-backed kernels in a build that includes faer.
See
[Parallelism and Caching](parallelism-and-caching.md) for thread-count and
cache-retention controls.
