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
use tenferro_gpu::cubecl::{upload_tensor, CubeclBackend as CudaBackend};
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
use tenferro_gpu::cubecl::{download_tensor, upload_tensor, CubeclBackend as CudaBackend};
use tenferro_tensor::{Tensor, TensorBackend};

let backend = CudaBackend::new(0).unwrap();
let x = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]);
let gpu_x = upload_tensor(backend.runtime(), &x).unwrap();
let cpu_x = download_tensor(backend.runtime(), &gpu_x).unwrap();
assert_eq!(cpu_x.as_slice::<f64>().unwrap(), &[3.0]);
```

## Dtype Mismatch

Typed accessors must match the tensor dtype. If `as_slice::<f64>()` fails,
check whether the tensor was created from `f32`, complex values, or another
supported scalar type.

## Column-Major and Row-Major Confusion

`Tensor::from_vec_col_major` expects tenferro's physical column-major order.
When porting PyTorch, NumPy, or JAX examples that use row-major flat data, use
`Tensor::from_vec_row_major` at the import boundary. If another library expects
row-major output, export with `try_into_vec_row_major::<T>()`; use
`try_into_vec_col_major::<T>()` when the consumer expects tenferro's physical
order.
See [Memory Order](memory-order.md).

## CPU Backend Feature Selection

At least one CPU backend feature must be enabled. `cpu-faer` is the default,
and `cpu-blas` can be enabled by itself or together with `cpu-faer`:

```toml
[dependencies]
tenferro-runtime = { path = "/path/to/tenferro-rs/tenferro-runtime", features = ["cpu-blas"] }
```

When both are compiled, `CpuBackend::new()` selects faer. Use
`CpuBackend::with_kind(CpuBackendKind::Blas)` when a linked BLAS/LAPACK
provider should handle provider-backed kernels. See
[Parallelism and Caching](parallelism-and-caching.md) for thread-count and
cache-retention controls.
