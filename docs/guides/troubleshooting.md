# Troubleshooting

## CUDA Library Load Failures

If a CUDA run fails while loading cuTENSOR, cuSOLVER, or cuBLAS, first check
that the CUDA runtime libraries are on the dynamic-linker path:

```bash
CUDA_PATH=/usr/local/cuda-12.0
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
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
use tenferro::cuda::{upload_tensor, CudaBackend};
use tenferro::{Tensor, TensorBackend};

let backend = CudaBackend::new(0).unwrap();
let x = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
let gpu_x = upload_tensor(backend.runtime(), &x).unwrap();
assert_eq!(gpu_x.shape(), &[2]);
```

## Host Access to GPU Tensors

Host access methods read CPU memory. If a tensor lives on CUDA memory, download
it before inspecting values:

```rust
use tenferro::cuda::{download_tensor, upload_tensor, CudaBackend};
use tenferro::{Tensor, TensorBackend};

let backend = CudaBackend::new(0).unwrap();
let x = Tensor::from_vec(vec![1], vec![3.0_f64]);
let gpu_x = upload_tensor(backend.runtime(), &x).unwrap();
let cpu_x = download_tensor(backend.runtime(), &gpu_x).unwrap();
assert_eq!(cpu_x.as_slice::<f64>().unwrap(), &[3.0]);
```

## Dtype Mismatch

Typed accessors must match the tensor dtype. If `as_slice::<f64>()` fails,
check whether the tensor was created from `f32`, complex values, or another
supported scalar type.

## Column-Major and Row-Major Confusion

`Tensor::from_vec` reads flat data as column-major. When porting PyTorch,
NumPy, or JAX examples that use row-major flat data, convert the data to
column-major before construction. If another library expects row-major output,
convert the exported `try_into_vec::<T>()` buffer at that boundary.
See [Memory Order](memory-order.md).

## CPU Backend Feature Conflicts

Exactly one CPU backend must be enabled. Use the default `cpu-faer` feature, or
disable defaults and enable one BLAS option:

```toml
[dependencies]
tenferro = { path = "/path/to/tenferro-rs/tenferro", default-features = false, features = ["cpu-blas"] }
```

Do not enable both `cpu-faer` and `cpu-blas`.
