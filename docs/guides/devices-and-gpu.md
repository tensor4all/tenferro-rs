# Devices and GPU

tenferro follows the PyTorch convention: no implicit CPU/GPU transfer. Upload
CPU tensors before CUDA backend operations and download results before host
inspection.

CUDA support targets NVIDIA CUDA through the CubeCL backend. AMD/ROCm is not a
supported execution path yet.

## CUDA Quickstart

<!-- snippet-source: tenferro/examples/cuda_quickstart.rs -->
```rust
use tenferro::cuda::{download_tensor, upload_tensor, CudaBackend};
use tenferro::{Tensor, TensorBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CudaBackend::new(0)?;

    let a = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);

    let gpu_a = upload_tensor(backend.runtime(), &a)?;
    let gpu_b = upload_tensor(backend.runtime(), &b)?;
    let gpu_c = backend.add(&gpu_a, &gpu_b)?;
    let c = download_tensor(backend.runtime(), &gpu_c)?;

    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

Compile-check the example without requiring a GPU:

```bash
cargo check -p tenferro --features cuda --example cuda_quickstart
```

Run it on a configured CUDA machine:

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH \
  cargo run -p tenferro --features cuda --example cuda_quickstart
```

The example downloads the result back to CPU and asserts the expected values.

## Coverage

The CUDA backend uses the same concrete, eager, and traced tensor surfaces as
the CPU backend. The table below describes the current CUDA backend dispatch
coverage for CUDA-resident `Tensor` values. It is not an autodiff coverage table.

Legend:

- `F32`, `F64`, `I64`, `C32`, and `C64` are the current public `Tensor` dtypes.
- Listed dtypes have CUDA implementations for that operation.
- Missing dtypes or rows marked "No CUDA implementation" return an error
  rather than silently falling back to CPU.

| Operation or family | CUDA dtype support | Notes |
| --- | --- | --- |
| Allocation, upload, download | `F32`, `F64`, `I64`, `C32`, `C64` | Explicit CPU/GPU transfer only |
| `add`, `mul`, `div` | `F32`, `F64`, `C32`, `C64` | Same dtype inputs only; `I64` arithmetic is not implemented |
| `neg` | `F32`, `F64`, `C32`, `C64` | `I64` is not implemented |
| `conj` | `F32`, `F64`, `C32`, `C64` | Real dtypes are identity; `I64` is not implemented |
| `abs`, `sign` | `F32`, `F64` | Complex and `I64` inputs are not implemented |
| `maximum`, `minimum`, `compare`, `select`, `clamp` | `F32`, `F64` | Complex ordering is not defined; `compare` returns a numeric 0/1 tensor |
| `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, `expm1`, `log1p` | `F32`, `F64` | Complex analytic kernels are not implemented |
| `pow` | `F32`, `F64` | Same dtype inputs only |
| `transpose`, `reshape`, `broadcast_in_dim`, `extract_diagonal`, `embed_diagonal`, `tril`, `triu` | `F32`, `F64`, `I64`, `C32`, `C64` | Structural tensor operations |
| `convert` | `F32`, `F64`, `C32`, `C64` among those dtypes; `I64` identity only | Conversion to or from `I64` is not implemented except `I64 -> I64` |
| `reduce_sum`, `reduce_prod` | `F32`, `F64`, `I64`, `C32`, `C64` | Multi-axis reductions are composed from single-axis kernels |
| `reduce_max`, `reduce_min` | `F32`, `F64` | Complex ordering is not defined; `I64` min/max is not implemented |
| `dot_general` | `F32`, `F64`, `C32`, `C64` | cuTENSOR-backed contraction; same dtype inputs only |
| `gather` | operand `F32`, `F64`, `C32`, `C64`; indices `F32`, `F64`, or `I64` | Complex index tensors and `I64` operands are not implemented |
| `scatter` | operand/update `F32`, `F64`, `C32`, `C64`; indices `F32`, `F64`, or `I64` | Add-scatter semantics; complex index tensors and `I64` operands are not implemented |
| `slice`, `pad`, `concatenate`, `reverse` | `F32`, `F64`, `I64`, `C32`, `C64` | Dense structural/indexing operations |
| `dynamic_slice` | input `F32`, `F64`, `C32`, `C64`; starts `F32`, `F64`, or `I64` | Complex start tensors and `I64` inputs are not implemented |
| `dynamic_update_slice` | No CUDA implementation | Returns an error |
| `cholesky`, `triangular_solve`, `lu`, `svd`, `qr`, `eigh`, `solve` | `F32`, `F64`, `C32`, `C64` | cuSOLVER/cuBLAS-backed; `svd` and `eigh` return real singular/eigenvalue tensors for complex inputs |
| `full_piv_lu`, `full_piv_lu_solve` | No CUDA implementation | Returns an error |
| General `eig` | No CUDA implementation | cuSOLVER does not provide LAPACK `dgeev`-style general eigendecomposition; download to CPU explicitly |
| AMD/ROCm | No supported backend | ROCm remains a feature stub |

If cuTENSOR, cuSOLVER, or cuBLAS are installed outside normal dynamic-linker
paths, set `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH`, or
`TENFERRO_CUBLAS_PATH`.
