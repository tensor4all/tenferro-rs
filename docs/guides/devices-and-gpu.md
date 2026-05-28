# Devices and GPU

tenferro follows the PyTorch convention: no implicit CPU/GPU transfer. A tensor
must already live on the device required by the backend operation.

CUDA is a backend/device axis, not a separate tensor layer. The same concrete,
eager, and traced surfaces can run supported CUDA operations when tensors are
explicitly uploaded to CUDA memory and the executor/backend is CUDA-backed.

CUDA support targets NVIDIA CUDA. AMD/ROCm is not a
supported execution path yet.

## Transfer Model

| Boundary | What happens |
| --- | --- |
| CPU tensor to CUDA backend | Upload first with `tenferro_gpu::cubecl::upload_tensor` |
| CUDA tensor to CUDA backend | Runs on CUDA for supported op/dtype combinations |
| CUDA tensor to CPU backend | `Result`-returning CPU backend ops fail; download first |
| CUDA tensor to host inspection | Direct host slice APIs panic; download first |
| Unsupported CUDA op or dtype | Error, not silent CPU fallback |

Keep tensors on CUDA across a CUDA workload. Download only when the host needs
to inspect values or hand data to CPU-only code.

View canonicalization follows the same rule. A CUDA backend may compact a CUDA
view into CUDA memory, and host code may compact a host view into host memory,
but tenferro does not use canonicalization as a hidden CPU/GPU transfer.

## Eager GPU Synchronization

Eager CUDA execution submits work immediately and returns a CUDA-resident
`Tensor` handle. It does not expose a user-visible ready flag, and normal
kernel launches do not imply host synchronization after every op. Subsequent
CUDA ops can consume the returned handle on the same backend stream.

The host waits when a value is downloaded or otherwise inspected on the host.
Some library-backed operations also synchronize internally when they must read
device-side status.

For a time-axis diagram, see [Execution Models](execution-models.md).

## CUDA Quickstart

<!-- snippet-source: tenferro-gpu/examples/cuda_quickstart.rs -->
```rust
use tenferro_gpu::{download_tensor, upload_tensor, CubeclBackend as CudaBackend};
use tenferro_tensor::{Tensor, TensorElementwise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !tenferro_gpu::gpu_available() {
        return Ok(());
    }

    let mut backend = CudaBackend::new(0)?;
    let cpu_a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let cpu_b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);

    let gpu_a = upload_tensor(backend.runtime(), &cpu_a)?;
    let gpu_b = upload_tensor(backend.runtime(), &cpu_b)?;
    let gpu_c = backend.add(&gpu_a, &gpu_b)?;
    let cpu_c = download_tensor(backend.runtime(), &gpu_c)?;

    assert_eq!(cpu_c.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    Ok(())
}
```
<!-- end-snippet-source -->

Compile-check the example without requiring a GPU:

```bash
cargo check -p tenferro-gpu --features cuda --example cuda_quickstart
```

Run it on a configured CUDA machine:

```bash
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
  cargo run -p tenferro-gpu --features cuda --example cuda_quickstart
```

The example downloads the result back to CPU and asserts the expected values.

## CUDA Across Tensor Layers

| Tensor model | How CUDA fits |
| --- | --- |
| `TypedTensor<T, R>` | Fixed-dtype runtime tensor with optional compile-time rank; storage may be host-backed or backend-backed, and host access still requires explicit download for CUDA buffers. |
| `Tensor` | Main concrete CUDA value for backend execution |
| `EagerTensor` | Wraps CUDA-resident `Tensor` values when using an `EagerRuntime` with `CudaBackend` |
| `TracedTensor` | Graphs can be executed by `GraphExecutor<CudaBackend>` for supported ops |

CUDA coverage is about backend dispatch. It is not the same as AD coverage.

## Coverage

The CUDA backend uses the same concrete, eager, and traced tensor surfaces as
the CPU backend. The table below describes the current CUDA backend dispatch
coverage for CUDA-resident `Tensor` values. It is not an autodiff coverage table.

Legend:

- `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, and `C64` are the current public `Tensor` dtypes.
- Listed dtypes have CUDA implementations for that operation.
- Missing dtypes or rows marked "No CUDA implementation" return an error
  rather than silently falling back to CPU.

| Operation or family | CUDA dtype support | Notes |
| --- | --- | --- |
| Allocation, upload, download | `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | Explicit CPU/GPU transfer only |
| `add`, `mul`, `div` | `F32`, `F64`, `C32`, `C64` | Same dtype inputs only; integer and `Bool` arithmetic are not implemented |
| `neg` | `F32`, `F64`, `C32`, `C64` | Integer and `Bool` negation are not implemented |
| `conj` | `F32`, `F64`, `C32`, `C64` | Real floating dtypes are identity; integer and `Bool` inputs are not implemented |
| `abs`, `sign` | `F32`, `F64` | Complex, integer, and `Bool` inputs are not implemented |
| `maximum`, `minimum`, `compare`, `select`, `clamp` | `F32`, `F64` | Complex ordering is not defined; `compare` returns a `Bool` tensor and `select` takes a `Bool` predicate |
| `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, `expm1`, `log1p` | `F32`, `F64` | Complex analytic kernels are not implemented |
| `pow` | `F32`, `F64` | Same dtype inputs only |
| `reshape` | `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | Metadata-only shape change |
| `transpose`, `broadcast_in_dim`, `extract_diagonal`, `embed_diagonal`, `tril`, `triu` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | Structural tensor operations; `Bool` is not implemented |
| `convert` | `F32`, `F64`, `C32`, `C64` among those dtypes; `I32`, `I64`, and `Bool` identity only | Conversion to or from integer or `Bool` dtypes is not implemented except identity |
| `reduce_sum`, `reduce_prod` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | Multi-axis reductions are composed from single-axis kernels; `Bool` is not implemented |
| `reduce_max`, `reduce_min` | `F32`, `F64` | Complex ordering is not defined; integer and `Bool` min/max are not implemented |
| `dot_general` | `F32`, `F64`, `C32`, `C64` | cuTENSOR-backed contraction; same dtype inputs only |
| `gather` | operand `F32`, `F64`, `I32`, `C32`, `C64`; indices `F32`, `F64`, `I32`, or `I64` | Complex and `Bool` index tensors; `I64` and `Bool` operands are not implemented |
| `scatter` | operand/update `F32`, `F64`, `C32`, `C64`; indices `F32`, `F64`, `I32`, or `I64` | Add-scatter semantics; complex and `Bool` index tensors and integer/`Bool` operands are not implemented |
| `slice`, `pad`, `concatenate`, `reverse` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | Dense structural/indexing operations; `Bool` is not implemented |
| `dynamic_slice` | input `F32`, `F64`, `I32`, `C32`, `C64`; starts `F32`, `F64`, `I32`, or `I64` | Complex and `Bool` start tensors; `I64` and `Bool` inputs are not implemented |
| `dynamic_update_slice` | No CUDA implementation | Returns an error |
| `cholesky`, `triangular_solve`, `lu`, `svd`, `qr`, `eigh`, `solve` | `F32`, `F64`, `C32`, `C64` | cuSOLVER/cuBLAS-backed; integer and `Bool` dtypes are not implemented |
| `full_piv_lu`, `full_piv_lu_solve` | No CUDA implementation | Returns an error |
| General `eig` | No CUDA implementation | cuSOLVER does not provide LAPACK `dgeev`-style general eigendecomposition; download to CPU explicitly |
| AMD/ROCm | No supported backend | ROCm remains a feature stub |

If cuTENSOR, cuSOLVER, or cuBLAS are installed outside normal dynamic-linker
paths, set `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH`, or
`TENFERRO_CUBLAS_PATH`.
