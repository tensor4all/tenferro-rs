# Devices and GPU

tenferro follows the PyTorch convention: no implicit CPU/GPU transfer. A tensor
must already live on the device required by the backend operation.

CUDA and WebGPU are backend/device choices, not separate tensor types. The same
concrete, eager, and traced APIs can run supported GPU operations when tensors
are explicitly uploaded to the selected provider and the executor/backend uses
the same provider.

CUDA support targets NVIDIA CUDA. WebGPU support is experimental and currently
focused on explicit transfer plus limited `dot_general`/einsum coverage.
AMD/ROCm is not a supported execution path yet.

The optional XLA/PJRT path is separate from these tensor backends. It lowers
static-shaped traced programs to StableHLO in `tenferro-xla` and loads PJRT
plugins at runtime. See [XLA and PJRT](xla.md).

## Provider Matrix

| Provider | Status | Feature | Notes |
| --- | --- | --- | --- |
| CPU | Supported | default CPU provider features | Host execution |
| CUDA | Supported | `cuda` | NVIDIA CUDA through CubeCL-CUDA plus CUDA libraries |
| WebGPU | Experimental | `webgpu` | Explicit transfer and limited `dot_general`/einsum coverage |
| ROCm | Not supported for execution | `rocm` reserved | Future compile-only substrate; no runtime quickstart |

## Transfer Model

| Boundary | What happens |
| --- | --- |
| CPU tensor to CUDA backend | Upload first with `tenferro_gpu::upload_tensor` |
| CPU tensor to WebGPU backend | Upload first with `tenferro_gpu::upload_webgpu_tensor` |
| CUDA tensor to CUDA backend | Runs on CUDA for supported op/dtype combinations |
| WebGPU tensor to WebGPU backend | Runs on WebGPU for supported op/dtype combinations |
| CUDA tensor to CPU backend | `Result`-returning CPU backend ops fail; download first |
| WebGPU tensor to CPU backend | `Result`-returning CPU backend ops fail; download first |
| GPU tensor to host inspection | Direct host slice APIs panic; download first |
| Unsupported CUDA op or dtype | Error, not silent CPU fallback |
| Unsupported WebGPU op or dtype | Error, not silent CPU fallback |

Keep tensors on one GPU provider across a GPU workload. Download only when the
host needs to inspect values or hand data to CPU-only code.

View compaction follows the same rule. A CUDA backend may copy a CUDA view into
compact CUDA memory, a WebGPU backend may copy a WebGPU view into compact
WebGPU memory, and host code may copy a host view into compact host memory, but
tenferro does not use that copy as a hidden CPU/GPU transfer.

## Eager GPU Synchronization

Eager GPU execution submits work immediately and returns a provider-resident
`Tensor` handle. Normal kernel launches do not imply host synchronization after
every op. Subsequent GPU ops can consume the returned handle on the same
backend stream or queue.

The host waits when a value is downloaded or otherwise inspected on the host.
Some library-backed operations also synchronize internally when they must read
device-side status.

Use `EagerRuntime::synchronize()` when code needs an explicit host-side barrier
for the eager runtime. CPU runtimes return immediately; CUDA runtimes wait for
the current backend stream, and WebGPU runtimes wait for the WebGPU queue.
Direct GPU backend code can call the provider runtime's `synchronize()` through
`backend.runtime()`.

For a time-axis diagram, see [Execution Models](execution-models.md).

## CUDA Quickstart

<!-- snippet-source: crates/tenferro-gpu/examples/cuda_quickstart.rs -->
```rust
use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::{Tensor, TensorElementwise};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !tenferro_gpu::gpu_available() {
        return Ok(());
    }

    let mut backend = CudaBackend::new(0)?;
    let cpu_a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let cpu_b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

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
LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH \
  cargo run -p tenferro-gpu --features cuda --example cuda_quickstart
```

The example downloads the result back to CPU and asserts the expected values.

Use the installed CUDA root on your machine. If several roots exist, inspect
them first:

```bash
ls -d /usr/local/cuda*
```

If CUDA libraries or cuTENSOR are outside the standard dynamic-linker paths,
set:

```bash
export CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH
export TENFERRO_CUTENSOR_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
export TENFERRO_CUSOLVER_PATH=$CUDA_PATH/lib64/libcusolver.so.12
export TENFERRO_CUBLAS_PATH=$CUDA_PATH/lib64/libcublas.so.12
export CUBECL_DEBUG_LOG=0
```

For XLA/PJRT GPU verification, add the PJRT plugin path separately:

```bash
export TENFERRO_PJRT_GPU_PLUGIN=/path/to/pjrt_c_api_gpu_plugin.so
```

## WebGPU Quickstart

WebGPU is experimental and currently useful for explicit transfer plus
`F32`/`C32` `dot_general` and einsum paths. For a WebGPU-only binary, disable
default features on `tenferro-gpu` unless the same crate also needs the default
CPU provider stack:

```toml
[dependencies]
tenferro-gpu = { version = "...", default-features = false, features = ["webgpu"] }
tenferro-tensor = "..."
```

For a local scratch crate inside the checkout, use matching path dependencies
and add an empty `[workspace]` table as described in
[Getting Started](../getting-started/index.md).

```rust
use tenferro_gpu::{
    download_webgpu_tensor, upload_webgpu_tensor, webgpu_available, WebGpuBackend,
};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorDot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !webgpu_available() {
        return Ok(());
    }

    let mut backend = WebGpuBackend::new_default()?;
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let gpu_lhs = upload_webgpu_tensor(backend.runtime(), &lhs)?;
    let gpu_rhs = upload_webgpu_tensor(backend.runtime(), &rhs)?;
    let gpu_out = backend.dot_general(&gpu_lhs, &gpu_rhs, &config)?;
    let out = download_webgpu_tensor(backend.runtime(), &gpu_out)?;

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f32>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    backend.runtime().synchronize()?;
    Ok(())
}
```

`DotGeneralConfig` uses StableHLO-style dimension-number fields:
`lhs_contracting_dims`, `rhs_contracting_dims`, `lhs_batch_dims`, and
`rhs_batch_dims`. Direct backend code can use the lower-level
`backend.runtime().synchronize()` barrier; eager code can use
`EagerRuntime::synchronize()`.

## CUDA Across Tensor Layers

| Tensor model | How CUDA fits |
| --- | --- |
| `TypedTensor<T, R>` | Fixed-dtype runtime tensor with optional compile-time rank; host access still requires explicit download for CUDA buffers. |
| `Tensor` | Main concrete CUDA value for backend execution |
| `EagerTensor` | Wraps CUDA-resident `Tensor` values when using an `EagerRuntime` with `CudaBackend` |
| `TracedTensor` | Graphs can be executed by `GraphExecutor<CudaBackend>` for supported ops |

CUDA coverage is about backend dispatch. It is not the same as AD coverage.

## Coverage

The CUDA backend uses the same concrete, eager, and traced tensor APIs as
the CPU backend. The generated table below describes the current first-scope
core primitive descriptor for CUDA-resident `Tensor` values. It is not an
autodiff coverage table.

Legend:

- `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, and `C64` are the current public `Tensor` dtypes.
- Native CUDA dtypes have CUDA implementations for that operation.
- Unsupported descriptor dtypes are semantically valid for the core op catalog
  but currently return an error on CUDA.
- Dtypes absent from the row are outside the current semantic dtype policy for
  that operation.

<!-- cuda-core-capability:start -->
| Primitive op | Native CUDA dtypes | Unsupported descriptor dtypes | Output dtype | Native axes |
| --- | --- | --- | --- | --- |
| `add` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `sub` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `mul` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `neg` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `conj` | `F32`, `F64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `div` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `rem` | `F32`, `F64`, `I32`, `I64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `abs` | `F32`, `F64`, `I32`, `I64` | `C32`, `C64` | `F32->F32`, `F64->F64`, `I32->I32`, `I64->I64`, `C32->F32`, `C64->F64` | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `sign` | `F32`, `F64`, `I32`, `I64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `maximum` | `F32`, `F64`, `I32`, `I64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `minimum` | `F32`, `F64`, `I32`, `I64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `compare` | `F32`, `F64`, `I32`, `I64` | `Bool` | `Bool` | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `select` | `F32`, `F64`, `I32`, `I64` | `Bool`, `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `clamp` | `F32`, `F64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `exp` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `log` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `sin` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `cos` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `tanh` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `sqrt` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `rsqrt` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `pow` | `F32`, `F64`, `I32`, `I64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `expm1` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `log1p` | `F32`, `F64` | `C32`, `C64` | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `dot_general` | `F32`, `F64`, `C32`, `C64` | none | same as input | result Native; read Native; write Native; strided Native; accumulation Native |
| `reduce_sum` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `reduce_prod` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `reduce_max` | `F32`, `F64`, `I32`, `I64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `reduce_min` | `F32`, `F64`, `I32`, `I64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
<!-- cuda-core-capability:end -->

Additional CUDA coverage outside the first-scope core descriptor is tracked
below until those operation families are modeled by the backend capability
descriptor.

| Operation or family | CUDA dtype support | Notes |
| --- | --- | --- |
| Allocation, upload, download | `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | Explicit CPU/GPU transfer only |
| `reshape` | `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | Metadata-only shape change |
| `transpose`, `broadcast_in_dim`, `extract_diagonal`, `embed_diagonal`, `tril`, `triu` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | Structural tensor operations; `Bool` is not implemented |
| checked `convert`, explicit `cast` | `F32`, `F64`, `C32`, `C64` among those dtypes; `I32`, `I64`, and `Bool` identity only | `convert` applies the public checked conversion contract before backend dispatch; `cast` is explicit dtype projection. Conversion to or from integer or `Bool` dtypes is not implemented on CUDA except identity |
| `gather` | operand `F32`, `F64`, `I32`, `C32`, `C64`; indices `F32`, `F64`, `I32`, or `I64` | Complex and `Bool` index tensors; `I64` and `Bool` operands are not implemented |
| `scatter` | operand/update `F32`, `F64`, `C32`, `C64`; indices `F32`, `F64`, `I32`, or `I64` | Add-scatter semantics; complex and `Bool` index tensors and integer/`Bool` operands are not implemented |
| `slice`, `pad`, `concatenate`, `reverse` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | Dense structural/indexing operations; `Bool` is not implemented |
| `dynamic_slice` | input `F32`, `F64`, `I32`, `C32`, `C64`; starts `F32`, `F64`, `I32`, or `I64` | Complex and `Bool` start tensors; `I64` and `Bool` inputs are not implemented |
| `dynamic_update_slice` | No CUDA implementation | Returns an error |
| `cholesky`, `triangular_solve`, `lu`, `svd`, `qr`, `eigh`, `solve` | `F32`, `F64`, `C32`, `C64` | cuSOLVER/cuBLAS-backed; integer and `Bool` dtypes are not implemented |
| `full_piv_lu`, `full_piv_lu_solve` | No CUDA implementation | Returns an error |
| General `eig` | No CUDA implementation | cuSOLVER does not provide LAPACK `dgeev`-style general eigendecomposition; download to CPU explicitly |
| AMD/ROCm | No supported execution backend | ROCm remains reserved for future loader-backed work |

## WebGPU Coverage

The WebGPU backend is experimental. It uses the same tensor APIs as CUDA and
CPU, but its operation coverage is intentionally much narrower. Unsupported
rows return explicit errors and do not fall back to CPU.

| Operation or family | WebGPU dtype support | Notes |
| --- | --- | --- |
| Allocation, upload, download | `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | Explicit CPU/WebGPU transfer only |
| `dot_general`, `dot_general_with_conj` | `F32`, `C32` | CubeK-backed BGEMM planner; `C32` conjugation is handled by the CubeK complex GEMM API. Supports rank-2, batched, and same-device packed operand layouts covered by tests |
| Binary einsum lowering to `dot_general` | `F32`, `C32` | Eager `F32`/`C32` and traced `F32` paths are covered when inputs are explicitly uploaded to WebGPU |
| `dot_general` with zero contracting size | No WebGPU implementation | Returns an error until CubeK behavior is validated |
| `dot_general` for `F64`, `C64` | No WebGPU implementation | Returns an error; no CPU fallback |
| Elementwise and analytic ops | No WebGPU implementation | Returns an error |
| Reductions | No WebGPU implementation | Returns an error |
| Structural/indexing ops beyond transfer-owned allocation metadata | No WebGPU implementation | Returns an error |
| Linalg | No WebGPU implementation | Returns an error |
| ROCm | No supported execution backend | No ROCm quickstart is provided |

If cuTENSOR, cuSOLVER, or cuBLAS are installed outside normal dynamic-linker
paths, set `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH`, or
`TENFERRO_CUBLAS_PATH`.
