# Devices and GPU

tenferro follows the PyTorch convention: no implicit CPU/GPU transfer. A tensor
must already live on the device required by the backend operation.

CUDA is a backend/device choice, not a separate tensor type. The same
concrete, eager, and traced APIs can run supported CUDA operations when tensors
are explicitly uploaded and execution uses the owning CUDA engine.

CUDA support targets NVIDIA CUDA. WebGPU, Metal execution, and the Apple shared
CPU/Metal path are not provided today. `tenferro-gpu` has no `webgpu` feature or
WebGPU upload API. AMD/ROCm is not a supported execution path either; `rocm` is
an empty reserved feature.

The optional XLA/PJRT path is separate from these tensor backends. It lowers
static-shaped traced programs to StableHLO in `tenferro-xla` and loads PJRT
plugins at runtime. See [XLA and PJRT](xla.md).

For the CUDA execution shapes — one caller-held engine per device, and pinned
host↔device transfers whose completion the caller decides when to wait for —
see [CUDA engines and pinned transfers](engines-and-transfers.md).

## Provider Matrix

| Provider | Status | Feature | Notes |
| --- | --- | --- | --- |
| CPU | Supported | default CPU provider features | Host execution |
| CUDA | Supported | `cuda` | Native NVIDIA CUDA through the `t4a-cuda-kernels` catalogue plus CUDA libraries |
| WebGPU / Metal / Apple shared CPU/Metal | Not provided | None | No backend or transfer quickstart |
| ROCm | Not supported for execution | `rocm` reserved | Empty feature; no runtime quickstart |

## Transfer Model

| Boundary | What happens |
| --- | --- |
| CPU tensor to CUDA backend | Upload first with `tenferro_gpu::cuda::upload_tensor` |
| CUDA tensor to CUDA backend | Runs on CUDA for supported op/dtype combinations |
| CUDA tensor to CPU backend | `Result`-returning CPU backend ops fail; download first |
| GPU tensor to host inspection | Direct host slice APIs return `Error::RuntimeState`; download first |
| Unsupported CUDA op or dtype | Error, not silent CPU fallback |

Keep tensors on one GPU provider across a GPU workload. Download only when the
host needs to inspect values or hand data to CPU-only code.

View compaction follows the same rule. A CUDA backend may copy a CUDA view into
compact CUDA memory, and host code may copy a host view into compact host
memory, but tenferro does not use that copy as a hidden CPU/GPU transfer.

CUDA transfers make one host-side copy at most. `upload_tensor` borrows the
host tensor, so the runtime copies it once into a staging buffer before queueing the
device write; the call returns without waiting, and the host tensor can be
changed or dropped immediately. `download_tensor` has the driver copy straight
into the vector the returned host tensor owns and waits for that copy.
`download_tensor_into` is the blocking alternative for repeated readback: it
copies into a caller-owned `Vec<T>` and hands it back, so a loop that reads the
same size back reuses one pageable allocation instead of paying first-touch
page faults per call (a `bool` destination is rejected; use `download_tensor`
to decode). For a one-off download the pageable routes allocate fresh host
memory; the pinned and asynchronous routes (`upload_pending`/`download_pending`
with a reusable `PinnedHostBuffer`) avoid the pageable transport for large
payloads when the caller can wait later.

## Eager GPU Synchronization

Eager GPU execution submits work immediately and returns a provider-resident
`Tensor` handle. Normal kernel launches do not imply host synchronization after
every op. Subsequent GPU ops can consume the returned handle on the same
backend stream or queue.

The host waits when a value is downloaded or otherwise inspected on the host.
Some library-backed operations also synchronize internally when they must read
device-side status.

Use `CudaRuntime::synchronize()` through the caller-held engine's
`runtime()` when code needs an explicit host-side barrier for the CUDA stream.

For a time-axis diagram, see [Execution Models](execution-models.md).

## CUDA Quickstart

<!-- snippet-source: crates/tenferro-gpu/examples/cuda_quickstart.rs -->
```rust
use tenferro_gpu::{
    cuda::cuda_devices, cuda::download_tensor, cuda::upload_tensor, cuda::CudaEngine,
};
use tenferro_tensor::{Tensor, TensorElementwise, TensorRead, TensorStructural, TensorWrite};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(device) = cuda_devices()?.into_iter().next() else {
        return Ok(());
    };

    let mut backend = CudaEngine::open(device.id())?;
    let cpu_a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let cpu_b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let gpu_a = upload_tensor(backend.runtime(), &cpu_a)?;
    let gpu_b = upload_tensor(backend.runtime(), &cpu_b)?;
    let gpu_c = backend.add_read(
        TensorRead::from_tensor(&gpu_a),
        TensorRead::from_tensor(&gpu_b),
    )?;
    let cpu_c = download_tensor(backend.runtime(), &gpu_c)?;

    assert_eq!(cpu_c.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

    let mut gpu_reuse = upload_tensor(backend.runtime(), &cpu_c)?;
    backend.copy_read_into(
        TensorRead::from_tensor(&gpu_c),
        TensorWrite::from_tensor(&mut gpu_reuse),
    )?;
    let copied = download_tensor(backend.runtime(), &gpu_reuse)?;
    assert_eq!(copied.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
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
CUDA_PATH=/usr/local/cuda-12.6 \
LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH \
  cargo run -p tenferro-gpu --features cuda --example cuda_quickstart
```

The runnable tutorial `cuda_tutorial` follows the same minimal workflow
(upload -> one engine call -> download) and asserts `[4.0, 6.0]`. It is
compiled and archived on the non-GPU CUDA lane, then executed with
`TENFERRO_REQUIRE_CUDA=1` on the trusted GPU lane and the supported legacy
manual GPU lane. Ordinary CPU tutorial CI never requires a CUDA device; the
CUDA tutorial is the hardware-gated artifact described in the tutorial index.

```bash
cargo build -p tenferro-tutorial-code --no-default-features \
  --features cuda,native --bin cuda_tutorial
```

The example downloads the result back to CPU and asserts the expected values.
An extension reaches the CUDA driver through the public `CudaEngine::with_raw`
and `cuda::raw` scope described in the
[GPU backend design](../design/gpu-backend-design.md#operation-crate-interop-boundary).

`TensorStructural::copy_read_into` can reuse an already allocated CUDA
destination. Both operands may be arbitrary-stride region views at a nonzero
offset inside a larger allocation: each side is described by its own extents,
strides, and offset, so a block region is read and written in place instead of
being canonicalized into scratch first. A permutation whose axis fusion reduces
to a single 2D transpose (a rotation of a square leading pair at any rank, and
the plain 2D transposes) runs on the native tiled transpose kernel, which is
coalesced on both operands and keeps the live page set bounded; when a fused
extent needs more blocks than a grid dimension allows, the kernel widens its
tile instead of falling back to a flat pass (issue #1891). Every other numeric
dtype layout uses the cuTENSOR permutation path, which is the bandwidth-bound
optimum there. A complex operand is planned through its real view with a real
`alpha = 1`, so the scaling multiply is exact and cannot turn `(inf, finite)`
into `(inf, NaN)` (issue #1891). A `NaN` payload may still be canonicalized.
Reversed and broadcast sources, which cuTENSOR descriptors cannot represent, use
the native copy, and a compact 2D/3D transpose into a row-major compact
destination uses the tiled transpose kernel there. The source and destination
must be distinct allocations.

`Engine::axpby_read_into_accum` keeps the documented compact
destination contract, and consumes an arbitrary-stride or offset `x` through
one native strided-source pass rather than canonicalizing it; a compact `x`
still runs the single cuBLAS `geam` call.

The CUDA raw session exposes `Session::fill_zero_write`, the `beta = 0`
reset for storage the caller owns. It never reads the previous contents, so a
stale `NaN`/`Inf` cannot survive and `-0.0` is normalized to `+0.0`, which
`0 * y` cannot promise. A compact destination (including an offset region
view) is filled by one stream-ordered memset; a strided region view runs one
native fill kernel and leaves every element outside the region untouched.

## CUDA and NVIDIA library compatibility

“Supported” below is tenferro's runtime contract. “Loader ABI” only means the
loader probes that soname and required symbols; it does not make every library
with that soname, or a complete CUDA 11 stack, supported. “CI-tested” records
the package family or exact archive used by the current GPU workflows.

| Component | Supported runtime contract | Loader ABI | CI-tested setup | Override |
| --- | --- | --- | --- | --- |
| NVIDIA driver API | CUDA 12.6.2 minimum; tenferro compiles against CUDA 12.8 `cudarc` bindings and resolves driver/NVRTC functions dynamically against the loaded stack | Loaded dynamically through `cudarc`; the selected NVRTC must not be newer than the driver API | RunPod accepts 12.6-or-newer drivers and selects a matching 12.6 or 12.8 runtime tier | System driver; no tenferro override |
| CUDA runtime and NVRTC | 12.6.2 baseline; 12.8 full-capability tier | CUDA 12.x runtime/NVRTC libraries | `cuda-cudart-12-4`/`cuda-nvrtc-12-4` and `cuda-cudart-12-8`/`cuda-nvrtc-12-8` runtime trees | `CUDA_PATH`, `LD_LIBRARY_PATH` |
| cuTENSOR | CUDA 12 build using the v2 API; on compute capability 7.0, use 2.1.x because tenferro rejects 2.2 and newer on that GPU | `libcutensor.so.2`, then `libcutensor.so` | cuTENSOR `2.6.0.4` CUDA 12 archive | `TENFERRO_CUTENSOR_PATH` |
| cuBLAS | CUDA 12 library matching the selected runtime tier | `libcublas.so.12`, then `libcublas.so` | `libcublas-12-4` and `libcublas-12-8` package families; patch version follows NVIDIA's package repository | `TENFERRO_CUBLAS_PATH` |
| cuSOLVER | CUDA 12 library matching the selected runtime tier | `libcusolver.so.12`, then `.so.11` and unversioned candidates; the `.so.11` probe alone does not make CUDA 11 a supported stack | `libcusolver-12-4` and `libcusolver-12-8` package families; patch version follows NVIDIA's package repository | `TENFERRO_CUSOLVER_PATH` |
| cuFFT | CUDA 12 uses the cuFFT 11 ABI | `libcufft.so.11`, then `.so.10` and unversioned candidates; the `.so.10` probe alone does not make CUDA 11 a supported stack | Not separately version-pinned by GPU CI; loader tests verify the search order | `TENFERRO_CUFFT_PATH` |

The current CI values are defined in
[`ci-cache-publish.yml`](https://github.com/tensor4all/tenferro-rs/blob/main/.github/workflows/ci-cache-publish.yml)
and [`runpod-gpu-test.yml`](https://github.com/tensor4all/tenferro-rs/blob/main/.github/workflows/runpod-gpu-test.yml).
Use NVIDIA's [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/)
to map a driver release to the CUDA API levels it supports; tenferro's 12.6.2
minimum can be narrower than NVIDIA's general minor-version compatibility. It
is a *library* floor, not a GPU one: CUDA 12.x supports the same architectures
throughout, and tenferro `dlopen`s cuSOLVER, so a cuSOLVER >= 11.7.1 package is
enough. The floor exists because `cusolverDnXsyevBatched`, which every batched
eigh now uses, first ships in that version (issue #1852).
tenferro compiles against CUDA 12.8 `cudarc` bindings, resolves driver
functions dynamically, and gates newer functions using the loaded driver and
NVRTC versions.

Use the installed CUDA root on your machine. If several roots exist, inspect
them first:

```bash
ls -d /usr/local/cuda*
```

If CUDA libraries or cuTENSOR are outside the standard dynamic-linker paths,
set:

```bash
export CUDA_PATH=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH
export TENFERRO_CUTENSOR_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
export TENFERRO_CUSOLVER_PATH=$CUDA_PATH/lib64/libcusolver.so.12
export TENFERRO_CUBLAS_PATH=$CUDA_PATH/lib64/libcublas.so.12
export TENFERRO_CUFFT_PATH=$CUDA_PATH/lib64/libcufft.so.11
```

CUDA operations implemented through NVIDIA libraries return typed load or
provider errors when the required library is missing. They do not silently use
a slower native kernel as a CUDA fallback.

On compute capability 7.0, select cuTENSOR 2.1.x with
`TENFERRO_CUTENSOR_PATH`. tenferro rejects cuTENSOR 2.2 and newer because that
combination has returned success with incorrect results.

For XLA/PJRT GPU verification, add the PJRT plugin path separately:

```bash
export TENFERRO_PJRT_GPU_PLUGIN=/path/to/pjrt_c_api_gpu_plugin.so
```

## CUDA Across Tensor Layers

| Tensor model | How CUDA fits |
| --- | --- |
| `TypedTensor<T, R>` | Fixed-dtype runtime tensor with optional compile-time rank; host access still requires explicit download for CUDA buffers. |
| `Tensor` | Main concrete CUDA value for backend execution |
| `EagerTensor` | Wraps CUDA-resident `Tensor` values when an `EagerRuntime` session borrows a `CudaEngine` |
| `TracedTensor` | Graphs can be compiled with `GraphCompiler` and executed by a runtime with registered GPU engines for supported ops |

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
| `abs` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | `F32->F32`, `F64->F64`, `I32->I32`, `I64->I64`, `C32->F32`, `C64->F64` | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `sign` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
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
| `erf` | `F32`, `F64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `dot_general` | `F32`, `F64`, `C32`, `C64` | none | same as input | result Native; read Native; write Native; strided Native; accumulation Native |
| `reduce_sum` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
| `reduce_sum_squares` | `F32`, `F64` | none | same as input | result Native; read FallbackCopy; write Unsupported; strided Unsupported; accumulation Unsupported |
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
| `transpose`, `broadcast_in_dim`, `extract_diagonal`, `embed_diagonal`, `tril`, `triu` | `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | Structural tensor operations; CUDA `F32`/`F64`/`C32`/`C64` transpose uses cuTENSOR permutation, and the triangular masks read transposed, offset and negative-stride views through their own layout |
| checked `convert`, explicit `cast` | All CPU-supported pairs among `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | `convert` remains restricted by the public promotion lattice; explicit `cast` supports lossy narrowing, real-component projection, zero-imaginary injection, integer conversion, and nonzero Bool truthiness. Fallible real/complex-to-integer casts validate on device and return the same typed errors as CPU |
| `gather` | operand `F32`, `F64`, `I32`, `Bool`, `C32`, `C64`; indices `F32`, `F64`, `I32`, or `I64` | Complex and `Bool` index tensors; `I64` operands are not implemented |
| `scatter` | operand/update `F32`, `F64`, `C32`, `C64`; indices `F32`, `F64`, `I32`, or `I64` | Add-scatter semantics; complex and `Bool` index tensors and integer/`Bool` operands are not implemented |
| `slice`, `pad`, `concatenate`, `reverse` | `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` | Dense structural/indexing operations |
| `to_contiguous_read` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | Same-device canonicalization of owned tensors and arbitrary valid CUDA views; CUDA `F32`/`F64`/`C32`/`C64` nonnegative-stride canonicalization uses cuTENSOR permutation, while negative-stride views use the native CUDA structural copy because cuTENSOR does not represent that layout; `Bool` is an explicit current limitation |
| `copy_read_into` | `F32`, `F64`, `I32`, `I64`, `C32`, `C64` | The source may be any valid region of this runtime (any offset, strides and rank; a source view wider than the catalogue's by-value layout rank keeps one compact staging copy); the destination is an owned tensor or a non-overlapping view up to the catalogue's by-value layout rank, written in place; source and destination allocations must not alias |
| `dynamic_slice` | input `F32`, `F64`, `I32`, `Bool`, `C32`, `C64` with starts `F32`, `F64`, `I32`, or `I64` | Complex and `Bool` start tensors; `I64` inputs are not implemented |
| `dynamic_update_slice` | operand/update `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64` with starts `F32`, `F64`, `I32`, or `I64` | StableHLO window write into a compact operand copy; a strided update and a compact offset starts view are read in place |
| `cholesky`, `triangular_solve`, `lu`, `svd`, `qr`, `eigh`, `solve` | `F32`, `F64`, `C32`, `C64` | cuSOLVER/cuBLAS-backed; integer and `Bool` dtypes are not implemented |
| `full_piv_lu`, `full_piv_lu_solve` | No CUDA implementation | Returns an error |
| General `eig` | No CUDA implementation | cuSOLVER does not provide LAPACK `dgeev`-style general eigendecomposition; download to CPU explicitly |
| AMD/ROCm | No supported execution backend | ROCm remains reserved for future loader-backed work |

If cuTENSOR, cuSOLVER, or cuBLAS are installed outside normal dynamic-linker
paths, set `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH`, or
`TENFERRO_CUBLAS_PATH`.
