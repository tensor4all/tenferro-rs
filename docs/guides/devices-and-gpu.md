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

| Area | Status |
| --- | --- |
| Allocation and transfer | CUDA supported |
| Elementwise and reductions | broad real coverage, selected complex coverage |
| Structural/indexing | broad coverage where dtype support exists |
| Contractions | selected cuTENSOR/cuBLAS-backed paths |
| Linalg | selected cuSOLVER/cuBLAS-backed paths |
| General `eig` | not supported by cuSOLVER; download to CPU |
| AMD/ROCm | not supported yet |

If cuTENSOR, cuSOLVER, or cuBLAS are installed outside normal dynamic-linker
paths, set `TENFERRO_CUTENSOR_PATH`, `TENFERRO_CUSOLVER_PATH`, or
`TENFERRO_CUBLAS_PATH`.
