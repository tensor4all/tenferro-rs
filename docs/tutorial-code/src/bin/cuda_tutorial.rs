//! Hardware-gated CUDA tutorial: explicit upload, session execution, and download.

use tenferro_gpu::cuda::{
    cuda_devices, download_tensor, gpu_available, upload_tensor, CudaBackend,
};
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::Tensor;

const TUTORIAL_SKIP_MARKER: &str = "TENFERRO_TUTORIAL_SKIP:";

fn skip_or_fail(
    require_cuda: bool,
    reason: impl std::fmt::Display,
) -> Result<(), Box<dyn std::error::Error>> {
    if require_cuda {
        return Err(std::io::Error::other(format!(
            "CUDA tutorial requires assertions, but {reason}"
        ))
        .into());
    }
    eprintln!("{TUTORIAL_SKIP_MARKER} CUDA tutorial skipped: {reason}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let require_cuda = std::env::var("TENFERRO_REQUIRE_CUDA")
        .map(|value| value == "1")
        .unwrap_or(false);
    if !gpu_available() {
        return skip_or_fail(require_cuda, "no usable CUDA device is available");
    }
    let Some(device) = cuda_devices()?.into_iter().next() else {
        return skip_or_fail(require_cuda, "CUDA reported no devices");
    };

    let mut backend = CudaBackend::new(device.id())?;
    let a = Tensor::from_vec_col_major([2], vec![1.0_f64, 2.0])?;
    let b = Tensor::from_vec_col_major([2], vec![3.0_f64, 4.0])?;
    let gpu_a = upload_tensor(backend.runtime(), &a)?;
    let gpu_b = upload_tensor(backend.runtime(), &b)?;

    let gpu_sum = backend.with_backend_session(|session| session.add(&gpu_a, &gpu_b))?;
    let sum = download_tensor(backend.runtime(), &gpu_sum)?;
    assert_eq!(sum.as_slice::<f64>()?, &[4.0, 6.0]);
    println!("cuda_tutorial: upload -> session -> download passed");
    Ok(())
}
