//! Explicit CUDA cuFFT execution with host/device transfers at visible boundaries.

use num_complex::Complex32;
use tenferro_fft::{FftNorm, TensorFftExt};
use tenferro_gpu::cuda::{
    cuda_devices, download_tensor, gpu_available, upload_tensor, with_cuda_exec_session,
    CudaBackend,
};
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::{DType, MemoryKind, Tensor, TensorRead};

const TUTORIAL_SKIP_MARKER: &str = "TENFERRO_TUTORIAL_SKIP:";

fn skip_or_fail(
    require_cuda: bool,
    reason: impl std::fmt::Display,
) -> Result<(), Box<dyn std::error::Error>> {
    let reason = reason.to_string();
    if require_cuda {
        return Err(std::io::Error::other(format!(
            "CUDA FFT tutorial requires CUDA assertions, but {reason}"
        ))
        .into());
    }

    eprintln!("{TUTORIAL_SKIP_MARKER} CUDA FFT tutorial skipped: {reason}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let require_cuda = std::env::var("TENFERRO_REQUIRE_CUDA")
        .map(|value| value == "1")
        .unwrap_or(false);
    if !gpu_available() {
        return skip_or_fail(require_cuda, "no usable CUDA device is available");
    }

    let devices = match cuda_devices() {
        Ok(devices) => devices,
        Err(error) => {
            return skip_or_fail(
                require_cuda,
                format!("CUDA device enumeration failed: {error}"),
            );
        }
    };
    let Some(device) = devices.into_iter().next() else {
        return skip_or_fail(
            require_cuda,
            "CUDA reported available but device enumeration returned no devices",
        );
    };

    // Keep one backend/runtime for the upload, FFT, synchronization, and download.
    let mut backend = CudaBackend::new(device.id())?;
    let host = Tensor::from_vec_col_major([4], vec![1.0_f32, 2.0, 3.0, 4.0])?;
    let gpu_input = upload_tensor(backend.runtime(), &host)?;

    let input_read = TensorRead::from_tensor(&gpu_input);
    assert_eq!(input_read.backend_family(), Some("cuda"));
    assert_eq!(input_read.placement().memory_kind, MemoryKind::Device);
    let input_domain = input_read
        .allocation_domain()
        .ok_or("uploaded tensor has no CUDA allocation domain")?;

    // FFT execution consumes the already uploaded tensor; it does not transfer it.
    let spectrum = backend.with_backend_session(|session| {
        with_cuda_exec_session(session, |exec_session| {
            gpu_input.rfft(None, 0, FftNorm::Backward, exec_session)
        })
        .ok_or_else(|| tenferro_tensor::Error::Unsupported {
            op: "cuda_fft_tutorial",
            message: "CUDA backend session is unavailable".to_owned(),
        })?
    })?;

    // Check residency before crossing the explicit device-to-host boundary.
    let spectrum_read = TensorRead::from_tensor(&spectrum);
    assert_eq!(spectrum_read.backend_family(), Some("cuda"));
    assert_eq!(spectrum_read.placement().memory_kind, MemoryKind::Device);
    assert_eq!(spectrum_read.allocation_domain(), Some(input_domain));
    assert_eq!(spectrum.dtype(), DType::C32);
    assert_eq!(spectrum.shape(), &[3]);

    // The cuFFT vendor call synchronizes at its FFI boundary. The explicit
    // download below synchronizes the stream-managed postprocessing and is the
    // visible device-to-host boundary for the final output.
    let host_spectrum = download_tensor(backend.runtime(), &spectrum)?;
    assert_eq!(host_spectrum.dtype(), DType::C32);
    assert_eq!(host_spectrum.shape(), &[3]);
    let values = host_spectrum.as_slice::<Complex32>()?;
    let expected = [
        Complex32::new(10.0, 0.0),
        Complex32::new(-2.0, 2.0),
        Complex32::new(-2.0, 0.0),
    ];
    assert_eq!(values.len(), expected.len());
    for (index, (actual, expected)) in values.iter().zip(expected).enumerate() {
        assert!(
            (*actual - expected).norm() <= 1.0e-5,
            "rfft value {index} differs: actual {actual:?}, expected {expected:?}"
        );
    }

    Ok(())
}
