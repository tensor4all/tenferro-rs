use super::common::*;
use super::support;
use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_fft::{EagerTensorFftExt, FftNorm};
use tenferro_gpu::cuda::gpu_available;
use tenferro_runtime::Tensor;
use tenferro_tensor::TensorRead;

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_eager_fft_rfft_irfft_select_cuda_and_keep_cpu_control_on_host() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let upload_backend = support::cuda_backend();
    let real_host = real_f64(&[8], 0.25);
    let real_device = support::upload_cuda(upload_backend.runtime(), &real_host);
    let real_domain = TensorRead::from_tensor(&real_device)
        .allocation_domain()
        .unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let eager_real = EagerTensor::from_tensor_in(real_device, ctx.clone()).unwrap();
    let eager_spectrum = eager_real.rfft(None, -1, FftNorm::Backward).unwrap();
    let _eager_spectrum_repeat = eager_real.rfft(None, -1, FftNorm::Backward).unwrap();
    let spectrum_tensor = eager_spectrum.to_tensor().unwrap();
    support::assert_cuda_resident(&spectrum_tensor, real_domain);
    let expected_spectrum = Operation::Rfft
        .execute_cpu(&mut cpu, &real_host, None, -1, FftNorm::Backward)
        .unwrap();
    let eager_cache_stats = ctx.cache_stats().unwrap().extensions;
    assert!(
        eager_cache_stats.entries > 0,
        "eager FFT cache was not populated"
    );
    assert!(
        eager_cache_stats.hits > 0,
        "repeated eager FFT should reuse a plan"
    );
    ctx.synchronize().unwrap();
    let spectrum_host = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&spectrum_tensor))
        })
        .unwrap()
        .unwrap();
    assert_host_close(&spectrum_host, &expected_spectrum, 1.0e-11);

    let eager_signal = eager_spectrum.irfft(None, -1, FftNorm::Backward).unwrap();
    let signal_tensor = eager_signal.to_tensor().unwrap();
    support::assert_cuda_resident(&signal_tensor, real_domain);
    let expected_signal = Operation::Irfft
        .execute_cpu(&mut cpu, &expected_spectrum, None, -1, FftNorm::Backward)
        .unwrap();
    ctx.synchronize().unwrap();
    let signal_host = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&signal_tensor))
        })
        .unwrap()
        .unwrap();
    assert_host_close(&signal_host, &expected_signal, 1.0e-11);

    let complex_host = complex_f64(&[4], 0.5);
    let complex_device = ctx
        .with_execution_session(|session| {
            session.upload_host_tensor(TensorRead::from_tensor(&complex_host))
        })
        .unwrap()
        .unwrap();
    let eager_complex = EagerTensor::from_tensor_in(complex_device, ctx.clone()).unwrap();
    let eager_fft = eager_complex.fft(None, -1, FftNorm::Backward).unwrap();
    let fft_tensor = eager_fft.to_tensor().unwrap();
    support::assert_cuda_resident(&fft_tensor, real_domain);
    let expected_fft = Operation::Fft
        .execute_cpu(&mut cpu, &complex_host, None, -1, FftNorm::Backward)
        .unwrap();
    ctx.synchronize().unwrap();
    let fft_host = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&fft_tensor))
        })
        .unwrap()
        .unwrap();
    assert_host_close(&fft_host, &expected_fft, 1.0e-11);

    let cpu_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let cpu_input = EagerTensor::from_tensor_in(real_f64(&[8], 0.25), cpu_ctx.clone()).unwrap();
    let cpu_output = cpu_input
        .rfft(None, -1, FftNorm::Backward)
        .unwrap()
        .to_tensor()
        .unwrap();
    assert_eq!(
        cpu_output.placement().memory_kind,
        tenferro_tensor::MemoryKind::UnpinnedHost
    );
    assert!(!cpu_output.is_backend_buffer());
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_eager_zero_batch_returns_empty_outputs_without_runtime_cache() {
    if !gpu_available() {
        return;
    }

    let upload_backend = support::cuda_backend();
    let host = Tensor::from_vec_col_major(vec![0, 8], Vec::<Complex64>::new()).unwrap();
    let device = support::upload_cuda(upload_backend.runtime(), &host);
    let domain = TensorRead::from_tensor(&device)
        .allocation_domain()
        .unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let input = EagerTensor::from_tensor_in(device, ctx.clone()).unwrap();
    let output = input.fft(None, 1, FftNorm::Backward).unwrap();
    let tensor = output.to_tensor().unwrap();
    support::assert_cuda_resident(&tensor, domain);
    assert_eq!(tensor.shape(), &[0, 8]);

    let real_host = real_f32(&[2, 0, 8], 0.5);
    let real_device = ctx
        .with_execution_session(|session| {
            session.upload_host_tensor(TensorRead::from_tensor(&real_host))
        })
        .unwrap()
        .unwrap();
    let real_input = EagerTensor::from_tensor_in(real_device, ctx.clone()).unwrap();
    let real_output = real_input.rfft(None, 2, FftNorm::Backward).unwrap();
    let real_tensor = real_output.to_tensor().unwrap();
    support::assert_cuda_resident(&real_tensor, domain);
    assert_eq!(real_tensor.shape(), &[2, 0, 5]);

    let stats = ctx.cache_stats().unwrap().extensions;
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.retained_bytes, 0);
    ctx.clear_caches().unwrap();
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 0);
}
