use super::common::*;
use super::support;
use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_fft::FftNorm;
use tenferro_gpu::cuda::gpu_available;
use tenferro_runtime::{DType, Tensor};
use tenferro_tensor::TensorRead;

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_execution_uses_explicit_engine_and_fft_module_registration() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let backend = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(backend.runtime(), &host);
    let domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .unwrap();
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&backend, true);
    let outputs = runtime.run_compiled(&program, &[&gpu_input]).unwrap();
    assert_eq!(outputs.len(), 1);
    support::assert_cuda_resident(&outputs[0], domain);
    let output = support::download_cuda(backend.runtime(), &outputs[0]).unwrap();
    assert_host_close(&output, &expected, 1.0e-11);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_missing_extension_module_is_an_explicit_error() {
    if !gpu_available() {
        return;
    }

    let backend = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let gpu_input = support::upload_cuda(backend.runtime(), &host);
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&backend, false);
    let error = runtime
        .run_compiled(&program, &[&gpu_input])
        .expect_err("missing FFT extension registration must fail");
    let message = error.to_string();
    assert!(
        message.contains("extension") || message.contains("module"),
        "missing registration error was not explicit: {message}"
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_runtime_mismatch_is_rejected_without_host_fallback() {
    if !gpu_available() {
        return;
    }

    let first = support::cuda_backend();
    let second = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let foreign_input = support::upload_cuda(second.runtime(), &host);
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&first, true);
    let error = runtime
        .run_compiled(&program, &[&foreign_input])
        .expect_err("foreign CUDA runtime input must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("runtime") || message.contains("input") || message.contains("device"),
        "runtime mismatch error was not explicit: {message}"
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_runtime_owned_cache_reuses_stats_clears_and_retires_after_launch() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let backend = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(backend.runtime(), &host);
    let domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .unwrap();
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&backend, true);

    let first = runtime
        .run_compiled(&program, &[&gpu_input])
        .unwrap()
        .remove(0);
    let second = runtime
        .run_compiled(&program, &[&gpu_input])
        .unwrap()
        .remove(0);
    let stats = runtime.cache_stats().unwrap().extensions;
    assert_eq!(stats.entries, 1, "runtime FFT cache stats: {stats:?}");
    assert_eq!(stats.hits, 1, "runtime FFT cache stats: {stats:?}");
    assert_eq!(stats.misses, 1, "runtime FFT cache stats: {stats:?}");
    assert!(
        stats.retained_bytes > 0,
        "runtime FFT cache has no retained bytes: {stats:?}"
    );
    runtime.clear_caches().unwrap();
    let cleared = runtime.cache_stats().unwrap().extensions;
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);

    support::assert_cuda_resident(&first, domain);
    support::assert_cuda_resident(&second, domain);
    backend.runtime().synchronize().unwrap();
    let first = support::download_cuda(backend.runtime(), &first).unwrap();
    let second = support::download_cuda(backend.runtime(), &second).unwrap();
    assert_host_close(&first, &expected, 1.0e-11);
    assert_host_close(&second, &expected, 1.0e-11);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_zero_batch_returns_empty_outputs_without_runtime_cache() {
    if !gpu_available() {
        return;
    }

    let backend = support::cuda_backend();
    for (host, dtype, shape, expected_shape, axis) in [
        (
            Tensor::from_vec_col_major(vec![0, 8], Vec::<Complex64>::new()).unwrap(),
            DType::C64,
            vec![0, 8],
            vec![0, 8],
            1isize,
        ),
        (
            Tensor::from_vec_col_major(vec![2, 0, 8], Vec::<f32>::new()).unwrap(),
            DType::F32,
            vec![2, 0, 8],
            vec![2, 0, 5],
            2isize,
        ),
    ] {
        let gpu_input = support::upload_cuda(backend.runtime(), &host);
        let domain = TensorRead::from_tensor(&gpu_input)
            .allocation_domain()
            .unwrap();
        let operation = if dtype == DType::C64 {
            Operation::Fft
        } else {
            Operation::Rfft
        };
        let (program, _) =
            compiled_cuda_fft(dtype, &shape, operation, None, axis, FftNorm::Backward);
        let runtime = cuda_runtime_with_fft(&backend, true);
        let output = runtime
            .run_compiled(&program, &[&gpu_input])
            .unwrap()
            .remove(0);
        support::assert_cuda_resident(&output, domain);
        assert_eq!(output.shape(), expected_shape.as_slice());
        let stats = runtime.cache_stats().unwrap().extensions;
        assert_eq!(stats.entries, 0, "zero-batch cache stats: {stats:?}");
        assert_eq!(stats.retained_bytes, 0, "zero-batch cache stats: {stats:?}");
        backend.runtime().synchronize().unwrap();
        let output = support::download_cuda(backend.runtime(), &output).unwrap();
        assert_eq!(output.shape(), expected_shape.as_slice());
        assert_eq!(
            output.as_slice::<Complex64>().map_or(0, <[Complex64]>::len),
            0
        );
        assert_eq!(
            output.as_slice::<Complex32>().map_or(0, <[Complex32]>::len),
            0
        );
        assert_eq!(output.as_slice::<f32>().map_or(0, <[f32]>::len), 0);
    }
}
