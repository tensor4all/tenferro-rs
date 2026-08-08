use std::num::NonZeroUsize;

use super::common::*;
use super::support;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftExecutor, FftNorm, FftPlanCache};
use tenferro_gpu::cuda::gpu_available;
use tenferro_runtime::ExtensionCacheLimits;
use tenferro_tensor::TensorRead;

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_reuses_and_separates_structural_plans() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let mut executor = FftExecutor::default();
    let c32 = complex_f32(&[4], 0.5);
    let c64 = complex_f64(&[4], 0.5);
    let batched = complex_f32(&[2, 4], -0.25);
    let real = real_f32(&[4], 0.5);

    let initial = executor.cache_stats();
    assert_eq!(initial.entries, 0);
    assert_eq!(initial.retained_bytes, 0);
    assert_eq!(initial.hits, 0);
    assert_eq!(initial.misses, 0);
    assert_eq!(initial.evictions, 0);
    assert_eq!(initial.clears, 0);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    let first = executor.cache_stats();
    assert_eq!(first.entries, 1);
    assert!(first.retained_bytes > 0, "cache stats: {first:?}");
    assert_eq!(first.hits, 0);
    assert_eq!(first.misses, 1);
    assert_eq!(first.evictions, 0);
    assert_eq!(first.clears, 0);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    let repeated = executor.cache_stats();
    assert_eq!(repeated.entries, 1);
    assert!(repeated.retained_bytes > 0, "cache stats: {repeated:?}");
    assert_eq!(repeated.hits, 1);
    assert_eq!(repeated.misses, 1);
    assert_eq!(repeated.evictions, 0);
    assert_eq!(repeated.clears, 0);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Ifft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    assert_eq!(executor.cache_stats().entries, 2);
    assert_eq!(executor.cache_stats().hits, 1);
    assert_eq!(executor.cache_stats().misses, 2);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c64,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    assert_eq!(executor.cache_stats().entries, 3);
    assert_eq!(executor.cache_stats().misses, 3);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Fft,
        Some(3),
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    assert_eq!(executor.cache_stats().entries, 4);
    assert_eq!(executor.cache_stats().misses, 4);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &batched,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    assert_eq!(executor.cache_stats().entries, 5);
    assert_eq!(executor.cache_stats().misses, 5);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &real,
        Operation::Rfft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    let separated = executor.cache_stats();
    assert_eq!(separated.entries, 6);
    assert!(separated.retained_bytes > 0, "cache stats: {separated:?}");
    assert_eq!(separated.hits, 1);
    assert_eq!(separated.misses, 6);
    assert_eq!(separated.evictions, 0);
    assert_eq!(separated.clears, 0);

    executor.clear_cache();
    let cleared = executor.cache_stats();
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);
    assert_eq!(cleared.hits, 1);
    assert_eq!(cleared.misses, 6);
    assert_eq!(cleared.evictions, 0);
    assert_eq!(cleared.clears, 1);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_keys_include_runtime_identity() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut first = support::cuda_backend();
    let mut second = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let first_input = support::upload_cuda(first.runtime(), &host);
    let second_input = support::upload_cuda(second.runtime(), &host);
    let mut executor = FftExecutor::default();

    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    assert_eq!(executor.cache_stats().entries, 0);
    for (index, (backend, input)) in [(&mut first, &first_input), (&mut second, &second_input)]
        .into_iter()
        .enumerate()
    {
        let domain = TensorRead::from_tensor(input).allocation_domain().unwrap();
        let output = Operation::Fft
            .execute_executor(&mut executor, backend, input, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&output, domain);
        let output = support::download_cuda(backend.runtime(), &output).unwrap();
        assert_host_close(&output, &expected, 1.0e-11);
        assert_eq!(executor.cache_stats().entries, index + 1);
    }

    let stats = executor.cache_stats();
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 2);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_reuses_after_runtime_move() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let host = complex_f64(&[4], 0.5);
    let mut backend = support::cuda_backend();
    let first_token = backend.runtime_identity().cache_discriminator();
    let mut executor = FftExecutor::default();

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut backend,
        &host,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    assert_eq!(executor.cache_stats().entries, 1);
    assert_eq!(executor.cache_stats().hits, 0);

    let backend = Box::new(backend);
    let mut backend = *backend;
    assert_eq!(
        backend.runtime_identity().cache_discriminator(),
        first_token
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut backend,
        &host,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    let moved_stats = executor.cache_stats();
    assert_eq!(moved_stats.entries, 1);
    assert_eq!(moved_stats.hits, 1);
    assert_eq!(moved_stats.misses, 1);

    let mut independent = support::cuda_backend();
    let independent_token = independent.runtime_identity().cache_discriminator();
    assert_ne!(first_token, independent_token);
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut independent,
        &host,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    let separated_stats = executor.cache_stats();
    assert_eq!(separated_stats.entries, 2);
    assert_eq!(separated_stats.hits, 1);
    assert_eq!(separated_stats.misses, 2);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_limit_reduction_evicts_and_preserves_results() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let input = complex_f64(&[4], 0.5);
    let gpu_input = support::upload_cuda(cuda.runtime(), &input);
    let mut executor = FftExecutor::default();
    let first = Operation::Fft
        .execute_executor(
            &mut executor,
            &mut cuda,
            &gpu_input,
            None,
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    let second = Operation::Fft
        .execute_executor(
            &mut executor,
            &mut cuda,
            &gpu_input,
            Some(3),
            -1,
            FftNorm::Backward,
        )
        .unwrap();

    let populated = executor.cache_stats();
    assert_eq!(populated.entries, 2);
    assert!(populated.retained_bytes > 0, "cache stats: {populated:?}");
    assert_eq!(populated.hits, 0);
    assert_eq!(populated.misses, 2);
    assert_eq!(populated.evictions, 0);
    assert_eq!(populated.clears, 0);

    let entry_limit = ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(populated.retained_bytes).unwrap());
    executor.plan_cache_mut().set_limits(entry_limit);
    let entry_reduced = executor.cache_stats();
    assert_eq!(executor.plan_cache().limits(), entry_limit);
    assert_eq!(entry_reduced.entries, 1);
    assert!(entry_reduced.retained_bytes > 0);
    assert!(entry_reduced.retained_bytes <= populated.retained_bytes);
    assert_eq!(entry_reduced.hits, 0);
    assert_eq!(entry_reduced.misses, 2);
    assert_eq!(entry_reduced.evictions, 1);
    assert_eq!(entry_reduced.clears, 0);

    let byte_limit = ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(1).unwrap());
    executor.plan_cache_mut().set_limits(byte_limit);
    let bytes_reduced = executor.cache_stats();
    assert_eq!(executor.plan_cache().limits(), byte_limit);
    assert_eq!(bytes_reduced.entries, 0);
    assert_eq!(bytes_reduced.retained_bytes, 0);
    assert_eq!(bytes_reduced.hits, 0);
    assert_eq!(bytes_reduced.misses, 2);
    assert_eq!(bytes_reduced.evictions, 2);
    assert_eq!(bytes_reduced.clears, 0);

    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
        .unwrap();
    let first_host = support::download_cuda(cuda.runtime(), &first).unwrap();
    assert_host_close(&first_host, &expected, 1.0e-11);
    drop(second);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_clear_and_eviction_are_safe_after_launch() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let input = complex_f64(&[32, 4096], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(cuda.runtime(), &input);
    let domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .unwrap();
    let mut executor = FftExecutor::default();
    let output = Operation::Fft
        .execute_executor(
            &mut executor,
            &mut cuda,
            &gpu_input,
            None,
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    support::assert_cuda_resident(&output, domain);
    let launched = executor.cache_stats();
    assert_eq!(launched.entries, 1);
    assert!(launched.retained_bytes > 0);
    assert_eq!(launched.hits, 0);
    assert_eq!(launched.misses, 1);
    assert_eq!(launched.evictions, 0);
    assert_eq!(launched.clears, 0);

    executor.clear_cache();
    let cleared = executor.cache_stats();
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);
    assert_eq!(cleared.hits, 0);
    assert_eq!(cleared.misses, 1);
    assert_eq!(cleared.evictions, 0);
    assert_eq!(cleared.clears, 1);
    let output_host = support::download_cuda(cuda.runtime(), &output).unwrap();
    assert_host_close(&output_host, &expected, 1.0e-11);

    let mut limited = FftExecutor::new(FftPlanCache::with_capacity(NonZeroUsize::new(1).unwrap()));
    let first = Operation::Fft
        .execute_executor(
            &mut limited,
            &mut cuda,
            &gpu_input,
            None,
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    let second = Operation::Fft
        .execute_executor(
            &mut limited,
            &mut cuda,
            &gpu_input,
            Some(2048),
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    let limited_stats = limited.cache_stats();
    assert_eq!(limited.plan_cache().capacity().get(), 1);
    assert_eq!(limited_stats.entries, 1);
    assert!(limited_stats.retained_bytes > 0);
    assert_eq!(limited_stats.hits, 0);
    assert_eq!(limited_stats.misses, 2);
    assert_eq!(limited_stats.evictions, 1);
    assert_eq!(limited_stats.clears, 0);
    let first_host = support::download_cuda(cuda.runtime(), &first).unwrap();
    let second_host = support::download_cuda(cuda.runtime(), &second).unwrap();
    assert_host_close(&first_host, &expected, 1.0e-11);
    let expected_second = Operation::Fft
        .execute_cpu(&mut cpu, &input, Some(2048), -1, FftNorm::Backward)
        .unwrap();
    assert_host_close(&second_host, &expected_second, 1.0e-11);
}
