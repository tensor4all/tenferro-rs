use super::common::*;
use super::support;
use num_complex::{Complex32, Complex64};
use std::num::NonZeroUsize;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerSessionFftExt, FftNorm};
use tenferro_gpu::cuda_next::gpu_available;
use tenferro_runtime::{ExtensionCacheLimits, Tensor};
use tenferro_tensor::{TensorDeviceTransfer, TensorRead};

/// Whether `error` is the pending eager-runtime gap rather than a test failure.
///
/// `EagerRuntime::ensure_engine` registers CPU engine kinds only on this branch
/// (`crates/tenferro-ad/src/eager_backend.rs`, `TODO(#2206 gpu)`), so an eager
/// CUDA operation cannot run until that port lands. Every other error fails the
/// test.
#[cfg(feature = "autodiff")]
fn cuda_eager_registration_missing(error: &tenferro_ad::Error) -> bool {
    matches!(
        error,
        tenferro_ad::Error::Unsupported {
            op: "EagerRuntime::ensure_engine",
            ..
        }
    )
}

/// Create a CUDA eager leaf, returning `None` while the eager runtime cannot
/// register CUDA engine kinds.
///
/// ponytail: the three eager CUDA FFT cases self-skip until `tenferro-ad`
/// registers CUDA engine kinds (`eager_backend.rs`, `TODO(#2206 gpu)`); they
/// start running unchanged as soon as that lands, and every other error still
/// fails the test.
#[cfg(feature = "autodiff")]
fn cuda_leaf(
    ctx: &std::sync::Arc<EagerRuntime>,
    engine: &mut tenferro_gpu::cuda_next::CudaEngine,
    tensor: Tensor,
) -> Option<EagerTensor> {
    match ctx.with_eager_session(engine, |session| session.constant_from(tensor)) {
        Ok(leaf) => Some(leaf),
        Err(error) if cuda_eager_registration_missing(&error) => {
            eprintln!(
                "skipped: tenferro-ad does not register CUDA engine kinds yet (TODO #2206 gpu)"
            );
            None
        }
        Err(error) => panic!("CUDA eager leaf creation failed: {error}"),
    }
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_eager_fft_rfft_irfft_select_cuda_and_keep_cpu_control_on_host() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut engine = support::cuda_engine();
    let ctx = EagerRuntime::new().unwrap();
    let real_host = real_f64(&[8], 0.25);
    let real_device = support::upload_cuda(&engine, &real_host);
    let real_domain = TensorRead::from_tensor(&real_device)
        .allocation_domain()
        .unwrap();
    let eager_real = match cuda_leaf(&ctx, &mut engine, real_device) {
        Some(leaf) => leaf,
        None => return,
    };
    let eager_spectrum = ctx
        .with_eager_session(&mut engine, |session| {
            session.rfft(&eager_real, None, -1, FftNorm::Backward)
        })
        .unwrap();
    // cuFFT plans are device-bound, so an eager run keeps them in the engine
    // and never in a runtime- or caller-owned cache (#2206).
    let first_stats = support::engine_cache_stats(&mut engine);
    assert_eq!(first_stats.entries, 1);
    assert!(first_stats.retained_bytes > 0);
    assert_eq!(first_stats.hits, 0);
    assert_eq!(first_stats.misses, 1);
    assert_eq!(first_stats.evictions, 0);
    assert_eq!(first_stats.clears, 0);

    let _eager_spectrum_repeat = ctx
        .with_eager_session(&mut engine, |session| {
            session.rfft(&eager_real, None, -1, FftNorm::Backward)
        })
        .unwrap();
    let eager_cache_stats = support::engine_cache_stats(&mut engine);
    assert_eq!(eager_cache_stats.entries, 1);
    assert!(eager_cache_stats.retained_bytes > 0);
    assert_eq!(eager_cache_stats.hits, 1);
    assert_eq!(eager_cache_stats.misses, 1);
    assert_eq!(eager_cache_stats.evictions, 0);
    assert_eq!(eager_cache_stats.clears, 0);

    let spectrum_tensor = eager_spectrum.to_tensor(&mut engine).unwrap();
    support::assert_cuda_resident(&spectrum_tensor, real_domain);
    let expected_spectrum = Operation::Rfft
        .execute_cpu(&mut cpu, &real_host, None, -1, FftNorm::Backward)
        .unwrap();
    let spectrum_host = engine
        .download_to_host(TensorRead::from_tensor(&spectrum_tensor))
        .unwrap();
    assert_host_close(&spectrum_host, &expected_spectrum, 1.0e-11);

    let eager_signal = ctx
        .with_eager_session(&mut engine, |session| {
            session.irfft(&eager_spectrum, None, -1, FftNorm::Backward)
        })
        .unwrap();
    let signal_stats = support::engine_cache_stats(&mut engine);
    assert_eq!(signal_stats.entries, 2);
    assert!(signal_stats.retained_bytes > 0);
    assert_eq!(signal_stats.hits, 1);
    assert_eq!(signal_stats.misses, 2);
    assert_eq!(signal_stats.evictions, 0);
    assert_eq!(signal_stats.clears, 0);
    let signal_tensor = eager_signal.to_tensor(&mut engine).unwrap();
    support::assert_cuda_resident(&signal_tensor, real_domain);
    let expected_signal = Operation::Irfft
        .execute_cpu(&mut cpu, &expected_spectrum, None, -1, FftNorm::Backward)
        .unwrap();
    let signal_host = engine
        .download_to_host(TensorRead::from_tensor(&signal_tensor))
        .unwrap();
    assert_host_close(&signal_host, &expected_signal, 1.0e-11);

    let complex_host = complex_f64(&[4], 0.5);
    let complex_device = engine
        .upload_host_tensor(TensorRead::from_tensor(&complex_host))
        .unwrap();
    let eager_complex = match cuda_leaf(&ctx, &mut engine, complex_device) {
        Some(leaf) => leaf,
        None => return,
    };
    let eager_fft = ctx
        .with_eager_session(&mut engine, |session| {
            session.fft(&eager_complex, None, -1, FftNorm::Backward)
        })
        .unwrap();
    let fft_stats = support::engine_cache_stats(&mut engine);
    assert_eq!(fft_stats.entries, 3);
    assert!(fft_stats.retained_bytes > 0);
    assert_eq!(fft_stats.hits, 1);
    assert_eq!(fft_stats.misses, 3);
    assert_eq!(fft_stats.evictions, 0);
    assert_eq!(fft_stats.clears, 0);
    let fft_tensor = eager_fft.to_tensor(&mut engine).unwrap();
    support::assert_cuda_resident(&fft_tensor, real_domain);
    let expected_fft = Operation::Fft
        .execute_cpu(&mut cpu, &complex_host, None, -1, FftNorm::Backward)
        .unwrap();
    let fft_host = engine
        .download_to_host(TensorRead::from_tensor(&fft_tensor))
        .unwrap();
    assert_host_close(&fft_host, &expected_fft, 1.0e-11);

    // The CPU control path stays on host placement and keeps its own plans in
    // the CPU engine, not in this CUDA engine.
    let cpu_input = EagerTensor::from_tensor_in(real_f64(&[8], 0.25), ctx.clone()).unwrap();
    let cpu_output = ctx
        .with_eager_session(&mut cpu, |session| {
            session.rfft(&cpu_input, None, -1, FftNorm::Backward)
        })
        .unwrap()
        .to_tensor(&mut cpu)
        .unwrap();
    assert_eq!(
        cpu_output.placement().memory_kind,
        tenferro_tensor::MemoryKind::UnpinnedHost
    );
    assert!(!cpu_output.is_backend_buffer());
    assert_eq!(support::engine_cache_stats(&mut engine).entries, 3);
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_eager_engine_cache_limit_reduction_evicts_and_preserves_results() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut engine = support::cuda_engine();
    let host = real_f64(&[8], 0.25);
    let device = support::upload_cuda(&engine, &host);
    let domain = TensorRead::from_tensor(&device)
        .allocation_domain()
        .unwrap();
    let ctx = EagerRuntime::new().unwrap();
    let input = match cuda_leaf(&ctx, &mut engine, device) {
        Some(leaf) => leaf,
        None => return,
    };
    let first = ctx
        .with_eager_session(&mut engine, |session| {
            session.rfft(&input, None, -1, FftNorm::Backward)
        })
        .unwrap()
        .to_tensor(&mut engine)
        .unwrap();
    let second = ctx
        .with_eager_session(&mut engine, |session| {
            session.rfft(&input, Some(6), -1, FftNorm::Backward)
        })
        .unwrap()
        .to_tensor(&mut engine)
        .unwrap();
    support::assert_cuda_resident(&first, domain);
    support::assert_cuda_resident(&second, domain);

    let populated = support::engine_cache_stats(&mut engine);
    assert_eq!(populated.entries, 2);
    assert!(populated.retained_bytes > 0);
    assert_eq!(populated.hits, 0);
    assert_eq!(populated.misses, 2);
    assert_eq!(populated.evictions, 0);
    assert_eq!(populated.clears, 0);

    let entry_limit = ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(populated.retained_bytes).unwrap());
    support::with_engine_caches(&mut engine, |caches| caches.set_limits(entry_limit));
    let entry_reduced = support::engine_cache_stats(&mut engine);
    assert_eq!(entry_reduced.entries, 1);
    assert!(entry_reduced.retained_bytes > 0);
    assert!(entry_reduced.retained_bytes <= populated.retained_bytes);
    assert_eq!(entry_reduced.hits, 0);
    assert_eq!(entry_reduced.misses, 2);
    assert_eq!(entry_reduced.evictions, 1);
    assert_eq!(entry_reduced.clears, 0);

    let byte_limit = ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(1).unwrap());
    support::with_engine_caches(&mut engine, |caches| caches.set_limits(byte_limit));
    let bytes_reduced = support::engine_cache_stats(&mut engine);
    assert_eq!(bytes_reduced.entries, 0);
    assert_eq!(bytes_reduced.retained_bytes, 0);
    assert_eq!(bytes_reduced.hits, 0);
    assert_eq!(bytes_reduced.misses, 2);
    assert_eq!(bytes_reduced.evictions, 2);
    assert_eq!(bytes_reduced.clears, 0);

    let expected = Operation::Rfft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    let first_host = engine
        .download_to_host(TensorRead::from_tensor(&first))
        .unwrap();
    assert_host_close(&first_host, &expected, 1.0e-11);

    support::with_engine_caches(&mut engine, |caches| caches.clear());
    let cleared = support::engine_cache_stats(&mut engine);
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);
    assert_eq!(cleared.hits, 0);
    assert_eq!(cleared.misses, 2);
    assert_eq!(cleared.evictions, 2);
    assert_eq!(cleared.clears, 1);
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_eager_zero_batch_returns_empty_outputs_without_engine_cache() {
    if !gpu_available() {
        return;
    }

    let mut engine = support::cuda_engine();
    let ctx = EagerRuntime::new().unwrap();
    let host = Tensor::from_vec_col_major(vec![0, 8], Vec::<Complex64>::new()).unwrap();
    let device = support::upload_cuda(&engine, &host);
    let domain = TensorRead::from_tensor(&device)
        .allocation_domain()
        .unwrap();
    let input = match cuda_leaf(&ctx, &mut engine, device) {
        Some(leaf) => leaf,
        None => return,
    };
    let output = ctx
        .with_eager_session(&mut engine, |session| {
            session.fft(&input, None, 1, FftNorm::Backward)
        })
        .unwrap();
    let tensor = output.to_tensor(&mut engine).unwrap();
    support::assert_cuda_resident(&tensor, domain);
    assert_eq!(tensor.shape(), &[0, 8]);
    assert_eq!(tensor.dtype(), tenferro_runtime::DType::C64);
    let host = engine
        .download_to_host(TensorRead::from_tensor(&tensor))
        .unwrap();
    assert_eq!(host.shape(), &[0, 8]);
    assert_eq!(host.as_slice::<Complex64>().unwrap().len(), 0);

    let first_stats = support::engine_cache_stats(&mut engine);
    assert_eq!(first_stats.entries, 0);
    assert_eq!(first_stats.retained_bytes, 0);
    assert_eq!(first_stats.hits, 0);
    assert_eq!(first_stats.misses, 0);
    assert_eq!(first_stats.evictions, 0);
    assert_eq!(first_stats.clears, 0);

    let real_host = real_f32(&[2, 0, 8], 0.5);
    let real_device = engine
        .upload_host_tensor(TensorRead::from_tensor(&real_host))
        .unwrap();
    let real_input = match cuda_leaf(&ctx, &mut engine, real_device) {
        Some(leaf) => leaf,
        None => return,
    };
    let real_output = ctx
        .with_eager_session(&mut engine, |session| {
            session.rfft(&real_input, None, 2, FftNorm::Backward)
        })
        .unwrap();
    let real_tensor = real_output.to_tensor(&mut engine).unwrap();
    support::assert_cuda_resident(&real_tensor, domain);
    assert_eq!(real_tensor.shape(), &[2, 0, 5]);
    assert_eq!(real_tensor.dtype(), tenferro_runtime::DType::C32);
    let real_host = engine
        .download_to_host(TensorRead::from_tensor(&real_tensor))
        .unwrap();
    assert_eq!(real_host.shape(), &[2, 0, 5]);
    assert_eq!(real_host.as_slice::<Complex32>().unwrap().len(), 0);

    let stats = support::engine_cache_stats(&mut engine);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.retained_bytes, 0);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.clears, 0);
    support::with_engine_caches(&mut engine, |caches| caches.clear());
    let cleared = support::engine_cache_stats(&mut engine);
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);
    assert_eq!(cleared.hits, 0);
    assert_eq!(cleared.misses, 0);
    assert_eq!(cleared.evictions, 0);
    assert_eq!(cleared.clears, 1);
}
