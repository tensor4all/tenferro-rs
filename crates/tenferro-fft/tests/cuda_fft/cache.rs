use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use super::common::*;
use super::support;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftNorm, FftPlanCache};
use tenferro_gpu::cuda::{gpu_available, CudaRuntimeIdentity};
use tenferro_runtime::ExtensionCacheLimits;
use tenferro_tensor::TensorRead;

/// A caller-held `FftExecutor` cache must stay empty on CUDA: cuFFT plans are
/// device-bound and therefore live in the engine that owns the device/stream.
fn assert_executor_cache_empty(executor: &FftExecutor) {
    let stats = executor.cache_stats();
    assert_eq!(stats.entries, 0, "caller-owned cache stats: {stats:?}");
    assert_eq!(
        stats.retained_bytes, 0,
        "caller-owned cache stats: {stats:?}"
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_plans_live_in_the_engine_not_the_caller_owned_cache() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let mut executor = FftExecutor::default();
    let c32 = complex_f32(&[4], 0.5);
    let c64 = complex_f64(&[4], 0.5);
    let batched = complex_f32(&[2, 4], -0.25);
    let real = real_f32(&[4], 0.5);

    assert_eq!(support::engine_cache_stats(&mut cuda).entries, 0);
    assert_executor_cache_empty(&executor);

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
    let first = support::engine_cache_stats(&mut cuda);
    assert_eq!(first.entries, 1);
    assert!(first.retained_bytes > 0, "engine cache stats: {first:?}");
    assert_eq!(first.hits, 0);
    assert_eq!(first.misses, 1);
    assert_eq!(first.evictions, 0);
    assert_eq!(first.clears, 0);
    assert_executor_cache_empty(&executor);

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
    let repeated = support::engine_cache_stats(&mut cuda);
    assert_eq!(repeated.entries, 1);
    assert!(
        repeated.retained_bytes > 0,
        "engine cache stats: {repeated:?}"
    );
    assert_eq!(repeated.hits, 1);
    assert_eq!(repeated.misses, 1);
    assert_eq!(repeated.evictions, 0);
    assert_eq!(repeated.clears, 0);

    // A plan belongs to the engine, not to the executor handle: a fresh
    // caller-owned cache on the same engine reuses the same plan.
    let mut fresh = FftExecutor::default();
    run_executor_case(
        &mut fresh,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    let reused_across_executors = support::engine_cache_stats(&mut cuda);
    assert_eq!(reused_across_executors.entries, 1);
    assert_eq!(reused_across_executors.hits, 2);
    assert_eq!(reused_across_executors.misses, 1);
    assert_executor_cache_empty(&fresh);

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
    assert_eq!(support::engine_cache_stats(&mut cuda).entries, 2);
    assert_eq!(support::engine_cache_stats(&mut cuda).hits, 2);
    assert_eq!(support::engine_cache_stats(&mut cuda).misses, 2);

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
    assert_eq!(support::engine_cache_stats(&mut cuda).entries, 3);
    assert_eq!(support::engine_cache_stats(&mut cuda).misses, 3);

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
    assert_eq!(support::engine_cache_stats(&mut cuda).entries, 4);
    assert_eq!(support::engine_cache_stats(&mut cuda).misses, 4);

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
    assert_eq!(support::engine_cache_stats(&mut cuda).entries, 5);
    assert_eq!(support::engine_cache_stats(&mut cuda).misses, 5);

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
    let separated = support::engine_cache_stats(&mut cuda);
    assert_eq!(separated.entries, 6);
    assert!(
        separated.retained_bytes > 0,
        "engine cache stats: {separated:?}"
    );
    assert_eq!(separated.hits, 2);
    assert_eq!(separated.misses, 6);
    assert_eq!(separated.evictions, 0);
    assert_eq!(separated.clears, 0);
    assert_executor_cache_empty(&executor);

    support::with_engine_caches(&mut cuda, |caches| caches.clear());
    let cleared = support::engine_cache_stats(&mut cuda);
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);
    assert_eq!(cleared.hits, 2);
    assert_eq!(cleared.misses, 6);
    assert_eq!(cleared.evictions, 0);
    assert_eq!(cleared.clears, 1);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_engines_own_separate_plan_stores_keyed_by_runtime_identity() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut first = support::cuda_engine();
    let mut second = support::cuda_engine();
    let host = complex_f64(&[4], 0.5);
    let first_input = support::upload_cuda(&first, &host);
    let second_input = support::upload_cuda(&second, &host);
    let mut executor = FftExecutor::default();

    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    assert_eq!(support::engine_cache_stats(&mut first).entries, 0);
    assert_eq!(support::engine_cache_stats(&mut second).entries, 0);

    for (engine, input) in [(&mut first, &first_input), (&mut second, &second_input)] {
        let domain = TensorRead::from_tensor(input).allocation_domain().unwrap();
        let output = Operation::Fft
            .execute_executor(&mut executor, engine, input, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&output, domain);
        let output = support::download_cuda(engine, &output).unwrap();
        assert_host_close(&output, &expected, 1.0e-11);
        // Each engine keeps its own device-bound plan, so neither one observes
        // a hit from the other engine's execution.
        let stats = support::engine_cache_stats(engine);
        assert_eq!(stats.entries, 1, "engine cache stats: {stats:?}");
        assert_eq!(stats.hits, 0, "engine cache stats: {stats:?}");
        assert_eq!(stats.misses, 1, "engine cache stats: {stats:?}");
    }
    assert_executor_cache_empty(&executor);
}

fn identity_hash(identity: &CudaRuntimeIdentity) -> u64 {
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    hasher.finish()
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_engine_plans_reuse_after_the_engine_is_moved() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let host = complex_f64(&[4], 0.5);
    let mut engine = support::cuda_engine();
    let first_key = identity_hash(&engine.runtime_identity());
    let mut executor = FftExecutor::default();

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut engine,
        &host,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    assert_eq!(support::engine_cache_stats(&mut engine).entries, 1);
    assert_eq!(support::engine_cache_stats(&mut engine).hits, 0);

    let engine = Box::new(engine);
    let mut engine = *engine;
    assert_eq!(identity_hash(&engine.runtime_identity()), first_key);
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut engine,
        &host,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    let moved_stats = support::engine_cache_stats(&mut engine);
    assert_eq!(moved_stats.entries, 1);
    assert_eq!(moved_stats.hits, 1);
    assert_eq!(moved_stats.misses, 1);

    let mut independent = support::cuda_engine();
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
    let independent_stats = support::engine_cache_stats(&mut independent);
    assert_eq!(independent_stats.entries, 1);
    assert_eq!(independent_stats.hits, 0);
    assert_eq!(independent_stats.misses, 1);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_engine_cache_limit_reduction_evicts_and_preserves_results() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let input = complex_f64(&[4], 0.5);
    let gpu_input = support::upload_cuda(&cuda, &input);
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

    let populated = support::engine_cache_stats(&mut cuda);
    assert_eq!(populated.entries, 2);
    assert!(
        populated.retained_bytes > 0,
        "engine cache stats: {populated:?}"
    );
    assert_eq!(populated.hits, 0);
    assert_eq!(populated.misses, 2);
    assert_eq!(populated.evictions, 0);
    assert_eq!(populated.clears, 0);

    let entry_limit = ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(populated.retained_bytes).unwrap());
    support::with_engine_caches(&mut cuda, |caches| caches.set_limits(entry_limit));
    let entry_reduced = support::engine_cache_stats(&mut cuda);
    assert_eq!(entry_reduced.entries, 1);
    assert!(entry_reduced.retained_bytes > 0);
    assert!(entry_reduced.retained_bytes <= populated.retained_bytes);
    assert_eq!(entry_reduced.hits, 0);
    assert_eq!(entry_reduced.misses, 2);
    assert_eq!(entry_reduced.evictions, 1);
    assert_eq!(entry_reduced.clears, 0);

    let byte_limit = ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(1).unwrap());
    support::with_engine_caches(&mut cuda, |caches| caches.set_limits(byte_limit));
    let bytes_reduced = support::engine_cache_stats(&mut cuda);
    assert_eq!(bytes_reduced.entries, 0);
    assert_eq!(bytes_reduced.retained_bytes, 0);
    assert_eq!(bytes_reduced.hits, 0);
    assert_eq!(bytes_reduced.misses, 2);
    assert_eq!(bytes_reduced.evictions, 2);
    assert_eq!(bytes_reduced.clears, 0);
    assert_executor_cache_empty(&executor);

    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
        .unwrap();
    let first_host = support::download_cuda(&cuda, &first).unwrap();
    assert_host_close(&first_host, &expected, 1.0e-11);
    drop(second);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_engine_cache_clear_and_eviction_are_safe_after_launch() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let input = complex_f64(&[32, 4096], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(&cuda, &input);
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
    let launched = support::engine_cache_stats(&mut cuda);
    assert_eq!(launched.entries, 1);
    assert!(launched.retained_bytes > 0);
    assert_eq!(launched.hits, 0);
    assert_eq!(launched.misses, 1);
    assert_eq!(launched.evictions, 0);
    assert_eq!(launched.clears, 0);

    support::with_engine_caches(&mut cuda, |caches| caches.clear());
    let cleared = support::engine_cache_stats(&mut cuda);
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.retained_bytes, 0);
    assert_eq!(cleared.hits, 0);
    assert_eq!(cleared.misses, 1);
    assert_eq!(cleared.evictions, 0);
    assert_eq!(cleared.clears, 1);
    let output_host = support::download_cuda(&cuda, &output).unwrap();
    assert_host_close(&output_host, &expected, 1.0e-11);

    let mut limited = FftExecutor::new(FftPlanCache::with_capacity(NonZeroUsize::new(1).unwrap()));
    support::with_engine_caches(&mut cuda, |caches| {
        caches.set_limits(ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap()))
    });
    // The engine's counters are cumulative across this test, so compare the
    // bounded-cache execution against the snapshot taken here.
    let before = support::engine_cache_stats(&mut cuda);
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
    let limited_stats = support::engine_cache_stats(&mut cuda);
    assert_eq!(limited_stats.entries, 1);
    assert!(limited_stats.retained_bytes > 0);
    assert_eq!(limited_stats.hits, before.hits);
    assert_eq!(limited_stats.misses, before.misses + 2);
    assert_eq!(limited_stats.evictions, before.evictions + 1);
    assert_eq!(limited_stats.clears, before.clears);
    assert_executor_cache_empty(&limited);
    let first_host = support::download_cuda(&cuda, &first).unwrap();
    let second_host = support::download_cuda(&cuda, &second).unwrap();
    assert_host_close(&first_host, &expected, 1.0e-11);
    let expected_second = Operation::Fft
        .execute_cpu(&mut cpu, &input, Some(2048), -1, FftNorm::Backward)
        .unwrap();
    assert_host_close(&second_host, &expected_second, 1.0e-11);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_cufft_drop_and_reallocate_after_vendor_barrier_is_safe() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let input = complex_f32(&[4, 64], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(&cuda, &input);
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
    drop(first);
    support::with_engine_caches(&mut cuda, |caches| caches.clear());

    // The next upload is intentionally allowed to reuse the released output
    // allocation. The preceding cuFFT call has synchronized at its FFI
    // boundary, so no vendor work can access that allocation after `drop`.
    let replacement_input = support::upload_cuda(&cuda, &input);
    let replacement = Operation::Fft
        .execute_executor(
            &mut executor,
            &mut cuda,
            &replacement_input,
            None,
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    let replacement_host = support::download_cuda(&cuda, &replacement).unwrap();
    assert_host_close(&replacement_host, &expected, 1.0e-5);
}
