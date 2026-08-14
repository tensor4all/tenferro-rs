use num_complex::Complex64;
use std::num::NonZeroUsize;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::ExtensionCacheLimits;
use tenferro_tensor::BackendSessionHost;
use tenferro_tensor::{ErrorKind, Tensor, TensorRead, TensorView, TypedTensorView};

use crate::{FftExecutor, FftNorm, FftPlanCache, TensorFftExt, TensorReadFftExt};

fn assert_complex_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).norm() < 1.0e-12,
            "actual {actual:?} expected {expected:?}"
        );
    }
}

fn assert_real_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).abs() < 1.0e-12,
            "actual {actual:?} expected {expected:?}"
        );
    }
}

#[test]
fn public_tensor_fft_ext_executes_real_and_complex_transforms() {
    let mut backend = CpuBackend::new();
    let real = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let (full, onesided, recovered, recovered_real) = backend.with_backend_session(|session| {
        let full = real.fft(None, -1, FftNorm::Backward, session).unwrap();
        let onesided = real.rfft(None, -1, FftNorm::Backward, session).unwrap();
        let recovered = full.ifft(None, -1, FftNorm::Backward, session).unwrap();
        let recovered_real = onesided
            .irfft(Some(4), -1, FftNorm::Backward, session)
            .unwrap();
        (full, onesided, recovered, recovered_real)
    });

    assert_eq!(full.shape(), &[4]);
    assert_complex_close(
        full.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );
    assert_eq!(onesided.shape(), &[3]);
    assert_complex_close(
        onesided.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    );

    assert_complex_close(
        recovered.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    );

    assert_real_close(
        recovered_real.as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );
}

#[test]
fn public_tensor_read_fft_ext_accepts_strided_host_views() {
    let mut backend = CpuBackend::new();
    let data = [1.0_f64, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let view = TypedTensorView::from_slice([4], [2], 0, &data).unwrap();
    let input = TensorRead::from_view(TensorView::F64(view));

    let (full, onesided) = backend.with_backend_session(|session| {
        let full = input
            .fft_read(None, -1, FftNorm::Backward, session)
            .unwrap();
        let onesided = input
            .rfft_read(None, -1, FftNorm::Backward, session)
            .unwrap();
        (full, onesided)
    });

    assert_complex_close(
        full.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );
    assert_complex_close(
        onesided.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    );
}

#[test]
fn public_tensor_fft_ext_reports_invalid_dtype_and_shape_errors() {
    let mut backend = CpuBackend::new();
    let bools = Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    let spectrum = Tensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();

    let (dtype_err, shape_err) = backend.with_backend_session(|session| {
        let dtype_err = bools.fft(None, -1, FftNorm::Backward, session).unwrap_err();
        let shape_err = spectrum
            .irfft(Some(6), -1, FftNorm::Backward, session)
            .unwrap_err();
        (dtype_err, shape_err)
    });

    assert!(matches!(
        dtype_err,
        tenferro_tensor::Error::Extension {
            op: "TensorFftExt::fft",
            kind: ErrorKind::Unsupported,
            ..
        }
    ));
    assert!(matches!(
        shape_err,
        tenferro_tensor::Error::Validation { op: "irfft", .. }
    ));
}

#[test]
fn fft_plan_cache_is_bounded_lru_and_reports_known_retention() {
    let mut cache = FftPlanCache::with_capacity(NonZeroUsize::new(2).unwrap());
    assert_eq!(cache.capacity().get(), 2);
    let first = cache.plan_f64(4, true);
    cache.plan_f64(8, true);
    assert!(std::sync::Arc::ptr_eq(&first, &cache.plan_f64(4, true)));
    cache.plan_f64(16, true);

    assert!(cache.contains_f64(4, true));
    assert!(!cache.contains_f64(8, true));
    assert!(cache.contains_f64(16, true));
    assert_eq!(cache.stats().entries, 2);
    assert!(cache.stats().retained_bytes > 0);
    assert!(std::sync::Arc::ptr_eq(&first, &cache.plan_f64(4, true)));

    cache.clear();
    let stats = cache.stats();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.retained_bytes, 0);
    assert_eq!(stats.hits, 4);
    assert_eq!(stats.misses, 4);
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.clears, 1);
    cache.set_capacity(NonZeroUsize::MIN);
    assert_eq!(cache.capacity(), NonZeroUsize::MIN);
}

#[test]
fn fft_plan_cache_exposes_full_extension_cache_limits() {
    let mut cache = FftPlanCache::with_capacity(NonZeroUsize::new(8).unwrap());
    let limits = ExtensionCacheLimits::new(NonZeroUsize::new(8).unwrap())
        .with_max_retained_bytes(NonZeroUsize::new(1).unwrap());

    cache.set_limits(limits);
    assert_eq!(cache.limits(), limits);
    cache.plan_f64(4, true);

    assert_eq!(cache.stats().entries, 0);
    assert_eq!(cache.stats().evictions, 1);
}

#[test]
fn fft_plan_cache_key_distinguishes_scalar_dtype_length_and_direction() {
    let mut cache = FftPlanCache::with_capacity(NonZeroUsize::new(8).unwrap());

    let f32_forward_4 = cache.plan_f32(4, true);
    let f32_forward_8 = cache.plan_f32(8, true);
    let f64_forward_4 = cache.plan_f64(4, true);
    let f64_inverse_4 = cache.plan_f64(4, false);

    assert_eq!(cache.stats().entries, 4);
    assert!(std::sync::Arc::ptr_eq(
        &f32_forward_4,
        &cache.plan_f32(4, true)
    ));
    assert!(std::sync::Arc::ptr_eq(
        &f32_forward_8,
        &cache.plan_f32(8, true)
    ));
    assert!(std::sync::Arc::ptr_eq(
        &f64_forward_4,
        &cache.plan_f64(4, true)
    ));
    assert!(std::sync::Arc::ptr_eq(
        &f64_inverse_4,
        &cache.plan_f64(4, false)
    ));
    assert!(!std::sync::Arc::ptr_eq(&f64_forward_4, &f64_inverse_4));
}

#[test]
fn caller_owned_fft_executor_reuses_plans() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let mut executor = FftExecutor::default();

    backend.with_backend_session(|session| {
        let full = executor
            .fft(&input, None, -1, FftNorm::Backward, session)
            .unwrap();
        executor
            .ifft(&full, None, -1, FftNorm::Backward, session)
            .unwrap();
        let onesided = executor
            .rfft(&input, None, -1, FftNorm::Backward, session)
            .unwrap();
        executor
            .irfft(&onesided, Some(4), -1, FftNorm::Backward, session)
            .unwrap();
    });

    assert_eq!(executor.cache_stats().entries, 2);
    assert_eq!(executor.plan_cache().stats().entries, 2);
    executor.plan_cache_mut().set_capacity(NonZeroUsize::MIN);
    assert_eq!(executor.cache_stats().entries, 1);
    executor.clear_cache();
    assert_eq!(executor.cache_stats().entries, 0);
}

#[test]
fn configured_fft_executor_validates_each_public_operation_before_dispatch() {
    let mut executor = FftExecutor::new(FftPlanCache::default());
    let mut backend = CpuBackend::new();
    let real = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let complex = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();

    backend.with_backend_session(|session| {
        assert!(executor
            .fft(&real, Some(0), -1, FftNorm::Backward, session)
            .is_err());
        assert!(executor
            .ifft(&real, None, -1, FftNorm::Backward, session)
            .is_err());
        assert!(executor
            .rfft(&complex, None, -1, FftNorm::Backward, session)
            .is_err());
        assert!(executor
            .irfft(&complex, Some(0), -1, FftNorm::Backward, session)
            .is_err());
    });
}
