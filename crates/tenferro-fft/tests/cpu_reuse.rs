use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TensorFftExt, TensorReadFftExt};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorView, TypedTensorView};

#[test]
fn compact_real_reads_roundtrip_on_all_axes_and_thread_counts() {
    macro_rules! check {
        ($real:ty, $complex:ty, $variant:ident, $complex_variant:ident, $tolerance:expr) => {{
            let shape = [64, 32, 32];
            let data: Vec<$real> = (0..65536).map(|i| (i % 17) as $real).collect();
            let read = TensorRead::from_view(TensorView::$variant(
                TypedTensorView::from_slice(shape, [1, 64, 2048], 0, &data).unwrap()));
            for threads in [1, 8] {
                let mut backend = CpuBackend::with_threads(threads).unwrap();
                assert_eq!(backend.num_threads(), threads);
                for (axis, &length) in shape.iter().enumerate() {
                    for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                        backend.with_backend_session(|session| {
                            let spectrum = read.rfft_read(None, axis as isize, norm, session).unwrap();
                            let dims = spectrum.shape();
                            let spectrum_view = TensorRead::from_view(TensorView::$complex_variant(
                                TypedTensorView::from_slice(dims,
                                    [1, dims[0] as isize, (dims[0] * dims[1]) as isize],
                                    0, spectrum.as_slice::<$complex>().unwrap()).unwrap()));
                            let restored = spectrum_view.irfft_read(Some(length), axis as isize, norm, session).unwrap();
                            let error = restored.as_slice::<$real>().unwrap().iter().zip(&data)
                                .map(|(a, b)| (*a - *b).abs()).fold(0.0 as $real, <$real>::max);
                            assert!(error < $tolerance, "real roundtrip threads={threads} axis={axis} norm={norm:?} max_error={error}");
                            let full = read.fft_read(None, axis as isize, norm, session).unwrap();
                            let full_view = TensorRead::from_view(TensorView::$complex_variant(
                                TypedTensorView::from_slice(shape, [1, 64, 2048], 0,
                                    full.as_slice::<$complex>().unwrap()).unwrap()));
                            let restored = full_view.ifft_read(None, axis as isize, norm, session).unwrap();
                            let error = restored.as_slice::<$complex>().unwrap().iter().zip(&data)
                                .map(|(a, b)| (a.re - *b).abs().max(a.im.abs()))
                                .fold(0.0 as $real, <$real>::max);
                            assert!(error < $tolerance, "full roundtrip threads={threads} axis={axis} norm={norm:?} max_error={error}");
                        });
                    }
                }
            }
        }};
    }
    check!(f32, num_complex::Complex32, F32, C32, 1e-4);
    check!(f64, Complex64, F64, C64, 1e-10);
}

#[test]
fn explicit_reclaim_to_another_backend_does_not_also_return_to_origin() {
    let input = Tensor::from_vec_col_major([4], vec![Complex64::new(1., 0.); 4]).unwrap();
    let mut origin = CpuBackend::with_threads(1).unwrap();
    let mut destination = CpuBackend::with_threads(1).unwrap();
    let output = origin
        .with_backend_session(|s| input.fft(None, 0, FftNorm::Backward, s))
        .unwrap();
    let pointer = output.as_slice::<Complex64>().unwrap().as_ptr();
    destination.with_backend_session(|s| s.reclaim_buffer(output));
    assert_eq!(origin.buffer_pool_stats().unwrap().buffers, 0);
    assert_eq!(destination.buffer_pool_stats().unwrap().buffers, 1);
    let reused = destination
        .with_backend_session(|s| input.fft(None, 0, FftNorm::Backward, s))
        .unwrap();
    assert_eq!(reused.as_slice::<Complex64>().unwrap().as_ptr(), pointer);
    assert_eq!(destination.buffer_pool_stats().unwrap().buffers, 0);
    drop(reused);
    assert_eq!(origin.buffer_pool_stats().unwrap().buffers, 0);
    assert_eq!(destination.buffer_pool_stats().unwrap().buffers, 1);
}

#[test]
fn output_drop_recycles_while_session_remains_active() {
    let input = Tensor::from_vec_col_major([32, 32], vec![Complex64::new(1., 0.); 1024]).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    backend.with_backend_session(|session| {
        let first = input.fft(None, 0, FftNorm::Backward, session).unwrap();
        let first_pointer = first.as_slice::<Complex64>().unwrap().as_ptr();
        let second = input.fft(None, 0, FftNorm::Backward, session).unwrap();
        assert_ne!(
            second.as_slice::<Complex64>().unwrap().as_ptr(),
            first_pointer
        );
        drop(first);
        let third = input.fft(None, 0, FftNorm::Backward, session).unwrap();
        assert_eq!(
            third.as_slice::<Complex64>().unwrap().as_ptr(),
            first_pointer
        );
        assert_eq!(
            second.as_slice::<Complex64>().unwrap(),
            third.as_slice::<Complex64>().unwrap()
        );
    });
    assert!(backend.buffer_pool_stats().unwrap().buffers >= 2);
}

#[test]
fn nested_fft_session_is_rejected_and_backend_remains_usable() {
    let mut backend = CpuBackend::with_threads(8).unwrap();
    let mut nested = backend.clone();
    let input = Tensor::from_vec_col_major([2], vec![Complex64::new(1., 0.); 2]).unwrap();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_backend_session(|_| {
            nested.with_backend_session(|s| input.fft(None, 0, FftNorm::Backward, s))
        })
    }));
    let payload = outcome.expect_err("CPU admission must reject a nested execution");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(
        message.contains("another CPU backend execution"),
        "{message}"
    );
    let result = backend
        .with_backend_session(|s| input.fft(None, 0, FftNorm::Backward, s))
        .unwrap();
    assert_eq!(
        result.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2., 0.), Complex64::new(0., 0.)]
    );
}

#[test]
fn uneven_parallel_lane_partition_matches_serial() {
    // Nine lanes over eight jobs leaves both a short final job and empty jobs.
    let input = Tensor::from_vec_col_major(
        [3, 3, 4096],
        (0..36864)
            .map(|i| Complex64::new((i % 17) as f64, (i % 23) as f64))
            .collect(),
    )
    .unwrap();
    let mut serial = CpuBackend::with_threads(1).unwrap();
    let mut parallel = CpuBackend::with_threads(8).unwrap();
    let expected = serial
        .with_backend_session(|s| input.fft(None, 2, FftNorm::Backward, s))
        .unwrap();
    let actual = parallel
        .with_backend_session(|s| input.fft(None, 2, FftNorm::Backward, s))
        .unwrap();
    assert_eq!(
        actual.as_slice::<Complex64>().unwrap(),
        expected.as_slice::<Complex64>().unwrap()
    );
}

#[test]
fn parallel_lanes_match_serial_on_every_axis_with_padding_and_truncation() {
    let shape = [64, 32, 32];
    let input = Tensor::from_vec_col_major(
        shape,
        (0..65536)
            .map(|i| Complex64::new((i % 17) as f64, (i % 23) as f64))
            .collect(),
    )
    .unwrap();
    let mut serial = CpuBackend::with_threads(1).unwrap();
    let mut parallel = CpuBackend::with_threads(8).unwrap();
    assert_eq!(serial.num_threads(), 1);
    assert_eq!(parallel.num_threads(), 8);
    for (axis, &length) in shape.iter().enumerate() {
        for n in [None, Some(length + 3), Some(length - 1)] {
            for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                let expected = serial
                    .with_backend_session(|s| input.fft(n, axis as isize, norm, s))
                    .unwrap();
                let actual = parallel
                    .with_backend_session(|s| input.fft(n, axis as isize, norm, s))
                    .unwrap();
                assert_eq!(
                    actual.as_slice::<Complex64>().unwrap(),
                    expected.as_slice::<Complex64>().unwrap(),
                    "axis={axis} n={n:?} norm={norm:?}"
                );
                let expected = serial
                    .with_backend_session(|s| input.ifft(n, axis as isize, norm, s))
                    .unwrap();
                let actual = parallel
                    .with_backend_session(|s| input.ifft(n, axis as isize, norm, s))
                    .unwrap();
                assert_eq!(
                    actual.as_slice::<Complex64>().unwrap(),
                    expected.as_slice::<Complex64>().unwrap()
                );
            }
        }
    }
}
