use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftNorm, TensorFftExt, TensorReadFftExt};
use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensorView};

#[test]
fn compact_real_reads_roundtrip_on_all_axes_and_thread_counts() {
    macro_rules! check {
        ($real:ty, $complex:ty, $variant:ident, $complex_variant:ident, $tolerance:expr) => {{
            let shape = [64, 32, 32];
            let data: Vec<$real> = (0..65536).map(|i| (i % 17) as $real).collect();
            let read = TensorRead::from_view(TensorView::$variant(
                TypedTensorView::from_slice(shape, [1, 64, 2048], 0, &data).unwrap()));
            for threads in [1, 8] {
                let mut backend = CpuEngine::new(tenferro_cpu::thread_pool_builder().num_threads(threads).build().unwrap());
                // Managed backends clamp the request to the process CPU set.
                assert!((1..=threads).contains(&backend.num_threads()));
                for (axis, &length) in shape.iter().enumerate() {
                    for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                        {
 let session: &mut dyn tenferro_tensor::Engine = &mut backend;
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
                        };
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
    let mut origin = CpuEngine::serial();
    let mut destination = CpuEngine::serial();
    let output = {
        let s: &mut dyn tenferro_tensor::Engine = &mut origin;
        input.fft(None, 0, FftNorm::Backward, s)
    }
    .unwrap();
    let pointer = output.as_slice::<Complex64>().unwrap().as_ptr();
    // The origin session retains the lane scratch the FFT acquired from its
    // pool. Clear that working buffer so the reclamation accounting below
    // measures only the output buffer.
    origin.reset_buffer_pool();
    {
        let s: &mut dyn tenferro_tensor::Engine = &mut destination;
        s.reclaim_buffer(output)
    };
    assert_eq!(origin.buffer_pool_stats().buffers, 0);
    assert_eq!(destination.buffer_pool_stats().buffers, 1);
    let reused = {
        let s: &mut dyn tenferro_tensor::Engine = &mut destination;
        input.fft(None, 0, FftNorm::Backward, s)
    }
    .unwrap();
    assert_eq!(reused.as_slice::<Complex64>().unwrap().as_ptr(), pointer);
    // The second execution retains its own lane scratch in the destination.
    destination.reset_buffer_pool();
    assert_eq!(destination.buffer_pool_stats().buffers, 0);
    drop(reused);
    assert_eq!(origin.buffer_pool_stats().buffers, 0);
    assert_eq!(destination.buffer_pool_stats().buffers, 1);
}

#[test]
fn output_drop_recycles_while_session_remains_active() {
    let input = Tensor::from_vec_col_major([32, 32], vec![Complex64::new(1., 0.); 1024]).unwrap();
    let mut backend = CpuEngine::serial();
    {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
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
    };
    assert!(backend.buffer_pool_stats().buffers >= 2);
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
    let mut serial = CpuEngine::serial();
    let mut parallel = CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(8)
            .build()
            .unwrap(),
    );
    let expected = {
        let s: &mut dyn tenferro_tensor::Engine = &mut serial;
        input.fft(None, 2, FftNorm::Backward, s)
    }
    .unwrap();
    let actual = {
        let s: &mut dyn tenferro_tensor::Engine = &mut parallel;
        input.fft(None, 2, FftNorm::Backward, s)
    }
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
    let mut serial = CpuEngine::serial();
    let mut parallel = CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(8)
            .build()
            .unwrap(),
    );
    assert_eq!(serial.num_threads(), 1);
    // CI may expose fewer than eight CPUs; keep testing its effective budget.
    assert!((1..=8).contains(&parallel.num_threads()));
    for (axis, &length) in shape.iter().enumerate() {
        for n in [None, Some(length + 3), Some(length - 1)] {
            for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                let expected = {
                    let s: &mut dyn tenferro_tensor::Engine = &mut serial;
                    input.fft(n, axis as isize, norm, s)
                }
                .unwrap();
                let actual = {
                    let s: &mut dyn tenferro_tensor::Engine = &mut parallel;
                    input.fft(n, axis as isize, norm, s)
                }
                .unwrap();
                assert_eq!(
                    actual.as_slice::<Complex64>().unwrap(),
                    expected.as_slice::<Complex64>().unwrap(),
                    "axis={axis} n={n:?} norm={norm:?}"
                );
                let expected = {
                    let s: &mut dyn tenferro_tensor::Engine = &mut serial;
                    input.ifft(n, axis as isize, norm, s)
                }
                .unwrap();
                let actual = {
                    let s: &mut dyn tenferro_tensor::Engine = &mut parallel;
                    input.ifft(n, axis as isize, norm, s)
                }
                .unwrap();
                assert_eq!(
                    actual.as_slice::<Complex64>().unwrap(),
                    expected.as_slice::<Complex64>().unwrap()
                );
            }
        }
    }
}
