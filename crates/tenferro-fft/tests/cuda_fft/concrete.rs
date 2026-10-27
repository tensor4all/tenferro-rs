use super::common::*;
use super::support;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftNorm};
use tenferro_gpu::cuda_next::gpu_available;
use tenferro_tensor::{Tensor, TensorRead};

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_c2c_f32_f64_forward_inverse() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    for (input, tolerance) in [
        (complex_f32(&[4], 0.5), 1.0e-5),
        (complex_f64(&[4], 0.5), 1.0e-11),
    ] {
        let cpu_forward = Operation::Fft
            .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_input = support::upload_cuda(&cuda, &input);
        let gpu_domain = TensorRead::from_tensor(&gpu_input)
            .allocation_domain()
            .unwrap();
        let gpu_forward = Operation::Fft
            .execute_cuda(&mut cuda, &gpu_input, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_forward, gpu_domain);
        let forward = support::download_cuda(&cuda, &gpu_forward).unwrap();
        assert_host_close(&forward, &cpu_forward, tolerance);

        let cpu_inverse = Operation::Ifft
            .execute_cpu(&mut cpu, &cpu_forward, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_inverse = Operation::Ifft
            .execute_cuda(&mut cuda, &gpu_forward, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_inverse, gpu_domain);
        let inverse = support::download_cuda(&cuda, &gpu_inverse).unwrap();
        assert_host_close(&inverse, &cpu_inverse, tolerance);
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_c2c_explicit_three_by_four_case() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    for (input, tolerance) in [
        (complex_f32(&[3, 4], 0.5), 1.0e-5),
        (complex_f64(&[3, 4], 0.5), 1.0e-11),
    ] {
        run_case(
            &mut cpu,
            &mut cuda,
            &input,
            Operation::Fft,
            None,
            -1,
            FftNorm::Backward,
            tolerance,
        );
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_r2c_c2r_f32_f64() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    for (input, tolerance) in [
        (real_f32(&[8], 0.25), 1.0e-5),
        (real_f64(&[8], 0.25), 1.0e-11),
    ] {
        let cpu_spectrum = Operation::Rfft
            .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_input = support::upload_cuda(&cuda, &input);
        let gpu_domain = TensorRead::from_tensor(&gpu_input)
            .allocation_domain()
            .unwrap();
        let gpu_spectrum = Operation::Rfft
            .execute_cuda(&mut cuda, &gpu_input, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_spectrum, gpu_domain);
        let spectrum = support::download_cuda(&cuda, &gpu_spectrum).unwrap();
        assert_host_close(&spectrum, &cpu_spectrum, tolerance);

        let cpu_signal = Operation::Irfft
            .execute_cpu(&mut cpu, &cpu_spectrum, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_signal = Operation::Irfft
            .execute_cuda(&mut cuda, &gpu_spectrum, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_signal, gpu_domain);
        let signal = support::download_cuda(&cuda, &gpu_signal).unwrap();
        assert_host_close(&signal, &cpu_signal, tolerance);
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_real_fft_completes_even_and_odd_hermitian_spectrum() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    for n in [8usize, 7, 2, 1] {
        for (input, tolerance) in [
            (real_f32(&[n], 0.25), 1.0e-5),
            (real_f64(&[n], 0.25), 1.0e-11),
        ] {
            let actual = run_case(
                &mut cpu,
                &mut cuda,
                &input,
                Operation::Fft,
                None,
                -1,
                FftNorm::Backward,
                tolerance,
            );
            assert_eq!(actual.shape(), &[n]);
            assert_full_hermitian(&actual);
        }
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_irfft_explicit_even_and_odd_lengths_c32_c64() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    for (input, n, tolerance) in [
        (real_f32(&[8], 0.25), Some(8), 1.0e-5),
        (real_f32(&[7], 0.25), Some(7), 1.0e-5),
        (real_f64(&[8], 0.25), Some(8), 1.0e-11),
        (real_f64(&[7], 0.25), Some(7), 1.0e-11),
    ] {
        let spectrum = run_case(
            &mut cpu,
            &mut cuda,
            &input,
            Operation::Rfft,
            n,
            -1,
            FftNorm::Backward,
            tolerance,
        );
        run_case(
            &mut cpu,
            &mut cuda,
            &spectrum,
            Operation::Irfft,
            n,
            -1,
            FftNorm::Backward,
            tolerance,
        );
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_zero_padding_matches_explicit_padding_real_and_complex() {
    assert!(gpu_available(), "requires CUDA hardware");

    let mut cuda = support::cuda_engine();
    let cases = [
        (
            Tensor::from_vec_col_major([1], vec![f32::NAN]).unwrap(),
            Tensor::from_vec_col_major([2], vec![f32::NAN, 0.0]).unwrap(),
            1.0e-5,
        ),
        (
            Tensor::from_vec_col_major([1], vec![f64::INFINITY]).unwrap(),
            Tensor::from_vec_col_major([2], vec![f64::INFINITY, 0.0]).unwrap(),
            1.0e-11,
        ),
        (
            Tensor::from_vec_col_major([1], vec![num_complex::Complex32::new(f32::NAN, 0.0)])
                .unwrap(),
            Tensor::from_vec_col_major(
                [2],
                vec![
                    num_complex::Complex32::new(f32::NAN, 0.0),
                    num_complex::Complex32::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            1.0e-5,
        ),
        (
            Tensor::from_vec_col_major([1], vec![num_complex::Complex64::new(f64::INFINITY, 0.0)])
                .unwrap(),
            Tensor::from_vec_col_major(
                [2],
                vec![
                    num_complex::Complex64::new(f64::INFINITY, 0.0),
                    num_complex::Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            1.0e-11,
        ),
    ];
    // NaN/Inf propagation depends on the FFT algorithm: x86 realfft and cuFFT
    // differ even for [Inf, 0]. Compare identical inputs on the same backend.
    // The canonicalization unit tests also inspect the zero tail before cuFFT,
    // so a contaminated tail cannot be hidden by nonfinite transform outputs.
    for (input, padded, tolerance) in cases {
        let padded = support::upload_cuda(&cuda, &padded);
        let expected = Operation::Fft
            .execute_cuda(&mut cuda, &padded, None, -1, FftNorm::Backward)
            .unwrap();
        let expected = support::download_cuda(&cuda, &expected).unwrap();
        let gpu_input = support::upload_cuda(&cuda, &input);
        let actual = Operation::Fft
            .execute_cuda(&mut cuda, &gpu_input, Some(2), -1, FftNorm::Backward)
            .unwrap();
        let actual = support::download_cuda(&cuda, &actual).unwrap();
        assert_host_close_allow_nonfinite(&actual, &expected, tolerance);
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_axes_final_middle_and_negative() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let input = complex_f64(&[2, 3, 5], 0.125);
    for (axis, label) in [(2isize, "final"), (1, "middle"), (-1, "negative-final")] {
        let _ = label;
        run_case(
            &mut cpu,
            &mut cuda,
            &input,
            Operation::Fft,
            None,
            axis,
            FftNorm::Backward,
            1.0e-11,
        );
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_multiple_interleaved_column_major_batches() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let input = complex_f64(&[2, 3, 5], -0.75);
    run_case(
        &mut cpu,
        &mut cuda,
        &input,
        Operation::Fft,
        None,
        2,
        FftNorm::Backward,
        1.0e-11,
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_zero_transform_axis_with_nonzero_batch_is_padded_on_device() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let mut executor = FftExecutor::default();
    let complex = complex_f32(&[2, 0], 0.5);
    let actual = run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &complex,
        Operation::Fft,
        Some(4),
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    assert_eq!(actual.shape(), &[2, 4]);
    let stats = support::engine_cache_stats(&mut cuda);
    assert_eq!(stats.entries, 1, "engine cache stats: {stats:?}");
    assert!(stats.retained_bytes > 0, "engine cache stats: {stats:?}");

    let real = real_f64(&[2, 0], 0.5);
    let actual = run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &real,
        Operation::Fft,
        Some(4),
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    assert_eq!(actual.shape(), &[2, 4]);

    let middle_axis = complex_f32(&[2, 0, 3], -0.25);
    let actual = run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &middle_axis,
        Operation::Fft,
        Some(4),
        1,
        FftNorm::Backward,
        1.0e-5,
    );
    assert_eq!(actual.shape(), &[2, 4, 3]);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_lengths_equal_truncated_and_zero_padded() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let input = complex_f32(&[5], 0.5);
    for n in [None, Some(5), Some(3), Some(8)] {
        run_case(
            &mut cpu,
            &mut cuda,
            &input,
            Operation::Fft,
            n,
            -1,
            FftNorm::Backward,
            1.0e-5,
        );
    }

    let real_input = real_f64(&[5], 0.5);
    for n in [None, Some(5), Some(3), Some(8)] {
        run_case(
            &mut cpu,
            &mut cuda,
            &real_input,
            Operation::Rfft,
            n,
            -1,
            FftNorm::Backward,
            1.0e-11,
        );
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_backward_forward_and_ortho_norms() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuEngine::serial();
    let mut cuda = support::cuda_engine();
    let input = complex_f64(&[4], 0.5);
    let real_input = real_f64(&[8], 0.25);
    for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
        run_case(
            &mut cpu,
            &mut cuda,
            &input,
            Operation::Fft,
            None,
            -1,
            norm,
            1.0e-11,
        );
        run_case(
            &mut cpu,
            &mut cuda,
            &input,
            Operation::Ifft,
            None,
            -1,
            norm,
            1.0e-11,
        );
        let spectrum = run_case(
            &mut cpu,
            &mut cuda,
            &real_input,
            Operation::Rfft,
            None,
            -1,
            norm,
            1.0e-11,
        );
        run_case(
            &mut cpu,
            &mut cuda,
            &spectrum,
            Operation::Irfft,
            None,
            -1,
            norm,
            1.0e-11,
        );
    }
}
