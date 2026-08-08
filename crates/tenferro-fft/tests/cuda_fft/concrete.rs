use super::common::*;
use super::support;
use tenferro_cpu::CpuBackend;
use tenferro_fft::FftNorm;
use tenferro_gpu::cuda::gpu_available;
use tenferro_tensor::TensorRead;

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_c2c_f32_f64_forward_inverse() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    for (input, tolerance) in [
        (complex_f32(&[4], 0.5), 1.0e-5),
        (complex_f64(&[4], 0.5), 1.0e-11),
    ] {
        let cpu_forward = Operation::Fft
            .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_input = support::upload_cuda(cuda.runtime(), &input);
        let gpu_domain = TensorRead::from_tensor(&gpu_input)
            .allocation_domain()
            .unwrap();
        let gpu_forward = Operation::Fft
            .execute_cuda(&mut cuda, &gpu_input, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_forward, gpu_domain);
        let forward = support::download_cuda(cuda.runtime(), &gpu_forward).unwrap();
        assert_host_close(&forward, &cpu_forward, tolerance);

        let cpu_inverse = Operation::Ifft
            .execute_cpu(&mut cpu, &cpu_forward, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_inverse = Operation::Ifft
            .execute_cuda(&mut cuda, &gpu_forward, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_inverse, gpu_domain);
        let inverse = support::download_cuda(cuda.runtime(), &gpu_inverse).unwrap();
        assert_host_close(&inverse, &cpu_inverse, tolerance);
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_c2c_explicit_three_by_four_case() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
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

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    for (input, tolerance) in [
        (real_f32(&[8], 0.25), 1.0e-5),
        (real_f64(&[8], 0.25), 1.0e-11),
    ] {
        let cpu_spectrum = Operation::Rfft
            .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_input = support::upload_cuda(cuda.runtime(), &input);
        let gpu_domain = TensorRead::from_tensor(&gpu_input)
            .allocation_domain()
            .unwrap();
        let gpu_spectrum = Operation::Rfft
            .execute_cuda(&mut cuda, &gpu_input, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_spectrum, gpu_domain);
        let spectrum = support::download_cuda(cuda.runtime(), &gpu_spectrum).unwrap();
        assert_host_close(&spectrum, &cpu_spectrum, tolerance);

        let cpu_signal = Operation::Irfft
            .execute_cpu(&mut cpu, &cpu_spectrum, None, -1, FftNorm::Backward)
            .unwrap();
        let gpu_signal = Operation::Irfft
            .execute_cuda(&mut cuda, &gpu_spectrum, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&gpu_signal, gpu_domain);
        let signal = support::download_cuda(cuda.runtime(), &gpu_signal).unwrap();
        assert_host_close(&signal, &cpu_signal, tolerance);
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_real_fft_completes_even_and_odd_hermitian_spectrum() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    for n in [8usize, 7] {
        let input = real_f64(&[n], 0.25);
        let actual = run_case(
            &mut cpu,
            &mut cuda,
            &input,
            Operation::Fft,
            None,
            -1,
            FftNorm::Backward,
            1.0e-11,
        );
        assert_eq!(actual.shape(), &[n]);
        assert_full_hermitian(&actual);
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_axes_final_middle_and_negative() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
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

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
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
fn cuda_lengths_equal_truncated_and_zero_padded() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
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

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
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
