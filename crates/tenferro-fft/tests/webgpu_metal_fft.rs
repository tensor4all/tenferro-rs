#[cfg(target_os = "macos")]
mod metal {
    use num_complex::{Complex32, Complex64};
    use tenferro_fft::{FftNorm, TensorFftExt};
    use tenferro_gpu::{upload_webgpu_tensor, webgpu_interop, AppleContext, WebGpuRuntime};
    use tenferro_tensor::{Buffer, Error, Tensor};

    fn mapped<T: Copy + Send + Sync + 'static>(tensor: &tenferro_tensor::TypedTensor<T>) -> Vec<T> {
        let Buffer::Backend(buffer) = tensor.buffer() else {
            panic!("expected managed backend buffer")
        };
        buffer.map_read().unwrap().to_vec()
    }

    fn c32_values(tensor: &Tensor) -> Vec<Complex32> {
        let Tensor::C32(tensor) = tensor else {
            panic!("expected C32 tensor")
        };
        mapped(tensor)
    }

    fn f32_values(tensor: &Tensor) -> Vec<f32> {
        let Tensor::F32(tensor) = tensor else {
            panic!("expected F32 tensor")
        };
        mapped(tensor)
    }

    fn assert_c32_close(actual: &[Complex32], expected: &[Complex32], tolerance: f32) {
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error <= tolerance,
            "C32 max absolute error {max_error} exceeds {tolerance}"
        );
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error <= tolerance,
            "F32 max absolute error {max_error} exceeds {tolerance}"
        );
    }

    #[test]
    fn metal_fft_numerical_and_capability_matrix() {
        let Ok(context) = AppleContext::new() else {
            return;
        };
        cfft_axes_batches_and_normalizations(&context);
        rfft_padding_truncation_and_round_trip(&context);
        small_and_large_threshold_paths(&context);
        capability_and_placement_errors(&context);
    }

    fn cfft_axes_batches_and_normalizations(context: &AppleContext) {
        let input_values = (0..24)
            .map(|index| Complex32::new(index as f32 * 0.03125, (index % 5) as f32 * -0.02))
            .collect::<Vec<_>>();
        let input = Tensor::from_vec_col_major(vec![8, 3], input_values).unwrap();
        let managed = context.upload_tensor(&input).unwrap();
        let Tensor::C32(managed_typed) = &managed else {
            panic!("expected C32 input")
        };
        let input_allocation = managed_typed.allocation_id().unwrap();
        let before = context.transfer_stats();

        for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
            let mut cpu = tenferro_cpu::CpuBackend::new();
            let reference = input.fft(None, 0, norm, &mut cpu).unwrap();
            let output = managed
                .fft(None, 0, norm, &mut context.metal_backend().clone())
                .unwrap();
            context.metal_backend().synchronize().unwrap();
            assert_c32_close(&c32_values(&output), reference.as_slice().unwrap(), 2.0e-5);

            let round_trip = output
                .ifft(None, 0, norm, &mut context.metal_backend().clone())
                .unwrap();
            let reference_round_trip = reference.ifft(None, 0, norm, &mut cpu).unwrap();
            context.metal_backend().synchronize().unwrap();
            assert_c32_close(
                &c32_values(&round_trip),
                reference_round_trip.as_slice().unwrap(),
                3.0e-5,
            );
            let Tensor::C32(output_typed) = output else {
                panic!("expected C32 output")
            };
            assert_eq!(output_typed.allocation_domain(), Some(context.domain_id()));
            assert_ne!(output_typed.allocation_id(), Some(input_allocation));
        }

        let axis_one_input = Tensor::from_vec_col_major(
            vec![2, 4, 2],
            (0..16)
                .map(|index| Complex32::new(index as f32 * 0.125, 0.0))
                .collect(),
        )
        .unwrap();
        let managed = context.upload_tensor(&axis_one_input).unwrap();
        let mut cpu = tenferro_cpu::CpuBackend::new();
        let reference = axis_one_input
            .fft(None, 1, FftNorm::Backward, &mut cpu)
            .unwrap();
        let output = managed
            .fft(
                None,
                1,
                FftNorm::Backward,
                &mut context.metal_backend().clone(),
            )
            .unwrap();
        context.metal_backend().synchronize().unwrap();
        assert_c32_close(&c32_values(&output), reference.as_slice().unwrap(), 2.0e-5);
        assert_eq!(
            context.transfer_stats().downloaded_bytes,
            before.downloaded_bytes
        );
    }

    fn rfft_padding_truncation_and_round_trip(context: &AppleContext) {
        let cases = [
            (vec![3], vec![1.0_f32, -2.0, 0.5], 8usize, 0isize),
            (
                vec![2, 8],
                (0..16).map(|index| index as f32 * 0.0625).collect(),
                4,
                1,
            ),
        ];
        for (shape, values, n_fft, axis) in cases {
            let input = Tensor::from_vec_col_major(shape, values).unwrap();
            let managed = context.upload_tensor(&input).unwrap();
            let before = context.transfer_stats();
            let mut cpu = tenferro_cpu::CpuBackend::new();
            let reference = input
                .rfft(Some(n_fft), axis, FftNorm::Ortho, &mut cpu)
                .unwrap();
            let spectrum = managed
                .rfft(
                    Some(n_fft),
                    axis,
                    FftNorm::Ortho,
                    &mut context.metal_backend().clone(),
                )
                .unwrap();
            context.metal_backend().synchronize().unwrap();
            assert_c32_close(
                &c32_values(&spectrum),
                reference.as_slice().unwrap(),
                2.0e-5,
            );
            let round_trip = spectrum
                .irfft(
                    Some(n_fft),
                    axis,
                    FftNorm::Ortho,
                    &mut context.metal_backend().clone(),
                )
                .unwrap();
            let reference_round_trip = reference
                .irfft(Some(n_fft), axis, FftNorm::Ortho, &mut cpu)
                .unwrap();
            context.metal_backend().synchronize().unwrap();
            assert_f32_close(
                &f32_values(&round_trip),
                reference_round_trip.as_slice().unwrap(),
                3.0e-5,
            );
            let Tensor::F32(round_trip) = round_trip else {
                panic!("expected F32 output")
            };
            assert_eq!(round_trip.allocation_domain(), Some(context.domain_id()));
            assert_eq!(context.transfer_stats(), before);
        }
    }

    fn small_and_large_threshold_paths(context: &AppleContext) {
        let shared_elements = webgpu_interop::max_shared_memory_size(context.metal_backend())
            / (2 * core::mem::size_of::<f32>());
        let max_shared = if shared_elements.is_power_of_two() {
            shared_elements
        } else {
            shared_elements.next_power_of_two() >> 1
        };
        for n_fft in [max_shared, max_shared * 2] {
            let complex_values = (0..n_fft)
                .map(|index| {
                    let sample = ((index * 17) % 97) as f32 * 1.0e-4;
                    Complex32::new(sample, -sample * 0.25)
                })
                .collect::<Vec<_>>();
            let complex = Tensor::from_vec_col_major(vec![n_fft], complex_values).unwrap();
            let managed = context.upload_tensor(&complex).unwrap();
            let mut cpu = tenferro_cpu::CpuBackend::new();
            let reference = complex.fft(None, 0, FftNorm::Backward, &mut cpu).unwrap();
            let output = managed
                .fft(
                    None,
                    0,
                    FftNorm::Backward,
                    &mut context.metal_backend().clone(),
                )
                .unwrap();
            context.metal_backend().synchronize().unwrap();
            assert_c32_close(&c32_values(&output), reference.as_slice().unwrap(), 2.0e-3);

            let real_values = (0..n_fft)
                .map(|index| ((index * 13) % 89) as f32 * 1.0e-4)
                .collect::<Vec<_>>();
            let real = Tensor::from_vec_col_major(vec![n_fft], real_values).unwrap();
            let managed = context.upload_tensor(&real).unwrap();
            let reference = real.rfft(None, 0, FftNorm::Backward, &mut cpu).unwrap();
            let output = managed
                .rfft(
                    None,
                    0,
                    FftNorm::Backward,
                    &mut context.metal_backend().clone(),
                )
                .unwrap();
            context.metal_backend().synchronize().unwrap();
            assert_c32_close(&c32_values(&output), reference.as_slice().unwrap(), 2.0e-3);
            let round_trip = output
                .irfft(
                    Some(n_fft),
                    0,
                    FftNorm::Backward,
                    &mut context.metal_backend().clone(),
                )
                .unwrap();
            context.metal_backend().synchronize().unwrap();
            assert_f32_close(&f32_values(&round_trip), real.as_slice().unwrap(), 2.0e-3);
        }
    }

    fn capability_and_placement_errors(context: &AppleContext) {
        let f64_input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let c64_input = Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        )
        .unwrap();
        let f32_input = Tensor::from_vec_col_major(vec![4], vec![1.0_f32; 4]).unwrap();
        let c32_input =
            Tensor::from_vec_col_major(vec![4], vec![Complex32::new(1.0, 0.0); 4]).unwrap();
        for (input, transform) in [
            (context.upload_tensor(&f64_input).unwrap(), "rfft"),
            (context.upload_tensor(&c64_input).unwrap(), "fft"),
        ] {
            let error = if transform == "rfft" {
                input.rfft(
                    None,
                    0,
                    FftNorm::Backward,
                    &mut context.metal_backend().clone(),
                )
            } else {
                input.fft(
                    None,
                    0,
                    FftNorm::Backward,
                    &mut context.metal_backend().clone(),
                )
            }
            .unwrap_err();
            assert!(matches!(error, Error::Unsupported { .. }));
        }

        let managed_real = context.upload_tensor(&f32_input).unwrap();
        let error = managed_real
            .fft(
                None,
                0,
                FftNorm::Backward,
                &mut context.metal_backend().clone(),
            )
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported { .. }));
        for invalid_n in [1usize, 3usize, 1usize << 40] {
            let error = managed_real
                .rfft(
                    Some(invalid_n),
                    0,
                    FftNorm::Backward,
                    &mut context.metal_backend().clone(),
                )
                .unwrap_err();
            assert!(matches!(error, Error::Unsupported { .. }));
        }
        let managed_complex = context.upload_tensor(&c32_input).unwrap();
        let error = managed_complex
            .fft(
                Some(8),
                0,
                FftNorm::Backward,
                &mut context.metal_backend().clone(),
            )
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported { .. }));

        let other = AppleContext::new().unwrap();
        let foreign = other.upload_tensor(&c32_input).unwrap();
        let before = context.transfer_stats();
        let error = foreign
            .fft(
                None,
                0,
                FftNorm::Backward,
                &mut context.metal_backend().clone(),
            )
            .unwrap_err();
        assert!(matches!(error, Error::HostAccess { .. }));
        assert_eq!(context.transfer_stats(), before);

        let runtime = WebGpuRuntime::new_default().unwrap();
        let device_local = upload_webgpu_tensor(&runtime, &c32_input).unwrap();
        let error = device_local
            .fft(
                None,
                0,
                FftNorm::Backward,
                &mut context.metal_backend().clone(),
            )
            .unwrap_err();
        assert!(matches!(error, Error::RuntimeState { .. }));
        assert_eq!(context.transfer_stats(), before);

        let undersized_f32 = webgpu_interop::allocate_raw(context.metal_backend(), 4);
        let error = webgpu_interop::finish_f32(
            context.metal_backend(),
            vec![2],
            undersized_f32,
            "test_finish_f32",
        )
        .unwrap_err();
        assert!(matches!(error, Error::RuntimeState { .. }));

        let undersized_c32 = webgpu_interop::allocate_raw(context.metal_backend(), 4);
        let error = webgpu_interop::finish_c32(
            context.metal_backend(),
            vec![1],
            undersized_c32,
            "test_finish_c32",
        )
        .unwrap_err();
        assert!(matches!(error, Error::RuntimeState { .. }));

        let aliased = webgpu_interop::allocate_raw(context.metal_backend(), 8);
        let surviving_alias = aliased.clone();
        let error = webgpu_interop::finish_f32(
            context.metal_backend(),
            vec![2],
            aliased,
            "test_finish_alias",
        )
        .unwrap_err();
        assert!(matches!(error, Error::RuntimeState { .. }));
        drop(surviving_alias);

        let mut invalid_range = webgpu_interop::allocate_raw(context.metal_backend(), 8);
        invalid_range.offset_start = Some(invalid_range.size() + 1);
        let error = webgpu_interop::finish_f32(
            context.metal_backend(),
            vec![1],
            invalid_range,
            "test_finish_invalid_range",
        )
        .unwrap_err();
        assert!(matches!(error, Error::RuntimeState { .. }));
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_fft_surface_compiles_off_macos() {
    use tenferro_fft::{FftNorm, TensorFftExt};
    use tenferro_gpu::WebGpuBackend;
    use tenferro_tensor::Tensor;

    fn compile_surface(input: &Tensor, backend: &mut WebGpuBackend) {
        let _ = input.rfft(None, -1, FftNorm::Backward, backend);
    }

    let _ = compile_surface as fn(&Tensor, &mut WebGpuBackend);
}
