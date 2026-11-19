use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftOps, FftOptions, FftnOptions};
use tenferro_tensor::{Dest, ErrorKind, Tensor, TypedTensor};

fn close(actual: &[Complex64], expected: &[Complex64]) {
    let residual = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max);
    assert_eq!(actual.len(), expected.len());
    assert!(residual < 1e-10, "FFT output maximum residual {residual}");
}

#[test]
fn typed_real_and_complex_destinations_follow_result_dtype() {
    let mut engine = CpuEngine::serial();
    let x = TypedTensor::<f32>::from_vec_col_major([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut spectrum =
        TypedTensor::<Complex32>::from_vec_col_major([3], vec![Complex32::new(f32::NAN, 0.0); 3])
            .unwrap();
    x.rfft_into(
        &FftOptions::default(),
        &mut engine,
        Dest::overwrite(&mut spectrum),
    )
    .unwrap();
    assert_eq!(spectrum.host_data().unwrap()[0], Complex32::new(10.0, 0.0));
    let mut restored = TypedTensor::<f32>::from_vec_col_major([4], vec![f32::NAN; 4]).unwrap();
    spectrum
        .irfft_into(
            &FftOptions::default().n(Some(4)),
            &mut engine,
            Dest::overwrite(&mut restored),
        )
        .unwrap();
    assert_eq!(restored.host_data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(x.host_data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn every_fft_family_writes_the_existing_destination() {
    let mut engine = CpuEngine::serial();
    let real = Tensor::from_vec_col_major([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let complex = real.fft(&mut engine).unwrap();
    macro_rules! check {
        ($input:expr, $with:ident, $into:ident, $options:expr) => {{
            let options = $options;
            let expected = $input.$with(&options, &mut engine).unwrap();
            let mut output = expected.duplicate().unwrap();
            let pointer = output.as_slice::<Complex64>().ok().map(|v| v.as_ptr());
            $input
                .$into(&options, &mut engine, Dest::overwrite(&mut output))
                .unwrap();
            assert_eq!(output.shape(), expected.shape());
            if expected.dtype() == tenferro_tensor::DType::F64 {
                assert_eq!(
                    output.as_slice::<f64>().unwrap(),
                    expected.as_slice::<f64>().unwrap()
                );
            } else {
                assert_eq!(
                    pointer,
                    Some(output.as_slice::<Complex64>().unwrap().as_ptr())
                );
                close(
                    output.as_slice::<Complex64>().unwrap(),
                    expected.as_slice::<Complex64>().unwrap(),
                );
            }
        }};
    }
    check!(
        real,
        fft_with,
        fft_into,
        FftOptions::default().axis(0).n(Some(3))
    );
    check!(
        complex,
        ifft_with,
        ifft_into,
        FftOptions::default().axis(1).n(Some(2))
    );
    check!(real, rfft_with, rfft_into, FftOptions::default().n(Some(4)));
    check!(
        complex,
        irfft_with,
        irfft_into,
        FftOptions::default().n(Some(4))
    );
    check!(
        real,
        fftn_with,
        fftn_into,
        FftnOptions::default()
            .axes(vec![1, 0])
            .shape(Some(vec![4, 3]))
    );
    check!(
        complex,
        ifftn_with,
        ifftn_into,
        FftnOptions::default().axes(vec![1, 0])
    );
    check!(
        real,
        rfftn_with,
        rfftn_into,
        FftnOptions::default()
            .axes(vec![1, 0])
            .shape(Some(vec![3, 4]))
    );
    let real_options = FftnOptions::default()
        .axes(vec![1, 0])
        .shape(Some(vec![3, 4]));
    let half = real.rfftn_with(&real_options, &mut engine).unwrap();
    check!(half, irfftn_with, irfftn_into, real_options);
}

#[test]
fn rejected_modes_and_destination_metadata_preserve_output() {
    let mut engine = CpuEngine::serial();
    let input = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(1.0, 0.0); 4]).unwrap();
    let options = FftnOptions::default();
    let mut output = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(7.0, 8.0); 4]).unwrap();
    for scaled in [false, true] {
        let dest = if scaled {
            Dest::scaled(&mut output, 1.0_f64, 0.0_f64)
        } else {
            Dest::accumulate(&mut output)
        };
        let error = input.fftn_into(&options, &mut engine, dest).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            output.as_slice::<Complex64>().unwrap(),
            &[Complex64::new(7.0, 8.0); 4]
        );
    }
    let mut wrong_shape =
        Tensor::from_vec_col_major([4], vec![Complex64::new(7.0, 8.0); 4]).unwrap();
    assert!(input
        .fftn_into(&options, &mut engine, Dest::overwrite(&mut wrong_shape))
        .is_err());
    assert_eq!(
        wrong_shape.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(7.0, 8.0); 4]
    );
    let mut wrong_dtype = Tensor::from_vec_col_major([2, 2], vec![7.0_f64; 4]).unwrap();
    assert!(input
        .fftn_into(&options, &mut engine, Dest::overwrite(&mut wrong_dtype))
        .is_err());
    assert_eq!(wrong_dtype.as_slice::<f64>().unwrap(), &[7.0; 4]);
}

#[test]
fn executor_reuses_multiaxis_real_schedule_and_can_clear_it() {
    let mut engine = CpuEngine::serial();
    let mut executor = FftExecutor::default();
    let input = Tensor::from_vec_col_major([2, 3], vec![1.0_f64; 6]).unwrap();
    let options = FftnOptions::default()
        .axes(vec![1, 0])
        .shape(Some(vec![3, 4]));
    let mut output =
        Tensor::from_vec_col_major([3, 3], vec![Complex64::new(f64::NAN, 0.0); 9]).unwrap();
    executor
        .rfftn_into(&input, &options, &mut engine, Dest::overwrite(&mut output))
        .unwrap();
    close(
        output.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(6.0, 0.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let stats = executor.cache_stats();
    assert!(stats.entries > 0 && stats.retained_bytes >= 9 * std::mem::size_of::<Complex64>());
    executor
        .rfftn_into(&input, &options, &mut engine, Dest::overwrite(&mut output))
        .unwrap();
    assert_eq!(executor.cache_stats().entries, stats.entries);
    executor.clear_cache();
    assert_eq!(executor.cache_stats().entries, 0);
}

#[test]
fn executor_rejects_strided_final_destination_before_any_axis_executes() {
    use tenferro_tensor::{TensorViewMut, TypedTensorViewMut};
    let mut engine = CpuEngine::serial();
    let mut executor = FftExecutor::default();
    let input = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(1.0, 0.0); 4]).unwrap();
    let mut data = [Complex64::new(7.0, 8.0); 8];
    let view = TypedTensorViewMut::from_slice([2, 2], [2, 4], 0, &mut data).unwrap();
    let error = executor
        .fftn_into(
            &input,
            &FftnOptions::default(),
            &mut engine,
            Dest::overwrite(TensorViewMut::C64(view)),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(data, [Complex64::new(7.0, 8.0); 8]);
    assert_eq!(executor.cache_stats().entries, 0);
}

#[test]
fn mismatched_input_placement_preserves_destination_without_running_axes() {
    use tenferro_tensor::{
        BackendStorageHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
        StorageBuffer,
    };
    let input = Tensor::from_typed(
        TypedTensor::<Complex64>::from_buffer_col_major(
            [2, 2],
            StorageBuffer::Backend(Box::new(BackendStorageHandle::new_with_len(271, 4))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
                cpu_affinity: None,
            },
        )
        .unwrap(),
    );
    let mut engine = CpuEngine::serial();
    let mut executor = FftExecutor::default();
    let mut output = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(7.0, 8.0); 4]).unwrap();
    assert!(executor
        .fftn_into(
            &input,
            &FftnOptions::default(),
            &mut engine,
            Dest::overwrite(&mut output)
        )
        .is_err());
    assert_eq!(
        output.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(7.0, 8.0); 4]
    );
    assert_eq!(executor.cache_stats().entries, 0);
}

#[test]
fn same_device_foreign_domain_is_rejected_before_backend_dispatch() {
    use tenferro_tensor::{
        BackendStorageHandle, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement,
        StorageBuffer,
    };
    let opaque = |id| {
        Tensor::from_typed(
            TypedTensor::<Complex64>::from_buffer_col_major(
                [2, 2],
                StorageBuffer::Backend(Box::new(BackendStorageHandle::new_with_len(id, 4))),
                Placement {
                    memory_kind: MemoryKind::Device,
                    device: Some(DeviceId {
                        kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                        ordinal: 0,
                    }),
                    cpu_affinity: None,
                },
            )
            .unwrap(),
        )
    };
    let input = opaque(272);
    let mut output = opaque(273);
    let mut engine = CpuEngine::serial();
    let mut executor = FftExecutor::default();
    let error = executor
        .fftn_into(
            &input,
            &FftnOptions::default(),
            &mut engine,
            Dest::overwrite(&mut output),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    assert!(error.to_string().contains("destination allocation domain"));
    assert_eq!(executor.cache_stats().entries, 0);
}
