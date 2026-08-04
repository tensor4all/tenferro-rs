#![cfg(target_os = "macos")]

use num_complex::{Complex32, Complex64};
use tenferro_fft::{FftExecutor, FftNorm, TensorFftExt};
use tenferro_gpu::{apple::AppleContext, webgpu::upload_webgpu_tensor, webgpu::WebGpuRuntime};
use tenferro_tensor::{HostAccessError, StorageBuffer, Tensor};

fn apple_context() -> Option<AppleContext> {
    match AppleContext::new() {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping Apple CPU FFT test: {error}");
            None
        }
    }
}

fn mapped_slice<T: Copy + Send + Sync + 'static>(
    tensor: &tenferro_tensor::TypedTensor<T>,
) -> Vec<T> {
    let StorageBuffer::Backend(buffer) = tensor.buffer() else {
        panic!("expected managed backend output")
    };
    buffer.map_read().unwrap().to_vec()
}

#[test]
fn managed_cpu_fft_preserves_values_domain_and_transfer_counters() {
    let Some(context) = apple_context() else {
        return;
    };

    let host_f32 = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let mut reference_cpu = tenferro_cpu::CpuBackend::new();
    let reference = host_f32
        .rfft(Some(4), 1, FftNorm::Ortho, &mut reference_cpu)
        .unwrap();
    let managed_f32 = context.upload_tensor(&host_f32).unwrap();
    let before = context.transfer_stats();
    let mut apple_cpu = context.cpu_backend().clone();
    let output = managed_f32
        .rfft(Some(4), 1, FftNorm::Ortho, &mut apple_cpu)
        .unwrap();

    let Tensor::C32(output) = output else {
        panic!("expected C32 output")
    };
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.allocation_domain(), Some(context.domain_id()));
    assert_eq!(
        mapped_slice(&output),
        reference.as_slice::<Complex32>().unwrap()
    );
    assert_eq!(context.transfer_stats(), before);

    let host_f64 = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, -2.0, 0.5]).unwrap();
    let reference_spectrum = host_f64
        .rfft(Some(4), 0, FftNorm::Forward, &mut reference_cpu)
        .unwrap();
    let reference_round_trip = reference_spectrum
        .irfft(Some(4), 0, FftNorm::Forward, &mut reference_cpu)
        .unwrap();
    let managed_f64 = context.upload_tensor(&host_f64).unwrap();
    let before = context.transfer_stats();
    let spectrum = managed_f64
        .rfft(Some(4), 0, FftNorm::Forward, &mut apple_cpu)
        .unwrap();
    let round_trip = spectrum
        .irfft(Some(4), 0, FftNorm::Forward, &mut apple_cpu)
        .unwrap();
    let Tensor::F64(round_trip) = round_trip else {
        panic!("expected F64 output")
    };
    assert_eq!(round_trip.allocation_domain(), Some(context.domain_id()));
    assert_eq!(
        mapped_slice(&round_trip),
        reference_round_trip.as_slice::<f64>().unwrap()
    );
    assert_eq!(context.transfer_stats(), before);

    let host_c32 = Tensor::from_vec_col_major(
        vec![3],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
        ],
    )
    .unwrap();
    let reference = host_c32
        .fft(Some(2), 0, FftNorm::Forward, &mut reference_cpu)
        .unwrap();
    let managed_c32 = context.upload_tensor(&host_c32).unwrap();
    let before = context.transfer_stats();
    let output = managed_c32
        .fft(Some(2), 0, FftNorm::Forward, &mut apple_cpu)
        .unwrap();
    let Tensor::C32(output) = output else {
        panic!("expected C32 output")
    };
    assert_eq!(output.allocation_domain(), Some(context.domain_id()));
    assert_eq!(
        mapped_slice(&output),
        reference.as_slice::<Complex32>().unwrap()
    );
    assert_eq!(context.transfer_stats(), before);

    let host_c64 = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.5)],
    )
    .unwrap();
    let reference = host_c64
        .fft(None, -1, FftNorm::Backward, &mut reference_cpu)
        .unwrap();
    let managed_c64 = context.upload_tensor(&host_c64).unwrap();
    let before = context.transfer_stats();
    let mut executor = FftExecutor::default();
    let output = executor
        .fft(&managed_c64, None, -1, FftNorm::Backward, &mut apple_cpu)
        .unwrap();
    let cached_entries = executor.cache_stats().entries;
    assert!(cached_entries > 0);
    let repeated = executor
        .fft(&managed_c64, None, -1, FftNorm::Backward, &mut apple_cpu)
        .unwrap();
    assert_eq!(executor.cache_stats().entries, cached_entries);
    let Tensor::C64(output) = output else {
        panic!("expected C64 output")
    };
    assert_eq!(output.allocation_domain(), Some(context.domain_id()));
    assert_eq!(
        mapped_slice(&output),
        reference.as_slice::<Complex64>().unwrap()
    );
    let Tensor::C64(repeated) = repeated else {
        panic!("expected repeated C64 output")
    };
    assert_eq!(mapped_slice(&repeated), mapped_slice(&output));
    assert_eq!(context.transfer_stats(), before);
}

#[test]
fn managed_cpu_fft_rejects_foreign_and_device_local_buffers_without_transfers() {
    let (Some(first), Some(second)) = (apple_context(), apple_context()) else {
        return;
    };
    let host = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
    )
    .unwrap();
    let foreign = first.upload_tensor(&host).unwrap();
    let first_before = first.transfer_stats();
    let second_before = second.transfer_stats();
    let error = foreign
        .fft(
            None,
            -1,
            FftNorm::Backward,
            &mut second.cpu_backend().clone(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::ForeignDomain { .. },
            ..
        }
    ));
    assert_eq!(first.transfer_stats(), first_before);
    assert_eq!(second.transfer_stats(), second_before);

    let Ok(runtime) = WebGpuRuntime::new_default() else {
        return;
    };
    let device_local = upload_webgpu_tensor(&runtime, &host).unwrap();
    runtime.synchronize().unwrap();
    let before = first.transfer_stats();
    let error = device_local
        .fft(
            None,
            -1,
            FftNorm::Backward,
            &mut first.cpu_backend().clone(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::Unsupported { .. },
            ..
        }
    ));
    assert_eq!(first.transfer_stats(), before);
}
