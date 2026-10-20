#![cfg(target_os = "macos")]

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftNorm, TensorFftExt};
use tenferro_gpu::{apple::AppleContext, webgpu::upload_webgpu_tensor, webgpu::WebGpuRuntime};
use tenferro_tensor::{HostAccessError, Tensor, TensorScalar};

fn apple_context() -> Option<AppleContext> {
    match AppleContext::new() {
        Ok(context) => Some(context),
        Err(error) => {
            assert!(
                std::env::var_os("TENFERRO_REQUIRE_METAL").is_none(),
                "Apple CI requires Metal: {error}"
            );
            eprintln!("skipping Apple CPU FFT test: {error}");
            None
        }
    }
}

fn mapped_slice<T: TensorScalar + Copy + Send + Sync + 'static>(
    tensor: &tenferro_tensor::TypedTensor<T>,
) -> Vec<T> {
    tensor.with_host_read(<[T]>::to_vec).unwrap()
}

#[test]
fn managed_cpu_fft_preserves_values_domain_and_transfer_counters() {
    let Some(mut context) = apple_context() else {
        return;
    };

    let host_f32 = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let mut reference_cpu = CpuEngine::serial();
    let reference = host_f32
        .rfft(Some(4), 1, FftNorm::Ortho, &mut reference_cpu)
        .unwrap();
    let managed_f32 = context.upload_tensor(&host_f32).unwrap();
    let before = context.transfer_stats();
    let output = managed_f32
        .rfft(Some(4), 1, FftNorm::Ortho, context.parts().0)
        .unwrap();

    let output = output
        .into_typed::<tenferro_tensor::Complex32>()
        .expect("expected C32 output");
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.allocation_domain(), Some(context.domain_id()));
    assert_eq!(
        mapped_slice(&output),
        reference.as_slice::<Complex32>().unwrap()
    );
    assert_eq!(context.transfer_stats(), before);

    let host_f64 = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, -2.0, 0.5]).unwrap();
    let reference_round_trip = host_f64
        .rfft(Some(4), 0, FftNorm::Forward, &mut reference_cpu)
        .unwrap()
        .irfft(Some(4), 0, FftNorm::Forward, &mut reference_cpu)
        .unwrap();
    let managed_f64 = context.upload_tensor(&host_f64).unwrap();
    let before = context.transfer_stats();
    let round_trip = managed_f64
        .rfft(Some(4), 0, FftNorm::Forward, context.parts().0)
        .unwrap()
        .irfft(Some(4), 0, FftNorm::Forward, context.parts().0)
        .unwrap();
    let round_trip = round_trip.into_typed::<f64>().expect("expected F64 output");
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
        .fft(Some(2), 0, FftNorm::Forward, context.parts().0)
        .unwrap();
    let output = output
        .into_typed::<tenferro_tensor::Complex32>()
        .expect("expected C32 output");
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
    let (output, repeated) = {
        let engine = context.parts().0;
        let output = executor
            .fft(&managed_c64, None, -1, FftNorm::Backward, engine)
            .unwrap();
        let repeated = executor
            .fft(&managed_c64, None, -1, FftNorm::Backward, engine)
            .unwrap();
        (output, repeated)
    };
    let cached_entries = executor.cache_stats().entries;
    assert!(cached_entries > 0);
    assert_eq!(executor.cache_stats().entries, cached_entries);
    let output = output
        .into_typed::<tenferro_tensor::Complex64>()
        .expect("expected C64 output");
    assert_eq!(output.allocation_domain(), Some(context.domain_id()));
    assert_eq!(
        mapped_slice(&output),
        reference.as_slice::<Complex64>().unwrap()
    );
    let repeated = repeated
        .into_typed::<tenferro_tensor::Complex64>()
        .expect("expected repeated C64 output");
    assert_eq!(mapped_slice(&repeated), mapped_slice(&output));
    assert_eq!(context.transfer_stats(), before);
}

#[test]
fn managed_cpu_fft_rejects_foreign_and_device_local_buffers_without_transfers() {
    let (Some(mut first), Some(mut second)) = (apple_context(), apple_context()) else {
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
        .fft(None, -1, FftNorm::Backward, second.parts().0)
        .unwrap_err();
    assert!(
        matches!(
            error,
            tenferro_tensor::Error::HostAccess {
                source: HostAccessError::ForeignDomain { .. },
                ..
            }
        ),
        "unexpected foreign-domain error: {error:?}"
    );
    assert_eq!(first.transfer_stats(), first_before);
    assert_eq!(second.transfer_stats(), second_before);

    let Ok(runtime) = WebGpuRuntime::new_default() else {
        return;
    };
    let device_local = upload_webgpu_tensor(&runtime, &host).unwrap();
    runtime.synchronize().unwrap();
    let before = first.transfer_stats();
    let error = device_local
        .fft(None, -1, FftNorm::Backward, first.parts().0)
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
