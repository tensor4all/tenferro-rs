//! Explicit RustFFT/Metal CubeK execution over one Apple shared allocation domain.

#[cfg(target_os = "macos")]
use num_complex::Complex64;
#[cfg(target_os = "macos")]
use tenferro_fft::{FftNorm, TensorFftExt};
#[cfg(target_os = "macos")]
use tenferro_gpu::apple::AppleContext;
#[cfg(target_os = "macos")]
use tenferro_tensor::{
    AllocationDomainId, AllocationId, Error, StorageBuffer, Tensor, TypedTensor,
};

#[cfg(target_os = "macos")]
fn managed_values<T: Copy + Send + Sync + 'static>(tensor: &TypedTensor<T>) -> Vec<T> {
    let StorageBuffer::Backend(buffer) = tensor.buffer() else {
        panic!("expected Apple managed storage")
    };
    buffer.map_read().unwrap().to_vec()
}

#[cfg(target_os = "macos")]
fn f32_identity(tensor: &Tensor) -> (AllocationDomainId, AllocationId) {
    let Tensor::F32(tensor) = tensor else {
        panic!("expected F32 tensor")
    };
    (
        tensor.allocation_domain().expect("managed domain"),
        tensor.allocation_id().expect("managed allocation"),
    )
}

#[cfg(target_os = "macos")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = AppleContext::new()?;

    // This explicit upload is the only host-to-device transfer in the sequence.
    let host = Tensor::from_vec_col_major([8], vec![1.0_f32, -2.0, 0.5, 3.0, 0.0, 1.5, -1.0, 2.0])?;
    let unified = context.upload_tensor(&host)?;
    let input_identity = f32_identity(&unified);
    assert_eq!(input_identity.0, context.domain_id());
    let after_creation = context.transfer_stats();

    let mut cpu = context.cpu_backend().clone();
    let cpu_first = unified.rfft(None, 0, FftNorm::Backward, &mut cpu)?;
    assert_eq!(f32_identity(&unified), input_identity);

    let mut metal = context.metal_backend().clone();
    let metal_result = unified.rfft(None, 0, FftNorm::Backward, &mut metal)?;
    metal.synchronize()?;
    assert_eq!(f32_identity(&unified), input_identity);

    let mut cpu = context.cpu_backend().clone();
    let cpu_again = unified.rfft(None, 0, FftNorm::Backward, &mut cpu)?;
    assert_eq!(f32_identity(&unified), input_identity);

    let (Tensor::C32(cpu_first), Tensor::C32(metal_result), Tensor::C32(cpu_again)) =
        (&cpu_first, &metal_result, &cpu_again)
    else {
        panic!("RFFT must return C32")
    };
    for tensor in [cpu_first, metal_result, cpu_again] {
        assert_eq!(tensor.allocation_domain(), Some(context.domain_id()));
    }
    let result_allocations = [
        cpu_first.allocation_id().expect("managed CPU result"),
        metal_result.allocation_id().expect("managed Metal result"),
        cpu_again.allocation_id().expect("managed CPU result"),
    ];
    for allocation in result_allocations {
        assert_ne!(allocation, input_identity.1);
    }
    assert_ne!(result_allocations[0], result_allocations[1]);
    assert_ne!(result_allocations[0], result_allocations[2]);
    assert_ne!(result_allocations[1], result_allocations[2]);
    let cpu_values = managed_values(cpu_first);
    let metal_values = managed_values(metal_result);
    let cpu_again_values = managed_values(cpu_again);
    for (actual, expected) in metal_values.iter().zip(&cpu_values) {
        assert!((*actual - *expected).norm() <= 2.0e-5);
    }
    assert_eq!(cpu_again_values, cpu_values);
    assert_eq!(context.transfer_stats(), after_creation);

    // RustFFT supports C64 in the same managed domain. Metal is deliberately
    // F32/C32-only and returns a typed capability error without falling back.
    let c64_host = Tensor::from_vec_col_major(
        [4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-0.5, 0.25),
            Complex64::new(3.0, 0.0),
        ],
    )?;
    let c64 = context.upload_tensor(&c64_host)?;
    let after_c64_creation = context.transfer_stats();
    let mut cpu = context.cpu_backend().clone();
    let cpu_c64 = c64.fft(None, 0, FftNorm::Backward, &mut cpu)?;
    let Tensor::C64(cpu_c64) = cpu_c64 else {
        panic!("C64 FFT must return C64")
    };
    assert_eq!(cpu_c64.allocation_domain(), Some(context.domain_id()));

    let mut metal = context.metal_backend().clone();
    let error = c64
        .fft(None, 0, FftNorm::Backward, &mut metal)
        .expect_err("Metal must reject C64");
    assert!(matches!(error, Error::Unsupported { .. }));
    assert_eq!(context.transfer_stats(), after_c64_creation);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    run()?;
    #[cfg(not(target_os = "macos"))]
    eprintln!("Apple shared FFT example is runtime-gated to macOS");
    Ok(())
}
