// INVARIANT: The parent `#[cfg(test)] mod tests;` declaration in
// `cuda/mod.rs` is the sole inclusion point for this test module.
#[path = "tests/canonicalization.rs"]
mod canonicalization;
#[path = "tests/descriptor.rs"]
mod descriptor;
#[path = "tests/ffi.rs"]
mod ffi;
#[path = "tests/plan.rs"]
mod plan;
#[path = "tests/source_contract.rs"]
mod source_contract;

#[test]
fn placement_error_explains_explicit_transfer_and_runtime_ownership() {
    let error = super::CudaFftPlacementError {
        source: tenferro_tensor::Error::runtime_state("cuda_fft", "residency mismatch"),
    };
    assert_eq!(
        error.to_string(),
        "CUDA FFT expected GPU tensor owned by this runtime; host tensors must use ".to_string()
            + "upload_tensor(), and foreign-runtime tensors must use their owning "
            + "runtime: cuda_fft: runtime state failure: residency mismatch"
    );
}

#[test]
fn into_placement_rejects_host_before_cuda_work() {
    use tenferro_tensor::{AllocationDomainId, Tensor, TensorRead};
    let host = Tensor::from_vec_col_major([0, 4], Vec::<num_complex::Complex64>::new()).unwrap();
    let error = super::validate_cuda_read_domain(
        &TensorRead::from_tensor(&host),
        AllocationDomainId::fresh(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(error.to_string().contains("upload_tensor()"));
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn into_device_destinations_match_cpu_without_host_transfers() {
    use crate::{FftOps, FftOptions, FftnOptions};
    use num_complex::{Complex32, Complex64};
    use tenferro_cpu::CpuEngine;
    use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaDeviceId, CudaEngine};
    use tenferro_tensor::{Dest, MemoryKind, Tensor, TensorRead, TensorWrite};
    let mut cuda = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let mut cpu = CpuEngine::serial();
    macro_rules! check {
        ($host:expr, $with:ident, $into:ident, $options:expr, $scalar:ty, $magnitude:expr, $tolerance:expr) => {{
            let host = &$host;
            let options = $options;
            let expected = host.$with(options, &mut cpu).unwrap();
            let input = upload_tensor(cuda.runtime(), host).unwrap();
            let mut output = cuda
                .alloc_output(expected.dtype(), expected.shape())
                .unwrap();
            input
                .$into(
                    options,
                    &mut cuda,
                    Dest::Overwrite(TensorWrite::from_tensor(&mut output)),
                )
                .unwrap();
            assert_eq!(output.shape(), expected.shape());
            assert_eq!(output.dtype(), expected.dtype());
            let read = TensorRead::from_tensor(&output);
            assert_eq!(read.placement().memory_kind, MemoryKind::Device);
            assert_eq!(
                read.allocation_domain(),
                Some(cuda.runtime().allocation_domain_id())
            );
            let actual = download_tensor(cuda.runtime(), &output).unwrap();
            let residual = actual
                .as_slice::<$scalar>()
                .unwrap()
                .iter()
                .zip(expected.as_slice::<$scalar>().unwrap())
                .map(|(a, b)| {
                    let delta = ($magnitude)(*a - *b);
                    if delta.is_finite() {
                        delta
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(0.0_f64, f64::max);
            assert!(
                residual < $tolerance,
                "{} maximum residual {residual}",
                stringify!($into)
            );

            let mut host_output = expected.duplicate().unwrap();
            host_output
                .as_slice_mut::<$scalar>()
                .unwrap()
                .fill(<$scalar>::default());
            assert!(input
                .$into(
                    options,
                    &mut cuda,
                    Dest::Overwrite(TensorWrite::from_tensor(&mut host_output))
                )
                .is_err());
            assert!(host_output
                .as_slice::<$scalar>()
                .unwrap()
                .iter()
                .all(|value| *value == <$scalar>::default()));
        }};
    }
    macro_rules! check_width {
        ($real:ty, $complex:ty, $tolerance:expr) => {{
            let real = Tensor::from_vec_col_major(
                [2, 4],
                (0..8).map(|x| x as $real * 0.25 - 0.5).collect(),
            )
            .unwrap();
            let complex = Tensor::from_vec_col_major(
                [2, 4],
                (0..8)
                    .map(|x| <$complex>::new(x as $real * 0.25, 0.125 - x as $real * 0.0625))
                    .collect(),
            )
            .unwrap();
            let complex_magnitude = |value: $complex| value.norm() as f64;
            let real_magnitude = |value: $real| value.abs() as f64;
            let one = FftOptions::default()
                .axis(1)
                .n(Some(5))
                .norm(crate::FftNorm::Ortho);
            check!(
                complex,
                fft_with,
                fft_into,
                &one,
                $complex,
                complex_magnitude,
                $tolerance
            );
            check!(
                real,
                fft_with,
                fft_into,
                &one,
                $complex,
                complex_magnitude,
                $tolerance
            );
            check!(
                complex,
                ifft_with,
                ifft_into,
                &one,
                $complex,
                complex_magnitude,
                $tolerance
            );
            check!(
                real,
                rfft_with,
                rfft_into,
                &one,
                $complex,
                complex_magnitude,
                $tolerance
            );
            let half = real.rfft_with(&one, &mut cpu).unwrap();
            check!(
                half,
                irfft_with,
                irfft_into,
                &one,
                $real,
                real_magnitude,
                $tolerance
            );
            let nd = FftnOptions::default()
                .axes(vec![1, 0])
                .shape(Some(vec![5, 3]))
                .norm(crate::FftNorm::Ortho);
            check!(
                complex,
                fftn_with,
                fftn_into,
                &nd,
                $complex,
                complex_magnitude,
                $tolerance
            );
            check!(
                real,
                fftn_with,
                fftn_into,
                &nd,
                $complex,
                complex_magnitude,
                $tolerance
            );
            check!(
                complex,
                ifftn_with,
                ifftn_into,
                &nd,
                $complex,
                complex_magnitude,
                $tolerance
            );
            check!(
                real,
                rfftn_with,
                rfftn_into,
                &nd,
                $complex,
                complex_magnitude,
                $tolerance
            );
            let half_nd = real.rfftn_with(&nd, &mut cpu).unwrap();
            check!(
                half_nd,
                irfftn_with,
                irfftn_into,
                &nd,
                $real,
                real_magnitude,
                $tolerance
            );
        }};
    }
    // Runtime transfer counters are unavailable: numerical and placement checks
    // complement the CUDA source contract guarding host payload access.
    check_width!(f32, Complex32, 2e-5);
    check_width!(f64, Complex64, 1e-11);
}
