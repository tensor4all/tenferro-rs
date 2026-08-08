use super::common::*;
use super::support;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftExecutor, FftNorm};
use tenferro_gpu::cuda::gpu_available;
use tenferro_tensor::TensorRead;

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_zero_batch_returns_empty_resident_outputs_without_cache_entries() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let mut executor = FftExecutor::default();
    for (input, operation, axis, tolerance, expected_shape) in [
        (
            complex_f64(&[0, 8], 0.5),
            Operation::Fft,
            1isize,
            1.0e-11,
            &[0, 8][..],
        ),
        (
            real_f32(&[2, 0, 8], 0.5),
            Operation::Rfft,
            2isize,
            1.0e-5,
            &[2, 0, 5][..],
        ),
    ] {
        let expected = operation
            .execute_cpu(&mut cpu, &input, None, axis, FftNorm::Backward)
            .expect("CPU FFT oracle");
        assert_eq!(expected.shape(), expected_shape);
        let gpu_input = support::upload_cuda(cuda.runtime(), &input);
        let domain = TensorRead::from_tensor(&gpu_input)
            .allocation_domain()
            .expect("uploaded input allocation domain");
        let actual = operation
            .execute_executor(
                &mut executor,
                &mut cuda,
                &gpu_input,
                None,
                axis,
                FftNorm::Backward,
            )
            .expect("zero-batch CUDA FFT execution");
        support::assert_cuda_resident(&actual, domain);
        assert_eq!(actual.shape(), expected_shape);
        assert_eq!(actual.dtype(), expected.dtype());
        let actual =
            support::download_cuda(cuda.runtime(), &actual).expect("explicit CUDA download");
        assert_host_close(&actual, &expected, tolerance);
        assert_eq!(actual.shape(), expected_shape);
        assert_eq!(element_count(actual.shape()), 0);
        let stats = executor.cache_stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.retained_bytes, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.clears, 0);
    }
}
