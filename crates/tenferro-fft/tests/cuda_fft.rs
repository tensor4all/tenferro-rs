#![cfg(feature = "cuda")]

use std::error::Error as StdError;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, OnceLock};

use num_complex::{Complex32, Complex64};
#[cfg(feature = "autodiff")]
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
#[cfg(feature = "autodiff")]
use tenferro_fft::EagerTensorFftExt;
use tenferro_fft::{FftExecutor, FftNorm, FftPlanCache, TensorFftExt, TracedTensorFftExt};
use tenferro_gpu::cuda::{cuda_runtime_engine_registration, gpu_available, CudaBackend};
use tenferro_runtime::{
    DType, EngineId, ExtensionCacheLimits, GraphCompiler, Runtime, Tensor, TracedTensor,
};
use tenferro_tensor::{BackendSessionHost, TensorRead};

mod support;

#[derive(Clone, Copy, Debug)]
enum Operation {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

impl Operation {
    fn execute_cpu(
        self,
        backend: &mut CpuBackend,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> tenferro_tensor::Result<Tensor> {
        backend.with_backend_session(|session| {
            with_cpu_exec_session(session, |exec_session| match self {
                Self::Fft => input.fft(n, axis, norm, exec_session),
                Self::Ifft => input.ifft(n, axis, norm, exec_session),
                Self::Rfft => input.rfft(n, axis, norm, exec_session),
                Self::Irfft => input.irfft(n, axis, norm, exec_session),
            })
            .expect("CPU backend session should be available")
        })
    }

    fn execute_cuda(
        self,
        backend: &mut CudaBackend,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> tenferro_tensor::Result<Tensor> {
        support::with_cuda_fft_session(backend, |exec_session| match self {
            Self::Fft => input.fft(n, axis, norm, exec_session),
            Self::Ifft => input.ifft(n, axis, norm, exec_session),
            Self::Rfft => input.rfft(n, axis, norm, exec_session),
            Self::Irfft => input.irfft(n, axis, norm, exec_session),
        })
    }

    fn execute_executor(
        self,
        executor: &mut FftExecutor,
        backend: &mut CudaBackend,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> tenferro_tensor::Result<Tensor> {
        support::with_cuda_fft_session(backend, |exec_session| match self {
            Self::Fft => executor.fft(input, n, axis, norm, exec_session),
            Self::Ifft => executor.ifft(input, n, axis, norm, exec_session),
            Self::Rfft => executor.rfft(input, n, axis, norm, exec_session),
            Self::Irfft => executor.irfft(input, n, axis, norm, exec_session),
        })
    }
}

fn element_count(shape: &[usize]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .expect("test shape product")
}

fn real_f32(shape: &[usize], seed: f32) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| seed + index as f32 * 0.375 - (index % 3) as f32 * 0.125)
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn real_f64(shape: &[usize], seed: f64) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| seed + index as f64 * 0.375 - (index % 3) as f64 * 0.125)
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn complex_f32(shape: &[usize], seed: f32) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| {
            let value = seed + index as f32 * 0.375;
            Complex32::new(value, 0.25 - value * 0.5)
        })
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn complex_f64(shape: &[usize], seed: f64) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| {
            let value = seed + index as f64 * 0.375;
            Complex64::new(value, 0.25 - value * 0.5)
        })
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn max_residual(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    let mut absolute = 0.0_f64;
    let mut scale = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        absolute = absolute.max((actual - expected).abs() as f64);
        scale = scale.max(expected.abs() as f64);
    }
    (
        absolute,
        if scale == 0.0 {
            absolute
        } else {
            absolute / scale
        },
    )
}

fn max_residual_f64(actual: &[f64], expected: &[f64]) -> (f64, f64) {
    let mut absolute = 0.0_f64;
    let mut scale = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        absolute = absolute.max((actual - expected).abs());
        scale = scale.max(expected.abs());
    }
    (
        absolute,
        if scale == 0.0 {
            absolute
        } else {
            absolute / scale
        },
    )
}

fn max_residual_c32(actual: &[Complex32], expected: &[Complex32]) -> (f64, f64) {
    let mut absolute = 0.0_f64;
    let mut scale = 0.0_f64;
    for (actual, expected) in actual.iter().zip(expected) {
        absolute = absolute
            .max((actual.re - expected.re).abs() as f64)
            .max((actual.im - expected.im).abs() as f64);
        scale = scale
            .max(expected.re.abs() as f64)
            .max(expected.im.abs() as f64);
    }
    (
        absolute,
        if scale == 0.0 {
            absolute
        } else {
            absolute / scale
        },
    )
}

fn max_residual_c64(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64) {
    let mut absolute = 0.0_f64;
    let mut scale = 0.0_f64;
    for (actual, expected) in actual.iter().zip(expected) {
        absolute = absolute
            .max((actual.re - expected.re).abs())
            .max((actual.im - expected.im).abs());
        scale = scale.max(expected.re.abs()).max(expected.im.abs());
    }
    (
        absolute,
        if scale == 0.0 {
            absolute
        } else {
            absolute / scale
        },
    )
}

fn assert_host_close(actual: &Tensor, expected: &Tensor, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    let (absolute, relative) = match (actual.dtype(), expected.dtype()) {
        (DType::F32, DType::F32) => max_residual(
            actual.as_slice::<f32>().unwrap(),
            expected.as_slice::<f32>().unwrap(),
        ),
        (DType::F64, DType::F64) => max_residual_f64(
            actual.as_slice::<f64>().unwrap(),
            expected.as_slice::<f64>().unwrap(),
        ),
        (DType::C32, DType::C32) => max_residual_c32(
            actual.as_slice::<Complex32>().unwrap(),
            expected.as_slice::<Complex32>().unwrap(),
        ),
        (DType::C64, DType::C64) => max_residual_c64(
            actual.as_slice::<Complex64>().unwrap(),
            expected.as_slice::<Complex64>().unwrap(),
        ),
        _ => panic!("unexpected comparison dtype {:?}", actual.dtype()),
    };
    assert!(
        absolute <= tolerance || relative <= tolerance,
        "max absolute residual {absolute:e}, relative residual {relative:e}, tolerance {tolerance:e}"
    );
}

fn run_case(
    cpu: &mut CpuBackend,
    cuda: &mut CudaBackend,
    input: &Tensor,
    operation: Operation,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
    tolerance: f64,
) -> Tensor {
    let expected = operation
        .execute_cpu(cpu, input, n, axis, norm)
        .expect("CPU FFT oracle");
    let gpu_input = support::upload_cuda(cuda.runtime(), input);
    let gpu_domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .expect("uploaded input allocation domain");
    let actual = operation
        .execute_cuda(cuda, &gpu_input, n, axis, norm)
        .expect("CUDA FFT execution");
    support::assert_cuda_resident(&actual, gpu_domain);
    let actual = support::download_cuda(cuda.runtime(), &actual).expect("explicit CUDA download");
    assert_host_close(&actual, &expected, tolerance);
    actual
}

fn run_executor_case(
    executor: &mut FftExecutor,
    cpu: &mut CpuBackend,
    cuda: &mut CudaBackend,
    input: &Tensor,
    operation: Operation,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
    tolerance: f64,
) -> Tensor {
    let expected = operation
        .execute_cpu(cpu, input, n, axis, norm)
        .expect("CPU FFT oracle");
    let gpu_input = support::upload_cuda(cuda.runtime(), input);
    let gpu_domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .expect("uploaded input allocation domain");
    let actual = operation
        .execute_executor(executor, cuda, &gpu_input, n, axis, norm)
        .expect("CUDA FFT execution through FftExecutor");
    support::assert_cuda_resident(&actual, gpu_domain);
    let actual = support::download_cuda(cuda.runtime(), &actual).expect("explicit CUDA download");
    assert_host_close(&actual, &expected, tolerance);
    actual
}

fn assert_error(result: tenferro_tensor::Result<Tensor>, expected_text: &str) {
    let error = result.expect_err("unsupported CUDA FFT input must return an error");
    assert!(!error.to_string().is_empty());
    assert!(
        error.to_string().contains(expected_text),
        "error `{error}` did not contain `{expected_text}`"
    );
}

fn error_chain(error: &(dyn StdError + 'static)) -> String {
    let mut messages = Vec::new();
    let mut current = Some(error);
    while let Some(source) = current {
        messages.push(source.to_string());
        current = source.source();
    }
    messages.join(" -> ")
}

fn cuda_runtime_with_fft(backend: &CudaBackend, install_module: bool) -> Runtime {
    let engine_id = EngineId::new("tenferro-fft.cuda.acceptance.v1").unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(backend, engine_id.clone()).unwrap())
        .unwrap();
    if install_module {
        builder
            .install_extension_module(
                tenferro_fft::extension_module::<CudaBackend>(engine_id).unwrap(),
            )
            .unwrap();
    }
    builder.build().unwrap()
}

fn compiled_cuda_fft(
    dtype: DType,
    shape: &[usize],
    operation: Operation,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> (tenferro_runtime::CompiledGraph, TracedTensor) {
    let input = TracedTensor::input_concrete_shape(dtype, shape).unwrap();
    let output = match operation {
        Operation::Fft => input.fft(n, axis, norm).unwrap(),
        Operation::Ifft => input.ifft(n, axis, norm).unwrap(),
        Operation::Rfft => input.rfft(n, axis, norm).unwrap(),
        Operation::Irfft => input.irfft(n, axis, norm).unwrap(),
    };
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&output, &[(&input, dtype, shape)])
        .unwrap();
    (program, input)
}

fn assert_full_hermitian(actual: &Tensor) {
    match actual {
        Tensor::C32(_) => {
            let values = actual.as_slice::<Complex32>().unwrap();
            for index in 1..=(values.len() - 1) / 2 {
                assert_eq!(values[values.len() - index], values[index].conj());
            }
        }
        Tensor::C64(_) => {
            let values = actual.as_slice::<Complex64>().unwrap();
            for index in 1..=(values.len() - 1) / 2 {
                assert_eq!(values[values.len() - index], values[index].conj());
            }
        }
        _ => panic!("full real FFT must produce a complex tensor"),
    }
}

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
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_zero_batch_bypasses_cufft_library_and_plan_creation() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    for (input, operation, axis, tolerance) in [
        (complex_f64(&[0, 8], 0.5), Operation::Fft, 1isize, 1.0e-11),
        (real_f32(&[2, 0, 8], 0.5), Operation::Rfft, 2isize, 1.0e-5),
    ] {
        let actual = with_invalid_cufft_path(|| {
            run_case(
                &mut cpu,
                &mut cuda,
                &input,
                operation,
                None,
                axis,
                FftNorm::Backward,
                tolerance,
            )
        });
        assert_eq!(
            actual.as_slice::<Complex64>().map_or(0, <[Complex64]>::len),
            0
        );
        assert_eq!(
            actual.as_slice::<Complex32>().map_or(0, <[Complex32]>::len),
            0
        );
    }
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_reuses_and_separates_structural_plans() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let mut executor = FftExecutor::default();
    let c32 = complex_f32(&[4], 0.5);
    let c64 = complex_f64(&[4], 0.5);
    let batched = complex_f32(&[2, 4], -0.25);
    let real = real_f32(&[4], 0.5);

    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Ifft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c64,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-11,
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &c32,
        Operation::Fft,
        Some(3),
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &batched,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &real,
        Operation::Rfft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );

    let stats = executor.cache_stats();
    assert!(stats.entries >= 6, "cache stats: {stats:?}");
    assert!(stats.retained_bytes > 0, "cache stats: {stats:?}");
    assert!(
        stats.hits > 0,
        "repeated executor calls should hit: {stats:?}"
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_keys_include_runtime_identity() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut first = support::cuda_backend();
    let mut second = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let first_input = support::upload_cuda(first.runtime(), &host);
    let second_input = support::upload_cuda(second.runtime(), &host);
    let mut executor = FftExecutor::default();

    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    for (backend, input) in [(&mut first, &first_input), (&mut second, &second_input)] {
        let domain = TensorRead::from_tensor(input).allocation_domain().unwrap();
        let output = Operation::Fft
            .execute_executor(&mut executor, backend, input, None, -1, FftNorm::Backward)
            .unwrap();
        support::assert_cuda_resident(&output, domain);
        let output = support::download_cuda(backend.runtime(), &output).unwrap();
        assert_host_close(&output, &expected, 1.0e-11);
    }

    assert_eq!(executor.cache_stats().entries, 2);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_limits_evict_by_entries_and_bytes() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let input = complex_f32(&[4], 0.5);
    let mut executor = FftExecutor::new(FftPlanCache::with_capacity(NonZeroUsize::new(1).unwrap()));
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &input,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    run_executor_case(
        &mut executor,
        &mut cpu,
        &mut cuda,
        &input,
        Operation::Fft,
        Some(3),
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    let entry_stats = executor.cache_stats();
    assert_eq!(entry_stats.entries, 1);
    assert!(entry_stats.evictions >= 1, "cache stats: {entry_stats:?}");

    let mut byte_executor =
        FftExecutor::new(FftPlanCache::with_capacity(NonZeroUsize::new(4).unwrap()));
    byte_executor.plan_cache_mut().set_limits(
        ExtensionCacheLimits::new(NonZeroUsize::new(4).unwrap())
            .with_max_retained_bytes(NonZeroUsize::new(1).unwrap()),
    );
    run_executor_case(
        &mut byte_executor,
        &mut cpu,
        &mut cuda,
        &input,
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
        1.0e-5,
    );
    let byte_stats = byte_executor.cache_stats();
    assert_eq!(byte_stats.entries, 0);
    assert!(byte_stats.evictions >= 1, "cache stats: {byte_stats:?}");
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_caller_owned_cache_clear_and_eviction_are_safe_after_launch() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let input = complex_f64(&[4], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(cuda.runtime(), &input);
    let domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .unwrap();
    let mut executor = FftExecutor::default();
    let output = Operation::Fft
        .execute_executor(
            &mut executor,
            &mut cuda,
            &gpu_input,
            None,
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    support::assert_cuda_resident(&output, domain);
    executor.clear_cache();
    cuda.runtime().synchronize().unwrap();
    let output_host = support::download_cuda(cuda.runtime(), &output).unwrap();
    assert_host_close(&output_host, &expected, 1.0e-11);

    let mut limited = FftExecutor::new(FftPlanCache::with_capacity(NonZeroUsize::new(1).unwrap()));
    let first = Operation::Fft
        .execute_executor(
            &mut limited,
            &mut cuda,
            &gpu_input,
            None,
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    let second = Operation::Fft
        .execute_executor(
            &mut limited,
            &mut cuda,
            &gpu_input,
            Some(3),
            -1,
            FftNorm::Backward,
        )
        .unwrap();
    cuda.runtime().synchronize().unwrap();
    let first_host = support::download_cuda(cuda.runtime(), &first).unwrap();
    let second_host = support::download_cuda(cuda.runtime(), &second).unwrap();
    assert_host_close(&first_host, &expected, 1.0e-11);
    let expected_second = Operation::Fft
        .execute_cpu(&mut cpu, &input, Some(3), -1, FftNorm::Backward)
        .unwrap();
    assert_host_close(&second_host, &expected_second, 1.0e-11);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_repeated_caller_owned_calls_remain_async_until_explicit_sync() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let mut cuda = support::cuda_backend();
    let input = complex_f32(&[4], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &input, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(cuda.runtime(), &input);
    let mut executor = FftExecutor::default();
    let mut outputs = Vec::new();
    for _ in 0..4 {
        outputs.push(
            Operation::Fft
                .execute_executor(
                    &mut executor,
                    &mut cuda,
                    &gpu_input,
                    None,
                    -1,
                    FftNorm::Backward,
                )
                .unwrap(),
        );
    }
    cuda.runtime().synchronize().unwrap();
    for output in outputs {
        let host = support::download_cuda(cuda.runtime(), &output).unwrap();
        assert_host_close(&host, &expected, 1.0e-5);
    }
    assert!(executor.cache_stats().hits >= 3);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_validation_rejects_host_foreign_integer_bool_and_invalid_lengths() {
    if !gpu_available() {
        return;
    }

    let mut cuda = support::cuda_backend();
    let host = real_f64(&[4], 0.5);
    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, None, -1, FftNorm::Backward),
        "device",
    );

    let other = support::cuda_backend();
    let gpu_input = support::upload_cuda(other.runtime(), &host);
    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &gpu_input, None, -1, FftNorm::Backward),
        "runtime",
    );

    for input in [
        Tensor::from_vec_col_major(vec![4], vec![1_i32, 2, 3, 4]).unwrap(),
        Tensor::from_vec_col_major(vec![4], vec![true, false, true, false]).unwrap(),
    ] {
        assert_error(
            Operation::Fft.execute_cuda(&mut cuda, &input, None, -1, FftNorm::Backward),
            "dtype",
        );
    }

    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, None, 4, FftNorm::Backward),
        "axis",
    );
    assert_error(
        Operation::Fft.execute_cuda(&mut cuda, &host, Some(0), -1, FftNorm::Backward),
        "n",
    );

    let spectrum = complex_f64(&[3], 0.5);
    assert_error(
        Operation::Irfft.execute_cuda(&mut cuda, &spectrum, Some(8), -1, FftNorm::Backward),
        "spectrum",
    );
}

#[test]
fn cuda_host_descriptor_overflow_contract_remains_checked() {
    let source = include_str!("../src/cuda/descriptor.rs");
    assert!(source.contains("checked_mul"));
    assert!(source.contains("checked_cufft_i64"));
    assert!(source.contains("element_count"));
}

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn with_invalid_cufft_path<T>(f: impl FnOnce() -> T) -> T {
    let lock = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    let previous = std::env::var_os("TENFERRO_CUFFT_PATH");
    std::env::set_var(
        "TENFERRO_CUFFT_PATH",
        "/definitely/missing/libcufft-for-acceptance.so",
    );
    let result = catch_unwind(AssertUnwindSafe(f));
    match previous {
        Some(value) => std::env::set_var("TENFERRO_CUFFT_PATH", value),
        None => std::env::remove_var("TENFERRO_CUFFT_PATH"),
    }
    drop(lock);
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[test]
#[ignore = "requires a CUDA runtime without a discoverable cuFFT library"]
fn cuda_missing_cufft_library_is_typed_and_never_cpu_fallback() {
    if !gpu_available() {
        return;
    }

    let mut backend = support::cuda_backend();
    let input = support::upload_cuda(backend.runtime(), &real_f64(&[4], 0.5));
    let result = with_invalid_cufft_path(|| {
        Operation::Fft.execute_cuda(&mut backend, &input, None, -1, FftNorm::Backward)
    });
    let error = result.expect_err("missing cuFFT must not produce a CPU tensor");
    let chain = error_chain(&error);
    assert!(
        chain.contains("failed to load cuFFT library") || chain.contains("cuFFT"),
        "missing cuFFT error lost its typed loader source: {chain}"
    );
    assert!(
        error.source().is_some(),
        "CUDA library error must retain source"
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_execution_uses_explicit_engine_and_fft_module_registration() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let backend = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(backend.runtime(), &host);
    let domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .unwrap();
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&backend, true);
    let outputs = runtime.run_compiled(&program, &[&gpu_input]).unwrap();
    assert_eq!(outputs.len(), 1);
    support::assert_cuda_resident(&outputs[0], domain);
    let output = support::download_cuda(backend.runtime(), &outputs[0]).unwrap();
    assert_host_close(&output, &expected, 1.0e-11);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_missing_extension_module_is_an_explicit_error() {
    if !gpu_available() {
        return;
    }

    let backend = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let gpu_input = support::upload_cuda(backend.runtime(), &host);
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&backend, false);
    let error = runtime
        .run_compiled(&program, &[&gpu_input])
        .expect_err("missing FFT extension registration must fail");
    let message = error.to_string();
    assert!(
        message.contains("extension") || message.contains("module"),
        "missing registration error was not explicit: {message}"
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_runtime_mismatch_is_rejected_without_host_fallback() {
    if !gpu_available() {
        return;
    }

    let first = support::cuda_backend();
    let second = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let foreign_input = support::upload_cuda(second.runtime(), &host);
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&first, true);
    let error = runtime
        .run_compiled(&program, &[&foreign_input])
        .expect_err("foreign CUDA runtime input must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("runtime") || message.contains("input") || message.contains("device"),
        "runtime mismatch error was not explicit: {message}"
    );
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_runtime_owned_cache_reuses_stats_clears_and_retires_after_launch() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let backend = support::cuda_backend();
    let host = complex_f64(&[4], 0.5);
    let expected = Operation::Fft
        .execute_cpu(&mut cpu, &host, None, -1, FftNorm::Backward)
        .unwrap();
    let gpu_input = support::upload_cuda(backend.runtime(), &host);
    let domain = TensorRead::from_tensor(&gpu_input)
        .allocation_domain()
        .unwrap();
    let (program, _) = compiled_cuda_fft(
        DType::C64,
        &[4],
        Operation::Fft,
        None,
        -1,
        FftNorm::Backward,
    );
    let runtime = cuda_runtime_with_fft(&backend, true);

    let first = runtime
        .run_compiled(&program, &[&gpu_input])
        .unwrap()
        .remove(0);
    let second = runtime
        .run_compiled(&program, &[&gpu_input])
        .unwrap()
        .remove(0);
    let stats = runtime.cache_stats().unwrap().extensions;
    assert!(
        stats.entries > 0,
        "runtime FFT cache was not populated: {stats:?}"
    );
    assert!(
        stats.retained_bytes > 0,
        "runtime FFT cache has no retained bytes: {stats:?}"
    );
    runtime.clear_caches().unwrap();
    assert_eq!(runtime.cache_stats().unwrap().extensions.entries, 0);

    support::assert_cuda_resident(&first, domain);
    support::assert_cuda_resident(&second, domain);
    backend.runtime().synchronize().unwrap();
    let first = support::download_cuda(backend.runtime(), &first).unwrap();
    let second = support::download_cuda(backend.runtime(), &second).unwrap();
    assert_host_close(&first, &expected, 1.0e-11);
    assert_host_close(&second, &expected, 1.0e-11);
}

#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_traced_zero_batch_bypasses_cufft_library() {
    if !gpu_available() {
        return;
    }

    let backend = support::cuda_backend();
    for (host, dtype, shape, axis) in [
        (
            Tensor::from_vec_col_major(vec![0, 8], Vec::<Complex64>::new()).unwrap(),
            DType::C64,
            vec![0, 8],
            1isize,
        ),
        (
            Tensor::from_vec_col_major(vec![2, 0, 8], Vec::<f32>::new()).unwrap(),
            DType::F32,
            vec![2, 0, 8],
            2isize,
        ),
    ] {
        let gpu_input = support::upload_cuda(backend.runtime(), &host);
        let domain = TensorRead::from_tensor(&gpu_input)
            .allocation_domain()
            .unwrap();
        let operation = if dtype == DType::C64 {
            Operation::Fft
        } else {
            Operation::Rfft
        };
        let (program, _) =
            compiled_cuda_fft(dtype, &shape, operation, None, axis, FftNorm::Backward);
        let runtime = cuda_runtime_with_fft(&backend, true);
        let output = with_invalid_cufft_path(|| {
            runtime
                .run_compiled(&program, &[&gpu_input])
                .unwrap()
                .remove(0)
        });
        support::assert_cuda_resident(&output, domain);
        assert_eq!(output.shape(), shape.as_slice());
        backend.runtime().synchronize().unwrap();
        let output = support::download_cuda(backend.runtime(), &output).unwrap();
        assert_eq!(output.shape(), shape.as_slice());
        assert_eq!(
            output.as_slice::<Complex64>().map_or(0, <[Complex64]>::len),
            0
        );
        assert_eq!(
            output.as_slice::<Complex32>().map_or(0, <[Complex32]>::len),
            0
        );
        assert_eq!(output.as_slice::<f32>().map_or(0, <[f32]>::len), 0);
    }
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_eager_fft_rfft_irfft_select_cuda_and_keep_cpu_control_on_host() {
    if !gpu_available() {
        return;
    }

    let mut cpu = CpuBackend::new();
    let upload_backend = support::cuda_backend();
    let real_host = real_f64(&[8], 0.25);
    let real_device = support::upload_cuda(upload_backend.runtime(), &real_host);
    let real_domain = TensorRead::from_tensor(&real_device)
        .allocation_domain()
        .unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let eager_real = EagerTensor::from_tensor_in(real_device, ctx.clone()).unwrap();
    let eager_spectrum = eager_real.rfft(None, -1, FftNorm::Backward).unwrap();
    let _eager_spectrum_repeat = eager_real.rfft(None, -1, FftNorm::Backward).unwrap();
    let spectrum_tensor = eager_spectrum.to_tensor().unwrap();
    support::assert_cuda_resident(&spectrum_tensor, real_domain);
    let expected_spectrum = Operation::Rfft
        .execute_cpu(&mut cpu, &real_host, None, -1, FftNorm::Backward)
        .unwrap();
    let eager_cache_stats = ctx.cache_stats().unwrap().extensions;
    assert!(
        eager_cache_stats.entries > 0,
        "eager FFT cache was not populated"
    );
    assert!(
        eager_cache_stats.hits > 0,
        "repeated eager FFT should reuse a plan"
    );
    ctx.synchronize().unwrap();
    let spectrum_host = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&spectrum_tensor))
        })
        .unwrap()
        .unwrap();
    assert_host_close(&spectrum_host, &expected_spectrum, 1.0e-11);

    let eager_signal = eager_spectrum.irfft(None, -1, FftNorm::Backward).unwrap();
    let signal_tensor = eager_signal.to_tensor().unwrap();
    support::assert_cuda_resident(&signal_tensor, real_domain);
    let expected_signal = Operation::Irfft
        .execute_cpu(&mut cpu, &expected_spectrum, None, -1, FftNorm::Backward)
        .unwrap();
    ctx.synchronize().unwrap();
    let signal_host = ctx
        .with_execution_session(|session| {
            session.download_to_host(TensorRead::from_tensor(&signal_tensor))
        })
        .unwrap()
        .unwrap();
    assert_host_close(&signal_host, &expected_signal, 1.0e-11);

    let complex_host = complex_f64(&[4], 0.5);
    let complex_device = ctx
        .with_execution_session(|session| {
            session.upload_host_tensor(TensorRead::from_tensor(&complex_host))
        })
        .unwrap()
        .unwrap();
    let eager_complex = EagerTensor::from_tensor_in(complex_device, ctx.clone()).unwrap();
    let eager_fft = eager_complex.fft(None, -1, FftNorm::Backward).unwrap();
    let fft_tensor = eager_fft.to_tensor().unwrap();
    support::assert_cuda_resident(&fft_tensor, real_domain);

    let cpu_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let cpu_input = EagerTensor::from_tensor_in(real_f64(&[8], 0.25), cpu_ctx.clone()).unwrap();
    let cpu_output = cpu_input
        .rfft(None, -1, FftNorm::Backward)
        .unwrap()
        .to_tensor()
        .unwrap();
    assert_eq!(
        cpu_output.placement().memory_kind,
        tenferro_tensor::MemoryKind::UnpinnedHost
    );
    assert!(!cpu_output.is_backend_buffer());
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA cuFFT hardware and library"]
fn cuda_eager_zero_batch_bypasses_cufft_library_and_runtime_cache_is_clearable() {
    if !gpu_available() {
        return;
    }

    let upload_backend = support::cuda_backend();
    let host = Tensor::from_vec_col_major(vec![0, 8], Vec::<Complex64>::new()).unwrap();
    let device = support::upload_cuda(upload_backend.runtime(), &host);
    let domain = TensorRead::from_tensor(&device)
        .allocation_domain()
        .unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let input = EagerTensor::from_tensor_in(device, ctx.clone()).unwrap();
    let output = with_invalid_cufft_path(|| input.fft(None, 1, FftNorm::Backward).unwrap());
    let tensor = output.to_tensor().unwrap();
    support::assert_cuda_resident(&tensor, domain);
    assert_eq!(tensor.shape(), &[0, 8]);
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 0);
    ctx.clear_caches().unwrap();
    assert_eq!(ctx.cache_stats().unwrap().extensions.entries, 0);
}
