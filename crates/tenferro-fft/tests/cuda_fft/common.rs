use num_complex::{Complex32, Complex64};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_fft::{FftExecutor, FftNorm, TensorFftExt, TracedTensorFftExt};
use tenferro_gpu::cuda::{cuda_runtime_engine_registration, CudaBackend};
use tenferro_runtime::{DType, EngineId, GraphCompiler, Runtime, Tensor, TracedTensor};
use tenferro_tensor::{BackendSessionHost, TensorRead};

use crate::support;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Operation {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

impl Operation {
    pub(crate) fn execute_cpu(
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

    pub(crate) fn execute_cuda(
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

    pub(crate) fn execute_executor(
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

pub(crate) fn element_count(shape: &[usize]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .expect("test shape product")
}

pub(crate) fn real_f32(shape: &[usize], seed: f32) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| seed + index as f32 * 0.375 - (index % 3) as f32 * 0.125)
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

pub(crate) fn real_f64(shape: &[usize], seed: f64) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| seed + index as f64 * 0.375 - (index % 3) as f64 * 0.125)
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

pub(crate) fn complex_f32(shape: &[usize], seed: f32) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| {
            let value = seed + index as f32 * 0.375;
            Complex32::new(value, 0.25 - value * 0.5)
        })
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

pub(crate) fn complex_f64(shape: &[usize], seed: f64) -> Tensor {
    let data = (0..element_count(shape))
        .map(|index| {
            let value = seed + index as f64 * 0.375;
            Complex64::new(value, 0.25 - value * 0.5)
        })
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn assert_component_close(
    actual: f64,
    expected: f64,
    tolerance: f64,
    index: usize,
    component: &str,
    max_absolute: &mut f64,
    max_relative: &mut f64,
) {
    assert!(
        actual.is_finite(),
        "actual {component} at index {index} is nonfinite: {actual:?}"
    );
    assert!(
        expected.is_finite(),
        "expected {component} at index {index} is nonfinite: {expected:?}"
    );
    let absolute = (actual - expected).abs();
    let relative = if expected == 0.0 {
        absolute
    } else {
        absolute / expected.abs()
    };
    *max_absolute = (*max_absolute).max(absolute);
    *max_relative = (*max_relative).max(relative);
    assert!(
        absolute <= tolerance || relative <= tolerance,
        "{component} at index {index} differs: actual {actual:e}, expected {expected:e}, absolute {absolute:e}, relative {relative:e}, tolerance {tolerance:e}"
    );
}

fn assert_real_components_close<T>(
    actual: &[T],
    expected: &[T],
    tolerance: f64,
    as_f64: impl Fn(T) -> f64,
) where
    T: Copy,
{
    assert_eq!(actual.len(), expected.len());
    let mut max_absolute = 0.0;
    let mut max_relative = 0.0;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_component_close(
            as_f64(actual),
            as_f64(expected),
            tolerance,
            index,
            "value",
            &mut max_absolute,
            &mut max_relative,
        );
    }
    let _ = (max_absolute, max_relative);
}

pub(crate) fn assert_host_close(actual: &Tensor, expected: &Tensor, tolerance: f64) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    match (actual.dtype(), expected.dtype()) {
        (DType::F32, DType::F32) => assert_real_components_close(
            actual.as_slice::<f32>().unwrap(),
            expected.as_slice::<f32>().unwrap(),
            tolerance,
            f64::from,
        ),
        (DType::F64, DType::F64) => assert_real_components_close(
            actual.as_slice::<f64>().unwrap(),
            expected.as_slice::<f64>().unwrap(),
            tolerance,
            |value| value,
        ),
        (DType::C32, DType::C32) => {
            let actual = actual.as_slice::<Complex32>().unwrap();
            let expected = expected.as_slice::<Complex32>().unwrap();
            assert_eq!(actual.len(), expected.len());
            let mut max_absolute = 0.0;
            let mut max_relative = 0.0;
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_component_close(
                    f64::from(actual.re),
                    f64::from(expected.re),
                    tolerance,
                    index,
                    "complex real component",
                    &mut max_absolute,
                    &mut max_relative,
                );
                assert_component_close(
                    f64::from(actual.im),
                    f64::from(expected.im),
                    tolerance,
                    index,
                    "complex imaginary component",
                    &mut max_absolute,
                    &mut max_relative,
                );
                assert_component_close(
                    f64::from(actual.norm()),
                    f64::from(expected.norm()),
                    tolerance,
                    index,
                    "complex magnitude",
                    &mut max_absolute,
                    &mut max_relative,
                );
            }
        }
        (DType::C64, DType::C64) => {
            let actual = actual.as_slice::<Complex64>().unwrap();
            let expected = expected.as_slice::<Complex64>().unwrap();
            assert_eq!(actual.len(), expected.len());
            let mut max_absolute = 0.0;
            let mut max_relative = 0.0;
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_component_close(
                    actual.re,
                    expected.re,
                    tolerance,
                    index,
                    "complex real component",
                    &mut max_absolute,
                    &mut max_relative,
                );
                assert_component_close(
                    actual.im,
                    expected.im,
                    tolerance,
                    index,
                    "complex imaginary component",
                    &mut max_absolute,
                    &mut max_relative,
                );
                assert_component_close(
                    actual.norm(),
                    expected.norm(),
                    tolerance,
                    index,
                    "complex magnitude",
                    &mut max_absolute,
                    &mut max_relative,
                );
            }
        }
        _ => panic!("unexpected comparison dtype {:?}", actual.dtype()),
    }
}

pub(crate) fn run_case(
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

pub(crate) fn run_executor_case(
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

pub(crate) fn assert_error(result: tenferro_tensor::Result<Tensor>, expected_text: &str) {
    let error = result.expect_err("unsupported CUDA FFT input must return an error");
    assert!(!error.to_string().is_empty());
    assert!(
        error.to_string().contains(expected_text),
        "error `{error}` did not contain `{expected_text}`"
    );
}

pub(crate) fn cuda_runtime_with_fft(backend: &CudaBackend, install_module: bool) -> Runtime {
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

pub(crate) fn compiled_cuda_fft(
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

pub(crate) fn assert_full_hermitian(actual: &Tensor) {
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

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::*;

    #[test]
    fn numerical_comparator_rejects_nonfinite_values() {
        let finite = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
        let actual_nan = Tensor::from_vec_col_major(vec![1], vec![f64::NAN]).unwrap();
        let expected_nan = Tensor::from_vec_col_major(vec![1], vec![f64::NAN]).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| {
            assert_host_close(&actual_nan, &finite, 1.0e-5);
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            assert_host_close(&finite, &expected_nan, 1.0e-5);
        }))
        .is_err());
    }

    #[test]
    fn numerical_comparator_uses_each_expected_element_for_relative_error() {
        let expected = Tensor::from_vec_col_major(vec![2], vec![1_000.0_f64, 1.0e-9]).unwrap();
        let actual = Tensor::from_vec_col_major(vec![2], vec![1_000.001, 1.0e-2]).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| {
            assert_host_close(&actual, &expected, 1.0e-3);
        }))
        .is_err());
    }
}
