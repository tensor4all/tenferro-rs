use std::error::Error as _;

use num_complex::{Complex32, Complex64};

use crate::config::{GatherConfig, ScatterConfig};
use crate::cubecl::{
    download_tensor, upload_tensor, with_cuda_exec_session, CudaBackend, CudaDeviceId,
    CudaExecSession,
};
use crate::{DType, Error, Tensor, TypedTensor};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{backend::BackendSessionHost, ErrorKind, ValidationError, ValidationKind};

mod capability_tests;
mod cubecl_session_tests;
mod device_tests;
mod elementwise_tests;
mod fusion_tests;
mod gemm_accum_tests;
mod gemm_tests;
mod indexing_tests;
mod metadata_tests;
mod raw_launch_tests;
mod raw_session_tests;
mod reduction_tests;
mod runtime_tests;
mod structural_tests;

/// Enter a CUDA execution session through the erased backend-session surface.
///
/// Mirrors the downstream pattern: `with_cuda_exec_session` must receive the
/// session reconstructed by `with_backend_session`, not the raw backend.
pub(crate) fn with_cuda_exec<R: Send>(
    backend: &mut CudaBackend,
    f: impl FnOnce(&mut CudaExecSession<'_>) -> R + Send,
) -> R {
    backend
        .with_backend_session(|session| with_cuda_exec_session(session, f).expect("CUDA session"))
}

fn cpu_backend() -> CpuBackend {
    CpuBackend::new()
}

fn gpu_backend() -> CudaBackend {
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap()
}

fn upload(backend: &CudaBackend, tensor: &Tensor) -> Tensor {
    upload_tensor(backend.runtime(), tensor).unwrap()
}

fn download(backend: &CudaBackend, tensor: &Tensor) -> Tensor {
    download_tensor(backend.runtime(), tensor).unwrap()
}

fn tensor_f32(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_f64(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_i64(shape: Vec<usize>, data: Vec<i64>) -> Tensor {
    Tensor::I64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_i32(shape: Vec<usize>, data: Vec<i32>) -> Tensor {
    Tensor::I32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_bool(shape: Vec<usize>, data: Vec<bool>) -> Tensor {
    Tensor::Bool(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_c32(shape: Vec<usize>, data: Vec<Complex32>) -> Tensor {
    Tensor::C32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn tensor_c64(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn assert_validation_kind(error: &Error, op: &'static str, kind: ValidationKind) {
    assert_eq!(error.kind(), ErrorKind::Validation(kind));
    assert!(matches!(
        error,
        Error::Validation {
            op: actual,
            source,
        } if *actual == op && source.kind() == kind
    ));
}

fn assert_dtype_mismatch(error: &Error, op: &'static str, expected: DType, actual: DType) {
    assert_validation_kind(error, op, ValidationKind::DTypeMismatch);
    let expected = core_dtype(expected);
    let actual = core_dtype(actual);
    assert!(matches!(
        error,
        Error::Validation {
            source: ValidationError::DTypeMismatch {
                expected: source_expected,
                actual: source_actual,
            },
            ..
        } if *source_expected == expected && *source_actual == actual
    ));
}

fn core_dtype(dtype: DType) -> tenferro_tensor::core::DType {
    match dtype {
        DType::F32 => tenferro_tensor::core::DType::F32,
        DType::F64 => tenferro_tensor::core::DType::F64,
        DType::I32 => tenferro_tensor::core::DType::I32,
        DType::I64 => tenferro_tensor::core::DType::I64,
        DType::Bool => tenferro_tensor::core::DType::Bool,
        DType::C32 => tenferro_tensor::core::DType::C32,
        DType::C64 => tenferro_tensor::core::DType::C64,
    }
}

fn assert_shape_mismatch(error: &Error, op: &'static str, lhs: &[usize], rhs: &[usize]) {
    assert_validation_kind(error, op, ValidationKind::ShapeMismatch);
    assert!(matches!(
        error,
        Error::Validation {
            source: ValidationError::ShapeMismatch(source),
            ..
        } if matches!(
            source.as_ref(),
            tenferro_tensor::ShapeMismatch::IncompatibleShapes { lhs: source_lhs, rhs: source_rhs }
                if source_lhs.as_slice() == lhs && source_rhs.as_slice() == rhs
        )
    ));
}

fn assert_runtime_state(error: &Error, op: &'static str, message: &str) {
    assert!(matches!(
        error,
        Error::RuntimeState {
            op: actual_op,
            message: actual_message,
        } if *actual_op == op && actual_message == message
    ));
}

fn assert_unsupported(error: &Error, op: &'static str, message: &str) {
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains(op));
    assert!(error.to_string().contains(message));
}

fn assert_error_parity(expected: Error, actual: Error) {
    assert_eq!(actual.kind(), expected.kind());
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(actual.source().is_some(), expected.source().is_some());
}

fn assert_cuda_unsupported_dtype(error: &Error, op: &'static str, dtype: DType) {
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let source = error
        .source()
        .expect("unsupported CUDA errors have a source");
    let source = source
        .downcast_ref::<super::error::CudaError>()
        .expect("CUDA unsupported dtype keeps its typed source");
    assert!(matches!(
        source,
        super::error::CudaError::UnsupportedDType {
            op: actual_op,
            dtype: actual_dtype,
        } if *actual_op == op && *actual_dtype == dtype
    ));
}

fn assert_cuda_numerical_error(error: &Error, op: &'static str, dtype: DType, negative: bool) {
    assert_eq!(error.kind(), ErrorKind::NumericalFailure);
    let source = error.source().expect("CUDA numerical errors have a source");
    let source = source
        .downcast_ref::<super::error::CudaError>()
        .expect("CUDA numerical errors keep their typed source");
    if negative {
        assert!(matches!(
            source,
            super::error::CudaError::NegativeIntegerExponent {
                op: actual_op,
                dtype: actual_dtype,
            } if *actual_op == op && *actual_dtype == dtype
        ));
    } else {
        assert!(matches!(
            source,
            super::error::CudaError::DivisionByZero {
                op: actual_op,
                dtype: actual_dtype,
            } if *actual_op == op && *actual_dtype == dtype
        ));
    }
}

#[test]
fn gather_launch_meta_rejects_offset_dim_outside_output_rank() {
    let err = super::gather_launch_meta(
        &[2, 3],
        &[],
        &GatherConfig {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 0,
            slice_sizes: vec![1, 3],
        },
    )
    .unwrap_err();

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::AxisOutOfBounds)
    );
    assert!(matches!(
        err,
        Error::Validation {
            op: "gather",
            source: ValidationError::AxisOutOfBounds { axis: 1, rank: 1 },
        }
    ));
}

#[test]
fn gather_launch_meta_rejects_collapsed_non_unit_slice_sizes() {
    let err = super::gather_launch_meta(
        &[3],
        &[],
        &GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 0,
            slice_sizes: vec![2],
        },
    )
    .unwrap_err();

    assert_validation_kind(&err, "gather", ValidationKind::InvalidArgument);
    assert!(matches!(
        err,
        Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "collapsed_slice_dims",
                ..
            },
            ..
        }
    ));
    assert!(err.to_string().contains("slice_size == 1"), "{err}");
}

#[test]
fn scatter_launch_meta_rejects_mismatched_update_batch_extents() {
    let err = super::scatter_launch_meta(
        &[4],
        &[2, 1],
        &[3],
        &ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
    )
    .unwrap_err();

    assert_shape_mismatch(&err, "scatter", &[2], &[3]);
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tol: f64) {
    assert_eq!(actual.shape(), expected.shape());
    match (actual, expected) {
        (Tensor::F32(_), Tensor::F32(_)) => {
            let actual = actual.as_slice::<f32>().unwrap();
            let expected = expected.as_slice::<f32>().unwrap();
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let diff = (*lhs as f64 - *rhs as f64).abs();
                assert!(
                    diff <= tol,
                    "f32 tensors differ: lhs={lhs:?} rhs={rhs:?} diff={diff}"
                );
            }
        }
        (Tensor::F64(_), Tensor::F64(_)) => {
            let actual = actual.as_slice::<f64>().unwrap();
            let expected = expected.as_slice::<f64>().unwrap();
            for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
                let diff = (*lhs - *rhs).abs();
                assert!(
                    diff <= tol,
                    "f64 tensors differ at {idx}: lhs={lhs:?} rhs={rhs:?} diff={diff}; actual={actual:?} expected={expected:?}"
                );
            }
        }
        (Tensor::I64(_), Tensor::I64(_)) => {
            let actual = actual.as_slice::<i64>().unwrap();
            let expected = expected.as_slice::<i64>().unwrap();
            assert_eq!(actual, expected);
        }
        (Tensor::I32(_), Tensor::I32(_)) => {
            let actual = actual.as_slice::<i32>().unwrap();
            let expected = expected.as_slice::<i32>().unwrap();
            assert_eq!(actual, expected);
        }
        (Tensor::Bool(_), Tensor::Bool(_)) => {
            let actual = actual.as_slice::<bool>().unwrap();
            let expected = expected.as_slice::<bool>().unwrap();
            assert_eq!(actual, expected);
        }
        (Tensor::C32(_), Tensor::C32(_)) => {
            let actual = actual.as_slice::<Complex32>().unwrap();
            let expected = expected.as_slice::<Complex32>().unwrap();
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let real_diff = (lhs.re as f64 - rhs.re as f64).abs();
                let imag_diff = (lhs.im as f64 - rhs.im as f64).abs();
                assert!(
                    real_diff <= tol && imag_diff <= tol,
                    "c32 tensors differ: lhs={lhs:?} rhs={rhs:?}"
                );
            }
        }
        (Tensor::C64(_), Tensor::C64(_)) => {
            let actual = actual.as_slice::<Complex64>().unwrap();
            let expected = expected.as_slice::<Complex64>().unwrap();
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let real_diff = (lhs.re - rhs.re).abs();
                let imag_diff = (lhs.im - rhs.im).abs();
                assert!(
                    real_diff <= tol && imag_diff <= tol,
                    "c64 tensors differ: lhs={lhs:?} rhs={rhs:?}"
                );
            }
        }
        _ => panic!(
            "dtype mismatch actual={:?} expected={:?}",
            actual.dtype(),
            expected.dtype()
        ),
    }
}

fn simple_gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

fn diagonal_scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    }
}
