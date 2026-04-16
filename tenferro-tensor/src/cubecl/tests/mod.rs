use num_complex::{Complex32, Complex64};

use crate::config::{GatherConfig, ScatterConfig};
use crate::cpu::CpuBackend;
use crate::cubecl::{download_tensor, upload_tensor, CubeclBackend};
use crate::{Tensor, TypedTensor};

mod elementwise_tests;
mod gemm_tests;
mod indexing_tests;
mod reduction_tests;
mod runtime_tests;
mod structural_tests;

fn cpu_backend() -> CpuBackend {
    CpuBackend::new()
}

fn gpu_backend() -> CubeclBackend {
    CubeclBackend::new(0).unwrap()
}

fn upload(backend: &CubeclBackend, tensor: &Tensor) -> Tensor {
    upload_tensor(backend.runtime(), tensor).unwrap()
}

fn download(backend: &CubeclBackend, tensor: &Tensor) -> Tensor {
    download_tensor(backend.runtime(), tensor).unwrap()
}

fn tensor_f32(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec(shape, data))
}

fn tensor_f64(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn tensor_c32(shape: Vec<usize>, data: Vec<Complex32>) -> Tensor {
    Tensor::C32(TypedTensor::from_vec(shape, data))
}

fn tensor_c64(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec(shape, data))
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
            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                let diff = (*lhs - *rhs).abs();
                assert!(
                    diff <= tol,
                    "f64 tensors differ: lhs={lhs:?} rhs={rhs:?} diff={diff}"
                );
            }
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
