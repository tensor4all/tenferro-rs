use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_algebra::Standard;
use tenferro_tensor::{
    cpu::CpuBackend, DotGeneralConfig, Error, PadConfig, SemiringBackend, SliceConfig, Tensor,
    TensorBackend, TypedTensor,
};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

#[test]
fn dot_general_rejects_out_of_bounds_contracting_dim() {
    let lhs = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut backend = CpuBackend::new();

    let err = backend
        .dot_general(
            &lhs,
            &rhs,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![2],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
                lhs_rank: 2,
                rhs_rank: 2,
            },
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::AxisOutOfBounds {
            op: "dot_general",
            axis: 2,
            rank: 2,
        }
    ));
}

#[test]
fn semiring_add_rejects_shape_mismatch() {
    let lhs = TypedTensor::from_vec(vec![2], vec![1.0, 2.0]);
    let rhs = TypedTensor::from_vec(vec![3], vec![3.0, 4.0, 5.0]);
    let mut backend = CpuBackend::new();

    let err =
        <CpuBackend as SemiringBackend<Standard<f64>>>::add(&mut backend, &lhs, &rhs).unwrap_err();

    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "add",
            lhs,
            rhs,
        } if lhs == vec![2] && rhs == vec![3]
    ));
}

#[test]
fn add_rejects_shape_mismatch() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs = f64_tensor(vec![3], vec![3.0, 4.0, 5.0]);
    let mut backend = CpuBackend::new();

    let err = <CpuBackend as TensorBackend>::add(&mut backend, &lhs, &rhs).unwrap_err();

    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "add",
            lhs,
            rhs,
        } if lhs == vec![2] && rhs == vec![3]
    ));
}

#[test]
fn solve_rejects_singular_matrix() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]);
    let b = f64_tensor(vec![2, 1], vec![1.0, 2.0]);
    let mut backend = CpuBackend::new();

    let err = backend.solve(&a, &b).unwrap_err();

    assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
}

#[test]
fn semiring_reduce_sum_rejects_out_of_bounds_axis() {
    let input = TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let err =
        <CpuBackend as SemiringBackend<Standard<f64>>>::reduce_sum(&mut backend, &input, &[2])
            .unwrap_err();

    assert!(matches!(
        err,
        Error::AxisOutOfBounds {
            op: "reduce_sum",
            axis: 2,
            rank: 2,
        }
    ));
}

#[test]
fn semiring_batched_gemm_returns_error_instead_of_panicking() {
    let lhs = TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut backend = CpuBackend::new();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0, 1],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        <CpuBackend as SemiringBackend<Standard<f64>>>::batched_gemm(
            &mut backend,
            &lhs,
            &rhs,
            &config,
        )
    }));

    assert!(
        result.is_ok(),
        "semiring batched_gemm should return Err, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "batched_gemm",
            ..
        } | Error::BackendFailure {
            op: "batched_gemm",
            ..
        }
    ));
}

#[test]
fn transpose_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.transpose(&input, &[0])));

    assert!(result.is_ok(), "transpose should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "transpose",
            ..
        } | Error::InvalidConfig {
            op: "transpose",
            ..
        } | Error::RankMismatch {
            op: "transpose",
            ..
        } | Error::AxisOutOfBounds {
            op: "transpose",
            ..
        } | Error::DuplicateAxis {
            op: "transpose",
            ..
        }
    ));
}

#[test]
fn reshape_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.reshape(&input, &[3])));

    assert!(result.is_ok(), "reshape should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "reshape", .. }
            | Error::InvalidConfig { op: "reshape", .. }
            | Error::ShapeMismatch { op: "reshape", .. }
    ));
}

#[test]
fn pow_returns_error_on_shape_mismatch_instead_of_panicking() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs = f64_tensor(vec![1], vec![3.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.pow(&lhs, &rhs)));

    assert!(result.is_ok(), "pow should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch { op: "pow", .. } | Error::BackendFailure { op: "pow", .. }
    ));
}

#[test]
fn slice_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();
    let config = SliceConfig {
        starts: vec![0],
        limits: vec![2],
        strides: vec![1],
    };

    let result = catch_unwind(AssertUnwindSafe(|| backend.slice(&input, &config)));

    assert!(result.is_ok(), "slice should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "slice", .. }
            | Error::InvalidConfig { op: "slice", .. }
            | Error::RankMismatch { op: "slice", .. }
            | Error::AxisOutOfBounds { op: "slice", .. }
    ));
}

#[test]
fn pad_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();
    let config = PadConfig {
        edge_padding_low: vec![0, 0],
        edge_padding_high: vec![0],
        interior_padding: vec![0, 0],
    };

    let result = catch_unwind(AssertUnwindSafe(|| backend.pad(&input, &config)));

    assert!(result.is_ok(), "pad should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "pad", .. }
            | Error::InvalidConfig { op: "pad", .. }
            | Error::RankMismatch { op: "pad", .. }
    ));
}
