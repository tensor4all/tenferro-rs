use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{cpu::CpuBackend, Error, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

#[test]
fn cholesky_rejects_rank_less_than_two_even_when_zero_dim() {
    let input = f64_tensor(vec![0], Vec::new());
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.cholesky(&input)));

    assert!(result.is_ok(), "cholesky should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::RankMismatch {
            op: "cholesky",
            expected: 2,
            actual: 1,
        }
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
fn triangular_solve_rejects_batch_mismatch_without_backend_panic() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let b = f64_tensor(vec![2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = catch_unwind(AssertUnwindSafe(|| {
        backend.triangular_solve(&a, &b, true, true, false, false)
    }));

    assert!(
        result.is_ok(),
        "triangular_solve should return Err on batch mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "triangular_solve",
            ..
        }
    ));
}

#[test]
fn full_piv_lu_solve_rejects_batch_mismatch_without_backend_panic() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let b = f64_tensor(vec![2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = catch_unwind(AssertUnwindSafe(|| {
        backend.full_piv_lu_solve(&a, &b, false)
    }));

    assert!(
        result.is_ok(),
        "full_piv_lu_solve should return Err on batch mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "full_piv_lu_solve",
            ..
        }
    ));
}
