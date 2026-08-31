use super::checked_product;
use crate::{QrOptions, TensorLinalgExt};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSessionHost, Tensor};

fn product(a: &[f64], a_rows: usize, a_cols: usize, b: &[f64], b_cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; a_rows * b_cols];
    for col in 0..b_cols {
        for inner in 0..a_cols {
            for row in 0..a_rows {
                out[row + col * a_rows] += a[row + inner * a_rows] * b[inner + col * a_cols];
            }
        }
    }
    out
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(tensor) => tensor.host_data().unwrap(),
        _ => panic!("expected f64 tensor"),
    }
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(error < 1.0e-10, "max reconstruction error: {error}");
}

#[test]
fn compact_factor_and_append_reconstructs_f64() {
    let a = Tensor::from_vec_col_major(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0])
        .unwrap();
    let b = Tensor::from_vec_col_major(vec![4, 2], vec![3.0, -1.0, 2.0, 1.0, 0.5, 2.0, -2.0, 4.0])
        .unwrap();
    let mut backend = CpuBackend::new();
    backend
        .with_backend_session(|session| {
            let state = a.householder_qr(session)?;
            let state = state.append_columns(&b, session)?;
            let q = state.q_columns(0..4, QrOptions::default(), session)?;
            let r = state.r(QrOptions::default(), session)?;
            let expected = [
                1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0, 3.0, -1.0, 2.0, 1.0, 0.5, 2.0, -2.0, 4.0,
            ];
            assert_close(&product(f64_data(&q), 4, 4, f64_data(&r), 4), &expected);
            Ok::<(), tenferro_tensor::Error>(())
        })
        .unwrap();
}

#[test]
fn compact_from_factors_reconstructs_f64() {
    let q = Tensor::from_vec_col_major(vec![4, 2], vec![1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
        .unwrap();
    let r = Tensor::from_vec_col_major(vec![2, 3], vec![2.0, 0.0, 3.0, 1.0, 4.0, 2.0]).unwrap();
    let mut backend = CpuBackend::new();
    backend
        .with_backend_session(|session| {
            let state = crate::HouseholderQr::<Tensor>::from_factors(&q, &r, session)?;
            let q_out = state.q_columns(0..3, QrOptions::default(), session)?;
            let r_out = state.r(QrOptions::default(), session)?;
            let expected = product(f64_data(&q), 4, 2, f64_data(&r), 3);
            assert_close(
                &product(f64_data(&q_out), 4, 3, f64_data(&r_out), 3),
                &expected,
            );
            Ok::<(), tenferro_tensor::Error>(())
        })
        .unwrap();
}

#[test]
fn checked_matrix_element_count_reports_typed_overflow() {
    let err = checked_product("test_faer_allocation", "matrix", &[usize::MAX, 2])
        .expect_err("overflowing matrix dimensions must fail");

    assert!(matches!(
        err,
        tenferro_tensor::Error::Validation {
            op: "test_faer_allocation",
            ..
        }
    ));
}
