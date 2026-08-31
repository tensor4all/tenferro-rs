use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;

use super::{append_2d, compact_factor_2d, from_factors_2d, q_columns_2d, raw_r_2d};

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
fn compact_factor_and_append_reconstruct_without_refactoring_old_columns() {
    let a =
        TypedTensor::from_vec_col_major(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0])
            .unwrap();
    let b =
        TypedTensor::from_vec_col_major(vec![4, 2], vec![3.0, -1.0, 2.0, 1.0, 0.5, 2.0, -2.0, 4.0])
            .unwrap();
    let mut buffers = BufferPool::new();

    let (packed, tau) = compact_factor_2d(&mut buffers, &a).unwrap();
    let (packed, tau) = append_2d(&mut buffers, &packed, &tau, &b).unwrap();
    let r = raw_r_2d(&packed, &tau, false).unwrap();
    let q = q_columns_2d(&packed, &tau, 0, 4, false).unwrap();

    let expected = [
        1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0, 3.0, -1.0, 2.0, 1.0, 0.5, 2.0, -2.0, 4.0,
    ];
    assert_close(
        &product(q.host_data().unwrap(), 4, 4, r.host_data().unwrap(), 4),
        &expected,
    );
}

#[test]
fn from_factors_reconstructs_product_without_forming_dense_qr_product() {
    let q =
        TypedTensor::from_vec_col_major(vec![4, 2], vec![1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
            .unwrap();
    let r =
        TypedTensor::from_vec_col_major(vec![2, 3], vec![2.0, 0.0, 3.0, 1.0, 4.0, 2.0]).unwrap();
    let mut buffers = BufferPool::new();

    let (packed, tau) = from_factors_2d(&mut buffers, &q, &r).unwrap();
    let extracted_r = raw_r_2d(&packed, &tau, false).unwrap();
    let extracted_q = q_columns_2d(&packed, &tau, 0, 3, false).unwrap();

    let expected = product(q.host_data().unwrap(), 4, 2, r.host_data().unwrap(), 3);
    assert_close(
        &product(
            extracted_q.host_data().unwrap(),
            4,
            3,
            extracted_r.host_data().unwrap(),
            3,
        ),
        &expected,
    );
}
