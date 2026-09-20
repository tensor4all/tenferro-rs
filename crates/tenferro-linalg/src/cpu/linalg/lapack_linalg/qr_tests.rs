use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;

use super::{append_2d, compact_factor_2d, from_factors_2d, q_columns_2d, raw_r_2d};

fn check_batched_qr<T: super::LapackQr>(
    make: impl Fn(f64, f64) -> T,
    value: impl Fn(T) -> num_complex::Complex64,
) {
    for (m, n) in [(1, 1), (2, 2), (5, 3), (3, 5)] {
        let k = m.min(n);
        let data: Vec<_> = (0..m * n * 6)
            .map(|i| make((i % 11) as f64 - 4.0, (i % 7) as f64 - 2.0))
            .collect();
        let input = TypedTensor::from_vec_col_major([m, n, 2, 3], data.clone()).unwrap();
        let mut buffers = BufferPool::new();
        let outputs = super::qr(&mut buffers, &input).unwrap();
        assert_eq!(outputs[0].shape(), &[m, k, 2, 3]);
        assert_eq!(outputs[1].shape(), &[k, n, 2, 3]);
        let q = outputs[0].host_data().unwrap();
        let r = outputs[1].host_data().unwrap();
        for batch in 0..6 {
            let q = &q[batch * m * k..(batch + 1) * m * k];
            let r = &r[batch * k * n..(batch + 1) * k * n];
            for col in 0..n {
                for row in 0..m {
                    let actual: num_complex::Complex64 = (0..k)
                        .map(|i| value(q[row + i * m]) * value(r[i + col * k]))
                        .sum();
                    let expected = value(data[batch * m * n + row + col * m]);
                    let error = (actual - expected).norm();
                    assert!(error < 2e-5, "QR reconstruction error: {error}");
                }
            }
            for col in 0..k {
                for row in 0..k {
                    let actual: num_complex::Complex64 = (0..m)
                        .map(|i| value(q[i + row * m]).conj() * value(q[i + col * m]))
                        .sum();
                    let error = (actual - f64::from(row == col)).norm();
                    assert!(error < 2e-5, "Q orthogonality error: {error}");
                }
            }
        }
        for (&actual, &expected) in input.host_data().unwrap().iter().zip(&data) {
            assert_eq!(value(actual), value(expected), "QR modified input");
        }
    }
}

#[test]
fn batched_qr_reuses_workspace_for_square_tall_wide_and_complex_matrices() {
    use num_complex::{Complex32, Complex64};
    check_batched_qr(|re, _| re, Complex64::from);
    check_batched_qr(|re, _| re as f32, |x| Complex64::from(x as f64));
    check_batched_qr(Complex64::new, |x| x);
    check_batched_qr(
        |re, im| Complex32::new(re as f32, im as f32),
        |x| Complex64::new(x.re as f64, x.im as f64),
    );
}

#[test]
fn batched_qr_handles_empty_shapes_and_rejects_wrong_rank() {
    let mut buffers = BufferPool::new();
    for shape in [[2, 2, 0], [0, 3, 2], [3, 0, 2]] {
        let input = TypedTensor::<f64>::from_vec_col_major(shape, vec![]).unwrap();
        let outputs = super::qr(&mut buffers, &input).unwrap();
        assert!(outputs
            .iter()
            .all(|out| out.host_data().unwrap().is_empty()));
    }
    let input = TypedTensor::<f64>::from_vec_col_major([0], vec![]).unwrap();
    assert!(super::qr(&mut buffers, &input).is_err());
}

fn product(a: &[f64], a_rows: usize, a_cols: usize, b: &[f64], b_cols: usize) -> Vec<f64> {
    let output_len = a_rows
        .checked_mul(b_cols)
        .expect("test matrix product length fits usize");
    let mut out = vec![0.0; output_len];
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
    let q = q_columns_2d(&mut buffers, &packed, &tau, 0, 4, false).unwrap();

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
    let extracted_q = q_columns_2d(&mut buffers, &packed, &tau, 0, 3, false).unwrap();

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
