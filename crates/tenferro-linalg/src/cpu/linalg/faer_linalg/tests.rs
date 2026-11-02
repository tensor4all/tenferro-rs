use super::checked_product;
use crate::cpu::tests::with_cpu_linalg;
use crate::{QrOptions, TensorLinalgExt};
use num_complex::Complex64;
use tenferro_tensor::Tensor;

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
    tensor
        .as_typed::<f64>()
        .expect("expected f64 tensor")
        .host_data()
        .unwrap()
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
fn compact_factor_and_append_reconstructs_f64() -> Result<(), tenferro_tensor::Error> {
    let a = Tensor::from_vec_col_major(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0])
        .unwrap();
    let b = Tensor::from_vec_col_major(vec![4, 2], vec![3.0, -1.0, 2.0, 1.0, 0.5, 2.0, -2.0, 4.0])
        .unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let state = a.householder_qr(&mut backend)?;
    let state = state.append_columns(&b, &mut backend)?;
    let q = state.q_columns(0..4, QrOptions::default(), &mut backend)?;
    let r = state.r(QrOptions::default(), &mut backend)?;
    let expected = [
        1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0, 3.0, -1.0, 2.0, 1.0, 0.5, 2.0, -2.0, 4.0,
    ];
    assert_close(&product(f64_data(&q), 4, 4, f64_data(&r), 4), &expected);
    Ok::<(), tenferro_tensor::Error>(())
}

#[test]
fn compact_from_factors_reconstructs_f64() -> Result<(), tenferro_tensor::Error> {
    let q = Tensor::from_vec_col_major(vec![4, 2], vec![1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
        .unwrap();
    let r = Tensor::from_vec_col_major(vec![2, 3], vec![2.0, 0.0, 3.0, 1.0, 4.0, 2.0]).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let state = crate::HouseholderQr::<Tensor>::from_factors(&q, &r, &mut backend)?;
    let q_out = state.q_columns(0..3, QrOptions::default(), &mut backend)?;
    let r_out = state.r(QrOptions::default(), &mut backend)?;
    let expected = product(f64_data(&q), 4, 2, f64_data(&r), 3);
    assert_close(
        &product(f64_data(&q_out), 4, 3, f64_data(&r_out), 3),
        &expected,
    );
    Ok::<(), tenferro_tensor::Error>(())
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

/// `eig`'s output is already in the gauge [`crate::eig`] produces, so the seam is the public
/// normalizer and lane callers that call it reproduce `eig` (#2387).
#[test]
fn eig_output_is_already_in_the_gauge_of_the_public_normalizer() {
    // [[1, 1], [0, 2]] column-major: eigenvalues 1 and 2, eigenvectors [1, 0] and
    // [1, 1] / sqrt(2), so the provider's columns are not unit-norm on their own.
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 1.0, 2.0]).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let (_values, vectors) = with_cpu_linalg(&mut backend, |backend| a.eig(backend)).unwrap();
    let from_eig = vectors.as_slice::<Complex64>().unwrap().to_vec();
    assert_eq!(from_eig.len(), 4);

    // Re-running the public normalizer on `eig`'s output leaves it in place.
    let mut again = from_eig.clone();
    crate::eig::normalize_eig_vectors_c64(&mut again, 2).unwrap();
    for (normalized, eig) in again.iter().zip(&from_eig) {
        assert!(
            (normalized - eig).norm() < 1.0e-15,
            "{normalized:?} against {eig:?}"
        );
    }

    // And every column carries the unit 2-norm it documents.
    for column in 0..2 {
        let norm = (0..2)
            .map(|row| from_eig[column * 2 + row].norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12, "column {column}: {norm}");
    }
}

/// A lane caller that runs the provider's own eigen-solve and then the public normalizer gets
/// `eig`'s output bit for bit: one implementation of the gauge, not two (#2387).
#[test]
fn the_public_normalizer_reproduces_eig_bit_for_bit() {
    // Column-major [[1, 1], [0, 2]]: the provider's own columns [1, 0] and [1, 1] are not
    // unit-norm, so the normalizer has work to do.
    let data = [1.0_f64, 0.0, 1.0, 2.0];
    let a = Tensor::from_vec_col_major(vec![2, 2], data.to_vec()).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let (_values, from_eig) = with_cpu_linalg(&mut backend, |backend| a.eig(backend)).unwrap();
    let from_eig = from_eig.as_slice::<Complex64>().unwrap().to_vec();

    let (mut raw_values, mut raw_vectors) = (Vec::<Complex64>::new(), Vec::<Complex64>::new());
    tlinalg::eig::eig(
        tlinalg::Op::Eig,
        strided_view::RawStridedRef::new(&data, &[2, 2], &[1, 2], 0).unwrap(),
        &mut raw_values,
        &mut raw_vectors,
        &mut tcontract_exec::Exec::serial(),
    )
    .unwrap();
    crate::eig::normalize_eig_vectors_c64(&mut raw_vectors, 2).unwrap();
    // Compare raw bit patterns, not `==`: `-0.0 == 0.0` and a bit-identity claim must not accept
    // a sign flip the arithmetic never made.
    let bits = |values: &[Complex64]| -> Vec<(u64, u64)> {
        values
            .iter()
            .map(|value| (value.re.to_bits(), value.im.to_bits()))
            .collect()
    };
    assert_eq!(
        bits(&raw_vectors),
        bits(&from_eig),
        "the public normalizer must reproduce eig bit for bit"
    );
}

/// A 64x24 `f64` factor is `packed` (1536 elements) plus `coeff` (24 elements). If either output
/// left the pool as a plain `Vec`, dropping the state would free it instead of returning it, so the
/// retained pool could never hold that payload and a second call would have to allocate it again.
#[test]
fn compact_factor_outputs_return_their_storage_to_the_pool() {
    let packed_bytes = 64 * 24 * std::mem::size_of::<f64>();
    let coeff_bytes = 24 * std::mem::size_of::<f64>();
    let mut data = vec![0.0_f64; 64 * 24];
    for col in 0..24 {
        for row in 0..64 {
            data[row + col * 64] = if row == col {
                9.0
            } else {
                0.1 * (row as f64) - 0.05
            };
        }
    }
    let a = Tensor::from_vec_col_major(vec![64, 24], data).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let empty = backend.buffer_pool_stats();
    assert_eq!(empty.capacity_bytes, 0, "a fresh backend retains nothing");

    let state = with_cpu_linalg(&mut backend, |backend| a.householder_qr(backend)).unwrap();
    let checked_out = backend.buffer_pool_stats();
    drop(state);
    let returned = backend.buffer_pool_stats();
    eprintln!(
        "checked out: {} buffers / {} B; returned: {} buffers / {} B",
        checked_out.buffers, checked_out.capacity_bytes, returned.buffers, returned.capacity_bytes
    );
    assert!(
        returned.capacity_bytes >= packed_bytes + coeff_bytes + checked_out.capacity_bytes,
        "dropping the compact state must return the packed ({packed_bytes} B) and coefficient \
         ({coeff_bytes} B) payloads to the pool: {} B retained before, {} B while checked out, \
         {} B after the drop",
        empty.capacity_bytes,
        checked_out.capacity_bytes,
        returned.capacity_bytes
    );
    assert!(
        returned.buffers > checked_out.buffers,
        "the two outputs must leave the pool while the state is alive"
    );

    // The warm pool serves the whole call: a second run must not grow the retained set, and the
    // same storage must come back.
    let state = with_cpu_linalg(&mut backend, |backend| a.householder_qr(backend)).unwrap();
    drop(state);
    let second = backend.buffer_pool_stats();
    assert_eq!(
        (second.buffers, second.capacity_bytes),
        (returned.buffers, returned.capacity_bytes),
        "a warm pool must serve and recover the whole compact-QR payload"
    );
}
