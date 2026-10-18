use super::{checked_product, FaerEig};
use crate::cpu::tests::with_cpu_linalg;
use crate::{QrOptions, TensorLinalgExt};
use num_complex::{Complex32, Complex64};
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
        .unwrap()
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
        .unwrap()
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

/// Assert the seam reported a non-normalizable eigenvector column through the
/// typed linalg `NonFinite` source, not a generic failure.
fn assert_typed_non_finite(error: tenferro_tensor::Error) {
    assert!(matches!(
        error,
        tenferro_tensor::Error::Extension {
            op: "eig",
            family: crate::LINALG_EXTENSION_FAMILY_ID,
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));
    assert!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<crate::Error>())
            .is_some_and(|source| matches!(
                source,
                crate::Error::NonFinite {
                    op: "eig",
                    role: "eigenvector column",
                }
            )),
        "expected a typed NonFinite eigenvector-column source, got {error}"
    );
}

#[test]
fn normalize_eig_vectors_puts_columns_in_the_unit_gauge() {
    // Column-major 2x2 block: column 0 is [3, 4i], column 1 is [1, 2i].
    let mut columns = vec![
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, 4.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 2.0),
    ];
    <f64 as FaerEig>::normalize_eig_vectors(&mut columns, 2, 1).unwrap();
    assert!((columns[0].re - 0.6).abs() < 1.0e-12, "{:?}", columns[0]);
    assert!((columns[1].im - 0.8).abs() < 1.0e-12, "{:?}", columns[1]);
    let root_five = 5.0_f64.sqrt();
    assert!(
        (columns[2].re - 1.0 / root_five).abs() < 1.0e-12,
        "{:?}",
        columns[2]
    );
    assert!(
        (columns[3].im - 2.0 / root_five).abs() < 1.0e-12,
        "{:?}",
        columns[3]
    );
}

#[test]
fn normalize_eig_vectors_rejects_non_normalizable_columns_typed() {
    // All-zero column: there is no direction to normalize, so the seam must
    // refuse rather than divide by zero.
    let mut zero = vec![Complex64::new(0.0, 0.0); 4];
    assert_typed_non_finite(<f64 as FaerEig>::normalize_eig_vectors(&mut zero, 2, 1).unwrap_err());

    // Non-finite column: a provider that overflowed inside its
    // back-substitution must surface the typed NonFinite source, not NaN.
    let mut non_finite = vec![
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    assert_typed_non_finite(
        <f64 as FaerEig>::normalize_eig_vectors(&mut non_finite, 2, 1).unwrap_err(),
    );
}

#[test]
fn normalize_eig_vectors_normalizes_extreme_magnitude_finite_columns() {
    // #2148 review follow-up: the column norm is accumulated in `f64` after
    // scaling, so finite columns whose squares overflow or underflow in the
    // operand dtype must still normalize instead of being rejected.
    // Column 0 is [3e20, 4i e20], whose operand-dtype square sum is `inf` in
    // f32; column 1 keeps the matrix block non-degenerate.
    let mut large32 = vec![
        Complex32::new(3.0e20, 0.0),
        Complex32::new(0.0, 4.0e20),
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 1.0),
    ];
    <f32 as FaerEig>::normalize_eig_vectors(&mut large32, 2, 1).unwrap();
    assert!((large32[0].re - 0.6).abs() < 1.0e-6, "{:?}", large32[0]);
    assert!((large32[1].im - 0.8).abs() < 1.0e-6, "{:?}", large32[1]);

    // A tiny column used to underflow the operand-dtype square sum to zero.
    let mut tiny32 = vec![
        Complex32::new(1.0e-30, 0.0),
        Complex32::new(0.0, 2.0e-30),
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 1.0),
    ];
    <f32 as FaerEig>::normalize_eig_vectors(&mut tiny32, 2, 1).unwrap();
    let root_five = 5.0_f32.sqrt();
    assert!(
        (tiny32[0].re - 1.0 / root_five).abs() < 1.0e-6,
        "{:?}",
        tiny32[0]
    );
    assert!(
        (tiny32[1].im - 2.0 / root_five).abs() < 1.0e-6,
        "{:?}",
        tiny32[1]
    );

    // The same overflow is observable in the `f64` instantiation.
    let mut large64 = vec![
        Complex64::new(3.0e200, 0.0),
        Complex64::new(0.0, 4.0e200),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    <f64 as FaerEig>::normalize_eig_vectors(&mut large64, 2, 1).unwrap();
    assert!((large64[0].re - 0.6).abs() < 1.0e-12, "{:?}", large64[0]);
    assert!((large64[1].im - 0.8).abs() < 1.0e-12, "{:?}", large64[1]);

    // And the underflow side in f64: `1e-200` squared is exactly zero in the
    // operand dtype, so this column used to look all-zero and be rejected.
    let mut tiny64 = vec![
        Complex64::new(1.0e-200, 0.0),
        Complex64::new(0.0, 2.0e-200),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    <f64 as FaerEig>::normalize_eig_vectors(&mut tiny64, 2, 1).unwrap();
    let root_five64 = 5.0_f64.sqrt();
    assert!(
        (tiny64[0].re - 1.0 / root_five64).abs() < 1.0e-12,
        "{:?}",
        tiny64[0]
    );
    assert!(
        (tiny64[1].im - 2.0 / root_five64).abs() < 1.0e-12,
        "{:?}",
        tiny64[1]
    );
}
/// The compact state must own pooled storage, not two ordinary vectors.
///
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
    let mut backend = CpuBackend::new();
    let empty = backend.buffer_pool_stats().unwrap();
    assert_eq!(empty.capacity_bytes, 0, "a fresh backend retains nothing");

    let state = with_cpu_linalg(&mut backend, |session| a.householder_qr(session)).unwrap();
    let checked_out = backend.buffer_pool_stats().unwrap();
    drop(state);
    let returned = backend.buffer_pool_stats().unwrap();
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
    let state = with_cpu_linalg(&mut backend, |session| a.householder_qr(session)).unwrap();
    drop(state);
    let second = backend.buffer_pool_stats().unwrap();
    assert_eq!(
        (second.buffers, second.capacity_bytes),
        (returned.buffers, returned.capacity_bytes),
        "a warm pool must serve and recover the whole compact-QR payload"
    );
}
