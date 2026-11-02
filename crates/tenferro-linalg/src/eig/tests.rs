use super::{normalize_eig_vectors_c32, normalize_eig_vectors_c64};
use num_complex::{Complex32, Complex64};

/// Assert the normalizer reported a non-normalizable column through the typed
/// linalg `NonFinite` source, not a generic failure.
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
fn normalize_eig_vectors_puts_c64_columns_in_the_unit_gauge() {
    // Column-major 2x2 block: column 0 is [3, 4i], column 1 is [1, 2i].
    let mut columns = vec![
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, 4.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 2.0),
    ];
    normalize_eig_vectors_c64(&mut columns, 2).unwrap();
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
fn normalize_eig_vectors_puts_c32_columns_in_the_unit_gauge() {
    let mut columns = vec![
        Complex32::new(3.0, 0.0),
        Complex32::new(0.0, 4.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 2.0),
    ];
    normalize_eig_vectors_c32(&mut columns, 2).unwrap();
    assert!((columns[0].re - 0.6).abs() < 1.0e-6, "{:?}", columns[0]);
    assert!((columns[1].im - 0.8).abs() < 1.0e-6, "{:?}", columns[1]);
}

#[test]
fn normalize_eig_vectors_rejects_non_normalizable_columns_typed() {
    // All-zero column: there is no direction to normalize, so the normalizer
    // must refuse rather than divide by zero.
    let mut zero = vec![Complex64::new(0.0, 0.0); 4];
    assert_typed_non_finite(normalize_eig_vectors_c64(&mut zero, 2).unwrap_err());

    let mut zero32 = vec![Complex32::new(0.0, 0.0); 4];
    assert_typed_non_finite(normalize_eig_vectors_c32(&mut zero32, 2).unwrap_err());

    // Non-finite column: a provider that overflowed inside its
    // back-substitution must surface the typed NonFinite source, not NaN.
    let mut non_finite = vec![
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    assert_typed_non_finite(normalize_eig_vectors_c64(&mut non_finite, 2).unwrap_err());

    // A NaN component is not normalizable either.
    let mut nan = vec![
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    assert_typed_non_finite(normalize_eig_vectors_c64(&mut nan, 2).unwrap_err());

    // A column whose `f64` norm is finite but overflows the operand's real dtype: [3e38, 3e38]
    // has norm 4.2e38 > `f32::MAX`, so the `f32` scale is `inf`.
    let mut overflow32 = vec![
        Complex32::new(3.0e38, 0.0),
        Complex32::new(3.0e38, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(1.0, 0.0),
    ];
    assert_typed_non_finite(normalize_eig_vectors_c32(&mut overflow32, 2).unwrap_err());
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
    normalize_eig_vectors_c32(&mut large32, 2).unwrap();
    assert!((large32[0].re - 0.6).abs() < 1.0e-6, "{:?}", large32[0]);
    assert!((large32[1].im - 0.8).abs() < 1.0e-6, "{:?}", large32[1]);

    // A tiny column used to underflow the operand-dtype square sum to zero.
    let mut tiny32 = vec![
        Complex32::new(1.0e-30, 0.0),
        Complex32::new(0.0, 2.0e-30),
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 1.0),
    ];
    normalize_eig_vectors_c32(&mut tiny32, 2).unwrap();
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
    normalize_eig_vectors_c64(&mut large64, 2).unwrap();
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
    normalize_eig_vectors_c64(&mut tiny64, 2).unwrap();
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

#[test]
fn normalize_eig_vectors_covers_every_block_of_a_batch() {
    // Two column-major 2x2 blocks, each with its own column magnitudes: the
    // second block must be normalized with its own norms, not the first's.
    let mut columns = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, -10.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, -0.5),
    ];
    normalize_eig_vectors_c64(&mut columns, 2).unwrap();
    assert_eq!(columns[0], Complex64::new(1.0, 0.0));
    assert_eq!(columns[3], Complex64::new(1.0, 0.0));
    assert_eq!(columns[4], Complex64::new(0.0, -1.0));
    assert_eq!(columns[7], Complex64::new(0.0, -1.0));

    // A single 3x3 block: three columns of different phases and magnitudes.
    let mut cubic = vec![
        Complex64::new(0.0, 2.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(0.0, 0.0),
    ];
    normalize_eig_vectors_c64(&mut cubic, 3).unwrap();
    assert_eq!(cubic[0], Complex64::new(0.0, 1.0));
    let third = 3.0_f64.sqrt();
    for value in &cubic[3..6] {
        assert!((value.re - 1.0 / third).abs() < 1.0e-12, "{value:?}");
    }
    assert_eq!(cubic[7], Complex64::new(0.6, 0.8));
}

#[test]
fn normalize_eig_vectors_rejects_a_partial_block() {
    let mut columns = vec![Complex64::new(1.0, 0.0); 5];
    let error = normalize_eig_vectors_c64(&mut columns, 2)
        .expect_err("a length that is not a whole number of n * n blocks must fail");
    assert!(
        matches!(error, tenferro_tensor::Error::Validation { op: "eig", .. }),
        "expected a validation failure, got {error}"
    );
    assert_eq!(columns, vec![Complex64::new(1.0, 0.0); 5]);

    // A block length that cannot be represented has no columns to walk either.
    let error =
        normalize_eig_vectors_c64(&mut [], usize::MAX).expect_err("an overflowing n * n must fail");
    assert!(
        matches!(error, tenferro_tensor::Error::Validation { op: "eig", .. }),
        "expected a validation failure, got {error}"
    );
}

#[test]
fn normalize_eig_vectors_is_a_no_op_for_an_empty_block() {
    let mut empty: Vec<Complex64> = Vec::new();
    normalize_eig_vectors_c64(&mut empty, 0).unwrap();
    normalize_eig_vectors_c64(&mut empty, 2).unwrap();
}
