use num_complex::Complex64;

/// Check every entry of the Hermitian Gram matrix of column-major columns.
pub(super) fn assert_isometric(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    tol: f64,
    label: &str,
) {
    assert_eq!(matrix.len(), rows.checked_mul(cols).unwrap());
    for left in 0..cols {
        let left_column = &matrix[left * rows..(left + 1) * rows];
        // INVARIANT: A^H A is Hermitian. Checking each diagonal and unordered
        // column pair checks both ordered entries at the same norm tolerance;
        // this does not sample columns or omit any independent Gram entry.
        for right in left..cols {
            let right_column = &matrix[right * rows..(right + 1) * rows];
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for (left_value, right_value) in left_column.iter().zip(right_column) {
                real += left_value.re * right_value.re + left_value.im * right_value.im;
                imaginary += left_value.re * right_value.im - left_value.im * right_value.re;
            }
            let inner = Complex64::new(real, imaginary);
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!(
                (inner - Complex64::new(expected, 0.0)).norm() < tol,
                "{label}: column pair ({left}, {right}) inner product {inner} is not {expected}"
            );
        }
    }
}

fn reference_accepts(matrix: &[Complex64], rows: usize, cols: usize, tol: f64) -> bool {
    (0..cols).all(|left| {
        (0..cols).all(|right| {
            let inner: Complex64 = (0..rows)
                .map(|row| matrix[row + left * rows].conj() * matrix[row + right * rows])
                .sum();
            let expected = if left == right { 1.0 } else { 0.0 };
            (inner - Complex64::new(expected, 0.0)).norm() < tol
        })
    })
}

fn candidate_accepts(matrix: &[Complex64], rows: usize, cols: usize, tol: f64) -> bool {
    std::panic::catch_unwind(|| assert_isometric(matrix, rows, cols, tol, "oracle parity")).is_ok()
}

#[test]
fn complex_gram_matches_all_ordered_reference_entries() {
    for rows in [0_usize, 1, 2, 3, 8, 16] {
        for cols in 0..=rows {
            let scale = (rows as f64).sqrt();
            let matrix: Vec<_> = (0..cols)
                .flat_map(|column| {
                    (0..rows).map(move |row| {
                        Complex64::from_polar(
                            1.0 / scale,
                            std::f64::consts::TAU * (row * column) as f64 / rows as f64,
                        )
                    })
                })
                .collect();
            assert!(reference_accepts(&matrix, rows, cols, 1.0e-12));
            assert!(candidate_accepts(&matrix, rows, cols, 1.0e-12));
            // Every stored entry participates; include imaginary perturbations
            // so a real-only dot product cannot pass this parity check.
            for index in 0..matrix.len() {
                for delta in [Complex64::new(0.25, 0.0), Complex64::new(0.0, 0.25)] {
                    let mut perturbed = matrix.clone();
                    perturbed[index] += delta;
                    let expected = reference_accepts(&perturbed, rows, cols, 1.0e-12);
                    assert!(!expected);
                    assert_eq!(candidate_accepts(&perturbed, rows, cols, 1.0e-12), expected);
                }
            }
        }
    }
}

#[test]
fn gram_rejects_nonfinite_and_correlated_columns() {
    let identity = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in 0..identity.len() {
            let mut matrix = identity;
            matrix[index].im = nonfinite;
            assert!(!reference_accepts(&matrix, 2, 2, 1.0e-12));
            assert!(!candidate_accepts(&matrix, 2, 2, 1.0e-12));
        }
    }
    let correlated = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
    ];
    assert!(!candidate_accepts(&correlated, 2, 2, 1.0e-12));
}
