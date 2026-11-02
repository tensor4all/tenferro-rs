//! The public gauge of `eig` eigenvector columns.
//!
//! `eig` promises unit 2-norm columns, which the providers do not all deliver, so the
//! backend seam normalizes them. A caller that drives `tlinalg` eig itself - a kernel that
//! batches square blocks on a lane token, say - has to produce the same values and the same
//! typed failure as `eig`; publishing the routine is what makes that possible without a copy
//! of the arithmetic (#2387).

use num_complex::Complex;
use num_traits::cast::AsPrimitive;
use num_traits::{Float, NumAssign};

use crate::error::into_tensor_error;
use crate::Error;

/// Normalize the eigenvector columns of a batch of `n x n` column-major blocks in place.
///
/// `vectors` holds `n * n` complex elements per batch item, blocks back to back, and only
/// the first `batch_count` blocks are read and written. Each block is normalized column by
/// column to unit 2-norm, the gauge `eig` documents.
///
/// The arithmetic is the one `eig` uses: each column is scaled by its largest component
/// before the squares are summed, so a finite column of large magnitude does not square to
/// infinity and one of tiny magnitude does not square to zero. A column that cannot be
/// normalized - every component zero, or a non-finite sum - is reported as
/// `Error::NonFinite` with `op: "eig"` and `role: "eigenvector column"`, the failure `eig`
/// raises.
///
/// The whole slice is validated before anything is written, so an `Err` leaves `vectors`
/// untouched.
///
/// # Errors
///
/// A [`tenferro_tensor::Error`] with a validation source when `n` is zero, when
/// `n * n * batch_count` overflows `usize`, or when `vectors` is shorter than that product;
/// and a numerical failure carrying [`Error::NonFinite`] for a column with no unit gauge.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_linalg::normalize_eig_vectors;
///
/// // One 2x2 column-major block: columns (2, 0) and (0, 3i).
/// let mut vectors = [
///     Complex64::new(2.0, 0.0),
///     Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 0.0),
///     Complex64::new(0.0, 3.0),
/// ];
/// normalize_eig_vectors(&mut vectors, 2, 1)?;
/// assert_eq!(vectors[0], Complex64::new(1.0, 0.0));
/// assert_eq!(vectors[3], Complex64::new(0.0, 1.0));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn normalize_eig_vectors<R>(
    vectors: &mut [Complex<R>],
    n: usize,
    batch_count: usize,
) -> tenferro_tensor::Result<()>
where
    R: Float + NumAssign + AsPrimitive<f64> + 'static,
    f64: AsPrimitive<R>,
{
    let block = n.checked_mul(n).filter(|_| n > 0).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "eig",
            "vectors",
            "n must be non-zero to name a column",
        )
    })?;
    let needed = block.checked_mul(batch_count).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "eig",
            "vectors",
            "the eigenvector element count overflows usize",
        )
    })?;
    if vectors.len() < needed {
        return Err(tenferro_tensor::Error::invalid_argument(
            "eig",
            "vectors",
            format!(
                "{} elements hold {batch_count} blocks of {n}x{n}, {needed} needed",
                vectors.len()
            ),
        ));
    }

    for block in vectors[..needed].chunks_exact_mut(block) {
        for column in block.chunks_exact_mut(n) {
            // Scale before summing squares. The operand dtype squares its own components,
            // so a finite column of large magnitude squares to `inf` and one of tiny
            // magnitude squares to `0`, and both would be rejected as `NonFinite` even
            // though the column is normalizable. Scaling by the largest component first
            // keeps every squared term in [0, 1], and the scaling itself is done in f64 so
            // it cannot overflow.
            let magnitude = column
                .iter()
                .map(|value| {
                    let largest: f64 = value.re.abs().max(value.im.abs()).as_();
                    largest
                })
                .fold(0.0_f64, f64::max);
            if magnitude == 0.0 {
                return Err(non_finite_column());
            }
            let norm = magnitude
                * column
                    .iter()
                    .map(|value| {
                        let re: f64 = value.re.as_() / magnitude;
                        let im: f64 = value.im.as_() / magnitude;
                        re * re + im * im
                    })
                    .sum::<f64>()
                    .sqrt();
            let scale: R = norm.as_();
            if !norm.is_finite() || !scale.is_finite() || scale == R::zero() {
                return Err(non_finite_column());
            }
            for value in column.iter_mut() {
                *value /= scale;
            }
        }
    }
    Ok(())
}

/// The failure `eig` raises for an eigenvector column with no unit gauge.
fn non_finite_column() -> tenferro_tensor::Error {
    into_tensor_error(
        "eig",
        Error::NonFinite {
            op: "eig",
            role: "eigenvector column",
        },
    )
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};

    use super::normalize_eig_vectors;

    fn norm64(column: &[Complex64]) -> f64 {
        column.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
    }

    #[test]
    fn every_column_of_every_block_is_unit() {
        let mut vectors = [
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, -6.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(1.0, 1.0),
        ];
        normalize_eig_vectors(&mut vectors, 2, 2).unwrap();
        for block in [&vectors[..4], &vectors[4..]] {
            for column in [&block[..2], &block[2..]] {
                assert!((norm64(column) - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_huge_f32_column_does_not_square_to_infinity() {
        // 1e30 squared is inf in f32; the scaled sum keeps both terms in [0, 1].
        let mut vectors = [
            Complex32::new(1e30, 0.0),
            Complex32::new(1e30, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, -1e30),
        ];
        normalize_eig_vectors(&mut vectors, 2, 1).unwrap();
        let scale = 1.0 / 2.0_f32.sqrt();
        assert!((vectors[0].re - scale).abs() < 1e-6);
        assert!((vectors[1].re - scale).abs() < 1e-6);
        assert_eq!(vectors[3], Complex32::new(0.0, -1.0));
    }

    #[test]
    fn a_tiny_f64_column_does_not_square_to_zero() {
        let mut vectors = [
            Complex64::new(1e-200, 0.0),
            Complex64::new(0.0, 1e-200),
            Complex64::new(0.0, 0.0),
            Complex64::new(1e-200, 0.0),
        ];
        normalize_eig_vectors(&mut vectors, 2, 1).unwrap();
        let scale = 1.0 / 2.0_f64.sqrt();
        assert!((vectors[0].re - scale).abs() < 1e-12);
        assert!((vectors[1].im - scale).abs() < 1e-12);
        assert_eq!(vectors[3], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn a_zero_column_is_the_typed_non_finite_failure() {
        let mut vectors = [Complex64::new(0.0, 0.0); 4];
        let error = normalize_eig_vectors(&mut vectors, 2, 1).unwrap_err();
        assert!(format!("{error:?}").contains("NumericalFailure"));
        assert!(format!("{error:?}").contains("eigenvector column"));
    }

    #[test]
    fn a_short_slice_is_rejected_before_anything_is_written() {
        let mut vectors = [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)];
        let error = normalize_eig_vectors(&mut vectors, 2, 2).unwrap_err();
        assert!(format!("{error:?}").contains("8 needed"));
        assert_eq!(vectors[0], Complex64::new(2.0, 0.0));
    }

    #[test]
    fn a_zero_column_count_is_rejected() {
        let mut vectors = [Complex64::new(1.0, 0.0)];
        assert!(normalize_eig_vectors(&mut vectors, 0, 1).is_err());
    }
}
