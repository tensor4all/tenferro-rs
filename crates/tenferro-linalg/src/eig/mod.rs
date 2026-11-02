//! The eigenvector gauge `eig` applies to its output, usable without an engine.
//!
//! These functions are **not** a general-purpose vector normalization. They are
//! [`eig`](crate::TensorLinalgExt::eig)'s output convention and nothing else: the unit 2-norm
//! gauge `eig` puts its eigenvector columns in, with the same arithmetic (the norm is accumulated
//! in `f64` *after* scaling by the column's largest component) and the same typed error. A caller
//! that runs the provider's eigen-solve itself - `tlinalg::eig::eig` on an execution lane the
//! engine already entered, for example - applies these to reach exactly what tenferro's `eig`
//! returns on that faer CPU route, including its errors, instead of copying the routine (#2148,
//! #2387). Another provider keeps its own gauge: LAPACK's `eig` normalizes its own way and never
//! reaches this step.
//!
//! Both `Complex32` and `Complex64` blocks are covered, and a column that cannot be normalized is
//! a typed [`Error::NonFinite`](crate::Error::NonFinite) error rather than a panic. The
//! normalization itself allocates nothing, and only the error path does.
//!
//! # Examples
//!
//! Normalizing a raw provider result reproduces `eig`:
//!
//! ```
//! # #[cfg(feature = "native")] {
//! use num_complex::Complex64;
//! use strided_view::RawStridedRef;
//! use tcontract_exec::Exec;
//! use tenferro_cpu::CpuEngine;
//! use tenferro_linalg::TensorLinalgExt;
//! use tenferro_linalg::eig::normalize_eig_vectors_c64;
//! use tenferro_tensor::Tensor;
//! use tlinalg::{eig::eig, Op};
//!
//! // Column-major [[1, 1], [0, 2]]: eigenvalues 1 and 2, with the eigenvectors
//! // [1, 0] and [1, 1] / sqrt(2).
//! let data = [1.0_f64, 0.0, 1.0, 2.0];
//! let (mut values, mut vectors) = (Vec::<Complex64>::new(), Vec::<Complex64>::new());
//! eig(
//!     Op::Eig,
//!     RawStridedRef::new(&data, &[2, 2], &[1, 2], 0)?,
//!     &mut values,
//!     &mut vectors,
//!     &mut Exec::serial(),
//! )?;
//! normalize_eig_vectors_c64(&mut vectors, 2)?;
//!
//! // The same matrix through tenferro's `eig` comes back in this gauge.
//! let a = Tensor::from_vec_col_major(vec![2, 2], data.to_vec())?;
//! let mut host = CpuEngine::serial();
//! let (_values, from_eig) = a.eig(&mut host)?;
//! for (normalized, eigen) in vectors.iter().zip(from_eig.as_slice::<Complex64>()?) {
//!     assert!((normalized - eigen).norm() < 1.0e-15);
//! }
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use num_complex::{Complex32, Complex64};

/// Put `Complex64` eigenvector columns into the unit 2-norm gauge `eig` returns.
///
/// `vectors` holds one column-major `[n, n, batch...]` eigenvector tensor, exactly what
/// `tlinalg::eig::eig` writes: `n * n` consecutive elements per matrix, and each `n`-element run
/// of those a column. Every column is divided by its 2-norm, which is accumulated in `f64` after
/// scaling by the column's largest component - the arithmetic
/// [`eig`](crate::TensorLinalgExt::eig) applies, so an `eig` output is already in this gauge.
/// `n == 0` has no columns and normalizes nothing.
///
/// This is `eig`'s gauge, not a general normalization: use it to make a provider result match
/// tenferro's `eig`, and for nothing else (see [the module documentation](self)).
///
/// A column with zero norm (every entry zero) and a column whose norm or operand-dtype scale is
/// non-finite are reported as [`Error::NonFinite`](crate::Error::NonFinite) with `op: "eig"` and
/// `role: "eigenvector column"`, wrapped as the linalg extension source exactly as `eig` wraps it.
///
/// # Errors
///
/// `Error::Validation` with `op: "eig"` and argument `"vectors"` when `n * n` overflows `usize`
/// or `vectors.len()` is not a whole number of `n * n` blocks. `Error::Extension` carrying the
/// linalg [`Error::NonFinite`](crate::Error::NonFinite) source when a column cannot be normalized.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_linalg::eig::normalize_eig_vectors_c64;
///
/// // One column-major 2x2 block: column 0 is [3, 4i], column 1 is [1, 2i].
/// let mut columns = vec![
///     Complex64::new(3.0, 0.0),
///     Complex64::new(0.0, 4.0),
///     Complex64::new(1.0, 0.0),
///     Complex64::new(0.0, 2.0),
/// ];
/// normalize_eig_vectors_c64(&mut columns, 2)?;
/// assert!((columns[0].re - 0.6).abs() < 1.0e-12);
/// assert!((columns[1].im - 0.8).abs() < 1.0e-12);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn normalize_eig_vectors_c64(
    vectors: &mut [Complex64],
    n: usize,
) -> tenferro_tensor::Result<()> {
    normalize_columns(
        vectors,
        n,
        |value| (value.re, value.im),
        |norm| norm,
        |value, scale| *value /= scale,
    )
}

/// Put `Complex32` eigenvector columns into the unit 2-norm gauge `eig` returns.
///
/// The [`Complex64`](normalize_eig_vectors_c64) counterpart for `Complex32` blocks: the same
/// column-major `[n, n, batch...]` layout, the same `f64` accumulation for the norm, and the
/// division in the operand dtype (`f32` here), which is what
/// [`eig`](crate::TensorLinalgExt::eig) does for a `Complex32` output. It is `eig`'s gauge, not a
/// general normalization (see [the module documentation](self)).
///
/// # Errors
///
/// `Error::Validation` with `op: "eig"` and argument `"vectors"` when `n * n` overflows `usize`
/// or `vectors.len()` is not a whole number of `n * n` blocks. `Error::Extension` carrying the
/// linalg [`Error::NonFinite`](crate::Error::NonFinite) source when a column cannot be normalized.
///
/// # Examples
///
/// ```
/// use num_complex::Complex32;
/// use tenferro_linalg::eig::normalize_eig_vectors_c32;
///
/// // The same column-major 2x2 block in `Complex32`.
/// let mut columns = vec![
///     Complex32::new(3.0, 0.0),
///     Complex32::new(0.0, 4.0),
///     Complex32::new(1.0, 0.0),
///     Complex32::new(0.0, 2.0),
/// ];
/// normalize_eig_vectors_c32(&mut columns, 2)?;
/// assert!((columns[0].re - 0.6).abs() < 1.0e-6);
/// assert!((columns[1].im - 0.8).abs() < 1.0e-6);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn normalize_eig_vectors_c32(
    vectors: &mut [Complex32],
    n: usize,
) -> tenferro_tensor::Result<()> {
    normalize_columns(
        vectors,
        n,
        |value| (f64::from(value.re), f64::from(value.im)),
        |norm| f64::from(norm as f32),
        |value, scale| *value /= scale as f32,
    )
}

/// The one column loop both dtypes use.
///
/// `parts` reads a value's real and imaginary parts in `f64`, `real` round-trips the `f64` norm
/// through the operand's real dtype, and `divide` divides a value by the resulting scale in the
/// operand dtype. Only those two steps differ per dtype.
fn normalize_columns<T>(
    vectors: &mut [T],
    n: usize,
    parts: impl Fn(&T) -> (f64, f64),
    real: impl Fn(f64) -> f64,
    divide: impl Fn(&mut T, f64),
) -> tenferro_tensor::Result<()> {
    if n == 0 {
        return Ok(());
    }
    let block_len = n
        .checked_mul(n)
        .ok_or_else(|| tenferro_tensor::Error::invalid_argument("eig", "n", "n * n overflows"))?;
    if !vectors.len().is_multiple_of(block_len) {
        return Err(tenferro_tensor::Error::invalid_argument(
            "eig",
            "vectors",
            format!(
                "expected a whole number of {block_len}-element (n * n) eigenvector blocks, got {}",
                vectors.len()
            ),
        ));
    }
    // INVARIANT: `n >= 1` here, so each chunk below is a whole column of the column-major
    // [n, n] block and each block is one matrix of the batch.
    for block in vectors.chunks_exact_mut(block_len) {
        for column in block.chunks_exact_mut(n) {
            // Scale before summing squares. `norm_sqr` is evaluated in the operand dtype (`f32`
            // for the `Complex32` instantiation), so a finite column of large magnitude squares
            // to `inf` and one of tiny magnitude squares to `0`, and both were rejected as
            // `NonFinite` even though the column is normalizable. Scaling by the largest
            // component first keeps every squared term in [0, 1], and the scaling itself is done
            // in `f64` so it cannot overflow.
            let magnitude = column
                .iter()
                .map(|value| {
                    let (re, im) = parts(value);
                    re.abs().max(im.abs())
                })
                .fold(0.0_f64, f64::max);
            let norm = magnitude
                * column
                    .iter()
                    .map(|value| {
                        let (re, im) = parts(value);
                        let re = re / magnitude;
                        let im = im / magnitude;
                        re * re + im * im
                    })
                    .sum::<f64>()
                    .sqrt();
            let scale = real(norm);
            if !norm.is_finite() || !scale.is_finite() || scale == 0.0 {
                return Err(crate::error::into_tensor_error(
                    "eig",
                    crate::Error::NonFinite {
                        op: "eig",
                        role: "eigenvector column",
                    },
                ));
            }
            for value in column.iter_mut() {
                divide(value, scale);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
