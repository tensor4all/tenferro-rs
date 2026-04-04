//! Zero-copy strided GEMM via faer.
//!
//! The [`FaerGemm`] trait is implemented for `f32`, `f64`, `Complex32`,
//! and `Complex64`. It wraps `faer::linalg::matmul::matmul` with
//! arbitrary row/col strides, avoiding any temporary allocation.

use num_complex::{Complex32, Complex64};

/// Trait for types that support strided GEMM via faer (zero-copy, zero-allocation).
///
/// Computes `C = beta * C + alpha * A * B` using faer's matmul with arbitrary strides.
///
/// # Examples
///
/// ```ignore
/// use tenferro::gemm::faer_gemm::FaerGemm;
///
/// // Perform a small 2x2 GEMM with column-major strides.
/// let a = [1.0f64, 2.0, 3.0, 4.0]; // 2x2 col-major
/// let b = [5.0, 6.0, 7.0, 8.0];
/// let mut c = [0.0f64; 4];
/// unsafe {
///     f64::strided_gemm(
///         1.0, a.as_ptr(), 2, 2, 1, 2,
///         b.as_ptr(), 2, 1, 2,
///         0.0, c.as_mut_ptr(), 1, 2,
///     );
/// }
/// ```
pub(crate) trait FaerGemm: Sized {
    /// Perform `C = beta * C + alpha * A * B` with arbitrary strides.
    ///
    /// # Safety
    ///
    /// All pointers must be valid for the given dimensions and strides.
    /// The caller must ensure no aliasing between A, B and C memory regions.
    #[allow(clippy::too_many_arguments)]
    unsafe fn strided_gemm(
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    );
}

macro_rules! impl_faer_gemm {
    ($ty:ty) => {
        impl FaerGemm for $ty {
            unsafe fn strided_gemm(
                alpha: $ty,
                a_ptr: *const $ty,
                m: usize,
                k: usize,
                a_rs: isize,
                a_cs: isize,
                b_ptr: *const $ty,
                n: usize,
                b_rs: isize,
                b_cs: isize,
                beta: $ty,
                c_ptr: *mut $ty,
                c_rs: isize,
                c_cs: isize,
            ) {
                use faer::{Accum, MatMut, MatRef, Par};
                let a_mat = MatRef::<$ty>::from_raw_parts(a_ptr, m, k, a_rs, a_cs);
                let b_mat = MatRef::<$ty>::from_raw_parts(b_ptr, k, n, b_rs, b_cs);
                let zero = <$ty as num_traits::Zero>::zero();
                let one = <$ty as num_traits::One>::one();
                let accum = if beta == zero {
                    Accum::Replace
                } else {
                    if beta != one {
                        // Scale C by beta before accumulating
                        let mut col_off = 0isize;
                        for _ in 0..n {
                            let mut off = col_off;
                            for _ in 0..m {
                                *c_ptr.offset(off) *= beta;
                                off += c_rs;
                            }
                            col_off += c_cs;
                        }
                    }
                    Accum::Add
                };
                let mut c_mat = MatMut::<$ty>::from_raw_parts_mut(c_ptr, m, n, c_rs, c_cs);
                faer::linalg::matmul::matmul(
                    &mut c_mat,
                    accum,
                    &a_mat,
                    &b_mat,
                    alpha,
                    Par::rayon(0),
                );
            }
        }
    };
}

impl_faer_gemm!(f64);
impl_faer_gemm!(f32);
impl_faer_gemm!(Complex64);
impl_faer_gemm!(Complex32);
