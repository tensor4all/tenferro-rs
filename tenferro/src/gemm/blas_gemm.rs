//! Contiguous column-major GEMM via CBLAS.
//!
//! [`BlasGemm`] is implemented for `f32`, `f64`, `Complex32`, `Complex64`.
//! Unlike faer, CBLAS requires contiguous column-major data, so callers
//! must pack strided data before calling these routines.

use num_complex::{Complex32, Complex64};

/// Trait for types supporting contiguous column-major GEMM via CBLAS.
///
/// Computes `C = beta * C + alpha * A * B` where A, B, C are
/// contiguous column-major slices.
///
/// # Examples
///
/// ```ignore
/// use tenferro::gemm::blas_gemm::BlasGemm;
///
/// let a = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2 col-major
/// let b = vec![5.0, 6.0, 7.0, 8.0];
/// let mut c = vec![0.0f64; 4];
/// f64::contiguous_gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
/// ```
pub(crate) trait BlasGemm: Sized {
    /// Perform `C = beta * C + alpha * A * B` on contiguous col-major data.
    ///
    /// # Panics
    ///
    /// Panics if dimensions exceed `i32::MAX`.
    #[allow(clippy::too_many_arguments)]
    fn contiguous_gemm(
        alpha: Self,
        a: &[Self],
        b: &[Self],
        beta: Self,
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    );
}

impl BlasGemm for f64 {
    fn contiguous_gemm(
        alpha: Self,
        a: &[Self],
        b: &[Self],
        beta: Self,
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let m_i32 = i32::try_from(m).expect("m too large for i32");
        let n_i32 = i32::try_from(n).expect("n too large for i32");
        let k_i32 = i32::try_from(k).expect("k too large for i32");
        unsafe {
            cblas_sys::cblas_dgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m_i32,
                n_i32,
                k_i32,
                alpha,
                a.as_ptr(),
                m_i32,
                b.as_ptr(),
                k_i32,
                beta,
                c.as_mut_ptr(),
                m_i32,
            );
        }
    }
}

impl BlasGemm for f32 {
    fn contiguous_gemm(
        alpha: Self,
        a: &[Self],
        b: &[Self],
        beta: Self,
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let m_i32 = i32::try_from(m).expect("m too large for i32");
        let n_i32 = i32::try_from(n).expect("n too large for i32");
        let k_i32 = i32::try_from(k).expect("k too large for i32");
        unsafe {
            cblas_sys::cblas_sgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m_i32,
                n_i32,
                k_i32,
                alpha,
                a.as_ptr(),
                m_i32,
                b.as_ptr(),
                k_i32,
                beta,
                c.as_mut_ptr(),
                m_i32,
            );
        }
    }
}

impl BlasGemm for Complex64 {
    fn contiguous_gemm(
        alpha: Self,
        a: &[Self],
        b: &[Self],
        beta: Self,
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let m_i32 = i32::try_from(m).expect("m too large for i32");
        let n_i32 = i32::try_from(n).expect("n too large for i32");
        let k_i32 = i32::try_from(k).expect("k too large for i32");
        let alpha_ri = [alpha.re, alpha.im];
        let beta_ri = [beta.re, beta.im];
        unsafe {
            cblas_sys::cblas_zgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m_i32,
                n_i32,
                k_i32,
                &alpha_ri,
                a.as_ptr() as *const _,
                m_i32,
                b.as_ptr() as *const _,
                k_i32,
                &beta_ri,
                c.as_mut_ptr() as *mut _,
                m_i32,
            );
        }
    }
}

impl BlasGemm for Complex32 {
    fn contiguous_gemm(
        alpha: Self,
        a: &[Self],
        b: &[Self],
        beta: Self,
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let m_i32 = i32::try_from(m).expect("m too large for i32");
        let n_i32 = i32::try_from(n).expect("n too large for i32");
        let k_i32 = i32::try_from(k).expect("k too large for i32");
        let alpha_ri = [alpha.re, alpha.im];
        let beta_ri = [beta.re, beta.im];
        unsafe {
            cblas_sys::cblas_cgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m_i32,
                n_i32,
                k_i32,
                &alpha_ri,
                a.as_ptr() as *const _,
                m_i32,
                b.as_ptr() as *const _,
                k_i32,
                &beta_ri,
                c.as_mut_ptr() as *mut _,
                m_i32,
            );
        }
    }
}
