use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
#[cfg(feature = "gemm-blas")]
use tenferro_device::{Error, Result};

/// Trait for types that support strided GEMM via faer (zero-copy, zero-allocation).
///
/// Computes `C = beta * C + alpha * A * B` using faer's `matmul` with arbitrary strides.
#[cfg(feature = "gemm-faer")]
pub(super) trait FaerGemm: Scalar {
    /// # Safety
    /// All pointers must be valid for the given dimensions and strides.
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

#[cfg(feature = "gemm-faer")]
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

#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(f64);
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(f32);
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(Complex64);
#[cfg(feature = "gemm-faer")]
impl_faer_gemm!(Complex32);

#[cfg(feature = "gemm-blas")]
pub(super) trait BlasGemm: Scalar {
    fn contiguous_gemm(
        alpha: Self,
        a: &[Self],
        b: &[Self],
        beta: Self,
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;
}

/// Compute batch offset for one operand: fused path uses flat index × step,
/// strided path uses multi-index dot product with strides.
#[inline]
pub(super) fn batch_offset(
    flat: usize,
    idx: &[usize],
    fused: Option<(usize, isize)>,
    batch_strides: &[isize],
) -> isize {
    if let Some((_, step)) = fused {
        flat as isize * step
    } else {
        idx.iter()
            .zip(batch_strides)
            .map(|(&i, &s)| i as isize * s)
            .sum()
    }
}

#[cfg(feature = "gemm-blas")]
pub(super) fn gemm_f64(
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
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
    Ok(())
}

#[cfg(feature = "gemm-blas")]
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
    ) -> Result<()> {
        gemm_f64(alpha, a, b, beta, c, m, n, k)
    }
}

#[cfg(feature = "gemm-blas")]
pub(super) fn gemm_f32(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
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
    Ok(())
}

#[cfg(feature = "gemm-blas")]
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
    ) -> Result<()> {
        gemm_f32(alpha, a, b, beta, c, m, n, k)
    }
}

#[cfg(feature = "gemm-blas")]
pub(super) fn gemm_c64(
    alpha: Complex64,
    a: &[Complex64],
    b: &[Complex64],
    beta: Complex64,
    c: &mut [Complex64],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
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
    Ok(())
}

#[cfg(feature = "gemm-blas")]
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
    ) -> Result<()> {
        gemm_c64(alpha, a, b, beta, c, m, n, k)
    }
}

#[cfg(feature = "gemm-blas")]
pub(super) fn gemm_c32(
    alpha: Complex32,
    a: &[Complex32],
    b: &[Complex32],
    beta: Complex32,
    c: &mut [Complex32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let m_i32 = i32::try_from(m).map_err(|_| Error::InvalidArgument("m too large".into()))?;
    let n_i32 = i32::try_from(n).map_err(|_| Error::InvalidArgument("n too large".into()))?;
    let k_i32 = i32::try_from(k).map_err(|_| Error::InvalidArgument("k too large".into()))?;
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
    Ok(())
}

#[cfg(feature = "gemm-blas")]
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
    ) -> Result<()> {
        gemm_c32(alpha, a, b, beta, c, m, n, k)
    }
}
