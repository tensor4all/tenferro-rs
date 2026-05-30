use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::{Complex32, Complex64};

use crate::Error;

pub(crate) trait BlasGemm: Sized {
    // Kept as a provider-local contiguous BLAS entry point for direct BLAS
    // validation even when the optimized path uses explicit strides.
    #[allow(dead_code)]
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
    ) -> crate::Result<()>;

    #[allow(clippy::too_many_arguments)]
    /// Run GEMM against raw strided matrix pointers.
    ///
    /// # Safety
    ///
    /// The caller must pass valid, non-null pointers to matrices whose logical
    /// `m x k`, `k x n`, and `m x n` elements are addressable through the given
    /// strides for the duration of the BLAS call. `c_ptr` must be uniquely
    /// writable for the output elements and must not alias input elements in a
    /// way forbidden by the linked BLAS implementation.
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
    ) -> crate::Result<()>;

    #[allow(clippy::too_many_arguments)]
    /// Run GEMM against raw strided matrix pointers with optional input conjugation.
    ///
    /// Returns `Ok(false)` when the requested conjugation cannot be represented
    /// by the selected BLAS transpose flags without materializing an input.
    ///
    /// # Safety
    ///
    /// The safety contract is the same as [`BlasGemm::strided_gemm`].
    unsafe fn strided_gemm_with_conj(
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        conj_a: bool,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        conj_b: bool,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    ) -> crate::Result<bool> {
        let _ = (conj_a, conj_b);
        unsafe {
            Self::strided_gemm(
                alpha, a_ptr, m, k, a_rs, a_cs, b_ptr, n, b_rs, b_cs, beta, c_ptr, c_rs, c_cs,
            )?;
        }
        Ok(true)
    }
}

fn dim_to_i32(name: &'static str, value: usize) -> crate::Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidConfig {
        op: "dot_general",
        message: format!("{name}={value} exceeds BLAS i32 range"),
    })
}

fn stride_to_i32(name: &'static str, value: isize) -> crate::Result<i32> {
    match i32::try_from(value) {
        Ok(value) if value > 0 => Ok(value),
        _ => Err(Error::InvalidConfig {
            op: "dot_general",
            message: format!("{name}={value} must be a positive BLAS stride"),
        }),
    }
}

fn infer_a_layout(
    m: usize,
    k: usize,
    a_rs: isize,
    a_cs: isize,
) -> crate::Result<(CBLAS_TRANSPOSE, i32)> {
    if a_rs == 1 {
        let lda = stride_to_i32("lda", a_cs)?;
        let min_lda = dim_to_i32("m", m)?;
        if lda < min_lda {
            return Err(Error::InvalidConfig {
                op: "dot_general",
                message: format!("lda={lda} must be >= max(1, m)={min_lda} for NoTrans A"),
            });
        }
        Ok((CBLAS_TRANSPOSE::CblasNoTrans, lda))
    } else if a_cs == 1 {
        let lda = stride_to_i32("lda", a_rs)?;
        let min_lda = dim_to_i32("k", k)?;
        if lda < min_lda {
            return Err(Error::InvalidConfig {
                op: "dot_general",
                message: format!("lda={lda} must be >= max(1, k)={min_lda} for Trans A"),
            });
        }
        Ok((CBLAS_TRANSPOSE::CblasTrans, lda))
    } else {
        Err(Error::InvalidConfig {
            op: "dot_general",
            message: "BLAS requires unit stride on one axis of A".into(),
        })
    }
}

fn infer_b_layout(
    k: usize,
    n: usize,
    b_rs: isize,
    b_cs: isize,
) -> crate::Result<(CBLAS_TRANSPOSE, i32)> {
    if b_rs == 1 {
        let ldb = stride_to_i32("ldb", b_cs)?;
        let min_ldb = dim_to_i32("k", k)?;
        if ldb < min_ldb {
            return Err(Error::InvalidConfig {
                op: "dot_general",
                message: format!("ldb={ldb} must be >= max(1, k)={min_ldb} for NoTrans B"),
            });
        }
        Ok((CBLAS_TRANSPOSE::CblasNoTrans, ldb))
    } else if b_cs == 1 {
        let ldb = stride_to_i32("ldb", b_rs)?;
        let min_ldb = dim_to_i32("n", n)?;
        if ldb < min_ldb {
            return Err(Error::InvalidConfig {
                op: "dot_general",
                message: format!("ldb={ldb} must be >= max(1, n)={min_ldb} for Trans B"),
            });
        }
        Ok((CBLAS_TRANSPOSE::CblasTrans, ldb))
    } else {
        Err(Error::InvalidConfig {
            op: "dot_general",
            message: "BLAS requires unit stride on one axis of B".into(),
        })
    }
}

fn infer_c_layout(m: usize, c_rs: isize, c_cs: isize) -> crate::Result<i32> {
    if c_rs != 1 {
        return Err(Error::InvalidConfig {
            op: "dot_general",
            message: format!("BLAS output requires unit row stride, got {c_rs}"),
        });
    }
    let ldc = stride_to_i32("ldc", c_cs)?;
    let min_ldc = dim_to_i32("m", m)?;
    if ldc < min_ldc {
        return Err(Error::InvalidConfig {
            op: "dot_general",
            message: format!("ldc={ldc} must be >= max(1, m)={min_ldc}"),
        });
    }
    Ok(ldc)
}

fn apply_conj_transpose(trans: CBLAS_TRANSPOSE, conj: bool) -> Option<CBLAS_TRANSPOSE> {
    if !conj {
        return Some(trans);
    }

    match trans {
        CBLAS_TRANSPOSE::CblasTrans | CBLAS_TRANSPOSE::CblasConjTrans => {
            Some(CBLAS_TRANSPOSE::CblasConjTrans)
        }
        CBLAS_TRANSPOSE::CblasNoTrans => None,
    }
}

macro_rules! impl_real_blas_gemm {
    ($ty:ty, $gemm:path) => {
        impl BlasGemm for $ty {
            fn contiguous_gemm(
                alpha: Self,
                a: &[Self],
                b: &[Self],
                beta: Self,
                c: &mut [Self],
                m: usize,
                n: usize,
                k: usize,
            ) -> crate::Result<()> {
                let m_i32 = dim_to_i32("m", m)?;
                let n_i32 = dim_to_i32("n", n)?;
                let k_i32 = dim_to_i32("k", k)?;
                // SAFETY: the slices provide valid contiguous column-major
                // storage for the BLAS read/write regions implied by m, n,
                // and k, and dimensions were checked to fit BLAS i32 args.
                unsafe {
                    $gemm(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
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
            ) -> crate::Result<()> {
                let m_i32 = dim_to_i32("m", m)?;
                let n_i32 = dim_to_i32("n", n)?;
                let k_i32 = dim_to_i32("k", k)?;
                let (trans_a, lda) = infer_a_layout(m, k, a_rs, a_cs)?;
                let (trans_b, ldb) = infer_b_layout(k, n, b_rs, b_cs)?;
                let ldc = infer_c_layout(m, c_rs, c_cs)?;

                // SAFETY: `strided_gemm`'s caller guarantees the raw pointers
                // are valid for the strided matrix regions. Layout inference
                // above checked the unit-stride axis and BLAS leading dims.
                $gemm(
                    CBLAS_LAYOUT::CblasColMajor,
                    trans_a,
                    trans_b,
                    m_i32,
                    n_i32,
                    k_i32,
                    alpha,
                    a_ptr,
                    lda,
                    b_ptr,
                    ldb,
                    beta,
                    c_ptr,
                    ldc,
                );
                Ok(())
            }
        }
    };
}

macro_rules! impl_complex_blas_gemm {
    ($ty:ty, $gemm:path) => {
        impl BlasGemm for $ty {
            fn contiguous_gemm(
                alpha: Self,
                a: &[Self],
                b: &[Self],
                beta: Self,
                c: &mut [Self],
                m: usize,
                n: usize,
                k: usize,
            ) -> crate::Result<()> {
                let m_i32 = dim_to_i32("m", m)?;
                let n_i32 = dim_to_i32("n", n)?;
                let k_i32 = dim_to_i32("k", k)?;
                let alpha_ri = [alpha.re, alpha.im];
                let beta_ri = [beta.re, beta.im];
                // SAFETY: the slices provide valid contiguous column-major
                // storage for the BLAS read/write regions implied by m, n,
                // and k, and dimensions were checked to fit BLAS i32 args.
                unsafe {
                    $gemm(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        m_i32,
                        n_i32,
                        k_i32,
                        alpha_ri.as_ptr() as *const _,
                        a.as_ptr() as *const _,
                        m_i32,
                        b.as_ptr() as *const _,
                        k_i32,
                        beta_ri.as_ptr() as *const _,
                        c.as_mut_ptr() as *mut _,
                        m_i32,
                    );
                }
                Ok(())
            }

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
            ) -> crate::Result<()> {
                let m_i32 = dim_to_i32("m", m)?;
                let n_i32 = dim_to_i32("n", n)?;
                let k_i32 = dim_to_i32("k", k)?;
                let (trans_a, lda) = infer_a_layout(m, k, a_rs, a_cs)?;
                let (trans_b, ldb) = infer_b_layout(k, n, b_rs, b_cs)?;
                let ldc = infer_c_layout(m, c_rs, c_cs)?;
                let alpha_ri = [alpha.re, alpha.im];
                let beta_ri = [beta.re, beta.im];

                // SAFETY: `strided_gemm`'s caller guarantees the raw pointers
                // are valid for the strided matrix regions. Layout inference
                // above checked the unit-stride axis and BLAS leading dims.
                $gemm(
                    CBLAS_LAYOUT::CblasColMajor,
                    trans_a,
                    trans_b,
                    m_i32,
                    n_i32,
                    k_i32,
                    alpha_ri.as_ptr() as *const _,
                    a_ptr as *const _,
                    lda,
                    b_ptr as *const _,
                    ldb,
                    beta_ri.as_ptr() as *const _,
                    c_ptr as *mut _,
                    ldc,
                );
                Ok(())
            }

            unsafe fn strided_gemm_with_conj(
                alpha: Self,
                a_ptr: *const Self,
                m: usize,
                k: usize,
                a_rs: isize,
                a_cs: isize,
                conj_a: bool,
                b_ptr: *const Self,
                n: usize,
                b_rs: isize,
                b_cs: isize,
                conj_b: bool,
                beta: Self,
                c_ptr: *mut Self,
                c_rs: isize,
                c_cs: isize,
            ) -> crate::Result<bool> {
                let m_i32 = dim_to_i32("m", m)?;
                let n_i32 = dim_to_i32("n", n)?;
                let k_i32 = dim_to_i32("k", k)?;
                let (trans_a, lda) = infer_a_layout(m, k, a_rs, a_cs)?;
                let (trans_b, ldb) = infer_b_layout(k, n, b_rs, b_cs)?;
                let Some(trans_a) = apply_conj_transpose(trans_a, conj_a) else {
                    return Ok(false);
                };
                let Some(trans_b) = apply_conj_transpose(trans_b, conj_b) else {
                    return Ok(false);
                };
                let ldc = infer_c_layout(m, c_rs, c_cs)?;
                let alpha_ri = [alpha.re, alpha.im];
                let beta_ri = [beta.re, beta.im];

                // SAFETY: `strided_gemm_with_conj`'s caller guarantees the
                // raw pointers are valid for the strided matrix regions.
                // Layout inference checked the unit-stride axis and BLAS
                // leading dims; `apply_conj_transpose` only accepts
                // conjugation representable as CblasConjTrans.
                $gemm(
                    CBLAS_LAYOUT::CblasColMajor,
                    trans_a,
                    trans_b,
                    m_i32,
                    n_i32,
                    k_i32,
                    alpha_ri.as_ptr() as *const _,
                    a_ptr as *const _,
                    lda,
                    b_ptr as *const _,
                    ldb,
                    beta_ri.as_ptr() as *const _,
                    c_ptr as *mut _,
                    ldc,
                );
                Ok(true)
            }
        }
    };
}

impl_real_blas_gemm!(f64, cblas_sys::cblas_dgemm);
impl_real_blas_gemm!(f32, cblas_sys::cblas_sgemm);
impl_complex_blas_gemm!(Complex64, cblas_sys::cblas_zgemm);
impl_complex_blas_gemm!(Complex32, cblas_sys::cblas_cgemm);
