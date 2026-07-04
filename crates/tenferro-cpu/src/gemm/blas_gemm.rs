use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::{Complex32, Complex64};

use crate::Error;

pub(crate) struct BlasGemmBatch<T> {
    pub(crate) a_ptr: *const T,
    pub(crate) b_ptr: *const T,
    pub(crate) c_ptr: *mut T,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) a_rs: isize,
    pub(crate) a_cs: isize,
    pub(crate) b_rs: isize,
    pub(crate) b_cs: isize,
    pub(crate) c_rs: isize,
    pub(crate) c_cs: isize,
}

pub(crate) trait BlasGemm: Sized + Copy {
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

    #[allow(clippy::too_many_arguments)]
    /// Run a group of raw strided GEMMs with shared alpha/beta.
    ///
    /// # Safety
    ///
    /// Each batch entry must satisfy [`BlasGemm::strided_gemm`]'s pointer
    /// safety contract. Mutable output regions must be pairwise disjoint.
    unsafe fn grouped_gemm(
        alpha: Self,
        beta: Self,
        batches: &[BlasGemmBatch<Self>],
    ) -> crate::Result<bool> {
        for batch in batches {
            unsafe {
                Self::strided_gemm(
                    alpha,
                    batch.a_ptr,
                    batch.m,
                    batch.k,
                    batch.a_rs,
                    batch.a_cs,
                    batch.b_ptr,
                    batch.n,
                    batch.b_rs,
                    batch.b_cs,
                    beta,
                    batch.c_ptr,
                    batch.c_rs,
                    batch.c_cs,
                )?;
            }
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

#[cfg(feature = "blas-openblas")]
mod openblas_batch {
    use super::*;
    use std::ffi::c_void;

    unsafe extern "C" {
        fn cblas_sgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const f32,
            a: *const *const f32,
            lda: *const i32,
            b: *const *const f32,
            ldb: *const i32,
            beta: *const f32,
            c: *const *mut f32,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );

        fn cblas_dgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const f64,
            a: *const *const f64,
            lda: *const i32,
            b: *const *const f64,
            ldb: *const i32,
            beta: *const f64,
            c: *const *mut f64,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );

        fn cblas_cgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const c_void,
            a: *const *const c_void,
            lda: *const i32,
            b: *const *const c_void,
            ldb: *const i32,
            beta: *const c_void,
            c: *const *mut c_void,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );

        fn cblas_zgemm_batch(
            order: CBLAS_LAYOUT,
            trans_a: *const CBLAS_TRANSPOSE,
            trans_b: *const CBLAS_TRANSPOSE,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            alpha: *const c_void,
            a: *const *const c_void,
            lda: *const i32,
            b: *const *const c_void,
            ldb: *const i32,
            beta: *const c_void,
            c: *const *mut c_void,
            ldc: *const i32,
            group_count: i32,
            group_size: *const i32,
        );
    }

    struct PreparedBatch<T> {
        trans_a: Vec<CBLAS_TRANSPOSE>,
        trans_b: Vec<CBLAS_TRANSPOSE>,
        m: Vec<i32>,
        n: Vec<i32>,
        k: Vec<i32>,
        a: Vec<*const T>,
        b: Vec<*const T>,
        c: Vec<*mut T>,
        lda: Vec<i32>,
        ldb: Vec<i32>,
        ldc: Vec<i32>,
        group_size: Vec<i32>,
    }

    fn prepare<T>(batches: &[BlasGemmBatch<T>]) -> crate::Result<PreparedBatch<T>> {
        let mut prepared = PreparedBatch {
            trans_a: Vec::with_capacity(batches.len()),
            trans_b: Vec::with_capacity(batches.len()),
            m: Vec::with_capacity(batches.len()),
            n: Vec::with_capacity(batches.len()),
            k: Vec::with_capacity(batches.len()),
            a: Vec::with_capacity(batches.len()),
            b: Vec::with_capacity(batches.len()),
            c: Vec::with_capacity(batches.len()),
            lda: Vec::with_capacity(batches.len()),
            ldb: Vec::with_capacity(batches.len()),
            ldc: Vec::with_capacity(batches.len()),
            group_size: Vec::with_capacity(batches.len()),
        };
        for batch in batches {
            let (trans_a, lda) = infer_a_layout(batch.m, batch.k, batch.a_rs, batch.a_cs)?;
            let (trans_b, ldb) = infer_b_layout(batch.k, batch.n, batch.b_rs, batch.b_cs)?;
            let ldc = infer_c_layout(batch.m, batch.c_rs, batch.c_cs)?;
            prepared.trans_a.push(trans_a);
            prepared.trans_b.push(trans_b);
            prepared.m.push(dim_to_i32("m", batch.m)?);
            prepared.n.push(dim_to_i32("n", batch.n)?);
            prepared.k.push(dim_to_i32("k", batch.k)?);
            prepared.a.push(batch.a_ptr);
            prepared.b.push(batch.b_ptr);
            prepared.c.push(batch.c_ptr);
            prepared.lda.push(lda);
            prepared.ldb.push(ldb);
            prepared.ldc.push(ldc);
            prepared.group_size.push(1);
        }
        Ok(prepared)
    }

    pub(crate) unsafe fn sgemm_batch(
        alpha: f32,
        beta: f32,
        batches: &[BlasGemmBatch<f32>],
    ) -> crate::Result<bool> {
        let prepared = prepare(batches)?;
        let group_count = dim_to_i32("group_count", batches.len())?;
        unsafe {
            cblas_sgemm_batch(
                CBLAS_LAYOUT::CblasColMajor,
                prepared.trans_a.as_ptr(),
                prepared.trans_b.as_ptr(),
                prepared.m.as_ptr(),
                prepared.n.as_ptr(),
                prepared.k.as_ptr(),
                vec![alpha; batches.len()].as_ptr(),
                prepared.a.as_ptr(),
                prepared.lda.as_ptr(),
                prepared.b.as_ptr(),
                prepared.ldb.as_ptr(),
                vec![beta; batches.len()].as_ptr(),
                prepared.c.as_ptr(),
                prepared.ldc.as_ptr(),
                group_count,
                prepared.group_size.as_ptr(),
            );
        }
        Ok(true)
    }

    pub(crate) unsafe fn dgemm_batch(
        alpha: f64,
        beta: f64,
        batches: &[BlasGemmBatch<f64>],
    ) -> crate::Result<bool> {
        let prepared = prepare(batches)?;
        let group_count = dim_to_i32("group_count", batches.len())?;
        unsafe {
            cblas_dgemm_batch(
                CBLAS_LAYOUT::CblasColMajor,
                prepared.trans_a.as_ptr(),
                prepared.trans_b.as_ptr(),
                prepared.m.as_ptr(),
                prepared.n.as_ptr(),
                prepared.k.as_ptr(),
                vec![alpha; batches.len()].as_ptr(),
                prepared.a.as_ptr(),
                prepared.lda.as_ptr(),
                prepared.b.as_ptr(),
                prepared.ldb.as_ptr(),
                vec![beta; batches.len()].as_ptr(),
                prepared.c.as_ptr(),
                prepared.ldc.as_ptr(),
                group_count,
                prepared.group_size.as_ptr(),
            );
        }
        Ok(true)
    }

    pub(crate) unsafe fn cgemm_batch(
        alpha: Complex32,
        beta: Complex32,
        batches: &[BlasGemmBatch<Complex32>],
    ) -> crate::Result<bool> {
        let prepared = prepare(batches)?;
        let group_count = dim_to_i32("group_count", batches.len())?;
        let alpha = vec![alpha; batches.len()];
        let beta = vec![beta; batches.len()];
        let a: Vec<*const c_void> = prepared.a.iter().map(|&ptr| ptr as *const c_void).collect();
        let b: Vec<*const c_void> = prepared.b.iter().map(|&ptr| ptr as *const c_void).collect();
        let c: Vec<*mut c_void> = prepared.c.iter().map(|&ptr| ptr as *mut c_void).collect();
        unsafe {
            cblas_cgemm_batch(
                CBLAS_LAYOUT::CblasColMajor,
                prepared.trans_a.as_ptr(),
                prepared.trans_b.as_ptr(),
                prepared.m.as_ptr(),
                prepared.n.as_ptr(),
                prepared.k.as_ptr(),
                alpha.as_ptr() as *const c_void,
                a.as_ptr(),
                prepared.lda.as_ptr(),
                b.as_ptr(),
                prepared.ldb.as_ptr(),
                beta.as_ptr() as *const c_void,
                c.as_ptr(),
                prepared.ldc.as_ptr(),
                group_count,
                prepared.group_size.as_ptr(),
            );
        }
        Ok(true)
    }

    pub(crate) unsafe fn zgemm_batch(
        alpha: Complex64,
        beta: Complex64,
        batches: &[BlasGemmBatch<Complex64>],
    ) -> crate::Result<bool> {
        let prepared = prepare(batches)?;
        let group_count = dim_to_i32("group_count", batches.len())?;
        let alpha = vec![alpha; batches.len()];
        let beta = vec![beta; batches.len()];
        let a: Vec<*const c_void> = prepared.a.iter().map(|&ptr| ptr as *const c_void).collect();
        let b: Vec<*const c_void> = prepared.b.iter().map(|&ptr| ptr as *const c_void).collect();
        let c: Vec<*mut c_void> = prepared.c.iter().map(|&ptr| ptr as *mut c_void).collect();
        unsafe {
            cblas_zgemm_batch(
                CBLAS_LAYOUT::CblasColMajor,
                prepared.trans_a.as_ptr(),
                prepared.trans_b.as_ptr(),
                prepared.m.as_ptr(),
                prepared.n.as_ptr(),
                prepared.k.as_ptr(),
                alpha.as_ptr() as *const c_void,
                a.as_ptr(),
                prepared.lda.as_ptr(),
                b.as_ptr(),
                prepared.ldb.as_ptr(),
                beta.as_ptr() as *const c_void,
                c.as_ptr(),
                prepared.ldc.as_ptr(),
                group_count,
                prepared.group_size.as_ptr(),
            );
        }
        Ok(true)
    }
}

macro_rules! impl_real_blas_gemm {
    ($ty:ty, $gemm:path, $batch:path) => {
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

            #[cfg(feature = "blas-openblas")]
            unsafe fn grouped_gemm(
                alpha: Self,
                beta: Self,
                batches: &[BlasGemmBatch<Self>],
            ) -> crate::Result<bool> {
                if batches.is_empty() {
                    return Ok(true);
                }
                unsafe { $batch(alpha, beta, batches) }
            }
        }
    };
}

macro_rules! impl_complex_blas_gemm {
    ($ty:ty, $gemm:path, $batch:path) => {
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

            #[cfg(feature = "blas-openblas")]
            unsafe fn grouped_gemm(
                alpha: Self,
                beta: Self,
                batches: &[BlasGemmBatch<Self>],
            ) -> crate::Result<bool> {
                if batches.is_empty() {
                    return Ok(true);
                }
                unsafe { $batch(alpha, beta, batches) }
            }
        }
    };
}

impl_real_blas_gemm!(f64, cblas_sys::cblas_dgemm, openblas_batch::dgemm_batch);
impl_real_blas_gemm!(f32, cblas_sys::cblas_sgemm, openblas_batch::sgemm_batch);
impl_complex_blas_gemm!(
    Complex64,
    cblas_sys::cblas_zgemm,
    openblas_batch::zgemm_batch
);
impl_complex_blas_gemm!(
    Complex32,
    cblas_sys::cblas_cgemm,
    openblas_batch::cgemm_batch
);
