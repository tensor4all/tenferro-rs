use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use num_complex::{Complex32, Complex64};

pub(crate) trait BlasGemm: Sized {
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
    );

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

fn dim_to_i32(name: &str, value: usize) -> i32 {
    match i32::try_from(value) {
        Ok(value) => value,
        Err(_) => panic!("{name} too large for BLAS i32 dimensions"),
    }
}

fn stride_to_i32(name: &str, value: isize) -> i32 {
    match i32::try_from(value) {
        Ok(value) if value > 0 => value,
        _ => panic!("{name} must be a positive BLAS stride"),
    }
}

fn infer_a_layout(m: usize, k: usize, a_rs: isize, a_cs: isize) -> (CBLAS_TRANSPOSE, i32) {
    if a_rs == 1 {
        let lda = stride_to_i32("lda", a_cs);
        assert!(
            lda >= dim_to_i32("m", m),
            "lda must be >= max(1, m) for NoTrans A"
        );
        (CBLAS_TRANSPOSE::CblasNoTrans, lda)
    } else if a_cs == 1 {
        let lda = stride_to_i32("lda", a_rs);
        assert!(
            lda >= dim_to_i32("k", k),
            "lda must be >= max(1, k) for Trans A"
        );
        (CBLAS_TRANSPOSE::CblasTrans, lda)
    } else {
        panic!("BLAS requires unit stride on one axis of A");
    }
}

fn infer_b_layout(k: usize, n: usize, b_rs: isize, b_cs: isize) -> (CBLAS_TRANSPOSE, i32) {
    if b_rs == 1 {
        let ldb = stride_to_i32("ldb", b_cs);
        assert!(
            ldb >= dim_to_i32("k", k),
            "ldb must be >= max(1, k) for NoTrans B"
        );
        (CBLAS_TRANSPOSE::CblasNoTrans, ldb)
    } else if b_cs == 1 {
        let ldb = stride_to_i32("ldb", b_rs);
        assert!(
            ldb >= dim_to_i32("n", n),
            "ldb must be >= max(1, n) for Trans B"
        );
        (CBLAS_TRANSPOSE::CblasTrans, ldb)
    } else {
        panic!("BLAS requires unit stride on one axis of B");
    }
}

fn infer_c_layout(m: usize, c_rs: isize, c_cs: isize) -> i32 {
    assert!(c_rs == 1, "BLAS output requires unit row stride");
    let ldc = stride_to_i32("ldc", c_cs);
    assert!(ldc >= dim_to_i32("m", m), "ldc must be >= max(1, m)");
    ldc
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
            ) {
                let m_i32 = dim_to_i32("m", m);
                let n_i32 = dim_to_i32("n", n);
                let k_i32 = dim_to_i32("k", k);
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
            ) {
                let m_i32 = dim_to_i32("m", m);
                let n_i32 = dim_to_i32("n", n);
                let k_i32 = dim_to_i32("k", k);
                let (trans_a, lda) = infer_a_layout(m, k, a_rs, a_cs);
                let (trans_b, ldb) = infer_b_layout(k, n, b_rs, b_cs);
                let ldc = infer_c_layout(m, c_rs, c_cs);

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
            ) {
                let m_i32 = dim_to_i32("m", m);
                let n_i32 = dim_to_i32("n", n);
                let k_i32 = dim_to_i32("k", k);
                let alpha_ri = [alpha.re, alpha.im];
                let beta_ri = [beta.re, beta.im];
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
            ) {
                let m_i32 = dim_to_i32("m", m);
                let n_i32 = dim_to_i32("n", n);
                let k_i32 = dim_to_i32("k", k);
                let (trans_a, lda) = infer_a_layout(m, k, a_rs, a_cs);
                let (trans_b, ldb) = infer_b_layout(k, n, b_rs, b_cs);
                let ldc = infer_c_layout(m, c_rs, c_cs);
                let alpha_ri = [alpha.re, alpha.im];
                let beta_ri = [beta.re, beta.im];

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
            }
        }
    };
}

impl_real_blas_gemm!(f64, cblas_sys::cblas_dgemm);
impl_real_blas_gemm!(f32, cblas_sys::cblas_sgemm);
impl_complex_blas_gemm!(Complex64, cblas_sys::cblas_zgemm);
impl_complex_blas_gemm!(Complex32, cblas_sys::cblas_cgemm);
