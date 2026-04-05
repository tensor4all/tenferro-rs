use num_complex::{Complex32, Complex64};

pub(crate) trait BlasGemm: Sized {
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
        let m_i32 = i32::try_from(m).expect("m too large");
        let n_i32 = i32::try_from(n).expect("n too large");
        let k_i32 = i32::try_from(k).expect("k too large");
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
        let m_i32 = i32::try_from(m).expect("m too large");
        let n_i32 = i32::try_from(n).expect("n too large");
        let k_i32 = i32::try_from(k).expect("k too large");
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
        let m_i32 = i32::try_from(m).expect("m too large");
        let n_i32 = i32::try_from(n).expect("n too large");
        let k_i32 = i32::try_from(k).expect("k too large");
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
        let m_i32 = i32::try_from(m).expect("m too large");
        let n_i32 = i32::try_from(n).expect("n too large");
        let k_i32 = i32::try_from(k).expect("k too large");
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
