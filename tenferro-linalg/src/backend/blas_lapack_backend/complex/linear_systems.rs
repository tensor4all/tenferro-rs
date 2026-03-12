macro_rules! impl_complex_linear_systems {
    ($complex_ty:ty, $real_ty:ty, $gesv:ident, $trtrs:ident, $gemm:path) => {
        fn mat_mul(
            &mut self,
            a: &[$complex_ty],
            m: usize,
            k: usize,
            b: &[$complex_ty],
            n: usize,
            c: &mut [$complex_ty],
        ) -> Result<()> {
            check_len("mat_mul", "a", a.len(), m * k)?;
            check_len("mat_mul", "b", b.len(), k * n)?;
            check_len("mat_mul", "c", c.len(), m * n)?;

            let m_i32 = as_i32("mat_mul m", m)?;
            let k_i32 = as_i32("mat_mul k", k)?;
            let n_i32 = as_i32("mat_mul n", n)?;

            let alpha = [1 as $real_ty, 0 as $real_ty];
            let beta = [0 as $real_ty, 0 as $real_ty];

            unsafe {
                $gemm(
                    cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                    m_i32,
                    n_i32,
                    k_i32,
                    &alpha,
                    a.as_ptr() as *const _,
                    m_i32,
                    b.as_ptr() as *const _,
                    k_i32,
                    &beta,
                    c.as_mut_ptr() as *mut _,
                    m_i32,
                );
            }
            Ok(())
        }

        fn solve(
            &mut self,
            a: &[$complex_ty],
            b: &[$complex_ty],
            n: usize,
            nrhs: usize,
            x: &mut [$complex_ty],
        ) -> Result<()> {
            check_len("solve", "a", a.len(), n * n)?;
            check_len("solve", "b", b.len(), n * nrhs)?;
            check_len("solve", "x", x.len(), n * nrhs)?;

            let n_i32 = as_i32("solve n", n)?;
            let nrhs_i32 = as_i32("solve nrhs", nrhs)?;

            let mut a_work = a[..n * n].to_vec();
            x[..n * nrhs].copy_from_slice(&b[..n * nrhs]);
            let mut ipiv = vec![0i32; n];
            let mut info = 0;

            unsafe {
                lapack::$gesv(
                    n_i32,
                    nrhs_i32,
                    &mut a_work,
                    n_i32,
                    &mut ipiv,
                    &mut x[..n * nrhs],
                    n_i32,
                    &mut info,
                );
            }
            check_info_success("solve", info)
        }

        fn solve_triangular(
            &mut self,
            a: &[$complex_ty],
            b: &[$complex_ty],
            n: usize,
            nrhs: usize,
            upper: bool,
            x: &mut [$complex_ty],
        ) -> Result<()> {
            check_len("solve_triangular", "a", a.len(), n * n)?;
            check_len("solve_triangular", "b", b.len(), n * nrhs)?;
            check_len("solve_triangular", "x", x.len(), n * nrhs)?;

            let n_i32 = as_i32("solve_triangular n", n)?;
            let nrhs_i32 = as_i32("solve_triangular nrhs", nrhs)?;

            x[..n * nrhs].copy_from_slice(&b[..n * nrhs]);
            let mut info = 0;
            let uplo = if upper { b'U' } else { b'L' };

            unsafe {
                lapack::$trtrs(
                    uplo,
                    b'N',
                    b'N',
                    n_i32,
                    nrhs_i32,
                    &a[..n * n],
                    n_i32,
                    &mut x[..n * nrhs],
                    n_i32,
                    &mut info,
                );
            }
            check_info_success("solve_triangular", info)
        }
    };
}

pub(super) use impl_complex_linear_systems;
