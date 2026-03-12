macro_rules! impl_real_decompositions {
    (
        $ty:ty,
        $gesvd:ident,
        $geqrf:ident,
        $orgqr:ident,
        $getrf:ident,
        $potrf:ident,
        $lwork_from_query:ident
    ) => {
        fn thin_svd(
            &mut self,
            a: &[$ty],
            m: usize,
            n: usize,
            u: &mut [$ty],
            s: &mut [Self::Real],
            vt: &mut [$ty],
        ) -> Result<()> {
            let k = m.min(n);
            check_len("thin_svd", "a", a.len(), m * n)?;
            check_len("thin_svd", "u", u.len(), m * k)?;
            check_len("thin_svd", "s", s.len(), k)?;
            check_len("thin_svd", "vt", vt.len(), k * n)?;

            let m_i32 = as_i32("thin_svd m", m)?;
            let n_i32 = as_i32("thin_svd n", n)?;
            let k_i32 = as_i32("thin_svd k", k)?;

            let mut a_work = a[..m * n].to_vec();
            let mut info = 0;

            let mut work_query = [0 as $ty; 1];
            unsafe {
                lapack::$gesvd(
                    b'S',
                    b'S',
                    m_i32,
                    n_i32,
                    &mut a_work,
                    m_i32,
                    &mut s[..k],
                    &mut u[..m * k],
                    m_i32,
                    &mut vt[..k * n],
                    k_i32,
                    &mut work_query,
                    -1,
                    &mut info,
                );
            }
            check_info_nonnegative("thin_svd(work query)", info)?;

            let lwork = $lwork_from_query("thin_svd", work_query[0])?;
            let mut work = vec![0 as $ty; lwork as usize];

            unsafe {
                lapack::$gesvd(
                    b'S',
                    b'S',
                    m_i32,
                    n_i32,
                    &mut a_work,
                    m_i32,
                    &mut s[..k],
                    &mut u[..m * k],
                    m_i32,
                    &mut vt[..k * n],
                    k_i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
            }
            check_info_success("thin_svd", info)
        }

        fn qr(
            &mut self,
            a: &[$ty],
            m: usize,
            n: usize,
            q: &mut [$ty],
            r: &mut [$ty],
        ) -> Result<()> {
            let k = m.min(n);
            check_len("qr", "a", a.len(), m * n)?;
            check_len("qr", "q", q.len(), m * k)?;
            check_len("qr", "r", r.len(), k * n)?;

            if k == 0 {
                return Ok(());
            }

            let m_i32 = as_i32("qr m", m)?;
            let n_i32 = as_i32("qr n", n)?;
            let k_i32 = as_i32("qr k", k)?;

            let mut a_fact = a[..m * n].to_vec();
            let mut tau = vec![0 as $ty; k];
            let mut info = 0;

            let mut work_query = [0 as $ty; 1];
            unsafe {
                lapack::$geqrf(
                    m_i32,
                    n_i32,
                    &mut a_fact,
                    m_i32,
                    &mut tau,
                    &mut work_query,
                    -1,
                    &mut info,
                );
            }
            check_info_nonnegative("qr(geqrf work query)", info)?;

            let geqrf_lwork = $lwork_from_query("qr(geqrf)", work_query[0])?;
            let mut geqrf_work = vec![0 as $ty; geqrf_lwork as usize];

            unsafe {
                lapack::$geqrf(
                    m_i32,
                    n_i32,
                    &mut a_fact,
                    m_i32,
                    &mut tau,
                    &mut geqrf_work,
                    geqrf_lwork,
                    &mut info,
                );
            }
            check_info_success("qr(geqrf)", info)?;

            for j in 0..n {
                for i in 0..k {
                    r[i + j * k] = if i <= j { a_fact[i + j * m] } else { 0 as $ty };
                }
            }

            let mut q_data = vec![0 as $ty; m * k];
            for j in 0..k {
                for i in 0..m {
                    q_data[i + j * m] = a_fact[i + j * m];
                }
            }

            let mut q_work_query = [0 as $ty; 1];
            unsafe {
                lapack::$orgqr(
                    m_i32,
                    k_i32,
                    k_i32,
                    &mut q_data,
                    m_i32,
                    &tau,
                    &mut q_work_query,
                    -1,
                    &mut info,
                );
            }
            check_info_nonnegative("qr(orgqr work query)", info)?;

            let orgqr_lwork = $lwork_from_query("qr(orgqr)", q_work_query[0])?;
            let mut orgqr_work = vec![0 as $ty; orgqr_lwork as usize];

            unsafe {
                lapack::$orgqr(
                    m_i32,
                    k_i32,
                    k_i32,
                    &mut q_data,
                    m_i32,
                    &tau,
                    &mut orgqr_work,
                    orgqr_lwork,
                    &mut info,
                );
            }
            check_info_success("qr(orgqr)", info)?;

            q[..m * k].copy_from_slice(&q_data);
            Ok(())
        }

        fn lu(
            &mut self,
            a: &[$ty],
            m: usize,
            n: usize,
            perm: &mut [usize],
            l: &mut [$ty],
            u_out: &mut [$ty],
        ) -> Result<()> {
            let k = m.min(n);
            check_len("lu", "a", a.len(), m * n)?;
            check_len("lu", "perm", perm.len(), m)?;
            check_len("lu", "l", l.len(), m * k)?;
            check_len("lu", "u_out", u_out.len(), k * n)?;

            if m == 0 || n == 0 {
                for (i, p) in perm.iter_mut().take(m).enumerate() {
                    *p = i;
                }
                return Ok(());
            }

            let m_i32 = as_i32("lu m", m)?;
            let n_i32 = as_i32("lu n", n)?;

            let mut lu = a[..m * n].to_vec();
            let mut piv = vec![0i32; k];
            let mut info = 0;

            unsafe {
                lapack::$getrf(m_i32, n_i32, &mut lu, m_i32, &mut piv, &mut info);
            }
            check_info_nonnegative("lu(getrf)", info)?;

            let p = pivots_to_forward_perm(m, &piv)?;
            perm[..m].copy_from_slice(&p);
            split_lu(&lu, m, n, l, u_out);

            Ok(())
        }

        fn cholesky(&mut self, a: &[$ty], n: usize, l: &mut [$ty]) -> Result<()> {
            check_len("cholesky", "a", a.len(), n * n)?;
            check_len("cholesky", "l", l.len(), n * n)?;

            if n == 0 {
                return Ok(());
            }

            let n_i32 = as_i32("cholesky n", n)?;
            l[..n * n].copy_from_slice(&a[..n * n]);

            let mut info = 0;
            unsafe {
                lapack::$potrf(b'L', n_i32, &mut l[..n * n], n_i32, &mut info);
            }
            check_info_cholesky("cholesky", info)?;
            fill_zero_upper(&mut l[..n * n], n);
            Ok(())
        }
    };
}

pub(super) use impl_real_decompositions;
