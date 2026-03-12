macro_rules! impl_real_spectral {
    ($ty:ty, $syev:ident, $geev:ident, $lwork_from_query:ident) => {
        fn eigen_sym(
            &mut self,
            a: &[$ty],
            n: usize,
            values: &mut [Self::Real],
            vectors: &mut [$ty],
        ) -> Result<()> {
            check_len("eigen_sym", "a", a.len(), n * n)?;
            check_len("eigen_sym", "values", values.len(), n)?;
            check_len("eigen_sym", "vectors", vectors.len(), n * n)?;

            if n == 0 {
                return Ok(());
            }

            let n_i32 = as_i32("eigen_sym n", n)?;
            vectors[..n * n].copy_from_slice(&a[..n * n]);

            let mut info = 0;
            let mut work_query = [0 as $ty; 1];
            unsafe {
                lapack::$syev(
                    b'V',
                    b'L',
                    n_i32,
                    &mut vectors[..n * n],
                    n_i32,
                    &mut values[..n],
                    &mut work_query,
                    -1,
                    &mut info,
                );
            }
            check_info_nonnegative("eigen_sym(work query)", info)?;

            let lwork = $lwork_from_query("eigen_sym", work_query[0])?;
            let mut work = vec![0 as $ty; lwork as usize];
            unsafe {
                lapack::$syev(
                    b'V',
                    b'L',
                    n_i32,
                    &mut vectors[..n * n],
                    n_i32,
                    &mut values[..n],
                    &mut work,
                    lwork,
                    &mut info,
                );
            }
            check_info_success("eigen_sym", info)
        }

        fn eig_general(
            &mut self,
            a: &[$ty],
            n: usize,
            values_ri: &mut [$ty],
            vectors_ri: &mut [$ty],
        ) -> Result<()> {
            check_len("eig_general", "a", a.len(), n * n)?;
            check_len("eig_general", "values_ri", values_ri.len(), 2 * n)?;
            check_len("eig_general", "vectors_ri", vectors_ri.len(), 2 * n * n)?;

            if n == 0 {
                return Ok(());
            }

            let n_i32 = as_i32("eig_general n", n)?;
            let mut a_work = a[..n * n].to_vec();
            let mut wr = vec![0 as $ty; n];
            let mut wi = vec![0 as $ty; n];
            let mut vr = vec![0 as $ty; n * n];
            let mut vl = vec![0 as $ty; 1];
            let mut info = 0;

            let mut work_query = [0 as $ty; 1];
            unsafe {
                lapack::$geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a_work,
                    n_i32,
                    &mut wr,
                    &mut wi,
                    &mut vl,
                    1,
                    &mut vr,
                    n_i32,
                    &mut work_query,
                    -1,
                    &mut info,
                );
            }
            check_info_nonnegative("eig_general(work query)", info)?;

            let lwork = $lwork_from_query("eig_general", work_query[0])?;
            let mut work = vec![0 as $ty; lwork as usize];
            unsafe {
                lapack::$geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a_work,
                    n_i32,
                    &mut wr,
                    &mut wi,
                    &mut vl,
                    1,
                    &mut vr,
                    n_i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
            }
            check_info_success("eig_general", info)?;

            write_real_eig_general_output(n, &wr, &wi, &vr, values_ri, vectors_ri);

            Ok(())
        }
    };
}

pub(crate) use impl_real_spectral;
