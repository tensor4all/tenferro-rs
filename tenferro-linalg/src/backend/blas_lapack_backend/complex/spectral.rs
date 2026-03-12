macro_rules! impl_complex_spectral {
    ($complex_ty:ty, $real_ty:ty, $heev:ident, $geev:ident, $lwork_from_query:ident) => {
        fn eigen_sym(
            &mut self,
            a: &[$complex_ty],
            n: usize,
            values: &mut [Self::Real],
            vectors: &mut [$complex_ty],
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
            let mut work_query = [<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
            let mut rwork = vec![0 as $real_ty; (3 * n).saturating_sub(2).max(1)];

            unsafe {
                lapack::$heev(
                    b'V',
                    b'L',
                    n_i32,
                    &mut vectors[..n * n],
                    n_i32,
                    &mut values[..n],
                    &mut work_query,
                    -1,
                    &mut rwork,
                    &mut info,
                );
            }
            check_info_nonnegative("eigen_sym(work query)", info)?;

            let lwork = $lwork_from_query("eigen_sym", work_query[0])?;
            let mut work = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); lwork as usize];
            unsafe {
                lapack::$heev(
                    b'V',
                    b'L',
                    n_i32,
                    &mut vectors[..n * n],
                    n_i32,
                    &mut values[..n],
                    &mut work,
                    lwork,
                    &mut rwork,
                    &mut info,
                );
            }
            check_info_success("eigen_sym", info)
        }

        fn eig_general(
            &mut self,
            a: &[$complex_ty],
            n: usize,
            values_ri: &mut [$complex_ty],
            vectors_ri: &mut [$complex_ty],
        ) -> Result<()> {
            check_len("eig_general", "a", a.len(), n * n)?;
            check_len("eig_general", "values_ri", values_ri.len(), n)?;
            check_len("eig_general", "vectors_ri", vectors_ri.len(), n * n)?;

            if n == 0 {
                return Ok(());
            }

            let n_i32 = as_i32("eig_general n", n)?;
            let mut a_work = a[..n * n].to_vec();
            let mut w = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); n];
            let mut vr = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); n * n];
            let mut vl = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
            let mut rwork = vec![0 as $real_ty; (2 * n).max(1)];
            let mut info = 0;

            let mut work_query = [<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); 1];
            unsafe {
                lapack::$geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a_work,
                    n_i32,
                    &mut w,
                    &mut vl,
                    1,
                    &mut vr,
                    n_i32,
                    &mut work_query,
                    -1,
                    &mut rwork,
                    &mut info,
                );
            }
            check_info_nonnegative("eig_general(work query)", info)?;

            let lwork = $lwork_from_query("eig_general", work_query[0])?;
            let mut work = vec![<$complex_ty>::new(0 as $real_ty, 0 as $real_ty); lwork as usize];
            unsafe {
                lapack::$geev(
                    b'N',
                    b'V',
                    n_i32,
                    &mut a_work,
                    n_i32,
                    &mut w,
                    &mut vl,
                    1,
                    &mut vr,
                    n_i32,
                    &mut work,
                    lwork,
                    &mut rwork,
                    &mut info,
                );
            }
            check_info_success("eig_general", info)?;

            values_ri[..n].copy_from_slice(&w[..n]);
            vectors_ri[..n * n].copy_from_slice(&vr[..n * n]);
            Ok(())
        }
    };
}

pub(crate) use impl_complex_spectral;
