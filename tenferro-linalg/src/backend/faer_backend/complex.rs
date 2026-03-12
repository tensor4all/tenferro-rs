macro_rules! impl_complex_linalg_backend {
    ($complex_ty:ty, $real_ty:ty, $to_faer:ident, $from_faer_mat:ident) => {
        impl LinalgBackend<$complex_ty> for FaerBackend {
            type Real = $real_ty;

            fn thin_svd(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                u: &mut [$complex_ty],
                s: &mut [Self::Real],
                vt: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("thin_svd", "a", a.len(), m * n)?;
                check_len("thin_svd", "u", u.len(), m * k)?;
                check_len("thin_svd", "s", s.len(), k)?;
                check_len("thin_svd", "vt", vt.len(), k * n)?;

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, m, n);
                let svd = mat.thin_svd().map_err(|_| {
                    Error::InvalidArgument("thin_svd: SVD computation failed".into())
                })?;

                let u_ref = svd.U();
                let v_ref = svd.V();
                let s_diag = svd.S();

                $from_faer_mat(u_ref, u, m, k);

                let s_col = s_diag.column_vector();
                for i in 0..k {
                    s[i] = s_col[i].re;
                }

                for j in 0..n {
                    for i in 0..k {
                        let v = v_ref[(j, i)];
                        vt[i + j * k] = <$complex_ty>::new(v.re, -v.im);
                    }
                }

                Ok(())
            }

            fn qr(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                q: &mut [$complex_ty],
                r: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("qr", "a", a.len(), m * n)?;
                check_len("qr", "q", q.len(), m * k)?;
                check_len("qr", "r", r.len(), k * n)?;

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, m, n);
                let qr_result = mat.qr();

                let q_mat = qr_result.compute_thin_Q();
                let r_mat = qr_result.thin_R();

                $from_faer_mat(q_mat.as_ref(), q, m, k);
                $from_faer_mat(r_mat, r, k, n);

                Ok(())
            }

            fn lu(
                &mut self,
                a: &[$complex_ty],
                m: usize,
                n: usize,
                perm: &mut [usize],
                l: &mut [$complex_ty],
                u_out: &mut [$complex_ty],
            ) -> Result<()> {
                let k = m.min(n);
                check_len("lu", "a", a.len(), m * n)?;
                check_len("lu", "perm", perm.len(), m)?;
                check_len("lu", "l", l.len(), m * k)?;
                check_len("lu", "u_out", u_out.len(), k * n)?;

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, m, n);
                let lu_result = mat.partial_piv_lu();

                let l_mat = lu_result.L();
                let u_mat = lu_result.U();

                $from_faer_mat(l_mat, l, m, k);
                $from_faer_mat(u_mat, u_out, k, n);

                let perm_ref = lu_result.P();
                let (fwd, _inv) = perm_ref.arrays();
                perm[..m].copy_from_slice(fwd);

                Ok(())
            }

            fn cholesky(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                l: &mut [$complex_ty],
            ) -> Result<()> {
                check_len("cholesky", "a", a.len(), n * n)?;
                check_len("cholesky", "l", l.len(), n * n)?;

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, n, n);
                match mat.llt(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.L();
                        $from_faer_mat(l_mat, l, n, n);
                        Ok(())
                    }
                    Err(_) => Err(Error::InvalidArgument(
                        "cholesky: matrix is not positive definite".to_string(),
                    )),
                }
            }

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

                let a_faer = $to_faer(a);
                let mat = faer::MatRef::from_column_major_slice(&a_faer, n, n);
                let eig = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|_| {
                    Error::InvalidArgument("eigen_sym: eigendecomposition failed".into())
                })?;

                let u_ref = eig.U();
                let s_diag = eig.S();

                $from_faer_mat(u_ref, vectors, n, n);

                let s_col = s_diag.column_vector();
                for i in 0..n {
                    values[i] = s_col[i].re;
                }

                Ok(())
            }

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

                let a_faer = $to_faer(a);
                let b_faer = $to_faer(b);
                let a_mat = faer::MatRef::from_column_major_slice(&a_faer, m, k);
                let b_mat = faer::MatRef::from_column_major_slice(&b_faer, k, n);
                let result = &a_mat * &b_mat;

                $from_faer_mat(result.as_ref(), c, m, n);

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

                let a_faer = $to_faer(a);
                let b_faer = $to_faer(b);
                let a_mat = faer::MatRef::from_column_major_slice(&a_faer, n, n);
                let b_mat = faer::MatRef::from_column_major_slice(&b_faer, n, nrhs);
                let lu = a_mat.partial_piv_lu();
                let u_mat = lu.U();
                for i in 0..n {
                    let diag = u_mat[(i, i)];
                    if !diag.re.is_finite()
                        || !diag.im.is_finite()
                        || (diag.re == 0.0 && diag.im == 0.0)
                    {
                        return Err(singular_matrix_error("solve"));
                    }
                }
                let result = lu.solve(&b_mat);

                $from_faer_mat(result.as_ref(), x, n, nrhs);
                for value in x.iter().copied() {
                    if !complex_is_finite(value) {
                        return Err(non_finite_result_error("solve"));
                    }
                }

                Ok(())
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

                for col in 0..nrhs {
                    let b_col = &b[col * n..(col + 1) * n];
                    let x_col = &mut x[col * n..(col + 1) * n];

                    if upper {
                        for i in (0..n).rev() {
                            let mut sum = b_col[i];
                            for j in (i + 1)..n {
                                sum -= a[i + j * n] * x_col[j];
                            }
                            let diag = a[i + i * n];
                            if !complex_is_finite(diag) || (diag.re == 0.0 && diag.im == 0.0) {
                                return Err(zero_diagonal_error("solve_triangular", i));
                            }
                            let value = sum / diag;
                            if !complex_is_finite(value) {
                                return Err(non_finite_result_error("solve_triangular"));
                            }
                            x_col[i] = value;
                        }
                    } else {
                        for i in 0..n {
                            let mut sum = b_col[i];
                            for j in 0..i {
                                sum -= a[i + j * n] * x_col[j];
                            }
                            let diag = a[i + i * n];
                            if !complex_is_finite(diag) || (diag.re == 0.0 && diag.im == 0.0) {
                                return Err(zero_diagonal_error("solve_triangular", i));
                            }
                            let value = sum / diag;
                            if !complex_is_finite(value) {
                                return Err(non_finite_result_error("solve_triangular"));
                            }
                            x_col[i] = value;
                        }
                    }
                }

                Ok(())
            }

            fn eig_general(
                &mut self,
                a: &[$complex_ty],
                n: usize,
                values_ri: &mut [$complex_ty],
                vectors_ri: &mut [$complex_ty],
            ) -> Result<()> {
                use faer::c64;

                check_len("eig_general", "a", a.len(), n * n)?;
                check_len("eig_general", "values_ri", values_ri.len(), n)?;
                check_len("eig_general", "vectors_ri", vectors_ri.len(), n * n)?;

                let a_c64: Vec<c64> = a[..n * n]
                    .iter()
                    .map(|c| c64::new(c.re as f64, c.im as f64))
                    .collect();
                let mat = faer::MatRef::from_column_major_slice(&a_c64, n, n);
                let eig = mat.eigen().map_err(|e| {
                    Error::InvalidArgument(format!("eigendecomposition failed: {e:?}"))
                })?;

                let s_diag = eig.S();
                let s_col = s_diag.column_vector();
                let u_ref = eig.U();

                for i in 0..n {
                    let val = s_col[i];
                    values_ri[i] = <$complex_ty>::new(val.re as $real_ty, val.im as $real_ty);
                }

                for j in 0..n {
                    for i in 0..n {
                        let val = u_ref[(i, j)];
                        vectors_ri[i + j * n] =
                            <$complex_ty>::new(val.re as $real_ty, val.im as $real_ty);
                    }
                }

                Ok(())
            }
        }
    };
}

pub(crate) use impl_complex_linalg_backend;
