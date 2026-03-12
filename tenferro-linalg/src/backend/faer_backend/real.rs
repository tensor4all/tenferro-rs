macro_rules! impl_linalg_backend {
    ($ty:ty) => {
        impl LinalgBackend<$ty> for FaerBackend {
            type Real = $ty;

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

                let mat = faer::MatRef::from_column_major_slice(a, m, n);
                let svd = mat.thin_svd().map_err(|_| {
                    Error::InvalidArgument("thin_svd: SVD computation failed".into())
                })?;

                let u_ref = svd.U();
                let v_ref = svd.V();
                let s_diag = svd.S();

                for j in 0..k {
                    for i in 0..m {
                        u[i + j * m] = u_ref[(i, j)];
                    }
                }

                let s_col = s_diag.column_vector();
                for i in 0..k {
                    s[i] = s_col[i];
                }

                for j in 0..n {
                    for i in 0..k {
                        vt[i + j * k] = v_ref[(j, i)];
                    }
                }

                Ok(())
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

                let mat = faer::MatRef::from_column_major_slice(a, m, n);
                let qr_result = mat.qr();

                let q_mat = qr_result.compute_thin_Q();
                let r_mat = qr_result.thin_R();

                for j in 0..k {
                    for i in 0..m {
                        q[i + j * m] = q_mat[(i, j)];
                    }
                }

                for j in 0..n {
                    for i in 0..k {
                        r[i + j * k] = r_mat[(i, j)];
                    }
                }

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

                let mat = faer::MatRef::from_column_major_slice(a, m, n);
                let lu_result = mat.partial_piv_lu();

                let l_mat = lu_result.L();
                let u_mat = lu_result.U();

                for j in 0..k {
                    for i in 0..m {
                        l[i + j * m] = l_mat[(i, j)];
                    }
                }

                for j in 0..n {
                    for i in 0..k {
                        u_out[i + j * k] = u_mat[(i, j)];
                    }
                }

                let perm_ref = lu_result.P();
                let (fwd, _inv) = perm_ref.arrays();
                perm[..m].copy_from_slice(fwd);

                Ok(())
            }

            fn cholesky(&mut self, a: &[$ty], n: usize, l: &mut [$ty]) -> Result<()> {
                check_len("cholesky", "a", a.len(), n * n)?;
                check_len("cholesky", "l", l.len(), n * n)?;

                let mat = faer::MatRef::from_column_major_slice(a, n, n);
                match mat.llt(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.L();
                        for j in 0..n {
                            for i in 0..n {
                                l[i + j * n] = l_mat[(i, j)];
                            }
                        }
                        Ok(())
                    }
                    Err(_) => Err(Error::InvalidArgument(
                        "cholesky: matrix is not positive definite".to_string(),
                    )),
                }
            }

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

                let mat = faer::MatRef::from_column_major_slice(a, n, n);
                let eig = mat.self_adjoint_eigen(faer::Side::Lower).map_err(|_| {
                    Error::InvalidArgument("eigen_sym: eigendecomposition failed".into())
                })?;

                let u_ref = eig.U();
                let s_diag = eig.S();

                for j in 0..n {
                    for i in 0..n {
                        vectors[i + j * n] = u_ref[(i, j)];
                    }
                }

                let s_col = s_diag.column_vector();
                for i in 0..n {
                    values[i] = s_col[i];
                }

                Ok(())
            }

            fn mat_mul(
                &mut self,
                a: &[$ty],
                m: usize,
                k: usize,
                b: &[$ty],
                n: usize,
                c: &mut [$ty],
            ) -> Result<()> {
                check_len("mat_mul", "a", a.len(), m * k)?;
                check_len("mat_mul", "b", b.len(), k * n)?;
                check_len("mat_mul", "c", c.len(), m * n)?;

                let a_mat = faer::MatRef::from_column_major_slice(a, m, k);
                let b_mat = faer::MatRef::from_column_major_slice(b, k, n);
                let result = &a_mat * &b_mat;

                for j in 0..n {
                    for i in 0..m {
                        c[i + j * m] = result[(i, j)];
                    }
                }

                Ok(())
            }

            fn solve(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                x: &mut [$ty],
            ) -> Result<()> {
                check_len("solve", "a", a.len(), n * n)?;
                check_len("solve", "b", b.len(), n * nrhs)?;
                check_len("solve", "x", x.len(), n * nrhs)?;

                let a_mat = faer::MatRef::from_column_major_slice(a, n, n);
                let b_mat = faer::MatRef::from_column_major_slice(b, n, nrhs);
                let lu = a_mat.partial_piv_lu();
                let u_mat = lu.U();
                for i in 0..n {
                    let diag = u_mat[(i, i)];
                    if !diag.is_finite() || diag == 0.0 {
                        return Err(singular_matrix_error("solve"));
                    }
                }
                let result = lu.solve(&b_mat);

                for j in 0..nrhs {
                    for i in 0..n {
                        let value = result[(i, j)];
                        if !value.is_finite() {
                            return Err(non_finite_result_error("solve"));
                        }
                        x[i + j * n] = value;
                    }
                }

                Ok(())
            }

            fn solve_triangular(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                upper: bool,
                x: &mut [$ty],
            ) -> Result<()> {
                check_len("solve_triangular", "a", a.len(), n * n)?;
                check_len("solve_triangular", "b", b.len(), n * nrhs)?;
                check_len("solve_triangular", "x", x.len(), n * nrhs)?;

                let a_mat = faer::MatRef::from_column_major_slice(a, n, n);
                for col in 0..nrhs {
                    let b_col = &b[col * n..(col + 1) * n];
                    let x_col = &mut x[col * n..(col + 1) * n];

                    if upper {
                        for i in (0..n).rev() {
                            let mut sum = b_col[i];
                            for j in (i + 1)..n {
                                sum -= a_mat[(i, j)] * x_col[j];
                            }
                            let diag = a_mat[(i, i)];
                            if !diag.is_finite() || diag == 0.0 {
                                return Err(zero_diagonal_error("solve_triangular", i));
                            }
                            let value = sum / diag;
                            if !value.is_finite() {
                                return Err(non_finite_result_error("solve_triangular"));
                            }
                            x_col[i] = value;
                        }
                    } else {
                        for i in 0..n {
                            let mut sum = b_col[i];
                            for j in 0..i {
                                sum -= a_mat[(i, j)] * x_col[j];
                            }
                            let diag = a_mat[(i, i)];
                            if !diag.is_finite() || diag == 0.0 {
                                return Err(zero_diagonal_error("solve_triangular", i));
                            }
                            let value = sum / diag;
                            if !value.is_finite() {
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
                a: &[$ty],
                n: usize,
                values_ri: &mut [$ty],
                vectors_ri: &mut [$ty],
            ) -> Result<()> {
                use faer::c64;

                check_len("eig_general", "a", a.len(), n * n)?;
                check_len("eig_general", "values_ri", values_ri.len(), 2 * n)?;
                check_len("eig_general", "vectors_ri", vectors_ri.len(), 2 * n * n)?;

                let a_complex: Vec<c64> = a[..n * n]
                    .iter()
                    .map(|&v| c64::new(v as f64, 0.0))
                    .collect();
                let mat = faer::MatRef::from_column_major_slice(&a_complex, n, n);
                let eig = mat.eigen().map_err(|e| {
                    Error::InvalidArgument(format!("eigendecomposition failed: {e:?}"))
                })?;

                let s_diag = eig.S();
                let s_col = s_diag.column_vector();
                let u_ref = eig.U();

                for i in 0..n {
                    let val = s_col[i];
                    values_ri[2 * i] = val.re as $ty;
                    values_ri[2 * i + 1] = val.im as $ty;
                }

                for j in 0..n {
                    for i in 0..n {
                        let val = u_ref[(i, j)];
                        vectors_ri[2 * (i + j * n)] = val.re as $ty;
                        vectors_ri[2 * (i + j * n) + 1] = val.im as $ty;
                    }
                }

                Ok(())
            }
        }
    };
}

pub(crate) use impl_linalg_backend;
