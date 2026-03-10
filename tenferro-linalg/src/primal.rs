use super::*;

// ============================================================================
// Primary decomposition functions
// ============================================================================

/// Compute the SVD of a batched matrix.
///
/// Input shape: `(m, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
///
/// # Arguments
///
/// * `tensor` — Input tensor of shape `(m, n, *)`
/// * `options` — Optional truncation parameters
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{svd, SvdOptions};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, col);
///
/// // Full SVD
/// let result = svd(&mut ctx, &a, None).unwrap();
///
/// // Truncated SVD
/// let opts = SvdOptions { max_rank: Some(2), cutoff: None };
/// let result = svd(&mut ctx, &a, Some(&opts)).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn svd<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::thin_svd(ctx, tensor)?;

    let u_input = ensure_col_major(&result.u);
    let s_input = ensure_col_major(&result.s);
    let vt_input = ensure_col_major(&result.vt);
    let u_data_in = extract_slice(&u_input)?;
    let s_data_in = extract_slice(&s_input)?;
    let vt_data_in = extract_slice(&vt_input)?;
    let u_offset = u_input.offset() as usize;
    let s_offset = s_input.offset() as usize;
    let vt_offset = vt_input.offset() as usize;

    let m = result.u.dims()[0];
    let k = result.s.dims()[0];
    let n = result.vt.dims()[1];
    let batch_dims = &result.s.dims()[1..];
    let bc = batch_count(batch_dims);

    // Determine effective rank after truncation
    let Some(opts) = options else {
        return Ok(SvdResult {
            u: result.u,
            s: result.s,
            vt: result.vt,
        });
    };
    let max_k = opts.max_rank.map_or(k, |r| r.min(k));

    let mut u_data = vec![T::zero(); m * max_k * bc];
    let mut s_data = vec![<T::Real>::zero(); max_k * bc];
    let mut vt_data = vec![T::zero(); max_k * n * bc];

    for b in 0..bc {
        let u_full = &u_data_in[u_offset + b * m * k..u_offset + (b + 1) * m * k];
        let s_full = &s_data_in[s_offset + b * k..s_offset + (b + 1) * k];
        let vt_full = &vt_data_in[vt_offset + b * k * n..vt_offset + (b + 1) * k * n];

        // Apply cutoff truncation
        let actual_k = if let Some(cutoff) = opts.cutoff {
            let cutoff_r: T::Real = scalar_from(cutoff)?;
            let mut ak = max_k;
            while ak > 0 && s_full[ak - 1] < cutoff_r {
                ak -= 1;
            }
            ak
        } else {
            max_k
        };

        // Copy U (m × max_k, col-major)
        for j in 0..actual_k {
            for i in 0..m {
                u_data[b * m * max_k + i + j * m] = u_full[i + j * m];
            }
        }

        // Copy S (max_k)
        for i in 0..actual_k {
            s_data[b * max_k + i] = s_full[i];
        }

        // Copy Vt (max_k × n, col-major) from vt_full (k × n, col-major)
        // vt_full is already k×n col-major: vt_full[i + j*k]
        for j in 0..n {
            for i in 0..actual_k {
                vt_data[b * max_k * n + i + j * max_k] = vt_full[i + j * k];
            }
        }
    }

    let u_dims = output_dims(&[m, max_k], batch_dims);
    let s_dims = output_dims(&[max_k], batch_dims);
    let vt_dims = output_dims(&[max_k, n], batch_dims);

    Ok(SvdResult {
        u: tensor_from_data(u_data, &u_dims)?,
        s: tensor_from_data(s_data, &s_dims)?,
        vt: tensor_from_data(vt_data, &vt_dims)?,
    })
}

/// Compute the QR decomposition of a batched matrix.
///
/// Input shape: `(m, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::qr;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[4, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = qr(&mut ctx, &a).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn qr<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<QrResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::qr(ctx, tensor)?;

    Ok(QrResult {
        q: result.q,
        r: result.r,
    })
}

/// Compute the LU decomposition of a batched matrix.
///
/// Input shape: `(m, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
///
/// # Arguments
///
/// * `tensor` — Input tensor of shape `(m, n, *)`
/// * `pivot` — Pivoting strategy: [`LuPivot::Partial`] (default, stable)
///   or [`LuPivot::NoPivot`] (faster, unstable)
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{lu, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::from_slice(
///     &[1.0, 0.0, 0.0, 1.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor
/// ).unwrap();
///
/// // Partial pivoting (default)
/// let result = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
///
/// // NoPivot is supported (no permutation output).
/// let no_pivot = lu(&mut ctx, &a, LuPivot::NoPivot).unwrap();
/// assert!(no_pivot.p.is_none());
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions.
pub fn lu<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    pivot: LuPivot,
) -> Result<LuResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    if pivot == LuPivot::NoPivot {
        let (m, n, batch_dims) = validate_2d(tensor)?;
        let bc = batch_count(batch_dims);
        let k = m.min(n);
        let mat_size = m * n;

        let input = ensure_col_major(tensor);
        let data = extract_slice(&input)?;
        let offset = input.offset() as usize;

        let mut all_l = vec![T::zero(); m * k * bc];
        let mut all_u = vec![T::zero(); k * n * bc];

        for batch in 0..bc {
            let start = offset + batch * mat_size;
            let mut lu_data = data[start..start + mat_size].to_vec();

            // Doolittle LU without pivoting.
            for p in 0..k {
                let pivot_val = lu_data[p + p * m];
                if pivot_val.abs_real() <= T::real_epsilon() {
                    return Err(Error::InvalidArgument(format!(
                        "NoPivot LU encountered near-zero pivot at row {p} in batch {batch}"
                    )));
                }

                for i in (p + 1)..m {
                    lu_data[i + p * m] = lu_data[i + p * m] / pivot_val;
                }
                for j in (p + 1)..n {
                    let up = lu_data[p + j * m];
                    for i in (p + 1)..m {
                        let idx = i + j * m;
                        lu_data[idx] = lu_data[idx] - lu_data[i + p * m] * up;
                    }
                }
            }

            for j in 0..k {
                for i in 0..m {
                    let val = if i < j {
                        T::zero()
                    } else if i == j {
                        T::one()
                    } else {
                        lu_data[i + j * m]
                    };
                    all_l[batch * m * k + i + j * m] = val;
                }
            }
            for j in 0..n {
                for i in 0..k {
                    let val = if i <= j {
                        lu_data[i + j * m]
                    } else {
                        T::zero()
                    };
                    all_u[batch * k * n + i + j * k] = val;
                }
            }
        }

        let l_dims = output_dims(&[m, k], batch_dims);
        let u_dims = output_dims(&[k, n], batch_dims);
        return Ok(LuResult {
            p: None,
            l: tensor_from_data(all_l, &l_dims)?,
            u: tensor_from_data(all_u, &u_dims)?,
        });
    }

    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;

    Ok(LuResult {
        p: Some(result.pivots.into_iter().map(|p| p as usize).collect()),
        l: result.l,
        u: result.u,
    })
}

/// Compute the packed LU factorization of a batched matrix.
///
/// The returned `factors` tensor has the same shape as the input. Its strict
/// lower-triangular part stores the multipliers for `L`, and its diagonal plus
/// upper-triangular part stores `U`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lu_factor;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = lu_factor(&mut ctx, &a).unwrap();
/// assert_eq!(result.factors.dims(), &[2, 2]);
/// assert_eq!(result.pivots.len(), 2);
/// ```
pub fn lu_factor<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<LuFactorResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = lu_factor_impl(ctx, tensor)?;
    Ok(LuFactorResult {
        factors: result.factors,
        pivots: result.pivots,
    })
}

/// Compute the packed LU factorization with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lu_factor_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = lu_factor_ex(&mut ctx, &a).unwrap();
/// assert_eq!(result.factors.dims(), &[2, 2]);
/// assert_eq!(result.info, vec![0]);
/// ```
pub fn lu_factor_ex<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<LuFactorExResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    lu_factor_impl(ctx, tensor)
}

/// Solve `A x = b` from a packed LU factorization.
///
/// `factors` and `pivots` should come from [`lu_factor`] or [`lu_factor_ex`].
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{lu_factor, lu_solve};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[3.0_f64, 1.0, 1.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let b = Tensor::from_slice(&[9.0_f64, 8.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let lu = lu_factor(&mut ctx, &a).unwrap();
/// let x = lu_solve(&mut ctx, &lu.factors, &lu.pivots, &b).unwrap();
/// assert_eq!(x.dims(), &[2]);
/// ```
pub fn lu_solve<T: LinalgScalar, C>(
    ctx: &mut C,
    factors: &Tensor<T>,
    pivots: &[usize],
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("lu_solve")?;
    lu_solve_impl(ctx, factors, pivots, b)
}

/// Compute the eigendecomposition of a batched square matrix.
///
/// Input shape: `(n, n, *)`.
///
/// The function internally normalizes input to column-major contiguous layout.
/// If the input is not already contiguous, an internal copy is performed.
///
/// `eigen` uses a symmetric/Hermitian eigensolver and validates
/// `A[i, j] == A[j, i]` (within floating-point tolerance) for each batch.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::eigen;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = eigen(&mut ctx, &a).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions, the first two
/// dimensions are not equal, or the matrix is not symmetric/Hermitian.
pub fn eigen<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<EigenResult<T, T::Real>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    validate_hermitian_batches(data, offset, n, bc, "eigen")?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::eigen_sym(ctx, tensor)?;

    Ok(EigenResult {
        values: result.values,
        vectors: result.vectors,
    })
}

/// Solve the least squares problem: `x = argmin ||Ax - b||²`.
///
/// Input shapes: `A` is `(m, n, *)`, `b` is `(m, *)`, with `m >= n`.
/// The function internally normalizes inputs to column-major contiguous layout.
/// If inputs are not already contiguous, internal copies are performed.
///
/// Internally computes `x = R⁻¹ Q† b` via thin QR decomposition of `A`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::lstsq;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], col).unwrap();
/// let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let result = lstsq(&mut ctx, &a, &b).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if `A` has fewer than 2 dimensions, `m < n`, or `b`
/// does not match `(m, *)` with the same batch dimensions as `A`.
pub fn lstsq<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<LstsqResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("lstsq")?;

    let (m, n, batch_dims) = validate_2d(a)?;
    if m < n {
        return Err(Error::InvalidArgument(format!(
            "lstsq requires m >= n, got m={m}, n={n}"
        )));
    }
    validate_lstsq_rhs(b, m, batch_dims)?;

    // Solve via QR: A = Q R, then x = R^{-1} Q^T b
    let qr_result = qr(ctx, a)?;
    let q_input = ensure_col_major(&qr_result.q);
    let r_input = ensure_col_major(&qr_result.r);
    let b_input = ensure_col_major(b);

    let q_data = extract_slice(&q_input)?;
    let r_data = extract_slice(&r_input)?;
    let b_data = extract_slice(&b_input)?;
    let q_off = q_input.offset() as usize;
    let r_off = r_input.offset() as usize;
    let b_off = b_input.offset() as usize;

    let k = m.min(n); // = n since m >= n
    let bc = batch_count(batch_dims);

    let mut x_data = vec![T::zero(); n * bc];
    let mut res_data = vec![T::zero(); m * bc];

    let mut x_buf = vec![T::zero(); k];

    for batch in 0..bc {
        let q_b = &q_data[q_off + batch * m * k..q_off + (batch + 1) * m * k];
        let r_b = &r_data[r_off + batch * k * n..r_off + (batch + 1) * k * n];
        let b_b = &b_data[b_off + batch * m..b_off + (batch + 1) * m];

        // Compute Q^T b (k × 1)
        let mut qtb = vec![T::zero(); k];
        for i in 0..k {
            let mut sum = T::zero();
            for j in 0..m {
                sum = sum + q_b[j + i * m] * b_b[j];
            }
            qtb[i] = sum;
        }

        // Solve R x = Q^T b (upper triangular)
        backend::cpu::solve_triangular_slices(r_b, &qtb, k, 1, true, &mut x_buf)?;
        x_data[batch * n..(batch + 1) * n].copy_from_slice(&x_buf);

        // Compute residual: r = b - A x
        let a_contiguous = a.contiguous(MemoryOrder::ColumnMajor);
        let a_slice = extract_slice(&a_contiguous)?;
        let a_off = a_contiguous.offset() as usize;
        let a_data_local = &a_slice[a_off + batch * m * n..a_off + (batch + 1) * m * n];
        for i in 0..m {
            let mut ax_i = T::zero();
            for j in 0..n {
                ax_i = ax_i + a_data_local[i + j * m] * x_buf[j];
            }
            res_data[batch * m + i] = b_b[i] - ax_i;
        }
    }

    let x_dims = output_dims(&[n], batch_dims);
    let res_dims = output_dims(&[m], batch_dims);

    Ok(LstsqResult {
        x: tensor_from_data(x_data, &x_dims)?,
        residual: tensor_from_data(res_data, &res_dims)?,
    })
}

/// Compute the Cholesky decomposition of a Hermitian positive-definite matrix.
///
/// Input shape: `(n, n, *)`. Returns lower triangular `L` such that `A = L L†`.
///
/// # Examples
///
/// ```no_run
/// use tenferro_linalg::cholesky;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let l = cholesky(&mut ctx, &a).unwrap();
/// ```
pub fn cholesky<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::cholesky(ctx, tensor)
}

/// Compute the Cholesky decomposition with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cholesky_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[4.0_f64, 2.0, 2.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = cholesky_ex(&mut ctx, &a).unwrap();
/// assert_eq!(result.l.dims(), &[2, 2]);
/// assert_eq!(result.info, vec![0]);
/// ```
pub fn cholesky_ex<T: LinalgScalar, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<CholeskyExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("cholesky_ex")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let mat_size = n * n;

    let mut factors = vec![T::zero(); mat_size * bc];
    let mut info = vec![0_i32; bc];

    for batch in 0..bc {
        let start = offset + batch * mat_size;
        let a_slice = &data[start..start + mat_size];
        let l_out = &mut factors[batch * mat_size..(batch + 1) * mat_size];
        if backend::cpu::cholesky_slices(a_slice, n, l_out).is_err() {
            l_out.fill(T::zero());
            info[batch] = 1;
        }
    }

    Ok(CholeskyExResult {
        l: tensor_from_data(factors, &output_dims(&[n, n], batch_dims))?,
        info,
    })
}

/// Solve a square linear system `A x = b`.
///
/// Input shapes: `A` is `(n, n, *)`, `b` is `(n, *)` or `(n, k, *)`.
/// Batch dimensions in `b` must match those of `A`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3, mem, col);
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let x = solve(&mut ctx, &a, &b).unwrap();
/// ```
pub fn solve<T: LinalgScalar, C>(ctx: &mut C, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve(ctx, a, b)
}

/// Solve a square linear system with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let b = Tensor::from_slice(&[2.0_f64, -1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let result = solve_ex(&mut ctx, &a, &b).unwrap();
/// assert_eq!(result.solution.dims(), &[2]);
/// assert_eq!(result.info, vec![0]);
/// ```
pub fn solve_ex<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<SolveExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("solve_ex")?;

    let (n, batch_dims) = validate_square(a)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "solve_ex")?;
    let bc = batch_count(batch_dims);

    let a_input = ensure_col_major(a);
    let b_input = ensure_col_major(b);
    let a_data = extract_slice(&a_input)?;
    let b_data = extract_slice(&b_input)?;
    let a_offset = a_input.offset() as usize;
    let b_offset = b_input.offset() as usize;

    let mat_size = n * n;
    let rhs_size = n * rhs.nrhs;
    let mut solution = vec![T::zero(); rhs_size * bc];
    let mut info = vec![0_i32; bc];

    for batch in 0..bc {
        let a_start = a_offset + batch * mat_size;
        let b_start = b_offset + batch * rhs_size;
        let a_slice = &a_data[a_start..a_start + mat_size];
        let b_slice = &b_data[b_start..b_start + rhs_size];
        let x_out = &mut solution[batch * rhs_size..(batch + 1) * rhs_size];
        if backend::cpu::solve_slices(a_slice, b_slice, n, rhs.nrhs, x_out).is_err() {
            x_out.fill(T::zero());
            info[batch] = 1;
        }
    }

    Ok(SolveExResult {
        solution: tensor_from_data(solution, &rhs.output_dims)?,
        info,
    })
}

/// Compute the inverse of a square matrix.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::eye(3,
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let a_inv = inv(&mut ctx, &a).unwrap();
/// ```
pub fn inv<T: LinalgScalar, C>(_ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("inv")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    // Solve A * X = I for each batch
    let mut eye_mat = vec![T::zero(); n * n];
    for i in 0..n {
        eye_mat[i + i * n] = T::one();
    }

    let mut inv_data = vec![T::zero(); n * n * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let a_b = &data[start..start + mat_size];
        let x_out = &mut inv_data[b * mat_size..(b + 1) * mat_size];
        backend::cpu::solve_slices(a_b, &eye_mat, n, n, x_out)?;
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(inv_data, &dims)
}

/// Compute the inverse with numerical status information.
///
/// `info` contains one entry per batch matrix. Zero indicates success.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::inv_ex;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let result = inv_ex(&mut ctx, &a).unwrap();
/// assert_eq!(result.inverse.dims(), &[2, 2]);
/// assert_eq!(result.info, vec![0]);
/// ```
pub fn inv_ex<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<InvExResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let mut eye_data = vec![T::zero(); n * n * bc];
    let eye = identity_matrix::<T>(n);
    for batch in 0..bc {
        eye_data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&eye);
    }
    let rhs = tensor_from_data(eye_data, &output_dims(&[n, n], batch_dims))?;
    let result = solve_ex(ctx, tensor, &rhs)?;
    Ok(InvExResult {
        inverse: result.solution,
        info: result.info,
    })
}

/// Compute the determinant of a square matrix.
///
/// Input shape: `(n, n, *)`. Returns shape `(*)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::det;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let d = det(&mut ctx, &a).unwrap();
/// ```
pub fn det<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("det")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut det_data = vec![T::zero(); bc];

    // Pre-allocate temp buffers for LU per batch
    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); n * n];
    let mut u_buf = vec![T::zero(); n * n];

    for (b, det_slot) in det_data.iter_mut().enumerate().take(bc) {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        // det = product of diagonal of U * sign from permutation
        backend::cpu::lu_slices(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

        let mut d = T::one();
        for i in 0..n {
            d = d * u_buf[i + i * n]; // U diagonal
        }

        // Count transpositions in permutation
        let mut sign = 1i32;
        let mut visited = vec![false; n];
        for i in 0..n {
            if !visited[i] {
                visited[i] = true;
                let mut j = perm[i];
                while j != i {
                    sign = -sign;
                    visited[j] = true;
                    j = perm[j];
                }
            }
        }

        if sign < 0 {
            d = T::zero() - d;
        }
        *det_slot = d;
    }

    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    if dims.is_empty() {
        // Scalar result: shape []
        let strides = vec![];
        Tensor::from_vec(det_data, &dims, &strides, 0)
    } else {
        tensor_from_data(det_data, &dims)
    }
}

/// Compute sign and log-absolute-determinant of a square matrix.
///
/// Numerically stable alternative to [`det`]. `det(A) = sign * exp(logabsdet)`.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::slogdet;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = slogdet(&mut ctx, &a).unwrap();
/// // det(A) ≈ result.sign * exp(result.logabsdet)
/// ```
pub fn slogdet<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<SlogdetResult<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("slogdet")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut sign_data = vec![T::zero(); bc];
    let mut logabsdet_data = vec![T::zero(); bc];

    // Pre-allocate temp buffers for LU per batch
    let mut perm = vec![0usize; n];
    let mut l_buf = vec![T::zero(); n * n];
    let mut u_buf = vec![T::zero(); n * n];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];

        backend::cpu::lu_slices(batch_data, n, n, &mut perm, &mut l_buf, &mut u_buf)?;

        let mut log_abs = T::zero();
        let mut sign = T::one();
        for i in 0..n {
            let diag = u_buf[i + i * n];
            log_abs = log_abs + diag.abs().ln();
            if diag < T::zero() {
                sign = T::zero() - sign;
            }
        }

        // Count transpositions
        let mut perm_sign = 1i32;
        let mut visited = vec![false; n];
        for i in 0..n {
            if !visited[i] {
                visited[i] = true;
                let mut j = perm[i];
                while j != i {
                    perm_sign = -perm_sign;
                    visited[j] = true;
                    j = perm[j];
                }
            }
        }
        if perm_sign < 0 {
            sign = T::zero() - sign;
        }

        sign_data[b] = sign;
        logabsdet_data[b] = log_abs;
    }

    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    if dims.is_empty() {
        let strides = vec![];
        Ok(SlogdetResult {
            sign: Tensor::from_vec(sign_data, &dims, &strides, 0)?,
            logabsdet: Tensor::from_vec(logabsdet_data, &dims, &strides, 0)?,
        })
    } else {
        Ok(SlogdetResult {
            sign: tensor_from_data(sign_data, &dims)?,
            logabsdet: tensor_from_data(logabsdet_data, &dims)?,
        })
    }
}

/// Compute the eigendecomposition of a general (non-symmetric) square matrix.
///
/// Unlike [`eigen`] (which requires Hermitian/symmetric input and returns
/// real eigenvalues), this function handles general matrices. Eigenvalues
/// and eigenvectors are always returned as complex, since a general real
/// matrix can have complex eigenvalue pairs.
///
/// Input shape: `(n, n, *)`.
///
/// Returns [`EigResult`] with complex eigenvalues (shape `(n, *)`) and
/// complex right eigenvectors (shape `(n, n, *)`).
///
/// # Examples
///
/// ```
/// use tenferro_linalg::eig;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let result = eig(&mut ctx, &a).unwrap();
/// assert_eq!(result.values.dims(), &[3]);
/// assert_eq!(result.vectors.dims(), &[3, 3]);
/// ```
///
/// # Errors
///
/// Returns an error if the input has fewer than 2 dimensions or the first
/// two dimensions are not equal.
pub fn eig<T: LinalgScalar<Real = T, Complex = num_complex::Complex<T>> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<EigResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::eig(ctx, tensor)?;

    Ok(EigResult {
        values: result.values,
        vectors: result.vectors,
    })
}

pub(crate) fn ensure_cpu_backend<T: LinalgScalar, C>(op: &str) -> Result<()>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
    C::Backend: 'static,
{
    if TypeId::of::<C::Backend>() == TypeId::of::<backend::CpuTensorLinalgBackend>() {
        return Ok(());
    }

    Err(Error::DeviceError(format!(
        "{op} is currently supported only on CpuContext"
    )))
}

/// Compute the Moore-Penrose pseudoinverse of a matrix.
///
/// Computed via SVD: `pinv(A) = V diag(1/S) U†`, with singular values
/// below a threshold treated as zero.
///
/// Input shape: `(m, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::pinv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let a_pinv = pinv(&mut ctx, &a, None).unwrap();
/// ```
pub fn pinv<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    rcond: Option<f64>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("pinv")?;

    let (m, n, batch_dims) = validate_2d(tensor)?;

    // Compute via SVD: pinv(A) = V diag(1/S) U^T
    let svd_result = svd(ctx, tensor, None)?;
    let u_input = ensure_col_major(&svd_result.u);
    let s_input = ensure_col_major(&svd_result.s);
    let vt_input = ensure_col_major(&svd_result.vt);

    let u_data = extract_slice(&u_input)?;
    let s_data = extract_slice(&s_input)?;
    let vt_data = extract_slice(&vt_input)?;
    let u_off = u_input.offset() as usize;
    let s_off = s_input.offset() as usize;
    let vt_off = vt_input.offset() as usize;

    let k = m.min(n);
    let bc = batch_count(batch_dims);
    // Default threshold: 1e-15 matches NumPy/Julia convention for f64
    // (approximately 4.5 × machine epsilon). Singular values below
    // `s_max * threshold` are treated as zero.
    let threshold: T = scalar_from(rcond.unwrap_or(1e-15))?;

    let mut result_data = vec![T::zero(); n * m * bc];

    for b in 0..bc {
        let s_b = &s_data[s_off + b * k..s_off + (b + 1) * k];
        let u_b = &u_data[u_off + b * m * k..u_off + (b + 1) * m * k];
        let vt_b = &vt_data[vt_off + b * k * n..vt_off + (b + 1) * k * n];

        let s_max = s_b
            .iter()
            .copied()
            .fold(T::zero(), |a, b| if a > b { a } else { b });
        let cutoff = s_max * threshold;

        // Build diag(1/S) U^T (k × m): element [i,j] = (1/s_i) * U[j,i]
        let mut sinv_ut = vec![T::zero(); k * m];
        for i in 0..k {
            // Division is safe: s_b[i] > cutoff > 0 guarantees s_b[i] is
            // bounded away from zero by at least `s_max * threshold`.
            if s_b[i] > cutoff {
                let sinv = T::one() / s_b[i];
                for j in 0..m {
                    sinv_ut[i + j * k] = sinv * u_b[j + i * m];
                }
            }
        }

        // Compute V * sinv_ut = Vt^T * sinv_ut
        // V is n×k (stored as Vt transposed): V[i,j] = Vt[j,i] = vt_b[j + i*k]
        for j in 0..m {
            for i in 0..n {
                let mut sum = T::zero();
                for p in 0..k {
                    // V[i,p] = vt_b[p + i*k]
                    sum = sum + vt_b[p + i * k] * sinv_ut[p + j * k];
                }
                result_data[b * n * m + i + j * n] = sum;
            }
        }
    }

    let dims = output_dims(&[n, m], batch_dims);
    tensor_from_data(result_data, &dims)
}

/// Compute the matrix exponential `exp(A)` of a square matrix.
///
/// Uses the scaling-and-squaring method with Pad\u{e9}\[13/13\] approximation
/// (Al-Mohy & Higham, 2010), following the PyTorch approach.
///
/// Input shape: `(n, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_exp;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 3],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let exp_a = matrix_exp(&mut ctx, &a).unwrap();
/// // exp(0) = I
/// ```
pub fn matrix_exp<T: LinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("matrix_exp")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = n * n;

    let mut result_data = vec![T::zero(); mat_size * bc];

    for b in 0..bc {
        let start = offset + b * mat_size;
        let a_slice = &data[start..start + mat_size];
        let exp_a = matrix_exp_single(ctx, a_slice, n)?;
        result_data[b * mat_size..(b + 1) * mat_size].copy_from_slice(&exp_a);
    }

    let dims = output_dims(&[n, n], batch_dims);
    tensor_from_data(result_data, &dims)
}

/// Raise a square matrix to an integer power.
///
/// Negative exponents are supported for invertible matrices.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::matrix_power;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let a3 = matrix_power(&mut ctx, &a, 3).unwrap();
/// assert_eq!(a3.dims(), &[2, 2]);
/// ```
pub fn matrix_power<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    exponent: i64,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("matrix_power")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let dims = output_dims(&[n, n], batch_dims);

    if exponent == 0 {
        let eye = identity_matrix::<T>(n);
        let mut data = vec![T::zero(); n * n * bc];
        for batch in 0..bc {
            data[batch * n * n..(batch + 1) * n * n].copy_from_slice(&eye);
        }
        return tensor_from_data(data, &dims);
    }

    let positive_exponent = if exponent < 0 {
        let abs = exponent.checked_abs().ok_or_else(|| {
            Error::InvalidArgument("matrix_power does not support i64::MIN exponent".into())
        })?;
        let inverse = inv(ctx, tensor)?;
        return matrix_power(ctx, &inverse, abs);
    } else {
        exponent as u64
    };

    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;
    let mat_size = n * n;
    let mut out = vec![T::zero(); mat_size * bc];

    for batch in 0..bc {
        let start = offset + batch * mat_size;
        let a_slice = &data[start..start + mat_size];
        let powered = matrix_power_single(ctx, a_slice, n, positive_exponent)?;
        out[batch * mat_size..(batch + 1) * mat_size].copy_from_slice(&powered);
    }

    tensor_from_data(out, &dims)
}

/// Compute the cross product along the leading vector axis.
///
/// Inputs must have shape `(3, *)` and identical dimensions. The cross product
/// is evaluated independently over every trailing index.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cross;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
/// let b = Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
/// let c = cross(&mut ctx, &a, &b).unwrap();
/// assert_eq!(c.dims(), &[3]);
/// ```
pub fn cross<T: LinalgScalar, C>(_ctx: &mut C, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("cross")?;

    if a.ndim() != b.ndim() {
        return Err(Error::InvalidArgument(format!(
            "cross expects matching ranks, got {:?} and {:?}",
            a.dims(),
            b.dims()
        )));
    }
    if a.ndim() == 0 || a.dims()[0] != 3 {
        return Err(Error::InvalidArgument(format!(
            "cross expects leading vector dimension of size 3, got {:?}",
            a.dims()
        )));
    }
    if b.ndim() == 0 || b.dims()[0] != 3 {
        return Err(Error::InvalidArgument(format!(
            "cross expects leading vector dimension of size 3, got {:?}",
            b.dims()
        )));
    }
    let mut out_dims = vec![3];
    for axis in 1..a.ndim() {
        let lhs = a.dims()[axis];
        let rhs = b.dims()[axis];
        if lhs != rhs && lhs != 1 && rhs != 1 {
            return Err(Error::InvalidArgument(format!(
                "cross broadcast mismatch on axis {axis}: left={}, right={}",
                lhs, rhs
            )));
        }
        out_dims.push(lhs.max(rhs));
    }

    let a_input = ensure_col_major(a);
    let b_input = ensure_col_major(b);
    let a_data = extract_slice(&a_input)?;
    let b_data = extract_slice(&b_input)?;
    let a_offset = a_input.offset() as usize;
    let b_offset = b_input.offset() as usize;
    let lanes = out_dims[1..].iter().product::<usize>().max(1);
    let out_strides = backend::col_major_strides(&out_dims);
    let a_strides = backend::col_major_strides(a.dims());
    let b_strides = backend::col_major_strides(b.dims());
    let mut out = vec![T::zero(); out_dims.iter().product()];
    let mut index = vec![0usize; out_dims.len().saturating_sub(1)];

    for _lane in 0..lanes {
        let mut a_tail_offset = 0isize;
        let mut b_tail_offset = 0isize;
        let mut out_tail_offset = 0isize;
        for axis in 1..out_dims.len() {
            let coord = index[axis - 1];
            out_tail_offset += coord as isize * out_strides[axis];
            let a_coord = if a.dims()[axis] == 1 { 0 } else { coord };
            let b_coord = if b.dims()[axis] == 1 { 0 } else { coord };
            a_tail_offset += a_coord as isize * a_strides[axis];
            b_tail_offset += b_coord as isize * b_strides[axis];
        }

        let a_base = (a_offset as isize + a_tail_offset) as usize;
        let b_base = (b_offset as isize + b_tail_offset) as usize;
        let o_base = out_tail_offset as usize;
        let ax = a_data[a_base];
        let ay = a_data[a_base + 1];
        let az = a_data[a_base + 2];
        let bx = b_data[b_base];
        let by = b_data[b_base + 1];
        let bz = b_data[b_base + 2];
        out[o_base] = ay * bz - az * by;
        out[o_base + 1] = az * bx - ax * bz;
        out[o_base + 2] = ax * by - ay * bx;

        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_dims[axis + 1] {
                break;
            }
            index[axis] = 0;
        }
    }

    tensor_from_data(out, &out_dims)
}

/// Form the explicit product of Householder reflectors.
///
/// `a` stores reflector vectors in the standard QR compact format with shape
/// `(m, n, *)`. `tau` stores the reflector coefficients with shape `(k, *)`,
/// where `k <= min(m, n)`.
///
/// The result has shape `(m, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::householder_product;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
///     &[4, 2],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let tau = Tensor::from_slice(&[0.0_f64, 0.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let q = householder_product(&mut ctx, &a, &tau).unwrap();
/// assert_eq!(q.dims(), &[4, 2]);
/// ```
pub fn householder_product<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &Tensor<T>,
    tau: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("householder_product")?;

    let (m, n, batch_dims) = validate_2d(a)?;
    if tau.ndim() != 1 + batch_dims.len() {
        return Err(Error::InvalidArgument(format!(
            "householder_product expects tau shape (k, *), got {:?}",
            tau.dims()
        )));
    }
    if &tau.dims()[1..] != batch_dims {
        return Err(Error::InvalidArgument(format!(
            "householder_product batch dims mismatch: expected {:?}, got {:?}",
            batch_dims,
            &tau.dims()[1..]
        )));
    }

    let k = tau.dims()[0];
    if k > m.min(n) {
        return Err(Error::InvalidArgument(format!(
            "householder_product expects tau length <= min(m, n) = {}, got {}",
            m.min(n),
            k
        )));
    }

    let a_input = ensure_col_major(a);
    let tau_input = ensure_col_major(tau);
    let a_data = extract_slice(&a_input)?;
    let tau_data = extract_slice(&tau_input)?;
    let a_offset = a_input.offset() as usize;
    let tau_offset = tau_input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mat_size = m * n;
    let mut out = vec![T::zero(); mat_size * bc];

    for batch in 0..bc {
        let a_start = a_offset + batch * mat_size;
        let tau_start = tau_offset + batch * k;
        let a_batch = &a_data[a_start..a_start + mat_size];
        let tau_batch = &tau_data[tau_start..tau_start + k];
        let q_batch = &mut out[batch * mat_size..(batch + 1) * mat_size];

        for col in 0..n {
            if col < m {
                q_batch[col * m + col] = T::one();
            }
        }

        for reflector in (0..k).rev() {
            let tau_i = tau_batch[reflector];
            if tau_i == T::zero() {
                continue;
            }
            for col in 0..n {
                let mut proj = q_batch[reflector + col * m];
                for row in (reflector + 1)..m {
                    proj = proj + a_batch[row + reflector * m].conj() * q_batch[row + col * m];
                }
                proj = tau_i * proj;
                q_batch[reflector + col * m] = q_batch[reflector + col * m] - proj;
                for row in (reflector + 1)..m {
                    q_batch[row + col * m] =
                        q_batch[row + col * m] - a_batch[row + reflector * m] * proj;
                }
            }
        }
    }

    tensor_from_data(out, &output_dims(&[m, n], batch_dims))
}

/// Build a Vandermonde matrix from leading-dimension vectors.
///
/// If `columns` is `None`, the output uses as many columns as the input vector
/// length. For scalar input, the leading vector length is treated as `1`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::vander;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let v = vander(&mut ctx, &x, Some(3), true).unwrap();
/// assert_eq!(v.dims(), &[2, 3]);
/// ```
pub fn vander<T: LinalgScalar, C>(
    _ctx: &mut C,
    x: &Tensor<T>,
    columns: Option<usize>,
    increasing: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("vander")?;

    let (vector_len, batch_dims): (usize, &[usize]) = if x.ndim() == 0 {
        (1, &[])
    } else {
        (x.dims()[0], &x.dims()[1..])
    };
    let columns = columns.unwrap_or(vector_len);

    let x_input = ensure_col_major(x);
    let x_data = extract_slice(&x_input)?;
    let x_offset = x_input.offset() as usize;
    let bc = batch_count(batch_dims);
    let mut out = vec![T::zero(); vector_len * columns * bc];

    for batch in 0..bc {
        let vector = if x.ndim() == 0 {
            &x_data[x_offset..x_offset + 1]
        } else {
            let start = x_offset + batch * vector_len;
            &x_data[start..start + vector_len]
        };
        for row in 0..vector_len {
            let value = vector[row];
            let mut powers = vec![T::one(); columns];
            for col in 1..columns {
                powers[col] = powers[col - 1] * value;
            }
            for col in 0..columns {
                let power = if increasing {
                    powers[col]
                } else {
                    powers[columns.saturating_sub(col + 1)]
                };
                out[batch * vector_len * columns + row + col * vector_len] = power;
            }
        }
    }

    tensor_from_data(out, &output_dims(&[vector_len, columns], batch_dims))
}

/// Invert a tensorized square operator.
///
/// `ind` splits the tensor shape into `(left_dims, right_dims)` and requires
/// `prod(left_dims) == prod(right_dims)`. The output shape is
/// `(right_dims..., left_dims...)`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::tensorinv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let eye = Tensor::from_slice(
///     &[1.0_f64, 0.0, 0.0, 1.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let a = eye.reshape(&[1, 2, 1, 2]).unwrap();
/// let inv = tensorinv(&mut ctx, &a, 2).unwrap();
/// assert_eq!(inv.dims(), &[1, 2, 1, 2]);
/// ```
pub fn tensorinv<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    ind: usize,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("tensorinv")?;

    if ind == 0 || ind >= tensor.ndim() {
        return Err(Error::InvalidArgument(format!(
            "tensorinv expects 0 < ind < rank, got ind={ind} for shape {:?}",
            tensor.dims()
        )));
    }

    let left_dims = &tensor.dims()[..ind];
    let right_dims = &tensor.dims()[ind..];
    let left_prod = left_dims.iter().product::<usize>();
    let right_prod = right_dims.iter().product::<usize>();
    if left_prod != right_prod {
        return Err(Error::InvalidArgument(format!(
            "tensorinv requires prod(shape[..ind]) == prod(shape[ind..]); got {} and {} for {:?}",
            left_prod,
            right_prod,
            tensor.dims()
        )));
    }

    let input = ensure_col_major(tensor);
    let matrix = input.reshape(&[left_prod, right_prod])?;
    let inverse = inv(ctx, &matrix)?;

    let mut out_dims = right_dims.to_vec();
    out_dims.extend_from_slice(left_dims);
    inverse.reshape(&out_dims)
}

/// Solve a tensorized linear system.
///
/// By default the solution uses the trailing `a.ndim() - b.ndim()` axes of `a`.
/// If `dims` is provided, those axes are moved to the end in the given order
/// before solving, and the solution shape follows that order.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::tensorsolve;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let eye = Tensor::from_slice(
///     &[1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
///     &[4, 4],
///     MemoryOrder::ColumnMajor,
/// )
/// .unwrap();
/// let a = eye.reshape(&[2, 2, 2, 2]).unwrap();
/// let b = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let x = tensorsolve(&mut ctx, &a, &b, None).unwrap();
/// assert_eq!(x.dims(), &[2, 2]);
/// ```
pub fn tensorsolve<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    dims: Option<&[usize]>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("tensorsolve")?;

    if b.ndim() > a.ndim() {
        return Err(Error::InvalidArgument(format!(
            "tensorsolve expects b rank <= a rank, got {:?} and {:?}",
            a.dims(),
            b.dims()
        )));
    }

    let solution_rank = a.ndim() - b.ndim();
    let solution_axes = validate_tensor_solve_axes(a.ndim(), solution_rank, dims)?;
    let perm = axes_to_end_permutation(a.ndim(), &solution_axes);
    let a_permuted = if is_identity_permutation(&perm) {
        a.clone()
    } else {
        a.permute(&perm)?
    };

    if &a_permuted.dims()[..b.ndim()] != b.dims() {
        return Err(Error::InvalidArgument(format!(
            "tensorsolve leading dims of permuted a must match b; got {:?} and {:?}",
            a_permuted.dims(),
            b.dims()
        )));
    }

    let lhs_prod = b.dims().iter().product::<usize>();
    let rhs_dims = &a_permuted.dims()[b.ndim()..];
    let rhs_prod = rhs_dims.iter().product::<usize>();
    if lhs_prod != rhs_prod {
        return Err(Error::InvalidArgument(format!(
            "tensorsolve requires matching flattened system size, got {} and {}",
            lhs_prod, rhs_prod
        )));
    }

    let a_contiguous = ensure_col_major(&a_permuted);
    let a_matrix = a_contiguous.reshape(&[lhs_prod, rhs_prod])?;
    let b_contiguous = ensure_col_major(b);
    let b_vector = b_contiguous.reshape(&[lhs_prod])?;
    let x = solve(ctx, &a_matrix, &b_vector)?;
    x.reshape(rhs_dims)
}

/// Solve a triangular linear system `A x = b`.
///
/// `A` must be upper or lower triangular (specified by `upper`).
///
/// Input shapes: `A` is `(n, n, *)`, `b` is `(n, *)` or `(n, k, *)`.
/// Batch dimensions in `b` must match those of `A`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::solve_triangular;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let col = MemoryOrder::ColumnMajor;
/// let mem = LogicalMemorySpace::MainMemory;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(
///     &[2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.0],
///     &[3, 3],
///     col,
/// ).unwrap();
/// let b = Tensor::<f64>::ones(&[3], mem, col);
/// let x = solve_triangular(&mut ctx, &a, &b, true).unwrap(); // upper=true
/// ```
pub fn solve_triangular<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve_triangular(ctx, a, b, upper)
}

/// Compute a norm.
///
/// Supported input shapes:
/// - rank-1 vectors `(n)` for `NormKind::Fro`, `NormKind::L1`, `NormKind::Inf`,
///   and `NormKind::Lp(p)`
/// - matrices `(m, n, *)` for all currently implemented matrix norms
///
/// Supported kinds in the current implementation:
/// - `NormKind::Fro`
/// - `NormKind::Nuclear`
/// - `NormKind::Spectral`
/// - `NormKind::L1` (max absolute column sum)
/// - `NormKind::Inf` (max absolute row sum)
/// - `NormKind::Lp(p)` for vectors
///
/// Return shape is `(*)` (batch dimensions) for matrices. For vectors and
/// non-batched matrices, the result is a scalar tensor `[]`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{norm, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{Tensor, MemoryOrder};
/// use tenferro_device::LogicalMemorySpace;
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4],
///     LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
/// let fro = norm(&mut ctx, &a, NormKind::Fro).unwrap();
/// ```
pub fn norm<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    ensure_cpu_backend::<T, C>("norm")?;

    if tensor.ndim() == 1 {
        let input = ensure_col_major(tensor);
        let offset = input.offset() as usize;
        let len = tensor.dims()[0];
        let vec_data = &extract_slice(&input)?[offset..offset + len];

        let value = match kind {
            NormKind::Fro => {
                let mut sum = T::zero();
                for &v in vec_data {
                    sum = sum + v * v;
                }
                sum.sqrt()
            }
            NormKind::L1 => vec_data.iter().fold(T::zero(), |acc, &v| acc + v.abs()),
            NormKind::Inf => vec_data.iter().fold(T::zero(), |acc, &v| acc.max(v.abs())),
            NormKind::Lp(p) => {
                if p < 1.0 {
                    return Err(invalid_vector_lp_exponent_error(p));
                }
                let (p_t, mut sum) = (scalar_from::<T>(p)?, T::zero());
                for &v in vec_data {
                    sum = sum + v.abs().powf(p_t);
                }
                sum.powf(T::one() / p_t)
            }
            NormKind::Nuclear | NormKind::Spectral => {
                return Err(matrix_only_norm_kind_error(kind));
            }
        };

        return tensor_from_data(vec![value], &[]);
    }

    let (m, n, batch_dims) = validate_2d(tensor)?;
    let bc = batch_count(batch_dims);
    let mat_size = m * n;
    let out_dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };

    let input = ensure_col_major(tensor);
    let data = extract_slice(&input)?;
    let offset = input.offset() as usize;

    match kind {
        NormKind::Fro => {
            // Frobenius norm per batch: sqrt(sum of squares over matrix dims)
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                let start = offset + batch * mat_size;
                let mut sum = T::zero();
                for i in 0..mat_size {
                    let v = data[start + i];
                    sum = sum + v * v;
                }
                *out_slot = sum.sqrt();
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Nuclear => {
            // Nuclear norm per batch: sum of singular values
            let svd_result = svd(ctx, tensor, None)?;
            let s_data = extract_slice(&svd_result.s)?;
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                let mut sum = T::zero();
                let start = s_off + batch * k;
                for i in 0..k {
                    sum = sum + s_data[start + i];
                }
                *out_slot = sum;
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Spectral => {
            // Spectral norm per batch: largest singular value
            let svd_result = svd(ctx, tensor, None)?;
            let s_data = extract_slice(&svd_result.s)?;
            let s_off = svd_result.s.offset() as usize;
            let k = m.min(n);
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                *out_slot = s_data[s_off + batch * k];
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::L1 => {
            // Matrix L1 norm per batch: max absolute column sum.
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    *out_slot = T::zero();
                    continue;
                }
                let start = offset + batch * mat_size;
                let mut max_col_sum = T::zero();
                for j in 0..n {
                    let mut col_sum = T::zero();
                    for i in 0..m {
                        col_sum = col_sum + data[start + i + j * m].abs();
                    }
                    if j == 0 || col_sum > max_col_sum {
                        max_col_sum = col_sum;
                    }
                }
                *out_slot = max_col_sum;
            }
            tensor_from_data(out, &out_dims)
        }
        NormKind::Inf => {
            // Matrix infinity norm per batch: max absolute row sum.
            let mut out = vec![T::zero(); bc];
            for (batch, out_slot) in out.iter_mut().enumerate().take(bc) {
                if m == 0 || n == 0 {
                    *out_slot = T::zero();
                    continue;
                }
                let start = offset + batch * mat_size;
                let mut max_row_sum = T::zero();
                for i in 0..m {
                    let mut row_sum = T::zero();
                    for j in 0..n {
                        row_sum = row_sum + data[start + i + j * m].abs();
                    }
                    if i == 0 || row_sum > max_row_sum {
                        max_row_sum = row_sum;
                    }
                }
                *out_slot = max_row_sum;
            }
            tensor_from_data(out, &out_dims)
        }
        _ => Err(Error::InvalidArgument(format!(
            "norm kind {kind:?} not yet implemented"
        ))),
    }
}

/// Compute the matrix condition number with a selected norm convention.
///
/// Currently supported for square matrices with `NormKind::Fro`,
/// `NormKind::L1`, `NormKind::Inf`, and `NormKind::Spectral`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{cond, NormKind};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 0.5], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let value = cond(&mut ctx, &a, NormKind::Fro).unwrap();
/// assert_eq!(value.dims(), &[]);
/// ```
pub fn cond<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    kind: NormKind,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    match kind {
        NormKind::Fro | NormKind::L1 | NormKind::Inf | NormKind::Spectral => {}
        _ => {
            return Err(Error::InvalidArgument(format!(
                "cond only supports Fro, L1, Inf, and Spectral norms, got {kind:?}"
            )));
        }
    }

    validate_square(tensor)?;
    let lhs = norm(ctx, tensor, kind)?;
    let inverse = inv(ctx, tensor)?;
    let rhs = norm(ctx, &inverse, kind)?;
    let lhs_data = extract_slice(&lhs)?;
    let rhs_data = extract_slice(&rhs)?;
    let lhs_offset = lhs.offset() as usize;
    let rhs_offset = rhs.offset() as usize;
    let len = lhs.dims().iter().product::<usize>().max(1);
    let mut out = vec![T::zero(); len];
    for i in 0..len {
        out[i] = lhs_data[lhs_offset + i] * rhs_data[rhs_offset + i];
    }
    tensor_from_data(out, lhs.dims())
}
