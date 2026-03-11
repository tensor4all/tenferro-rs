use super::*;

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
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::{svd, SvdOptions};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let col = MemoryOrder::ColumnMajor;
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, col);
///
/// let _full = svd(&mut ctx, &a, None).unwrap();
/// let opts = SvdOptions {
///     max_rank: Some(2),
///     cutoff: None,
/// };
/// let _truncated = svd(&mut ctx, &a, Some(&opts)).unwrap();
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

        for j in 0..actual_k {
            for i in 0..m {
                u_data[b * m * max_k + i + j * m] = u_full[i + j * m];
            }
        }
        for i in 0..actual_k {
            s_data[b * max_k + i] = s_full[i];
        }
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
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::qr;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[4, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let _result = qr(&mut ctx, &a).unwrap();
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
/// * `pivot` — Pivoting strategy
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{lu, LuPivot};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
///     .unwrap();
/// let _partial = lu(&mut ctx, &a, LuPivot::Partial).unwrap();
/// let no_pivot = lu(&mut ctx, &a, LuPivot::NoPivot).unwrap();
/// assert!(no_pivot.p.is_none());
/// ```
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
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::LuSolve, "lu_solve")?;
    lu_solve_impl(ctx, factors, pivots, b)
}

/// Compute the eigendecomposition of a batched square matrix.
///
/// Input shape: `(n, n, *)`.
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
