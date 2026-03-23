use super::linear_systems_sign::{lu_permutation_matrix_tensor, LiftPermutationMatrixTensor};
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
pub fn svd<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T, T::Real>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::Backend: 'static,
    T::Real: tenferro_tensor::KeepCountScalar,
{
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::thin_svd(ctx, tensor)?;

    let Some(opts) = options else {
        return Ok(SvdResult {
            u: result.u,
            s: result.s,
            vt: result.vt,
        });
    };

    let k = result.s.dims()[0];
    let max_k = opts.max_rank.map_or(k, |r| r.min(k));
    let u = result.u.narrow(1, 0, max_k)?;
    let s = result.s.narrow(0, 0, max_k)?;
    let vt = result.vt.narrow(0, 0, max_k)?;

    if max_k == 0 {
        return Ok(SvdResult { u, s, vt });
    }

    if opts.cutoff.is_none() {
        return Ok(SvdResult { u, s, vt });
    }

    let cutoff_r: T::Real = scalar_from(opts.cutoff.unwrap())?;
    let cutoff_tensor =
        crate::prims_bridge::full_like_constant(cutoff_r, s.dims(), s.logical_memory_space())?;
    let mask = crate::prims_bridge::scalar_binary_same_shape::<T::Real, C>(
        ctx,
        &s,
        &cutoff_tensor,
        tenferro_prims::ScalarBinaryOp::GreaterEqual,
    )?;
    let kept_axes: Vec<usize> = (1..s.ndim()).collect();
    let keep_counts =
        crate::prims_bridge::scalar_sum_keep_axes::<T::Real, C>(ctx, &mask, &kept_axes)?;

    Ok(SvdResult {
        u: backend::tensor_helpers::zero_trailing_by_counts(&u, &keep_counts, 1, 2)?,
        s: backend::tensor_helpers::zero_trailing_by_counts(&s, &keep_counts, 0, 1)?,
        vt: backend::tensor_helpers::zero_trailing_by_counts(&vt, &keep_counts, 0, 2)?,
    })
}

/// Compute singular values only for a batched matrix.
///
/// Input shape: `(m, n, *)`.
///
/// # Examples
///
/// ```
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::svdvals;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::zeros(
///     &[3, 4],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// let _values = svdvals(&mut ctx, &a).unwrap();
/// ```
pub fn svdvals<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T::Real>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::svdvals(ctx, tensor)
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
pub fn qr<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<QrResult<T>>
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
/// assert_eq!(no_pivot.p.dims(), &[0]);
/// ```
pub fn lu<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    pivot: LuPivot,
) -> Result<LuResult<T>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorMetadataContextFor
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::MetadataBackend: tenferro_prims::TensorMetadataPrims<Context = C>,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>>::ScalarBackend:
        tenferro_prims::TensorMetadataCastPrims<T::Real, Context = C>,
    T: LiftPermutationMatrixTensor<C>,
    C::Backend: 'static,
{
    let (m, _n, _batch_dims) = validate_2d(tensor)?;
    if pivot == LuPivot::NoPivot {
        let result =
            <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor_no_pivot(ctx, tensor)?;
        return Ok(LuResult {
            p: Tensor::empty(
                &[0],
                tensor.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?,
            l: result.l,
            u: result.u,
        });
    }

    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let p = lu_permutation_matrix_tensor(ctx, &result.pivots, m)?;

    Ok(LuResult {
        p,
        l: result.l,
        u: result.u,
    })
}

/// Compute the packed LU factorization of a batched matrix.
pub fn lu_factor<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<LuFactorResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    let _ = validate_2d(tensor)?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    Ok(LuFactorResult {
        factors: pack_lu_factors(&result.l, &result.u)?,
        pivots: result.pivots,
    })
}

/// Compute the packed LU factorization with numerical status information.
pub fn lu_factor_ex<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<LuFactorExResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::LuFactorEx, "lu_factor_ex")?;

    let _ = validate_2d(tensor)?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor_ex(ctx, tensor)?;
    let factors = pack_lu_factors(&result.l, &result.u)?;

    Ok(LuFactorExResult {
        factors,
        pivots: result.pivots,
        info: result.info,
    })
}

/// Solve `A x = b` from a packed LU factorization.
pub fn lu_solve<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    factors: &Tensor<T>,
    pivots: &Tensor<i32>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::LuSolve, "lu_solve")?;
    <C::Backend as backend::TensorLinalgBackend<T>>::lu_solve(ctx, factors, pivots, b)
}

/// Compute the eigendecomposition of a batched square matrix.
///
/// Input shape: `(n, n, *)`.
/// The lower triangle is treated as canonical input, matching the default
/// `UPLO='L'` behavior used by PyTorch's `linalg.eigh`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_linalg::eigen;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let a = Tensor::<f64>::from_slice(
///     &[2.0, 1.0, 1.0, 2.0],
///     &[2, 2],
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let result = eigen(&mut ctx, &a).unwrap();
/// assert_eq!(result.values.dims(), &[2]);
/// ```
pub fn eigen<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<EigenResult<T, T::Real>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::EigenSym, "eigen")?;
    let _ = validate_square(tensor)?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::eigen_sym(ctx, tensor)?;

    Ok(EigenResult {
        values: result.values,
        vectors: result.vectors,
    })
}
