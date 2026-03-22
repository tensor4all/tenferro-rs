use super::*;
use num_traits::One;

fn permutation_sign_from_forward_pivots(pivots: &[usize], n: usize) -> Result<i32> {
    if pivots.len() != n {
        return Err(Error::InvalidArgument(format!(
            "det expects {n} pivots per batch, got {}",
            pivots.len()
        )));
    }

    let mut visited = vec![false; n];
    let mut sign = 1i32;
    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut j = i;
        while !visited[j] {
            visited[j] = true;
            let next = pivots[j];
            if next >= n {
                return Err(Error::InvalidArgument(format!(
                    "det pivot index {next} is out of range for n={n}"
                )));
            }
            if next != i {
                sign = -sign;
            }
            j = next;
        }
    }
    Ok(sign)
}

fn backend_pivots_to_usize(pivots: &[i32]) -> Result<Vec<usize>> {
    pivots
        .iter()
        .map(|&pivot| {
            usize::try_from(pivot).map_err(|_| {
                Error::InvalidArgument(format!(
                    "backend LU pivot {pivot} is negative and cannot be converted to usize"
                ))
            })
        })
        .collect()
}

fn inverse_rhs<T: KernelLinalgScalar>(
    n: usize,
    batch_dims: &[usize],
    memory_space: tenferro_device::LogicalMemorySpace,
) -> Result<Tensor<T>> {
    let mut rhs = Tensor::eye(n, memory_space, MemoryOrder::ColumnMajor);
    for _ in batch_dims {
        rhs = rhs.unsqueeze(-1)?;
    }
    rhs.broadcast(&output_dims(&[n, n], batch_dims))
}

/// Solve a square linear system `A x = b`.
pub fn solve<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    <C::Backend as backend::TensorLinalgBackend<T>>::solve(ctx, a, b)
}

/// Solve a square linear system with numerical status information.
pub fn solve_ex<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<SolveExResult<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::SolveEx, "solve_ex")?;
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::solve_ex(ctx, a, b)?;
    Ok(SolveExResult {
        solution: result.solution,
        info: result.info,
    })
}

/// Compute the inverse of a square matrix.
pub fn inv<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let rhs = inverse_rhs::<T>(n, batch_dims, tensor.logical_memory_space())?;
    if n == 0 {
        return Ok(rhs);
    }
    solve(ctx, tensor, &rhs)
}

/// Compute the inverse with numerical status information.
pub fn inv_ex<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<InvExResult<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Inv, "inv_ex")?;
    let (n, batch_dims) = validate_square(tensor)?;
    let rhs = inverse_rhs::<T>(n, batch_dims, tensor.logical_memory_space())?;
    if n == 0 {
        return Ok(InvExResult {
            inverse: rhs,
            info: vec![0; batch_count(batch_dims)],
        });
    }
    let result = solve_ex(ctx, tensor, &rhs)?;
    Ok(InvExResult {
        inverse: result.solution,
        info: result.info,
    })
}

/// Compute the determinant of a square matrix.
pub fn det<T: KernelLinalgScalar, C>(ctx: &mut C, tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    T: crate::prims_bridge::ScaleTensorByRealSameShape<C>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Det, "det")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let bc = batch_count(batch_dims);
    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };
    let lu = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let diagonal = lu.u.diagonal(&[(0, 1)])?;
    let kept_axes: Vec<usize> = (0..batch_dims.len()).collect();
    let diagonal_prod = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &diagonal,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Prod,
    )?;
    let pivots = backend_pivots_to_usize(&lu.pivots)?;

    let sign_len = if dims.is_empty() { 1 } else { bc };
    let mut sign_data = vec![T::Real::one(); sign_len];
    for batch in 0..bc {
        let sign = permutation_sign_from_forward_pivots(&pivots[batch * n..(batch + 1) * n], n)?;
        if sign < 0 {
            sign_data[batch] = T::Real::zero() - T::Real::one();
        }
    }

    let sign_host = tensor_from_data(sign_data, &dims)?;
    let sign_tensor =
        if tensor.logical_memory_space() == tenferro_device::LogicalMemorySpace::MainMemory {
            sign_host
        } else {
            sign_host.to_memory_space_async(tensor.logical_memory_space())?
        };

    <T as crate::prims_bridge::ScaleTensorByRealSameShape<C>>::scale_tensor_by_real_same_shape(
        ctx,
        &diagonal_prod,
        &sign_tensor,
    )
}

/// Compute sign and log-absolute-determinant of a square matrix.
pub fn slogdet<T: KernelLinalgScalar<Real = T> + num_traits::Float, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<SlogdetResult<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
    <C as tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>>::ScalarBackend:
        'static + tenferro_prims::TensorAnalyticPrims<tenferro_algebra::Standard<T>, Context = C>,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Slogdet, "slogdet")?;

    let (n, batch_dims) = validate_square(tensor)?;
    let lu = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let diagonal = lu.u.diagonal(&[(0, 1)])?;
    let abs_diagonal = crate::prims_bridge::scalar_unary_same_shape(
        ctx,
        &diagonal,
        tenferro_prims::ScalarUnaryOp::Abs,
    )?;
    let logabsdet_factor = crate::prims_bridge::analytic_unary_same_shape(
        ctx,
        &abs_diagonal,
        tenferro_prims::AnalyticUnaryOp::Log,
    )?;
    let kept_axes: Vec<usize> = (0..batch_dims.len()).collect();
    let logabsdet = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &logabsdet_factor,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Sum,
    )?;
    let pivots = backend_pivots_to_usize(&lu.pivots)?;

    let zero_diagonal = crate::prims_bridge::full_like_constant(
        T::zero(),
        diagonal.dims(),
        tensor.logical_memory_space(),
    )?;
    let negative_mask = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &zero_diagonal,
        &diagonal,
        tenferro_prims::ScalarBinaryOp::Greater,
    )?;
    let double_negative = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &negative_mask,
        &negative_mask,
        tenferro_prims::ScalarBinaryOp::Add,
    )?;
    let one = crate::prims_bridge::full_like_constant(
        T::one(),
        diagonal.dims(),
        tensor.logical_memory_space(),
    )?;
    let sign_factors = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &one,
        &double_negative,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;
    let sign_from_diag = crate::prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &sign_factors,
        &kept_axes,
        tenferro_prims::ScalarReductionOp::Prod,
    )?;

    let bc = batch_count(batch_dims);
    let sign_len = if batch_dims.is_empty() { 1 } else { bc };
    let mut sign_data = vec![T::one(); sign_len];
    for batch in 0..bc {
        let sign = permutation_sign_from_forward_pivots(&pivots[batch * n..(batch + 1) * n], n)?;
        if sign < 0 {
            sign_data[batch] = T::zero() - T::one();
        }
    }
    let dims = if batch_dims.is_empty() {
        vec![]
    } else {
        batch_dims.to_vec()
    };
    let sign_perm_host = if dims.is_empty() {
        Tensor::from_vec(sign_data, &dims, &[], 0)?
    } else {
        tensor_from_data(sign_data, &dims)?
    };
    let sign_perm =
        if tensor.logical_memory_space() == tenferro_device::LogicalMemorySpace::MainMemory {
            sign_perm_host
        } else {
            sign_perm_host.to_memory_space_async(tensor.logical_memory_space())?
        };
    let sign = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &sign_perm,
        &sign_from_diag,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;

    Ok(SlogdetResult { sign, logabsdet })
}
