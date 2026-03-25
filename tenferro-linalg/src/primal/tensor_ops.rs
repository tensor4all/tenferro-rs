use super::*;

/// Compute the cross product along the leading vector axis.
///
/// Both inputs must have a leading dimension of size 3.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::cross;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0], &[3], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[0.0, 1.0, 0.0], &[3], col).unwrap();
/// let c = cross(&mut ctx, &a, &b).unwrap();
/// assert_eq!(c.dims(), &[3]);
/// ```
pub fn cross<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
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
    let out_tail_dims = &out_dims[1..];
    let ax = a_input.select(0, 0)?.broadcast(out_tail_dims)?;
    let ay = a_input.select(0, 1)?.broadcast(out_tail_dims)?;
    let az = a_input.select(0, 2)?.broadcast(out_tail_dims)?;
    let bx = b_input.select(0, 0)?.broadcast(out_tail_dims)?;
    let by = b_input.select(0, 1)?.broadcast(out_tail_dims)?;
    let bz = b_input.select(0, 2)?.broadcast(out_tail_dims)?;

    let ay_bz = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &ay,
        &bz,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let az_by = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &az,
        &by,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let az_bx = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &az,
        &bx,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let ax_bz = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &ax,
        &bz,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let ax_by = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &ax,
        &by,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    let ay_bx = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &ay,
        &bx,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;

    let out_x = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &ay_bz,
        &az_by,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;
    let out_y = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &az_bx,
        &ax_bz,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;
    let out_z = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &ax_by,
        &ay_bx,
        tenferro_prims::ScalarBinaryOp::Sub,
    )?;

    Tensor::stack(&[&out_x, &out_y, &out_z], 0)
}

/// Form the explicit product of Householder reflectors.
///
/// Given a lower-triangular Householder factor matrix `a` of shape `(m, n, *)`
/// and a vector `tau` of shape `(k, *)`, computes the orthogonal matrix `Q`.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::householder_product;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// // Typically obtained from an intermediate QR step.
/// let a = Tensor::<f64>::from_slice(
///     &[1.0, 0.5, 0.0, 1.0], &[2, 2], col,
/// ).unwrap();
/// let tau = Tensor::<f64>::from_slice(&[1.0, 0.5], &[2], col).unwrap();
/// let q = householder_product(&mut ctx, &a, &tau).unwrap();
/// assert_eq!(q.dims(), &[2, 2]);
/// ```
pub fn householder_product<T: KernelLinalgScalar + tenferro_algebra::Conjugate, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    tau: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorResolveConjContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>
        + tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>,
    C::Backend: 'static,
{
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
    let memory_space = a_input.logical_memory_space();
    let mut q = crate::prims_bridge::identity_matrix(m, memory_space)?.narrow(1, 0, n)?;
    for _ in batch_dims {
        q = q.unsqueeze(-1)?;
    }
    let q_target_dims = output_dims(&[m, n], batch_dims);
    q = q.broadcast(&q_target_dims)?;

    let vector_tail_dims = {
        let mut dims = Vec::with_capacity(1 + batch_dims.len());
        dims.push(1);
        dims.extend_from_slice(batch_dims);
        dims
    };

    for reflector in (0..k).rev() {
        let tail_rows = m - reflector;
        let tail = if tail_rows == 1 {
            Tensor::ones(&vector_tail_dims, memory_space, MemoryOrder::ColumnMajor)?
        } else {
            let head = Tensor::ones(&vector_tail_dims, memory_space, MemoryOrder::ColumnMajor)?;
            let lower = a_input
                .narrow(0, reflector + 1, tail_rows - 1)?
                .select(1, reflector)?;
            Tensor::cat(&[&head, &lower], 0)?
        };

        let q_tail = q.narrow(0, reflector, tail_rows)?;
        let v_col = tail.unsqueeze(1)?;
        let mut adj_perm: Vec<usize> = (0..v_col.ndim()).collect();
        adj_perm.swap(0, 1);
        let v_adj_view = v_col.conj().permute(&adj_perm)?;
        let v_adj = crate::prims_bridge::resolve_conj(ctx, &v_adj_view);
        let reflected = crate::prims_bridge::batched_gemm_with_semiring_tensors(
            ctx, &v_adj, &q_tail, 1, tail_rows, n,
        )?;

        let mut tau_scale = tau_input.select(0, reflector)?;
        tau_scale = tau_scale.unsqueeze(0)?;
        tau_scale = tau_scale.unsqueeze(0)?;
        let tau_scale = tau_scale.broadcast(reflected.dims())?;
        let scaled = crate::prims_bridge::scalar_binary_same_shape(
            ctx,
            &tau_scale,
            &reflected,
            tenferro_prims::ScalarBinaryOp::Mul,
        )?;
        let update = crate::prims_bridge::batched_gemm_with_semiring_tensors(
            ctx, &v_col, &scaled, tail_rows, 1, n,
        )?;
        let updated_tail = crate::prims_bridge::scalar_binary_same_shape(
            ctx,
            &q_tail,
            &update,
            tenferro_prims::ScalarBinaryOp::Sub,
        )?;

        q = if reflector == 0 {
            updated_tail
        } else {
            let prefix = q.narrow(0, 0, reflector)?;
            Tensor::cat(&[&prefix, &updated_tail], 0)?
        };
    }

    Ok(q)
}

/// Build a Vandermonde matrix from leading-dimension vectors.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::vander;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let x = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
/// let v = vander(&mut ctx, &x, None, false).unwrap();
/// assert_eq!(v.dims(), &[3, 3]);
/// ```
pub fn vander<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    x: &Tensor<T>,
    columns: Option<usize>,
    increasing: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T>>,
{
    let (vector_len, batch_dims): (usize, &[usize]) = if x.ndim() == 0 {
        (1, &[])
    } else {
        (x.dims()[0], &x.dims()[1..])
    };
    let columns = columns.unwrap_or(vector_len);
    let output_dims = output_dims(&[vector_len, columns], batch_dims);
    let output_numel = output_dims.iter().product::<usize>();

    let x_input = ensure_col_major(x);
    let base = if x.ndim() == 0 {
        x_input.reshape(&[1])?
    } else {
        x_input
    };
    let memory_space = base.logical_memory_space();
    if output_numel == 0 {
        return Ok(Tensor::zeros(
            &output_dims,
            memory_space,
            MemoryOrder::ColumnMajor,
        )?);
    }

    let mut current = Tensor::ones(base.dims(), memory_space, MemoryOrder::ColumnMajor)?;
    let mut columns_out = Vec::with_capacity(columns);

    columns_out.push(current.clone());
    for _ in 1..columns {
        let mut next = Tensor::zeros(base.dims(), memory_space, MemoryOrder::ColumnMajor)?;
        crate::prims_bridge::scalar_binary_same_shape_into(
            ctx,
            &current,
            &base,
            tenferro_prims::ScalarBinaryOp::Mul,
            &mut next,
        )?;
        columns_out.push(next.clone());
        current = next;
    }

    if !increasing {
        columns_out.reverse();
    }

    let column_refs: Vec<&Tensor<T>> = columns_out.iter().collect();
    Tensor::stack(&column_refs, 1)
}

/// Invert a tensorized square operator.
///
/// Reshapes the tensor into a square matrix using `ind` to split the
/// dimensions, computes the inverse, and reshapes back.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::tensorinv;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// // Shape [2, 2] with ind=1: left product = 2, right product = 2.
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], col).unwrap();
/// let inv = tensorinv(&mut ctx, &a, 1).unwrap();
/// assert_eq!(inv.dims(), &[2, 2]);
/// ```
pub fn tensorinv<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
    ind: usize,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::TensorInv, "tensorinv")?;

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
/// Reshapes `a` and `b` into a standard linear system and solves it.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::tensorsolve;
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mut ctx = CpuContext::new(1);
/// let col = MemoryOrder::ColumnMajor;
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], col).unwrap();
/// let b = Tensor::<f64>::from_slice(&[3.0, 4.0], &[2], col).unwrap();
/// let x = tensorsolve(&mut ctx, &a, &b, None).unwrap();
/// assert_eq!(x.dims(), &[2]);
/// ```
pub fn tensorsolve<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
    dims: Option<&[usize]>,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::TensorSolve, "tensorsolve")?;

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
