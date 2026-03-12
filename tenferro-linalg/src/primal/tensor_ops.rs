use super::*;

/// Compute the cross product along the leading vector axis.
pub fn cross<T: KernelLinalgScalar, C>(
    _ctx: &mut C,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Cross, "cross")?;

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
pub fn householder_product<T: KernelLinalgScalar, C>(
    _ctx: &mut C,
    a: &Tensor<T>,
    tau: &Tensor<T>,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(
        backend::LinalgCapabilityOp::HouseholderProduct,
        "householder_product",
    )?;

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
pub fn vander<T: KernelLinalgScalar, C>(
    _ctx: &mut C,
    x: &Tensor<T>,
    columns: Option<usize>,
    increasing: bool,
) -> Result<Tensor<T>>
where
    C: backend::TensorLinalgContextFor<T>,
    C::Backend: 'static,
{
    require_linalg_support::<T, C>(backend::LinalgCapabilityOp::Vander, "vander")?;

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
