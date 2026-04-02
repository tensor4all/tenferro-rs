use super::*;

/// Validate that tensor has at least 2 dimensions.
/// Returns `(m, n, batch_dims_slice)`.
pub(crate) fn validate_2d<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<(usize, usize, &[usize])> {
    if tensor.ndim() < 2 {
        return Err(Error::InvalidArgument(format!(
            "expected at least 2 dimensions, got {}",
            tensor.ndim()
        )));
    }
    let m = tensor.dims()[0];
    let n = tensor.dims()[1];
    let batch = &tensor.dims()[2..];
    Ok((m, n, batch))
}

/// Validate that tensor is square (first two dims equal) and at least 2D.
/// Returns `(n, batch_dims_slice)`.
pub(crate) fn validate_square<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<(usize, &[usize])> {
    let (m, n, batch) = validate_2d(tensor)?;
    if m != n {
        return Err(Error::ShapeMismatch {
            expected: vec![m, m],
            got: vec![m, n],
        });
    }
    Ok((n, batch))
}

/// Validate RHS shape for least squares.
pub(crate) fn validate_lstsq_rhs<T: LinalgScalar>(
    b: &Tensor<T>,
    m: usize,
    batch_dims: &[usize],
) -> Result<()> {
    if b.ndim() != 1 + batch_dims.len() {
        return Err(Error::InvalidArgument(format!(
            "lstsq expects b shape (m, *), got {:?}",
            b.dims()
        )));
    }
    if b.dims()[0] != m {
        return Err(Error::InvalidArgument(format!(
            "lstsq expects b dim[0] == m ({m}), got {}",
            b.dims()[0]
        )));
    }
    if &b.dims()[1..] != batch_dims {
        return Err(Error::InvalidArgument(format!(
            "lstsq batch dims mismatch: expected {:?}, got {:?}",
            batch_dims,
            &b.dims()[1..]
        )));
    }
    Ok(())
}

/// Validate cotangent shape for norm AD.
pub(crate) fn validate_norm_cotangent<T: LinalgScalar>(
    cotangent: &Tensor<T>,
    batch_dims: &[usize],
) -> Result<()> {
    if batch_dims.is_empty() {
        if cotangent.ndim() == 0 {
            return Ok(());
        }
        return Err(Error::InvalidArgument(format!(
            "norm cotangent shape mismatch: expected scalar [], got {:?}",
            cotangent.dims()
        )));
    }

    if cotangent.dims() != batch_dims {
        return Err(Error::InvalidArgument(format!(
            "norm cotangent shape mismatch: expected {:?}, got {:?}",
            batch_dims,
            cotangent.dims()
        )));
    }

    Ok(())
}

pub(crate) fn invalid_vector_lp_exponent_error(p: f64) -> Error {
    Error::InvalidArgument(format!("vector Lp norm requires p >= 1, got {p}"))
}

pub(crate) fn matrix_only_norm_kind_error(kind: NormKind) -> Error {
    Error::InvalidArgument(format!("norm kind {kind:?} expects matrix input"))
}

pub(crate) fn invalid_vector_lp_exponent_ad_error(p: f64) -> chainrules_core::AutodiffError {
    chainrules_core::AutodiffError::InvalidArgument(format!(
        "vector Lp norm requires p >= 1, got {p}"
    ))
}

pub(crate) fn matrix_only_norm_kind_ad_error(kind: NormKind) -> chainrules_core::AutodiffError {
    chainrules_core::AutodiffError::InvalidArgument(format!(
        "norm kind {kind:?} expects matrix input"
    ))
}

/// Validate Hermitian/symmetric structure for batched square matrices.
#[cfg(test)]
pub(crate) fn validate_hermitian_batches<T: LinalgScalar>(
    data: &[T],
    offset: usize,
    n: usize,
    bc: usize,
    op_name: &str,
) -> Result<()> {
    let mat_size = n * n;
    let tol_scale = <T::Real as num_traits::NumCast>::from(128.0).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{op_name}: cannot convert tolerance scale 128.0 to real type"
        ))
    })?;

    for b in 0..bc {
        let start = offset + b * mat_size;
        let batch_data = &data[start..start + mat_size];
        for j in 0..n {
            for i in 0..j {
                let a_ij = batch_data[i + j * n];
                let a_ji = batch_data[j + i * n];
                let diff = (a_ij - a_ji.conj()).abs_real();
                let scale = <T::Real as num_traits::One>::one()
                    + num_traits::Float::max(a_ij.abs_real(), a_ji.abs_real());
                let tol = T::real_epsilon() * tol_scale * scale;
                if diff > tol {
                    return Err(Error::InvalidArgument(format!(
                        "{op_name} expects symmetric/Hermitian input; mismatch at ({i}, {j}) in batch {b}"
                    )));
                }
            }
        }
    }

    Ok(())
}

/// Compute batch count from batch dims (product, or 1 if empty).
pub(crate) fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

/// Build output dims: `[mat_dims..., batch_dims...]`.
pub(crate) fn output_dims(mat_dims: &[usize], batch_dims: &[usize]) -> Vec<usize> {
    let mut dims = mat_dims.to_vec();
    dims.extend_from_slice(batch_dims);
    dims
}

pub(crate) fn is_identity_permutation(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(idx, &axis)| idx == axis)
}

pub(crate) fn axes_to_end_permutation(rank: usize, axes: &[usize]) -> Vec<usize> {
    let mut is_solution_axis = vec![false; rank];
    for &axis in axes {
        is_solution_axis[axis] = true;
    }

    let mut perm = Vec::with_capacity(rank);
    for (axis, selected) in is_solution_axis.iter().enumerate() {
        if !selected {
            perm.push(axis);
        }
    }
    perm.extend_from_slice(axes);
    perm
}

pub(crate) fn validate_tensor_solve_axes(
    rank: usize,
    expected_len: usize,
    dims: Option<&[usize]>,
) -> Result<Vec<usize>> {
    let axes = if let Some(dims) = dims {
        if dims.len() != expected_len {
            return Err(Error::InvalidArgument(format!(
                "tensorsolve expects {} solution axes, got {}",
                expected_len,
                dims.len()
            )));
        }
        dims.to_vec()
    } else {
        (rank - expected_len..rank).collect()
    };

    let mut seen = vec![false; rank];
    for &axis in &axes {
        if axis >= rank {
            return Err(Error::InvalidArgument(format!(
                "tensorsolve axis {} is out of bounds for rank {}",
                axis, rank
            )));
        }
        if std::mem::replace(&mut seen[axis], true) {
            return Err(Error::InvalidArgument(format!(
                "tensorsolve axes must be unique, got {:?}",
                axes
            )));
        }
    }
    Ok(axes)
}
