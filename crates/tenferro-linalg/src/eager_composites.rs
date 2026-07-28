use num_complex::{Complex32, Complex64};
use tenferro_ad::{CompareDir, DType, DotGeneralConfig, EagerTensor, Error, Result, Tensor};
use tenferro_runtime::ErrorPhase;

use crate::eager_ext::{eig, eigh, lu, qr, solve, svd, triangular_solve};

pub(crate) fn slogdet(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    let (_p, _l, u, parity) = lu(a)?;
    let diag_u = u.extract_diag(0, 1)?;
    let sign_u = diag_u.sign()?.reduce_prod(Some(&[0]))?;
    let sign = parity.mul(&sign_u)?;
    let logabsdet = diag_u.abs()?.log()?.reduce_sum(Some(&[0]))?;
    Ok((sign, logabsdet))
}

pub(crate) fn det(a: &EagerTensor) -> Result<EagerTensor> {
    let (sign, logabsdet) = slogdet(a)?;
    sign.mul(&logabsdet.exp()?)
}

pub(crate) fn inv(a: &EagerTensor) -> Result<EagerTensor> {
    ensure_min_rank("inv", a.shape().len(), 2)?;
    let eye = eye_like(a, a.shape()[0])?;
    solve(a, &eye)
}

pub(crate) fn lstsq(a: &EagerTensor, b: &EagerTensor) -> Result<EagerTensor> {
    ensure_float_or_complex("lstsq", a.dtype())?;
    ensure_min_rank("lstsq", a.shape().len(), 2)?;
    ensure_min_rank("lstsq", b.shape().len(), 2)?;
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m < n {
        return Err(Error::invalid_argument(
            "lstsq",
            ErrorPhase::GraphBuild,
            "shape",
            format!(
                "lstsq requires a tall or square matrix (rows {m} >= cols {n}); \
                 underdetermined (wide) systems are not supported"
            ),
        ));
    }
    // Least squares via thin QR: A = Q R (full column rank), so
    // argmin_x |A x - b| solves R x = Qᴴ b.
    let (q, r) = qr(a)?;
    let qh = q
        .conj()?
        .transpose(&matrix_transpose_perm(q.shape().len()))?;
    let qh_b = matmul_preserve_trailing_batch(&qh, b)?;
    triangular_solve(&r, &qh_b, true, false, false, false)
}

pub(crate) fn eigvalsh(a: &EagerTensor) -> Result<EagerTensor> {
    let (values, _vectors) = eigh(a)?;
    Ok(values)
}

pub(crate) fn eigvals(a: &EagerTensor) -> Result<EagerTensor> {
    let (values, _vectors) = eig(a)?;
    Ok(values)
}

pub(crate) fn pinv(a: &EagerTensor) -> Result<EagerTensor> {
    ensure_float_or_complex("pinv", a.dtype())?;
    let max_dim = match (a.shape().first(), a.shape().get(1)) {
        (Some(&m), Some(&n)) => m.max(n),
        (Some(&m), None) => m,
        _ => 0,
    };
    pinv_with_rtol(a, default_pinv_rtol(a.dtype(), max_dim))
}

pub(crate) fn pinv_with_rtol(a: &EagerTensor, rtol: f64) -> Result<EagerTensor> {
    ensure_float_or_complex("pinv_with_rtol", a.dtype())?;
    let (u, s, vt) = svd(a)?;
    let abs_s = s.abs()?;
    let s_max = abs_s.reduce_max(Some(&[0]))?;
    let threshold_scalar = scalar_real(&s, rtol.max(0.0))?;
    let threshold = s_max.mul(&threshold_scalar)?;
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, s.shape())?;
    let mask = abs_s
        .compare(&threshold, CompareDir::Gt)?
        .convert(s.dtype())?;
    let ones = ones_like(&s)?;
    let denom = s.add(&ones.add(&mask.neg()?)?)?;
    let s_inv = mask.div(&denom)?;

    let v = vt
        .conj()?
        .transpose(&matrix_transpose_perm(vt.shape().len()))?;
    let uh = u
        .conj()?
        .transpose(&matrix_transpose_perm(u.shape().len()))?;
    let vs = scale_matrix_columns(&v, &s_inv)?;
    matmul_preserve_trailing_batch(&vs, &uh)
}

pub(crate) fn norm(
    a: &EagerTensor,
    ord: Option<f64>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> Result<EagerTensor> {
    ensure_float_or_complex("norm", a.dtype())?;
    let axes = dim.map_or_else(
        || (0..a.shape().len()).collect::<Vec<_>>(),
        <[usize]>::to_vec,
    );
    if axes.is_empty() {
        return Ok(a.clone());
    }
    validate_axes("norm", a.shape().len(), &axes)?;

    let out = if can_square_without_abs(a.dtype(), axes.len(), ord) {
        frobenius_norm(a, &axes)?
    } else {
        match axes.len() {
            1 => vector_norm(a, axes[0], ord)?,
            2 => matrix_norm(a, &axes, ord)?,
            _ => {
                let abs = a.abs()?;
                match ord {
                    None => frobenius_norm(&abs, &axes)?,
                    Some(p) if p == f64::INFINITY => abs.reduce_max(Some(&axes))?,
                    Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(Some(&axes))?,
                    Some(0.0) => count_nonzero(&abs, &axes)?,
                    Some(p) => p_norm(&abs, &axes, p)?,
                }
            }
        }
    };
    restore_keepdim(out, a.shape(), &axes, keepdim)
}

fn scalar_real(anchor: &EagerTensor, value: f64) -> Result<EagerTensor> {
    let tensor = match anchor.dtype() {
        DType::F64 => Tensor::from_vec_col_major(vec![], vec![value])?,
        DType::F32 => Tensor::from_vec_col_major(vec![], vec![value as f32])?,
        DType::I32 => Tensor::from_vec_col_major(vec![], vec![value.round() as i32])?,
        DType::I64 => Tensor::from_vec_col_major(vec![], vec![value.round() as i64])?,
        DType::Bool => Tensor::from_vec_col_major(vec![], vec![value != 0.0])?,
        DType::C64 => Tensor::from_vec_col_major(vec![], vec![Complex64::new(value, 0.0)])?,
        DType::C32 => Tensor::from_vec_col_major(vec![], vec![Complex32::new(value as f32, 0.0)])?,
    };
    EagerTensor::from_tensor_in(tensor, anchor.runtime().clone())
}

fn ensure_float_or_complex(op: &'static str, dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        DType::I32 | DType::I64 | DType::Bool => Err(Error::TensorRuntime(
            crate::error::unsupported_dtype(op, dtype),
        )),
    }
}

fn can_square_without_abs(dtype: DType, axes_len: usize, ord: Option<f64>) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
        && (ord.is_none() || (ord == Some(2.0) && axes_len != 2))
}

fn ensure_min_rank(op: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        return Err(Error::TensorRuntime(tenferro_tensor::Error::rank_mismatch(
            op, expected, actual,
        )));
    }
    Ok(())
}

fn validate_axes(op: &'static str, rank: usize, axes: &[usize]) -> Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::TensorRuntime(
                tenferro_tensor::Error::axis_out_of_bounds(op, axis, rank),
            ));
        }
        if seen[axis] {
            return Err(Error::invalid_argument(
                op,
                ErrorPhase::GraphBuild,
                "dim",
                format!("axis {axis} appears more than once"),
            ));
        }
        seen[axis] = true;
    }
    Ok(())
}

fn ones_like(input: &EagerTensor) -> Result<EagerTensor> {
    broadcast_scalar(scalar_real(input, 1.0)?, input.shape())
}

fn eye_like(anchor: &EagerTensor, size: usize) -> Result<EagerTensor> {
    let mut vector_shape = vec![size];
    vector_shape.extend_from_slice(&anchor.shape()[2..]);
    broadcast_scalar(scalar_real(anchor, 1.0)?, &vector_shape)?.embed_diag(0, 1)
}

fn broadcast_scalar(input: EagerTensor, shape: &[usize]) -> Result<EagerTensor> {
    if input.shape() == shape {
        return Ok(input);
    }
    input.broadcast_in_dim(shape, &[])
}

fn broadcast_batch_scalar_to_leading_axis(
    input: &EagerTensor,
    shape: &[usize],
) -> Result<EagerTensor> {
    if input.shape() == shape {
        return Ok(input.clone());
    }
    let dims: Vec<usize> = (1..shape.len()).collect();
    input.broadcast_in_dim(shape, &dims)
}

fn matmul_preserve_trailing_batch(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<EagerTensor> {
    let batch_dims: Vec<usize> = (2..lhs.shape().len()).collect();
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: batch_dims.clone(),
            rhs_batch_dims: batch_dims,
        },
    )
}

fn matrix_transpose_perm(rank: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn frobenius_norm(abs: &EagerTensor, axes: &[usize]) -> Result<EagerTensor> {
    abs.mul(abs)?.reduce_sum(Some(axes))?.sqrt()
}

fn p_norm(abs: &EagerTensor, axes: &[usize], p: f64) -> Result<EagerTensor> {
    if !p.is_finite() || p == 0.0 {
        return Err(Error::invalid_argument(
            "norm",
            ErrorPhase::GraphBuild,
            "p",
            format!("p-norm order must be finite and nonzero, got {p}"),
        ));
    }
    if p == 2.0 {
        return frobenius_norm(abs, axes);
    }
    abs.pow(&scalar_real(abs, p)?)?
        .reduce_sum(Some(axes))?
        .pow(&scalar_real(abs, 1.0 / p)?)
}

fn default_pinv_rtol(dtype: DType, max_dim: usize) -> f64 {
    let eps = match dtype {
        DType::F32 | DType::C32 => f32::EPSILON as f64,
        DType::F64 | DType::C64 => f64::EPSILON,
        DType::I32 | DType::I64 | DType::Bool => 0.0,
    };
    eps * max_dim as f64
}

fn vector_norm(a: &EagerTensor, axis: usize, ord: Option<f64>) -> Result<EagerTensor> {
    let abs = a.abs()?;
    match ord {
        None => frobenius_norm(&abs, &[axis]),
        Some(0.0) => count_nonzero(&abs, &[axis]),
        Some(p) if p == f64::INFINITY => abs.reduce_max(Some(&[axis])),
        Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(Some(&[axis])),
        Some(p) => p_norm(&abs, &[axis], p),
    }
}

fn matrix_norm(a: &EagerTensor, axes: &[usize], ord: Option<f64>) -> Result<EagerTensor> {
    let matrix = move_axes_to_front(a, axes)?;
    let abs = matrix.abs()?;
    match ord {
        None => frobenius_norm(&abs, &[0, 1]),
        Some(p) if p == f64::INFINITY => matrix_row_sum_norm(&abs, true),
        Some(p) if p == f64::NEG_INFINITY => matrix_row_sum_norm(&abs, false),
        Some(1.0) => matrix_col_sum_norm(&abs, true),
        Some(-1.0) => matrix_col_sum_norm(&abs, false),
        Some(2.0) => svd(&matrix)?.1.abs()?.reduce_max(Some(&[0])),
        Some(-2.0) => svd(&matrix)?.1.abs()?.reduce_min(Some(&[0])),
        Some(0.0) => count_nonzero(&abs, &[0, 1]),
        Some(p) => p_norm(&abs, &[0, 1], p),
    }
}

fn scale_matrix_columns(matrix: &EagerTensor, scale: &EagerTensor) -> Result<EagerTensor> {
    let mut scale_shape = vec![1, scale.shape()[0]];
    scale_shape.extend_from_slice(&matrix.shape()[2..]);
    let dims: Vec<usize> = (0..matrix.shape().len()).collect();
    matrix.mul(
        &scale
            .reshape(&scale_shape)?
            .broadcast_in_dim(matrix.shape(), &dims)?,
    )
}

fn count_nonzero(abs: &EagerTensor, axes: &[usize]) -> Result<EagerTensor> {
    abs.compare(&scalar_real(abs, 0.0)?, CompareDir::Gt)?
        .convert(abs.dtype())?
        .reduce_sum(Some(axes))
}

fn matrix_row_sum_norm(abs: &EagerTensor, take_max: bool) -> Result<EagerTensor> {
    let row_sums = abs.reduce_sum(Some(&[1]))?;
    if take_max {
        row_sums.reduce_max(Some(&[0]))
    } else {
        row_sums.reduce_min(Some(&[0]))
    }
}

fn matrix_col_sum_norm(abs: &EagerTensor, take_max: bool) -> Result<EagerTensor> {
    let col_sums = abs.reduce_sum(Some(&[0]))?;
    if take_max {
        col_sums.reduce_max(Some(&[0]))
    } else {
        col_sums.reduce_min(Some(&[0]))
    }
}

fn move_axes_to_front(tensor: &EagerTensor, axes: &[usize]) -> Result<EagerTensor> {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        return Ok(tensor.clone());
    }
    let mut selected = vec![false; tensor.shape().len()];
    for &axis in axes {
        selected[axis] = true;
    }
    let mut perm = Vec::with_capacity(tensor.shape().len());
    perm.extend_from_slice(axes);
    for (axis, is_selected) in selected.iter().enumerate() {
        if !*is_selected {
            perm.push(axis);
        }
    }
    tensor.transpose(&perm)
}

fn restore_keepdim(
    reduced: EagerTensor,
    original_shape: &[usize],
    axes: &[usize],
    keepdim: bool,
) -> Result<EagerTensor> {
    if !keepdim {
        return Ok(reduced);
    }
    let mut kept_shape = original_shape.to_vec();
    for &axis in axes {
        kept_shape[axis] = 1;
    }
    reduced.reshape(&kept_shape)
}
