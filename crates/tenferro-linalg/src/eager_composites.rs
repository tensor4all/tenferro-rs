use num_complex::{Complex32, Complex64};
use tenferro_ad::{
    CompareDir, DType, DotGeneralConfig, EagerSession, EagerTensor, Error, Result, Tensor,
};
use tenferro_runtime::ErrorPhase;

use crate::eager_ext::EagerSessionLinalgExt;
use crate::validation::{ensure_float_or_complex, validate_lstsq};

pub(crate) fn slogdet(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
) -> Result<(EagerTensor, EagerTensor)> {
    let (_p, _l, u, parity) = session.lu(a)?;
    let diag_u = session.extract_diag(&u, 0, 1)?;
    let sign_u = session.sign(&diag_u)?;
    let sign_u = session.reduce_prod(&sign_u, Some(&[0]))?;
    let sign = session.mul(&parity, &sign_u)?;
    let abs_diag = session.abs(&diag_u)?;
    let log_diag = session.log(&abs_diag)?;
    let logabsdet = session.reduce_sum(&log_diag, Some(&[0]))?;
    Ok((sign, logabsdet))
}

pub(crate) fn det(session: &mut EagerSession<'_>, a: &EagerTensor) -> Result<EagerTensor> {
    let (sign, logabsdet) = slogdet(session, a)?;
    let exponent = session.exp(&logabsdet)?;
    session.mul(&sign, &exponent)
}

pub(crate) fn inv(session: &mut EagerSession<'_>, a: &EagerTensor) -> Result<EagerTensor> {
    ensure_min_rank("inv", a.shape().len(), 2)?;
    let eye = eye_like(session, a, a.shape()[0])?;
    session.solve(a, &eye)
}

pub(crate) fn lstsq(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    b: &EagerTensor,
) -> Result<EagerTensor> {
    validate_lstsq(
        "lstsq",
        a.dtype(),
        a.shape().len(),
        b.shape().len(),
        || Ok((a.shape()[0], a.shape()[1])),
        |message| Error::invalid_argument("lstsq", ErrorPhase::GraphBuild, "shape", message),
    )?;
    // Least squares via thin QR: A = Q R (full column rank), so
    // argmin_x |A x - b| solves R x = Qᴴ b.
    let (q, r) = session.qr(a)?;
    let q_conj = session.conj(&q)?;
    let qh = session.transpose(&q_conj, &matrix_transpose_perm(q.shape().len()))?;
    let qh_b = session.dot_general(&qh, b, trailing_batch_dot_config(qh.shape().len()))?;
    session.triangular_solve(&r, &qh_b, true, false, false, false)
}

pub(crate) fn pinv(session: &mut EagerSession<'_>, a: &EagerTensor) -> Result<EagerTensor> {
    ensure_float_or_complex("pinv", a.dtype())?;
    let max_dim = match (a.shape().first(), a.shape().get(1)) {
        (Some(&m), Some(&n)) => m.max(n),
        (Some(&m), None) => m,
        _ => 0,
    };
    pinv_with_rtol(session, a, default_pinv_rtol(a.dtype(), max_dim))
}

pub(crate) fn pinv_with_rtol(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    rtol: f64,
) -> Result<EagerTensor> {
    ensure_float_or_complex("pinv_with_rtol", a.dtype())?;
    let (u, s, vt) = session.svd(a)?;
    let abs_s = session.abs(&s)?;
    let s_max = session.reduce_max(&abs_s, Some(&[0]))?;
    let threshold_scalar =
        session.constant_from_host(scalar_real_tensor(s.dtype(), rtol.max(0.0))?)?;
    let threshold = session.mul(&s_max, &threshold_scalar)?;
    let threshold = broadcast_batch_scalar_to_leading_axis(session, &threshold, s.shape())?;
    let compare = session.compare(&abs_s, &threshold, CompareDir::Gt)?;
    let mask = session.convert(&compare, s.dtype())?;
    let ones = ones_like(session, &s)?;
    let neg_mask = session.neg(&mask)?;
    let denominator_offset = session.add(&ones, &neg_mask)?;
    let denom = session.add(&s, &denominator_offset)?;
    let s_inv = session.div(&mask, &denom)?;

    let conjugated = session.conj(&vt)?;
    let v = session.transpose(&conjugated, &matrix_transpose_perm(vt.shape().len()))?;
    let conjugated = session.conj(&u)?;
    let uh = session.transpose(&conjugated, &matrix_transpose_perm(u.shape().len()))?;
    let vs = scale_matrix_columns(session, &v, &s_inv)?;
    matmul_preserve_trailing_batch(session, &vs, &uh)
}

pub(crate) fn norm(
    session: &mut EagerSession<'_>,
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
    validate_axes("norm", a.shape().len(), &axes)?;
    if axes.is_empty() {
        return Ok(a.clone());
    }

    let out = if can_square_without_abs(a.dtype(), axes.len(), ord) {
        frobenius_norm(session, a, &axes)?
    } else {
        match axes.len() {
            1 => vector_norm(session, a, axes[0], ord)?,
            2 => matrix_norm(session, a, &axes, ord)?,
            _ => {
                let abs = session.abs(a)?;
                match ord {
                    None => frobenius_norm(session, &abs, &axes)?,
                    Some(p) if p == f64::INFINITY => session.reduce_max(&abs, Some(&axes))?,
                    Some(p) if p == f64::NEG_INFINITY => session.reduce_min(&abs, Some(&axes))?,
                    Some(0.0) => count_nonzero(session, &abs, &axes)?,
                    Some(p) => p_norm(session, &abs, &axes, p)?,
                }
            }
        }
    };
    restore_keepdim(session, out, a.shape(), &axes, keepdim)
}

fn scalar_real(
    session: &mut EagerSession<'_>,
    anchor: &EagerTensor,
    value: f64,
) -> Result<EagerTensor> {
    session.constant_from_host(scalar_real_tensor(anchor.dtype(), value)?)
}

fn scalar_real_tensor(dtype: DType, value: f64) -> Result<Tensor> {
    let tensor = match dtype {
        DType::F64 => Tensor::from_vec_col_major(vec![], vec![value])?,
        DType::F32 => Tensor::from_vec_col_major(vec![], vec![value as f32])?,
        DType::I32 => Tensor::from_vec_col_major(vec![], vec![value.round() as i32])?,
        DType::I64 => Tensor::from_vec_col_major(vec![], vec![value.round() as i64])?,
        DType::Bool => Tensor::from_vec_col_major(vec![], vec![value != 0.0])?,
        DType::C64 => Tensor::from_vec_col_major(vec![], vec![Complex64::new(value, 0.0)])?,
        DType::C32 => Tensor::from_vec_col_major(vec![], vec![Complex32::new(value as f32, 0.0)])?,
        // An externally defined scalar has no eager scalar literal.
        DType::External(_) => {
            return Err(tenferro_ad::Error::TensorRuntime(
                tenferro_tensor::Error::invalid_argument(
                    "scalar_real",
                    "dtype",
                    "an externally defined scalar has no eager literal",
                ),
            ));
        }
    };
    Ok(tensor)
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
    tenferro_tensor::validate::validate_unique_axes(op, "dim", rank, axes)
        .map_err(Error::TensorRuntime)
}

fn ones_like(session: &mut EagerSession<'_>, input: &EagerTensor) -> Result<EagerTensor> {
    let scalar = session.constant_from_host(scalar_real_tensor(input.dtype(), 1.0)?)?;
    broadcast_scalar(session, &scalar, input.shape())
}

fn eye_like(
    session: &mut EagerSession<'_>,
    anchor: &EagerTensor,
    size: usize,
) -> Result<EagerTensor> {
    let mut vector_shape = vec![size];
    vector_shape.extend_from_slice(&anchor.shape()[2..]);
    let scalar = session.constant_from_host(scalar_real_tensor(anchor.dtype(), 1.0)?)?;
    let ones = session.broadcast_in_dim(&scalar, &vector_shape, &[])?;
    session.embed_diag(&ones, 0, 1)
}

fn broadcast_scalar(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    shape: &[usize],
) -> Result<EagerTensor> {
    if input.shape() == shape {
        return Ok(input.clone());
    }
    session.broadcast_in_dim(input, shape, &[])
}

fn broadcast_batch_scalar_to_leading_axis(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    shape: &[usize],
) -> Result<EagerTensor> {
    if input.shape() == shape {
        return Ok(input.clone());
    }
    let dims: Vec<usize> = (1..shape.len()).collect();
    session.broadcast_in_dim(input, shape, &dims)
}

fn matmul_preserve_trailing_batch(
    session: &mut EagerSession<'_>,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
) -> Result<EagerTensor> {
    session.dot_general(lhs, rhs, trailing_batch_dot_config(lhs.shape().len()))
}

fn trailing_batch_dot_config(rank: usize) -> DotGeneralConfig {
    let batch_dims: Vec<usize> = (2..rank).collect();
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: batch_dims.clone().iter().copied().collect(),
        rhs_batch_dims: batch_dims.iter().copied().collect(),
    }
}

fn matrix_transpose_perm(rank: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn frobenius_norm(
    session: &mut EagerSession<'_>,
    abs: &EagerTensor,
    axes: &[usize],
) -> Result<EagerTensor> {
    let squares = session.reduce_sum_squares(abs, Some(axes))?;
    session.sqrt(&squares)
}

fn p_norm(
    session: &mut EagerSession<'_>,
    abs: &EagerTensor,
    axes: &[usize],
    p: f64,
) -> Result<EagerTensor> {
    if !p.is_finite() || p == 0.0 {
        return Err(Error::invalid_argument(
            "norm",
            ErrorPhase::GraphBuild,
            "p",
            format!("p-norm order must be finite and nonzero, got {p}"),
        ));
    }
    if p == 2.0 {
        return frobenius_norm(session, abs, axes);
    }
    let order = scalar_real(session, abs, p)?;
    let powered = session.pow(abs, &order)?;
    let summed = session.reduce_sum(&powered, Some(axes))?;
    let inverse_order = scalar_real(session, abs, 1.0 / p)?;
    session.pow(&summed, &inverse_order)
}

fn default_pinv_rtol(dtype: DType, max_dim: usize) -> f64 {
    let eps = match dtype {
        DType::F32 | DType::C32 => f32::EPSILON as f64,
        DType::F64 | DType::C64 => f64::EPSILON,
        DType::I32 | DType::I64 | DType::Bool => 0.0,
        // INVARIANT: the decomposition rejects an externally defined scalar before
        // it asks for a tolerance, so this value is never used.
        DType::External(_) => unreachable!("the decomposition validates its dtype first"),
    };
    eps * max_dim as f64
}

fn vector_norm(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    axis: usize,
    ord: Option<f64>,
) -> Result<EagerTensor> {
    let abs = session.abs(a)?;
    match ord {
        None => frobenius_norm(session, &abs, &[axis]),
        Some(0.0) => count_nonzero(session, &abs, &[axis]),
        Some(p) if p == f64::INFINITY => session.reduce_max(&abs, Some(&[axis])),
        Some(p) if p == f64::NEG_INFINITY => session.reduce_min(&abs, Some(&[axis])),
        Some(p) => p_norm(session, &abs, &[axis], p),
    }
}

fn matrix_norm(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    axes: &[usize],
    ord: Option<f64>,
) -> Result<EagerTensor> {
    let matrix = move_axes_to_front(session, a, axes)?;
    if matches!(ord, Some(2.0) | Some(-2.0)) {
        let singular_values = session.svd(&matrix)?.1;
        let singular_values = session.abs(&singular_values)?;
        return if ord == Some(2.0) {
            session.reduce_max(&singular_values, Some(&[0]))
        } else {
            session.reduce_min(&singular_values, Some(&[0]))
        };
    }

    let abs = session.abs(&matrix)?;
    match ord {
        None => frobenius_norm(session, &abs, &[0, 1]),
        Some(p) if p == f64::INFINITY => matrix_row_sum_norm(session, &abs, true),
        Some(p) if p == f64::NEG_INFINITY => matrix_row_sum_norm(session, &abs, false),
        Some(1.0) => matrix_col_sum_norm(session, &abs, true),
        Some(-1.0) => matrix_col_sum_norm(session, &abs, false),
        Some(0.0) => count_nonzero(session, &abs, &[0, 1]),
        Some(p) => p_norm(session, &abs, &[0, 1], p),
    }
}

fn scale_matrix_columns(
    session: &mut EagerSession<'_>,
    matrix: &EagerTensor,
    scale: &EagerTensor,
) -> Result<EagerTensor> {
    let mut scale_shape = vec![1, scale.shape()[0]];
    scale_shape.extend_from_slice(&matrix.shape()[2..]);
    let dims: Vec<usize> = (0..matrix.shape().len()).collect();
    let reshaped = session.reshape(scale, &scale_shape)?;
    let broadcast = session.broadcast_in_dim(&reshaped, matrix.shape(), &dims)?;
    session.mul(matrix, &broadcast)
}

fn count_nonzero(
    session: &mut EagerSession<'_>,
    abs: &EagerTensor,
    axes: &[usize],
) -> Result<EagerTensor> {
    let zero = scalar_real(session, abs, 0.0)?;
    let compared = session.compare(abs, &zero, CompareDir::Gt)?;
    let converted = session.convert(&compared, abs.dtype())?;
    session.reduce_sum(&converted, Some(axes))
}

fn matrix_row_sum_norm(
    session: &mut EagerSession<'_>,
    abs: &EagerTensor,
    take_max: bool,
) -> Result<EagerTensor> {
    let row_sums = session.reduce_sum(abs, Some(&[1]))?;
    if take_max {
        session.reduce_max(&row_sums, Some(&[0]))
    } else {
        session.reduce_min(&row_sums, Some(&[0]))
    }
}

fn matrix_col_sum_norm(
    session: &mut EagerSession<'_>,
    abs: &EagerTensor,
    take_max: bool,
) -> Result<EagerTensor> {
    let col_sums = session.reduce_sum(abs, Some(&[0]))?;
    if take_max {
        session.reduce_max(&col_sums, Some(&[0]))
    } else {
        session.reduce_min(&col_sums, Some(&[0]))
    }
}

fn move_axes_to_front(
    session: &mut EagerSession<'_>,
    tensor: &EagerTensor,
    axes: &[usize],
) -> Result<EagerTensor> {
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
    session.transpose(tensor, &perm)
}

fn restore_keepdim(
    session: &mut EagerSession<'_>,
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
    session.reshape(&reduced, &kept_shape)
}
