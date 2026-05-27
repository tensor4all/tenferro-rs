use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_runtime::extension::apply;
use tenferro_runtime::{CompareDir, DType, DotGeneralConfig, Error, Result, TracedTensor};

use crate::extension::{LinalgExtensionOp, LinalgOp};

/// Build a traced singular value decomposition op using the default epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::svd;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]);
/// let (u, s, vt) = svd(&a).unwrap();
/// assert_eq!(u.rank, 2);
/// assert_eq!(s.rank, 1);
/// assert_eq!(vt.rank, 2);
/// ```
pub fn svd(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    svd_with_eps(a, 1e-12)
}

/// Build a traced singular value decomposition op with an explicit epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::svd_with_eps;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]);
/// let (_u, s, _vt) = svd_with_eps(&a, 1e-10).unwrap();
/// assert_eq!(s.rank, 1);
/// ```
pub fn svd_with_eps(
    a: &TracedTensor,
    eps: f64,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    three_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Svd { eps })),
            &[a],
        ),
        "svd",
    )
}

/// Build a traced QR decomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::qr;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]);
/// let (q, r) = qr(&a).unwrap();
/// assert_eq!(q.rank, 2);
/// assert_eq!(r.rank, 2);
/// ```
pub fn qr(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Qr)), &[a]),
        "qr",
    )
}

/// Build a traced Hermitian eigenvalue decomposition op using the default epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::eigh;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let (values, vectors) = eigh(&a).unwrap();
/// assert_eq!(values.rank, 1);
/// assert_eq!(vectors.rank, 2);
/// ```
pub fn eigh(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    eigh_with_eps(a, 1e-12)
}

/// Build a traced Hermitian eigenvalue decomposition op with an explicit epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::eigh_with_eps;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let (values, _vectors) = eigh_with_eps(&a, 1e-10).unwrap();
/// assert_eq!(values.rank, 1);
/// ```
pub fn eigh_with_eps(a: &TracedTensor, eps: f64) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh { eps })),
            &[a],
        ),
        "eigh",
    )
}

/// Build a traced Cholesky decomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::cholesky;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]);
/// let factor = cholesky(&a).unwrap();
/// assert_eq!(factor.rank, 2);
/// ```
pub fn cholesky(a: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Cholesky)), &[a]),
        "cholesky",
    )
}

/// Build a traced LU decomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::lu;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
/// let (p, l, u, parity) = lu(&a).unwrap();
/// assert_eq!(p.rank, 2);
/// assert_eq!(l.rank, 2);
/// assert_eq!(u.rank, 2);
/// assert_eq!(parity.rank, 0);
/// ```
pub fn lu(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
    four_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Lu)), &[a]),
        "lu",
    )
}

/// Build a traced full-pivot LU decomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::full_piv_lu;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
/// let (p, l, u, q, parity) = full_piv_lu(&a).unwrap();
/// assert_eq!(p.rank, 2);
/// assert_eq!(l.rank, 2);
/// assert_eq!(u.rank, 2);
/// assert_eq!(q.rank, 2);
/// assert_eq!(parity.rank, 0);
/// ```
pub fn full_piv_lu(
    a: &TracedTensor,
) -> Result<(
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
)> {
    five_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLu)), &[a]),
        "full_piv_lu",
    )
}

/// Build a traced general eigendecomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::eig;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
/// let (values, vectors) = eig(&a).unwrap();
/// assert_eq!(values.rank, 1);
/// assert_eq!(vectors.rank, 2);
/// ```
pub fn eig(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eig {
                input_dtype: a.dtype,
            })),
            &[a],
        ),
        "eig",
    )
}

/// Build a traced linear solve op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::solve;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0]);
/// let x = solve(&a, &b).unwrap();
/// assert_eq!(x.rank, 2);
/// ```
pub fn solve(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Solve {
                transpose_a: false,
            })),
            &[a, b],
        ),
        "solve",
    )
}

/// Build a traced full-pivot LU solve op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::full_piv_lu_solve;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0]);
/// let x = full_piv_lu_solve(&a, &b).unwrap();
/// assert_eq!(x.rank, 2);
/// ```
pub fn full_piv_lu_solve(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLuSolve {
                transpose_a: false,
            })),
            &[a, b],
        ),
        "full_piv_lu_solve",
    )
}

/// Build a traced triangular solve op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::triangular_solve;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0]);
/// let x = triangular_solve(&a, &b, true, true, false, false).unwrap();
/// assert_eq!(x.rank, 2);
/// ```
pub fn triangular_solve(
    a: &TracedTensor,
    b: &TracedTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            })),
            &[a, b],
        ),
        "triangular_solve",
    )
}

/// Build traced sign and log-absolute-determinant ops.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::slogdet;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let (sign, logabsdet) = slogdet(&a).unwrap();
/// assert_eq!(sign.rank, 0);
/// assert_eq!(logabsdet.rank, 0);
/// ```
pub fn slogdet(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    let (_, _, u, parity) = lu(a)?;
    let diag_u = u.extract_diag(0, 1);
    let sign_u = diag_u.sign().reduce_prod(&[0]);
    let sign = &parity * &sign_u;
    let logabsdet = diag_u.abs().log().reduce_sum(&[0]);
    Ok((sign, logabsdet))
}

/// Build a traced determinant op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::det;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let determinant = det(&a).unwrap();
/// assert_eq!(determinant.rank, 0);
/// ```
pub fn det(a: &TracedTensor) -> Result<TracedTensor> {
    let (sign, logabsdet) = slogdet(a)?;
    Ok(&sign * &logabsdet.exp())
}

/// Build a traced matrix inverse op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::inv;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let inverse = inv(&a).unwrap();
/// assert_eq!(inverse.rank, 2);
/// ```
pub fn inv(a: &TracedTensor) -> Result<TracedTensor> {
    let shape = a.concrete_shape();
    let eye = eye_like(a, shape[0]);
    solve(a, &eye)
}

/// Build a traced Hermitian eigenvalue-only op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::eigvalsh;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]);
/// let values = eigvalsh(&a).unwrap();
/// assert_eq!(values.rank, 1);
/// ```
pub fn eigvalsh(a: &TracedTensor) -> Result<TracedTensor> {
    Ok(eigh(a)?.0)
}

/// Build a traced general eigenvalue-only op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::eigvals;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
/// let values = eigvals(&a).unwrap();
/// assert_eq!(values.rank, 1);
/// ```
pub fn eigvals(a: &TracedTensor) -> Result<TracedTensor> {
    Ok(eig(a)?.0)
}

/// Build a traced Moore-Penrose pseudoinverse op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::pinv;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
/// let inverse = pinv(&a).unwrap();
/// assert_eq!(inverse.rank, 2);
/// ```
pub fn pinv(a: &TracedTensor) -> Result<TracedTensor> {
    let shape = a.concrete_shape();
    let max_dim = match (shape.first(), shape.get(1)) {
        (Some(&m), Some(&n)) => m.max(n),
        (Some(&m), None) => m,
        _ => 0,
    };
    pinv_with_rtol(a, default_pinv_rtol(a.dtype, max_dim))
}

/// Build a traced Moore-Penrose pseudoinverse op with an explicit relative tolerance.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::pinv_with_rtol;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
/// let inverse = pinv_with_rtol(&a, 1e-12).unwrap();
/// assert_eq!(inverse.rank, 2);
/// ```
pub fn pinv_with_rtol(a: &TracedTensor, rtol: f64) -> Result<TracedTensor> {
    let (u, s, vt) = svd(a)?;
    let abs_s = s.abs();
    let s_max = abs_s.reduce_max(&[0]);
    let s_max_shape = s_max.concrete_shape();
    let threshold = &s_max * &broadcast_scalar(scalar_real(s.dtype, rtol.max(0.0)), &s_max_shape);
    let s_shape = s.concrete_shape();
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, &s_shape);
    let mask = abs_s.compare(&threshold, CompareDir::Gt);
    let mask = mask.convert(s.dtype);
    let ones = ones_like(&s);
    let denom = &s + &(&ones + &(-&mask));
    let s_inv = &mask / &denom;

    let v = vt.conj().transpose(&matrix_transpose_perm(vt.rank));
    let uh = u.conj().transpose(&matrix_transpose_perm(u.rank));
    let s_inv_diag = s_inv.embed_diag(0, 1);
    let vs = matmul_preserve_trailing_batch(&v, &s_inv_diag);
    Ok(matmul_preserve_trailing_batch(&vs, &uh))
}

/// Build a traced vector, matrix, or tensor norm op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::traced_tensor::norm;
/// use tenferro_runtime::TracedTensor;
///
/// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
/// let length = norm(&x, Some(2.0), Some(&[0]), false).unwrap();
/// assert_eq!(length.rank, 0);
/// ```
pub fn norm(
    a: &TracedTensor,
    ord: Option<f64>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> Result<TracedTensor> {
    let axes = dim.map_or_else(|| (0..a.rank).collect::<Vec<_>>(), |dims| dims.to_vec());
    if axes.is_empty() {
        return Ok(a.clone());
    }

    let out = match axes.len() {
        1 => vector_norm(a, axes[0], ord),
        2 => matrix_norm(a, &axes, ord)?,
        _ => {
            let abs = a.abs();
            match ord {
                None => frobenius_norm(&abs, &axes),
                Some(p) if p == f64::INFINITY => abs.reduce_max(&axes),
                Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(&axes),
                Some(p) if p == 0.0 => count_nonzero(&abs, &axes),
                Some(p) => p_norm(&abs, &axes, p),
            }
        }
    };
    let shape = a.concrete_shape();
    Ok(restore_keepdim(out, &shape, &axes, keepdim))
}

fn unexpected_output_count(name: &str, expected: usize) -> Error {
    Error::Internal(format!("{name} must produce exactly {expected} outputs"))
}

fn one_output(outputs: Vec<TracedTensor>, name: &str) -> Result<TracedTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(unexpected_output_count(name, 1)),
    }
}

fn two_outputs(outputs: Vec<TracedTensor>, name: &str) -> Result<(TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(unexpected_output_count(name, 2)),
    }
}

fn three_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), None) => Ok((first, second, third)),
        _ => Err(unexpected_output_count(name, 3)),
    }
}

fn four_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), Some(fourth), None) => {
            Ok((first, second, third, fourth))
        }
        _ => Err(unexpected_output_count(name, 4)),
    }
}

fn five_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), Some(fourth), Some(fifth), None) => {
            Ok((first, second, third, fourth, fifth))
        }
        _ => Err(unexpected_output_count(name, 5)),
    }
}

fn scalar_real(dtype: DType, value: f64) -> TracedTensor {
    match dtype {
        DType::F64 => TracedTensor::from_vec_col_major(vec![], vec![value]),
        DType::F32 => TracedTensor::from_vec_col_major(vec![], vec![value as f32]),
        DType::I32 => TracedTensor::from_vec_col_major(vec![], vec![value.round() as i32]),
        DType::I64 => TracedTensor::from_vec_col_major(vec![], vec![value.round() as i64]),
        DType::Bool => TracedTensor::from_vec_col_major(vec![], vec![value != 0.0]),
        DType::C64 => TracedTensor::from_vec_col_major(vec![], vec![Complex64::new(value, 0.0)]),
        DType::C32 => {
            TracedTensor::from_vec_col_major(vec![], vec![Complex32::new(value as f32, 0.0)])
        }
    }
}

fn zero_scalar(dtype: DType) -> TracedTensor {
    scalar_real(dtype, 0.0)
}

fn one_scalar(dtype: DType) -> TracedTensor {
    scalar_real(dtype, 1.0)
}

fn ones_like(input: &TracedTensor) -> TracedTensor {
    let shape = input.concrete_shape();
    broadcast_scalar(one_scalar(input.dtype), &shape)
}

fn eye_like(anchor: &TracedTensor, size: usize) -> TracedTensor {
    let mut vector_shape = vec![size];
    let anchor_shape = anchor.concrete_shape();
    vector_shape.extend_from_slice(&anchor_shape[2..]);
    let diagonal = broadcast_scalar(one_scalar(anchor.dtype), &vector_shape);
    diagonal.embed_diag(0, 1)
}

fn broadcast_scalar(input: TracedTensor, shape: &[usize]) -> TracedTensor {
    let input_shape = input.concrete_shape();
    if input_shape == shape {
        return input;
    }
    input.broadcast_in_dim(shape, &[])
}

fn broadcast_batch_scalar_to_leading_axis(input: &TracedTensor, shape: &[usize]) -> TracedTensor {
    let input_shape = input.concrete_shape();
    if input_shape == shape {
        return input.clone();
    }
    let dims: Vec<usize> = (1..shape.len()).collect();
    input.broadcast_in_dim(shape, &dims)
}

fn matmul_preserve_trailing_batch(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    let rank = lhs.rank;
    let batch_dims: Vec<usize> = (2..rank).collect();
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

fn frobenius_norm(abs: &TracedTensor, axes: &[usize]) -> TracedTensor {
    let squared = abs.pow(&scalar_real(abs.dtype, 2.0));
    squared.reduce_sum(axes).sqrt()
}

fn p_norm(abs: &TracedTensor, axes: &[usize], p: f64) -> TracedTensor {
    let power = abs.pow(&scalar_real(abs.dtype, p));
    let inv_p = scalar_real(abs.dtype, 1.0 / p);
    power.reduce_sum(axes).pow(&inv_p)
}

fn default_pinv_rtol(dtype: DType, max_dim: usize) -> f64 {
    let eps = match dtype {
        DType::F32 | DType::C32 => f32::EPSILON as f64,
        DType::F64 | DType::C64 => f64::EPSILON,
        DType::I32 | DType::I64 | DType::Bool => 0.0,
    };
    eps * max_dim as f64
}

fn vector_norm(a: &TracedTensor, axis: usize, ord: Option<f64>) -> TracedTensor {
    let abs = a.abs();
    match ord {
        None => frobenius_norm(&abs, &[axis]),
        Some(p) if p == 0.0 => count_nonzero(&abs, &[axis]),
        Some(p) if p == f64::INFINITY => abs.reduce_max(&[axis]),
        Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(&[axis]),
        Some(p) => p_norm(&abs, &[axis], p),
    }
}

fn matrix_norm(a: &TracedTensor, axes: &[usize], ord: Option<f64>) -> Result<TracedTensor> {
    let matrix = move_axes_to_front(a, axes);
    let abs = matrix.abs();
    Ok(match ord {
        None => frobenius_norm(&abs, &[0, 1]),
        Some(p) if p == f64::INFINITY => matrix_row_sum_norm(&abs, true),
        Some(p) if p == f64::NEG_INFINITY => matrix_row_sum_norm(&abs, false),
        Some(p) if p == 1.0 => matrix_col_sum_norm(&abs, true),
        Some(p) if p == -1.0 => matrix_col_sum_norm(&abs, false),
        Some(p) if p == 2.0 => {
            let singular_values = svd(&matrix)?.1.abs();
            singular_values.reduce_max(&[0])
        }
        Some(p) if p == -2.0 => {
            let singular_values = svd(&matrix)?.1.abs();
            singular_values.reduce_min(&[0])
        }
        Some(p) if p == 0.0 => count_nonzero(&abs, &[0, 1]),
        Some(p) => p_norm(&abs, &[0, 1], p),
    })
}

fn count_nonzero(abs: &TracedTensor, axes: &[usize]) -> TracedTensor {
    let mask = abs.compare(&zero_scalar(abs.dtype), CompareDir::Gt);
    mask.convert(abs.dtype).reduce_sum(axes)
}

fn matrix_row_sum_norm(abs: &TracedTensor, take_max: bool) -> TracedTensor {
    let row_sums = abs.reduce_sum(&[1]);
    if take_max {
        row_sums.reduce_max(&[0])
    } else {
        row_sums.reduce_min(&[0])
    }
}

fn matrix_col_sum_norm(abs: &TracedTensor, take_max: bool) -> TracedTensor {
    let col_sums = abs.reduce_sum(&[0]);
    if take_max {
        col_sums.reduce_max(&[0])
    } else {
        col_sums.reduce_min(&[0])
    }
}

fn move_axes_to_front(tensor: &TracedTensor, axes: &[usize]) -> TracedTensor {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        return tensor.clone();
    }

    let mut selected = vec![false; tensor.rank];
    for &axis in axes {
        selected[axis] = true;
    }

    let mut perm = Vec::with_capacity(tensor.rank);
    perm.extend_from_slice(axes);
    for axis in 0..tensor.rank {
        if !selected[axis] {
            perm.push(axis);
        }
    }
    tensor.transpose(&perm)
}

fn restore_keepdim(
    reduced: TracedTensor,
    original_shape: &[usize],
    axes: &[usize],
    keepdim: bool,
) -> TracedTensor {
    if !keepdim {
        return reduced;
    }
    let mut kept_shape = original_shape.to_vec();
    for &axis in axes {
        kept_shape[axis] = 1;
    }
    reduced.reshape(&kept_shape)
}
