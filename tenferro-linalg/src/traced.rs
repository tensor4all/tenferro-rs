use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro::extension::apply;
use tenferro::{CompareDir, DType, DotGeneralConfig, TracedTensor};

use crate::extension::{LinalgExtensionOp, LinalgOp};

pub fn svd(a: &TracedTensor) -> (TracedTensor, TracedTensor, TracedTensor) {
    svd_with_eps(a, 1e-12)
}

pub fn svd_with_eps(a: &TracedTensor, eps: f64) -> (TracedTensor, TracedTensor, TracedTensor) {
    ensure_ad_rule_registered();
    let mut outputs = apply(
        Arc::new(LinalgExtensionOp::new(LinalgOp::Svd { eps })),
        &[a],
    )
    .into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(u), Some(s), Some(vt), None) => (u, s, vt),
        _ => unreachable!("svd must produce exactly three outputs"),
    }
}

pub fn qr(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    ensure_ad_rule_registered();
    two_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Qr)), &[a]),
        "qr",
    )
}

pub fn eigh(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    eigh_with_eps(a, 1e-12)
}

pub fn eigh_with_eps(a: &TracedTensor, eps: f64) -> (TracedTensor, TracedTensor) {
    ensure_ad_rule_registered();
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh { eps })),
            &[a],
        ),
        "eigh",
    )
}

pub fn cholesky(a: &TracedTensor) -> TracedTensor {
    ensure_ad_rule_registered();
    one_output(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Cholesky)), &[a]),
        "cholesky",
    )
}

pub fn lu(a: &TracedTensor) -> (TracedTensor, TracedTensor, TracedTensor, TracedTensor) {
    ensure_ad_rule_registered();
    let mut outputs = apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Lu)), &[a]).into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(parity), None) => (p, l, u, parity),
        _ => unreachable!("lu must produce exactly four outputs"),
    }
}

pub fn full_piv_lu(
    a: &TracedTensor,
) -> (
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
) {
    ensure_ad_rule_registered();
    let mut outputs =
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLu)), &[a]).into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => (p, l, u, q, parity),
        _ => unreachable!("full_piv_lu must produce exactly five outputs"),
    }
}

pub fn eig(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    ensure_ad_rule_registered();
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

pub fn solve(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    ensure_ad_rule_registered();
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

pub fn full_piv_lu_solve(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    ensure_ad_rule_registered();
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

pub fn triangular_solve(
    a: &TracedTensor,
    b: &TracedTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> TracedTensor {
    ensure_ad_rule_registered();
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

pub fn slogdet(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    let (_, _, u, parity) = lu(a);
    let diag_u = u.extract_diag(0, 1);
    let sign_u = diag_u.sign().reduce_prod(&[0]);
    let sign = &parity * &sign_u;
    let logabsdet = diag_u.abs().log().reduce_sum(&[0]);
    (sign, logabsdet)
}

pub fn det(a: &TracedTensor) -> TracedTensor {
    let (sign, logabsdet) = slogdet(a);
    &sign * &logabsdet.exp()
}

pub fn inv(a: &TracedTensor) -> TracedTensor {
    let shape = a.concrete_shape();
    let eye = eye_like(a, shape[0]);
    solve(a, &eye)
}

pub fn eigvalsh(a: &TracedTensor) -> TracedTensor {
    eigh(a).0
}

pub fn eigvals(a: &TracedTensor) -> TracedTensor {
    eig(a).0
}

pub fn pinv(a: &TracedTensor) -> TracedTensor {
    let shape = a.concrete_shape();
    let max_dim = match (shape.first(), shape.get(1)) {
        (Some(&m), Some(&n)) => m.max(n),
        (Some(&m), None) => m,
        _ => 0,
    };
    pinv_with_rtol(a, default_pinv_rtol(a.dtype, max_dim))
}

pub fn pinv_with_rtol(a: &TracedTensor, rtol: f64) -> TracedTensor {
    let (u, s, vt) = svd(a);
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
    matmul_preserve_trailing_batch(&vs, &uh)
}

pub fn norm(
    a: &TracedTensor,
    ord: Option<f64>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> TracedTensor {
    let axes = dim.map_or_else(|| (0..a.rank).collect::<Vec<_>>(), |dims| dims.to_vec());
    if axes.is_empty() {
        return a.clone();
    }

    let out = match axes.len() {
        1 => vector_norm(a, axes[0], ord),
        2 => matrix_norm(a, &axes, ord),
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
    restore_keepdim(out, &shape, &axes, keepdim)
}

fn one_output(outputs: Vec<TracedTensor>, name: &str) -> TracedTensor {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => output,
        _ => unreachable!("{name} must produce exactly one output"),
    }
}

fn two_outputs(outputs: Vec<TracedTensor>, name: &str) -> (TracedTensor, TracedTensor) {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => (lhs, rhs),
        _ => unreachable!("{name} must produce exactly two outputs"),
    }
}

fn ensure_ad_rule_registered() {}

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

fn matrix_norm(a: &TracedTensor, axes: &[usize], ord: Option<f64>) -> TracedTensor {
    let matrix = move_axes_to_front(a, axes);
    let abs = matrix.abs();
    match ord {
        None => frobenius_norm(&abs, &[0, 1]),
        Some(p) if p == f64::INFINITY => matrix_row_sum_norm(&abs, true),
        Some(p) if p == f64::NEG_INFINITY => matrix_row_sum_norm(&abs, false),
        Some(p) if p == 1.0 => matrix_col_sum_norm(&abs, true),
        Some(p) if p == -1.0 => matrix_col_sum_norm(&abs, false),
        Some(p) if p == 2.0 => {
            let singular_values = svd(&matrix).1.abs();
            singular_values.reduce_max(&[0])
        }
        Some(p) if p == -2.0 => {
            let singular_values = svd(&matrix).1.abs();
            singular_values.reduce_min(&[0])
        }
        Some(p) if p == 0.0 => count_nonzero(&abs, &[0, 1]),
        Some(p) => p_norm(&abs, &[0, 1], p),
    }
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
