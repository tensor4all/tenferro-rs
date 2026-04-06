use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::traced::{apply_binary, apply_multi_output, apply_unary, TracedTensor};

/// Singular value decomposition with a default numerical epsilon.
///
/// # Examples
///
/// ```rust,ignore
/// let (u, s, vt) = tenferro::svd(&a);
/// ```
pub fn svd(a: &TracedTensor) -> (TracedTensor, TracedTensor, TracedTensor) {
    svd_with_eps(a, 1e-12)
}

/// Singular value decomposition with an explicit numerical epsilon.
///
/// # Examples
///
/// ```rust,ignore
/// let (u, s, vt) = tenferro::svd_with_eps(&a, 1e-10);
/// ```
pub fn svd_with_eps(a: &TracedTensor, eps: f64) -> (TracedTensor, TracedTensor, TracedTensor) {
    let m = a.shape[0];
    let n = a.shape[1];
    let k = m.min(n);
    let batch = &a.shape[2..];
    let op = StdTensorOp::Svd {
        eps,
        input_shape: a.shape.clone(),
    };
    let mut u_shape = vec![m, k];
    u_shape.extend_from_slice(batch);
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch);
    let mut vt_shape = vec![k, n];
    vt_shape.extend_from_slice(batch);
    let mut results = apply_multi_output(op, a, vec![u_shape, s_shape, vt_shape]).into_iter();
    match (
        results.next(),
        results.next(),
        results.next(),
        results.next(),
    ) {
        (Some(u), Some(s), Some(vt), None) => (u, s, vt),
        _ => unreachable!("svd must produce exactly three outputs"),
    }
}

/// QR decomposition.
///
/// # Examples
///
/// ```rust,ignore
/// let (q, r) = tenferro::qr(&a);
/// ```
pub fn qr(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    let m = a.shape[0];
    let n = a.shape[1];
    let k = m.min(n);
    let batch = &a.shape[2..];
    let mut q_shape = vec![m, k];
    q_shape.extend_from_slice(batch);
    let mut r_shape = vec![k, n];
    r_shape.extend_from_slice(batch);
    let mut results = apply_multi_output(
        StdTensorOp::Qr {
            input_shape: a.shape.clone(),
        },
        a,
        vec![q_shape, r_shape],
    )
    .into_iter();
    match (results.next(), results.next(), results.next()) {
        (Some(q), Some(r), None) => (q, r),
        _ => unreachable!("qr must produce exactly two outputs"),
    }
}

/// Hermitian eigenvalue decomposition with a default numerical epsilon.
///
/// # Examples
///
/// ```rust,ignore
/// let (values, vectors) = tenferro::eigh(&a);
/// ```
pub fn eigh(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    eigh_with_eps(a, 1e-12)
}

/// Hermitian eigenvalue decomposition with an explicit numerical epsilon.
///
/// # Examples
///
/// ```rust,ignore
/// let (values, vectors) = tenferro::eigh_with_eps(&a, 1e-10);
/// ```
pub fn eigh_with_eps(a: &TracedTensor, eps: f64) -> (TracedTensor, TracedTensor) {
    let n = a.shape[0];
    let batch = &a.shape[2..];
    let op = StdTensorOp::Eigh {
        eps,
        input_shape: a.shape.clone(),
    };
    let mut vals_shape = vec![n];
    vals_shape.extend_from_slice(batch);
    let mut vecs_shape = vec![n, n];
    vecs_shape.extend_from_slice(batch);
    let mut results = apply_multi_output(op, a, vec![vals_shape, vecs_shape]).into_iter();
    match (results.next(), results.next(), results.next()) {
        (Some(values), Some(vectors), None) => (values, vectors),
        _ => unreachable!("eigh must produce exactly two outputs"),
    }
}

/// Cholesky factorization.
///
/// # Examples
///
/// ```rust,ignore
/// let l = tenferro::cholesky(&a);
/// ```
pub fn cholesky(a: &TracedTensor) -> TracedTensor {
    apply_unary(
        StdTensorOp::Cholesky {
            input_shape: a.shape.clone(),
        },
        a,
        a.shape.clone(),
    )
}

/// Solve a linear system.
///
/// # Examples
///
/// ```rust,ignore
/// let x = tenferro::solve(&a, &b);
/// ```
pub fn solve(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        return b.scale_real(0.0);
    }
    if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        let b2d = b.reshape(&matrix_rhs_shape);
        let x2d = apply_binary(
            StdTensorOp::Solve {
                lhs_shape: a.shape.clone(),
                rhs_shape: matrix_rhs_shape.clone(),
            },
            a,
            &b2d,
            matrix_rhs_shape,
        );
        x2d.reshape(&b.shape)
    } else {
        apply_binary(
            StdTensorOp::Solve {
                lhs_shape: a.shape.clone(),
                rhs_shape: b.shape.clone(),
            },
            a,
            b,
            b.shape.clone(),
        )
    }
}

/// Solve a triangular linear system.
///
/// # Examples
///
/// ```rust,ignore
/// let x = tenferro::triangular_solve(&a, &b, true, true, false, false);
/// ```
pub fn triangular_solve(
    a: &TracedTensor,
    b: &TracedTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> TracedTensor {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        return b.scale_real(0.0);
    }
    let op = StdTensorOp::TriangularSolve {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
        lhs_shape: a.shape.clone(),
        rhs_shape: b.shape.clone(),
    };
    if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        let b2d = b.reshape(&matrix_rhs_shape);
        let x2d = apply_binary(
            StdTensorOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                lhs_shape: a.shape.clone(),
                rhs_shape: matrix_rhs_shape.clone(),
            },
            a,
            &b2d,
            matrix_rhs_shape,
        );
        x2d.reshape(&b.shape)
    } else {
        apply_binary(op, a, b, b.shape.clone())
    }
}

fn batched_vector_rhs_shape(a: &TracedTensor, b: &TracedTensor) -> Option<Vec<usize>> {
    if b.shape.len() == 1 {
        return Some(vec![b.shape[0], 1]);
    }

    let is_batched_vector_rhs = a.shape.len() == b.shape.len() + 1
        && !b.shape.is_empty()
        && b.shape[0] == a.shape[0]
        && b.shape[1..] == a.shape[2..];
    if !is_batched_vector_rhs {
        return None;
    }

    let mut rhs_shape = vec![b.shape[0], 1];
    rhs_shape.extend_from_slice(&b.shape[1..]);
    Some(rhs_shape)
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}
