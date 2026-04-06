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
    let op = StdTensorOp::Svd { eps, m, n };
    let mut results = apply_multi_output(op, a, vec![vec![m, k], vec![k], vec![k, n]]).into_iter();
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
    let mut results =
        apply_multi_output(StdTensorOp::Qr, a, vec![vec![m, k], vec![k, n]]).into_iter();
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
    let op = StdTensorOp::Eigh { eps };
    let mut results = apply_multi_output(op, a, vec![vec![n], vec![n, n]]).into_iter();
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
    let n = a.shape[0];
    apply_unary(StdTensorOp::Cholesky, a, vec![n, n])
}

/// Solve a linear system.
///
/// # Examples
///
/// ```rust,ignore
/// let x = tenferro::solve(&a, &b);
/// ```
pub fn solve(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let n = a.shape[0];
    let nrhs = if b.shape.len() > 1 { b.shape[1] } else { 1 };
    apply_binary(StdTensorOp::Solve, a, b, vec![n, nrhs])
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
    let op = StdTensorOp::TriangularSolve {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    };
    apply_binary(op, a, b, b.shape.clone())
}
