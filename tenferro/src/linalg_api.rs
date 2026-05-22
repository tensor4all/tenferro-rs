use num_complex::{Complex32, Complex64};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{CompareDir, DType, DotGeneralConfig};

use crate::scalar_semantics::round_real_to_i64;
use crate::sym_dim::SymDim;
use crate::traced::{
    apply_binary, apply_multi_output, apply_nullary, apply_unary, concrete_shape, TracedTensor,
};

/// Convert a traced tensor to a different dtype.
///
/// # Examples
///
/// ```rust,ignore
/// use tenferro::DType;
///
/// let y = tenferro::traced_tensor::convert(&x, DType::C64);
/// ```
pub fn convert(input: &TracedTensor, to: DType) -> TracedTensor {
    input.convert(to)
}

fn sym_shape(shape: &[usize]) -> Vec<SymDim> {
    shape.iter().copied().map(SymDim::from).collect()
}

/// Singular value decomposition with the default AD regularization epsilon.
///
/// The epsilon is used by the SVD AD rule to regularize divisions by small
/// singular-value gaps and small singular values. It does not change the
/// primal backend decomposition.
///
/// # Examples
///
/// ```rust,ignore
/// let (u, s, vt) = tenferro::traced_tensor::svd(&a);
/// ```
pub fn svd(a: &TracedTensor) -> (TracedTensor, TracedTensor, TracedTensor) {
    svd_with_eps(a, 1e-12)
}

/// Singular value decomposition with an explicit AD regularization epsilon.
///
/// The epsilon is used only when differentiating through the SVD. It
/// regularizes eigenvector-like singular-vector terms for repeated or nearly
/// repeated singular values and for singular values near zero; primal execution
/// returns the backend SVD result without applying this epsilon.
///
/// # Examples
///
/// ```rust,ignore
/// let (u, s, vt) = tenferro::traced_tensor::svd_with_eps(&a, 1e-10);
/// ```
pub fn svd_with_eps(a: &TracedTensor, eps: f64) -> (TracedTensor, TracedTensor, TracedTensor) {
    let shape = concrete_shape(a);
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let batch = &shape[2..];
    let op = StdTensorOp::Svd { eps };
    let mut u_shape = vec![m, k];
    u_shape.extend_from_slice(batch);
    let mut s_shape = vec![k];
    s_shape.extend_from_slice(batch);
    let mut vt_shape = vec![k, n];
    vt_shape.extend_from_slice(batch);
    let mut results = apply_multi_output(
        op,
        a,
        vec![
            sym_shape(&u_shape),
            sym_shape(&s_shape),
            sym_shape(&vt_shape),
        ],
    )
    .into_iter();
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
/// let (q, r) = tenferro::traced_tensor::qr(&a);
/// ```
pub fn qr(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    let shape = concrete_shape(a);
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let batch = &shape[2..];
    let mut q_shape = vec![m, k];
    q_shape.extend_from_slice(batch);
    let mut r_shape = vec![k, n];
    r_shape.extend_from_slice(batch);
    let mut results = apply_multi_output(
        StdTensorOp::Qr,
        a,
        vec![sym_shape(&q_shape), sym_shape(&r_shape)],
    )
    .into_iter();
    match (results.next(), results.next(), results.next()) {
        (Some(q), Some(r), None) => (q, r),
        _ => unreachable!("qr must produce exactly two outputs"),
    }
}

/// Hermitian eigenvalue decomposition with the default AD regularization epsilon.
///
/// The epsilon is used by the `eigh` AD rule to regularize divisions by small
/// eigenvalue gaps. It does not change the primal backend decomposition.
///
/// # Examples
///
/// ```rust,ignore
/// let (values, vectors) = tenferro::traced_tensor::eigh(&a);
/// ```
pub fn eigh(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    eigh_with_eps(a, 1e-12)
}

/// Hermitian eigenvalue decomposition with an explicit AD regularization epsilon.
///
/// The epsilon is used only when differentiating through the eigenvectors of
/// `eigh`. It regularizes terms involving small eigenvalue gaps; primal
/// execution returns the backend eigendecomposition result without applying
/// this epsilon.
///
/// # Examples
///
/// ```rust,ignore
/// let (values, vectors) = tenferro::traced_tensor::eigh_with_eps(&a, 1e-10);
/// ```
pub fn eigh_with_eps(a: &TracedTensor, eps: f64) -> (TracedTensor, TracedTensor) {
    let shape = concrete_shape(a);
    let n = shape[0];
    let batch = &shape[2..];
    let op = StdTensorOp::Eigh { eps };
    let mut vals_shape = vec![n];
    vals_shape.extend_from_slice(batch);
    let mut vecs_shape = vec![n, n];
    vecs_shape.extend_from_slice(batch);
    let mut results =
        apply_multi_output(op, a, vec![sym_shape(&vals_shape), sym_shape(&vecs_shape)]).into_iter();
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
/// let l = tenferro::traced_tensor::cholesky(&a);
/// ```
pub fn cholesky(a: &TracedTensor) -> TracedTensor {
    let shape = concrete_shape(a);
    apply_unary(StdTensorOp::Cholesky, a, a.rank, Some(sym_shape(&shape)))
}

/// LU decomposition with partial pivoting.
///
/// Returns `(P, L, U, parity)` where `P @ A = L @ U`.
///
/// # Examples
///
/// ```rust,ignore
/// let (p, l, u, parity) = tenferro::traced_tensor::lu(&a);
/// ```
pub fn lu(a: &TracedTensor) -> (TracedTensor, TracedTensor, TracedTensor, TracedTensor) {
    let shape = concrete_shape(a);
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let batch = &shape[2..];
    let mut p_shape = vec![m, m];
    p_shape.extend_from_slice(batch);
    let mut l_shape = vec![m, k];
    l_shape.extend_from_slice(batch);
    let mut u_shape = vec![k, n];
    u_shape.extend_from_slice(batch);
    let parity_shape = batch.to_vec();
    let mut results = apply_multi_output(
        StdTensorOp::Lu,
        a,
        vec![
            sym_shape(&p_shape),
            sym_shape(&l_shape),
            sym_shape(&u_shape),
            sym_shape(&parity_shape),
        ],
    )
    .into_iter();
    match (
        results.next(),
        results.next(),
        results.next(),
        results.next(),
        results.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(parity), None) => (p, l, u, parity),
        _ => unreachable!("lu must produce exactly four outputs"),
    }
}

/// LU decomposition with complete pivoting.
///
/// Returns `(P, L, U, Q, parity)` where `P @ A @ Q.T = L @ U`.
///
/// # Examples
///
/// ```
/// use tenferro::traced_tensor::full_piv_lu;
/// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
///
/// let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
///     vec![2, 2],
///     vec![0.0_f64, 2.0, 1.0, 3.0],
/// ));
/// let (p, l, u, q, parity) = full_piv_lu(&a);
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile_many(&[&p, &l, &u, &q, &parity]).unwrap();
/// let outputs = GraphExecutor::new(CpuBackend::new()).run_many(&program).unwrap();
///
/// assert_eq!(outputs[0].shape(), &[2, 2]);
/// assert_eq!(outputs[4].shape(), &[] as &[usize]);
/// ```
pub fn full_piv_lu(
    a: &TracedTensor,
) -> (
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
) {
    let shape = concrete_shape(a);
    let n = shape[0];
    let batch = &shape[2..];
    let mut p_shape = vec![n, n];
    p_shape.extend_from_slice(batch);
    let mut l_shape = vec![n, n];
    l_shape.extend_from_slice(batch);
    let mut u_shape = vec![n, n];
    u_shape.extend_from_slice(batch);
    let mut q_shape = vec![n, n];
    q_shape.extend_from_slice(batch);
    let parity_shape = batch.to_vec();
    let mut results = apply_multi_output(
        StdTensorOp::FullPivLu,
        a,
        vec![
            sym_shape(&p_shape),
            sym_shape(&l_shape),
            sym_shape(&u_shape),
            sym_shape(&q_shape),
            sym_shape(&parity_shape),
        ],
    )
    .into_iter();
    match (
        results.next(),
        results.next(),
        results.next(),
        results.next(),
        results.next(),
        results.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => (p, l, u, q, parity),
        _ => unreachable!("full_piv_lu must produce exactly five outputs"),
    }
}

/// Non-symmetric eigendecomposition.
///
/// For real `f64` input, both outputs are `Complex64`.
///
/// # Examples
///
/// ```rust,ignore
/// let (values, vectors) = tenferro::traced_tensor::eig(&a);
/// ```
pub fn eig(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    let shape = concrete_shape(a);
    let n = shape[0];
    let batch = &shape[2..];
    let mut vals_shape = vec![n];
    vals_shape.extend_from_slice(batch);
    let mut vecs_shape = vec![n, n];
    vecs_shape.extend_from_slice(batch);
    let eig_dtype = eig_output_dtype(a.dtype);
    let mut results = apply_multi_output(
        StdTensorOp::Eig {
            input_dtype: a.dtype,
        },
        a,
        vec![sym_shape(&vals_shape), sym_shape(&vecs_shape)],
    )
    .into_iter();
    match (results.next(), results.next(), results.next()) {
        (Some(mut values), Some(mut vectors), None) => {
            values.dtype = eig_dtype;
            vectors.dtype = eig_dtype;
            (values, vectors)
        }
        _ => unreachable!("eig must produce exactly two outputs"),
    }
}

fn validate_nonsingular(u: &TracedTensor) -> TracedTensor {
    apply_unary(
        StdTensorOp::ValidateNonsingular,
        u,
        u.rank,
        u.shape_hint.clone(),
    )
}

/// Solve a linear system using LU decomposition and triangular solves.
///
/// # Examples
///
/// ```rust,ignore
/// let x = tenferro::traced_tensor::solve(&a, &b);
/// ```
pub fn solve(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let a_shape = concrete_shape(a);
    let b_shape = concrete_shape(b);
    if has_zero_dim(&a_shape) || has_zero_dim(&b_shape) {
        return zeros_like(b);
    }

    let do_solve = |a: &TracedTensor, b: &TracedTensor| -> TracedTensor {
        let (p, l, u, _) = lu(a);
        let u = validate_nonsingular(&u);
        let pb = matmul_preserve_trailing_batch(&p, b);
        let z = triangular_solve(&l, &pb, true, true, false, true);
        triangular_solve(&u, &z, true, false, false, false)
    };

    if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        let b2d = b.reshape(&matrix_rhs_shape);
        let x2d = do_solve(a, &b2d);
        x2d.reshape(&b_shape)
    } else {
        do_solve(a, b)
    }
}

/// Solve a linear system using complete-pivoting LU factorization.
///
/// # Examples
///
/// ```
/// use tenferro::traced_tensor::full_piv_lu_solve;
/// use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
///
/// let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
///     vec![2, 2],
///     vec![0.0_f64, 2.0, 1.0, 3.0],
/// ));
/// let b = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
///     vec![2, 1],
///     vec![-1.0_f64, 5.0],
/// ));
/// let x = full_piv_lu_solve(&a, &b);
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&x).unwrap();
/// let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
///
/// assert_eq!(out.shape(), &[2, 1]);
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, -1.0]);
/// ```
pub fn full_piv_lu_solve(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let a_shape = concrete_shape(a);
    let b_shape = concrete_shape(b);
    if has_zero_dim(&a_shape) || has_zero_dim(&b_shape) {
        return zeros_like(b);
    }

    if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        let b2d = b.reshape(&matrix_rhs_shape);
        let x2d = apply_binary(
            StdTensorOp::FullPivLuSolve { transpose_a: false },
            a,
            &b2d,
            matrix_rhs_shape.len(),
            Some(sym_shape(&matrix_rhs_shape)),
        );
        x2d.reshape(&b_shape)
    } else {
        apply_binary(
            StdTensorOp::FullPivLuSolve { transpose_a: false },
            a,
            b,
            b.rank,
            b.shape_hint.clone(),
        )
    }
}

/// Solve a triangular linear system.
///
/// # Examples
///
/// ```rust,ignore
/// let x = tenferro::traced_tensor::triangular_solve(&a, &b, true, true, false, false);
/// ```
pub fn triangular_solve(
    a: &TracedTensor,
    b: &TracedTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> TracedTensor {
    let a_shape = concrete_shape(a);
    let b_shape = concrete_shape(b);
    if has_zero_dim(&a_shape) || has_zero_dim(&b_shape) {
        return zeros_like(b);
    }
    let op = StdTensorOp::TriangularSolve {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    };
    if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        let b2d = b.reshape(&matrix_rhs_shape);
        let x2d = apply_binary(
            StdTensorOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            },
            a,
            &b2d,
            matrix_rhs_shape.len(),
            Some(sym_shape(&matrix_rhs_shape)),
        );
        x2d.reshape(&b_shape)
    } else {
        apply_binary(op, a, b, b.rank, b.shape_hint.clone())
    }
}

/// Sign and log-absolute-determinant from the LU factorization.
///
/// # Examples
///
/// ```rust,ignore
/// let (sign, logabsdet) = tenferro::traced_tensor::slogdet(&a);
/// ```
pub fn slogdet(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    let shape = concrete_shape(a);
    let batch_shape = &shape[2..];
    if has_zero_dim(&shape) {
        let sign = broadcast_scalar(one_scalar(a.dtype), batch_shape);
        let logabsdet = broadcast_scalar(zero_scalar(a.dtype), batch_shape);
        return (sign, logabsdet);
    }

    let (_, _, u, parity) = lu(a);
    let diag_u = u.extract_diag(0, 1);
    let sign_u = diag_u.sign().reduce_prod(&[0]);
    let sign = &parity * &sign_u;
    let logabsdet = diag_u.abs().log().reduce_sum(&[0]);
    (sign, logabsdet)
}

/// Determinant from `slogdet`.
///
/// # Examples
///
/// ```rust,ignore
/// let value = tenferro::traced_tensor::det(&a);
/// ```
pub fn det(a: &TracedTensor) -> TracedTensor {
    let (sign, logabsdet) = slogdet(a);
    &sign * &logabsdet.exp()
}

/// Matrix inverse via `solve(a, eye)`.
///
/// # Examples
///
/// ```rust,ignore
/// let value = tenferro::traced_tensor::inv(&a);
/// ```
pub fn inv(a: &TracedTensor) -> TracedTensor {
    let shape = concrete_shape(a);
    let eye = eye_like(a, shape[0]);
    solve(a, &eye)
}

/// Hermitian eigenvalues only.
///
/// # Examples
///
/// ```rust,ignore
/// let values = tenferro::traced_tensor::eigvalsh(&a);
/// ```
pub fn eigvalsh(a: &TracedTensor) -> TracedTensor {
    eigh(a).0
}

/// General eigenvalues only.
///
/// # Examples
///
/// ```rust,ignore
/// let values = tenferro::traced_tensor::eigvals(&a);
/// ```
pub fn eigvals(a: &TracedTensor) -> TracedTensor {
    eig(a).0
}

/// Moore-Penrose pseudoinverse via the SVD.
///
/// # Examples
///
/// ```rust,ignore
/// let value = tenferro::traced_tensor::pinv(&a);
/// ```
pub fn pinv(a: &TracedTensor) -> TracedTensor {
    let shape = concrete_shape(a);
    let max_dim = match (shape.first(), shape.get(1)) {
        (Some(&m), Some(&n)) => m.max(n),
        (Some(&m), None) => m,
        _ => 0,
    };
    pinv_with_rtol(a, default_pinv_rtol(a.dtype, max_dim))
}

/// Moore-Penrose pseudoinverse via the SVD with an explicit relative cutoff.
///
/// Singular values `<= rtol * max(s)` are discarded.
///
/// # Examples
///
/// ```rust,ignore
/// let value = tenferro::traced_tensor::pinv_with_rtol(&a, 1.0e-8);
/// ```
pub fn pinv_with_rtol(a: &TracedTensor, rtol: f64) -> TracedTensor {
    let shape = concrete_shape(a);
    if has_zero_dim(&shape) {
        let mut out_shape = vec![shape[1], shape[0]];
        out_shape.extend_from_slice(&shape[2..]);
        return zeros_of_shape(a.dtype, out_shape);
    }

    let (u, s, vt) = svd(a);
    let abs_s = s.abs();
    let s_max = abs_s.reduce_max(&[0]);
    let s_max_shape = concrete_shape(&s_max);
    let threshold = &s_max * &broadcast_scalar(scalar_real(s.dtype, rtol.max(0.0)), &s_max_shape);
    let s_shape = concrete_shape(&s);
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, &s_shape);
    let mask = compare_dir(&abs_s, &threshold, CompareDir::Gt);
    let ones = ones_like(&s);
    let denom = &s + &(&ones + &(-&mask));
    let s_inv = &mask / &denom;

    let v = vt.conj().transpose(&matrix_transpose_perm(vt.rank));
    let uh = u.conj().transpose(&matrix_transpose_perm(u.rank));
    let s_inv_diag = s_inv.embed_diag(0, 1);
    let vs = matmul_preserve_trailing_batch(&v, &s_inv_diag);
    matmul_preserve_trailing_batch(&vs, &uh)
}

/// Vector or matrix norm.
///
/// This currently covers Frobenius norms, p-norms, and `±inf` reductions.
///
/// # Examples
///
/// ```rust,ignore
/// let value = tenferro::traced_tensor::norm(&a, None, None, false);
/// ```
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
    let shape = concrete_shape(a);
    restore_keepdim(out, &shape, &axes, keepdim)
}

fn eig_output_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F64 | DType::C64 => DType::C64,
        DType::F32 | DType::C32 => DType::C32,
        DType::I64 => DType::C64,
    }
}

fn scalar_real(dtype: DType, value: f64) -> TracedTensor {
    match dtype {
        DType::F64 => apply_nullary(
            StdTensorOp::constant_f64(value),
            0,
            DType::F64,
            Some(vec![]),
        ),
        DType::F32 => apply_nullary(
            StdTensorOp::constant_f32(value as f32),
            0,
            DType::F32,
            Some(vec![]),
        ),
        DType::I64 => apply_nullary(
            StdTensorOp::constant_i64(round_real_to_i64(value)),
            0,
            DType::I64,
            Some(vec![]),
        ),
        DType::C64 => apply_nullary(
            StdTensorOp::constant_c64(Complex64::new(value, 0.0)),
            0,
            DType::C64,
            Some(vec![]),
        ),
        DType::C32 => apply_nullary(
            StdTensorOp::constant_c32(Complex32::new(value as f32, 0.0)),
            0,
            DType::C32,
            Some(vec![]),
        ),
    }
}

fn zero_scalar(dtype: DType) -> TracedTensor {
    scalar_real(dtype, 0.0)
}

fn one_scalar(dtype: DType) -> TracedTensor {
    scalar_real(dtype, 1.0)
}

fn zeros_like(input: &TracedTensor) -> TracedTensor {
    zeros_of_shape(input.dtype, concrete_shape(input))
}

fn zeros_of_shape(dtype: DType, shape: Vec<usize>) -> TracedTensor {
    broadcast_scalar(zero_scalar(dtype), &shape)
}

fn ones_like(input: &TracedTensor) -> TracedTensor {
    let shape = concrete_shape(input);
    broadcast_scalar(one_scalar(input.dtype), &shape)
}

fn eye_like(anchor: &TracedTensor, size: usize) -> TracedTensor {
    let mut vector_shape = vec![size];
    let anchor_shape = concrete_shape(anchor);
    vector_shape.extend_from_slice(&anchor_shape[2..]);
    let diagonal = broadcast_scalar(one_scalar(anchor.dtype), &vector_shape);
    diagonal.embed_diag(0, 1)
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
        DType::I64 => 0.0,
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
    let mask = compare_dir(abs, &zero_scalar(abs.dtype), CompareDir::Gt);
    mask.reduce_sum(axes)
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

fn compare_dir(lhs: &TracedTensor, rhs: &TracedTensor, dir: CompareDir) -> TracedTensor {
    let (lhs, rhs) = broadcast_binary(lhs, rhs);
    apply_binary(
        StdTensorOp::Compare(dir),
        &lhs,
        &rhs,
        lhs.rank,
        lhs.shape_hint.clone(),
    )
}

fn broadcast_scalar(input: TracedTensor, shape: &[usize]) -> TracedTensor {
    let input_shape = concrete_shape(&input);
    if input_shape == shape {
        return input;
    }
    input.broadcast_in_dim(shape, &[])
}

fn broadcast_batch_scalar_to_leading_axis(input: &TracedTensor, shape: &[usize]) -> TracedTensor {
    let input_shape = concrete_shape(input);
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

fn batched_vector_rhs_shape(a: &TracedTensor, b: &TracedTensor) -> Option<Vec<usize>> {
    let a_shape = concrete_shape(a);
    let b_shape = concrete_shape(b);

    if b_shape.len() == 1 {
        return Some(vec![b_shape[0], 1]);
    }

    let is_batched_vector_rhs = a_shape.len() == b_shape.len() + 1
        && !b_shape.is_empty()
        && b_shape[0] == a_shape[0]
        && b_shape[1..] == a_shape[2..];
    if !is_batched_vector_rhs {
        return None;
    }

    let mut rhs_shape = vec![b_shape[0], 1];
    rhs_shape.extend_from_slice(&b_shape[1..]);
    Some(rhs_shape)
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(rank);
    for index in 0..rank {
        let a_dim = if index < rank - a.len() {
            1
        } else {
            a[index - (rank - a.len())]
        };
        let b_dim = if index < rank - b.len() {
            1
        } else {
            b[index - (rank - b.len())]
        };
        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return None;
        }
    }
    Some(result)
}

fn broadcast_to(tensor: &TracedTensor, target_shape: &[usize]) -> TracedTensor {
    let tensor_shape = concrete_shape(tensor);
    if tensor_shape == target_shape {
        return tensor.clone();
    }

    assert!(
        tensor.rank <= target_shape.len(),
        "cannot broadcast higher-rank shape {:?} to {:?}",
        tensor_shape,
        target_shape
    );

    let rank_diff = target_shape.len() - tensor.rank;
    let mut source_shape = Vec::with_capacity(tensor.rank);
    let mut dims = Vec::with_capacity(tensor.rank);
    for (src_axis, &src_dim) in tensor_shape.iter().enumerate() {
        let dst_axis = src_axis + rank_diff;
        let dst_dim = target_shape[dst_axis];
        assert!(
            src_dim == dst_dim || src_dim == 1,
            "cannot broadcast shape {:?} to {:?}",
            tensor_shape,
            target_shape
        );
        if src_dim == 1 && dst_dim != 1 {
            continue;
        }
        source_shape.push(src_dim);
        dims.push(dst_axis);
    }

    let source = if source_shape == tensor_shape {
        tensor.clone()
    } else {
        tensor.reshape(&source_shape)
    };
    source.broadcast_in_dim(target_shape, &dims)
}

fn broadcast_binary(a: &TracedTensor, b: &TracedTensor) -> (TracedTensor, TracedTensor) {
    if a.shape_hint == b.shape_hint && a.rank == b.rank {
        return (a.clone(), b.clone());
    }
    let a_shape = concrete_shape(a);
    let b_shape = concrete_shape(b);
    let target = broadcast_shape(&a_shape, &b_shape).unwrap_or_else(|| {
        panic!(
            "incompatible shapes for broadcast: {:?} and {:?}",
            a_shape, b_shape
        )
    });
    (broadcast_to(a, &target), broadcast_to(b, &target))
}
