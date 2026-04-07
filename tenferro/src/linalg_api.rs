use num_complex::{Complex32, Complex64};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{CompareDir, DType, DotGeneralConfig};

use crate::traced::{apply_binary, apply_multi_output, apply_nullary, apply_unary, TracedTensor};

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

/// LU decomposition with partial pivoting.
///
/// Returns `(P, L, U, parity)` where `P @ A = L @ U`.
///
/// # Examples
///
/// ```rust,ignore
/// let (p, l, u, parity) = tenferro::lu(&a);
/// ```
pub fn lu(a: &TracedTensor) -> (TracedTensor, TracedTensor, TracedTensor, TracedTensor) {
    let m = a.shape[0];
    let n = a.shape[1];
    let k = m.min(n);
    let batch = &a.shape[2..];
    let mut p_shape = vec![m, m];
    p_shape.extend_from_slice(batch);
    let mut l_shape = vec![m, k];
    l_shape.extend_from_slice(batch);
    let mut u_shape = vec![k, n];
    u_shape.extend_from_slice(batch);
    let parity_shape = batch.to_vec();
    let mut results = apply_multi_output(
        StdTensorOp::Lu {
            input_shape: a.shape.clone(),
        },
        a,
        vec![p_shape, l_shape, u_shape, parity_shape],
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

/// Non-symmetric eigendecomposition.
///
/// For real `f64` input, both outputs are `Complex64`.
///
/// # Examples
///
/// ```rust,ignore
/// let (values, vectors) = tenferro::eig(&a);
/// ```
pub fn eig(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    let n = a.shape[0];
    let batch = &a.shape[2..];
    let mut vals_shape = vec![n];
    vals_shape.extend_from_slice(batch);
    let mut vecs_shape = vec![n, n];
    vecs_shape.extend_from_slice(batch);
    let eig_dtype = eig_output_dtype(a.dtype);
    let mut results = apply_multi_output(
        StdTensorOp::Eig {
            input_shape: a.shape.clone(),
        },
        a,
        vec![vals_shape, vecs_shape],
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

/// Solve a linear system using LU decomposition and triangular solves.
///
/// # Examples
///
/// ```rust,ignore
/// let x = tenferro::solve(&a, &b);
/// ```
pub fn solve(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    if has_zero_dim(&a.shape) || has_zero_dim(&b.shape) {
        return zeros_like(b);
    }

    let do_solve = |a: &TracedTensor, b: &TracedTensor| -> TracedTensor {
        let (p, l, u, _) = lu(a);
        let pb = matmul_preserve_trailing_batch(&p, b);
        let z = triangular_solve(&l, &pb, true, true, false, true);
        triangular_solve(&u, &z, true, false, false, false)
    };

    if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        let b2d = b.reshape(&matrix_rhs_shape);
        let x2d = do_solve(a, &b2d);
        x2d.reshape(&b.shape)
    } else {
        do_solve(a, b)
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
        return zeros_like(b);
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

/// Sign and log-absolute-determinant from the LU factorization.
///
/// # Examples
///
/// ```rust,ignore
/// let (sign, logabsdet) = tenferro::slogdet(&a);
/// ```
pub fn slogdet(a: &TracedTensor) -> (TracedTensor, TracedTensor) {
    let batch_shape = &a.shape[2..];
    if has_zero_dim(&a.shape) {
        let sign = broadcast_scalar(one_scalar(a.dtype), batch_shape);
        let logabsdet = broadcast_scalar(zero_scalar(a.dtype), batch_shape);
        return (sign, logabsdet);
    }

    let (_, _, u, parity) = lu(a);
    let diag_u = u.extract_diag(0, 1);
    let sign_u = reduce_prod(&diag_u.sign(), &[0]);
    let sign = &parity * &sign_u;
    let logabsdet = diag_u.abs().log().reduce_sum(&[0]);
    (sign, logabsdet)
}

/// Determinant from `slogdet`.
///
/// # Examples
///
/// ```rust,ignore
/// let value = tenferro::det(&a);
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
/// let value = tenferro::inv(&a);
/// ```
pub fn inv(a: &TracedTensor) -> TracedTensor {
    let eye = eye_like(a, a.shape[0]);
    solve(a, &eye)
}

/// Hermitian eigenvalues only.
///
/// # Examples
///
/// ```rust,ignore
/// let values = tenferro::eigvalsh(&a);
/// ```
pub fn eigvalsh(a: &TracedTensor) -> TracedTensor {
    eigh(a).0
}

/// General eigenvalues only.
///
/// # Examples
///
/// ```rust,ignore
/// let values = tenferro::eigvals(&a);
/// ```
pub fn eigvals(a: &TracedTensor) -> TracedTensor {
    eig(a).0
}

/// Moore-Penrose pseudoinverse via the SVD.
///
/// # Examples
///
/// ```rust,ignore
/// let value = tenferro::pinv(&a);
/// ```
pub fn pinv(a: &TracedTensor) -> TracedTensor {
    if has_zero_dim(&a.shape) {
        let mut out_shape = vec![a.shape[1], a.shape[0]];
        out_shape.extend_from_slice(&a.shape[2..]);
        return zeros_of_shape(a.dtype, out_shape);
    }

    let (u, s, vt) = svd(a);
    let abs_s = s.abs();
    let s_max = reduce_max(&abs_s, &[0]);
    let threshold = &s_max * &broadcast_scalar(scalar_real(s.dtype, 1e-12), &s_max.shape);
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, &s.shape);
    let mask = compare_dir(&abs_s, &threshold, CompareDir::Gt);
    let ones = ones_like(&s);
    let denom = &s + &(&ones + &(-&mask));
    let s_inv = &mask / &denom;

    let v = vt.conj().transpose(&matrix_transpose_perm(vt.shape.len()));
    let uh = u.conj().transpose(&matrix_transpose_perm(u.shape.len()));
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
/// let value = tenferro::norm(&a, None, None, false);
/// ```
pub fn norm(
    a: &TracedTensor,
    ord: Option<f64>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> TracedTensor {
    let axes = dim.map_or_else(
        || (0..a.shape.len()).collect::<Vec<_>>(),
        |dims| dims.to_vec(),
    );
    if axes.is_empty() {
        return a.clone();
    }

    let abs = a.abs();
    let out = match ord {
        None => frobenius_norm(&abs, &axes),
        Some(p) if p == f64::INFINITY => reduce_max(&abs, &axes),
        Some(p) if p == f64::NEG_INFINITY => reduce_min(&abs, &axes),
        Some(p) if axes.len() == 2 && p == 2.0 => {
            let singular_values = svd(a).1.abs();
            reduce_max(&singular_values, &[0])
        }
        Some(p) if axes.len() == 2 && p == -2.0 => {
            let singular_values = svd(a).1.abs();
            reduce_min(&singular_values, &[0])
        }
        Some(p) => p_norm(&abs, &axes, p),
    };
    restore_keepdim(out, &a.shape, &axes, keepdim)
}

fn eig_output_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F64 | DType::C64 => DType::C64,
        DType::F32 | DType::C32 => DType::C32,
    }
}

fn scalar_real(dtype: DType, value: f64) -> TracedTensor {
    match dtype {
        DType::F64 => apply_nullary(StdTensorOp::constant_f64(value), vec![], DType::F64),
        DType::F32 => apply_nullary(StdTensorOp::constant_f32(value as f32), vec![], DType::F32),
        DType::C64 => apply_nullary(
            StdTensorOp::constant_c64(Complex64::new(value, 0.0)),
            vec![],
            DType::C64,
        ),
        DType::C32 => apply_nullary(
            StdTensorOp::constant_c32(Complex32::new(value as f32, 0.0)),
            vec![],
            DType::C32,
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
    zeros_of_shape(input.dtype, input.shape.clone())
}

fn zeros_of_shape(dtype: DType, shape: Vec<usize>) -> TracedTensor {
    broadcast_scalar(zero_scalar(dtype), &shape)
}

fn ones_like(input: &TracedTensor) -> TracedTensor {
    broadcast_scalar(one_scalar(input.dtype), &input.shape)
}

fn eye_like(anchor: &TracedTensor, size: usize) -> TracedTensor {
    let mut vector_shape = vec![size];
    vector_shape.extend_from_slice(&anchor.shape[2..]);
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

fn reduce_prod(input: &TracedTensor, axes: &[usize]) -> TracedTensor {
    let out_shape = reduced_shape(&input.shape, axes);
    apply_unary(
        StdTensorOp::ReduceProd {
            axes: axes.to_vec(),
            input_shape: input.shape.clone(),
        },
        input,
        out_shape,
    )
}

fn reduce_max(input: &TracedTensor, axes: &[usize]) -> TracedTensor {
    let out_shape = reduced_shape(&input.shape, axes);
    apply_unary(
        StdTensorOp::ReduceMax {
            axes: axes.to_vec(),
            input_shape: input.shape.clone(),
        },
        input,
        out_shape,
    )
}

fn reduce_min(input: &TracedTensor, axes: &[usize]) -> TracedTensor {
    let out_shape = reduced_shape(&input.shape, axes);
    apply_unary(
        StdTensorOp::ReduceMin {
            axes: axes.to_vec(),
            input_shape: input.shape.clone(),
        },
        input,
        out_shape,
    )
}

fn compare_dir(lhs: &TracedTensor, rhs: &TracedTensor, dir: CompareDir) -> TracedTensor {
    let (lhs, rhs) = broadcast_binary(lhs, rhs);
    apply_binary(StdTensorOp::Compare(dir), &lhs, &rhs, lhs.shape.clone())
}

fn broadcast_scalar(input: TracedTensor, shape: &[usize]) -> TracedTensor {
    if input.shape == shape {
        return input;
    }
    input.broadcast_in_dim(shape, &[])
}

fn broadcast_batch_scalar_to_leading_axis(input: &TracedTensor, shape: &[usize]) -> TracedTensor {
    if input.shape == shape {
        return input.clone();
    }
    let dims: Vec<usize> = (1..shape.len()).collect();
    input.broadcast_in_dim(shape, &dims)
}

fn matmul_preserve_trailing_batch(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    let rank = lhs.shape.len();
    let batch_dims: Vec<usize> = (2..rank).collect();
    let out = lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: batch_dims.clone(),
            rhs_batch_dims: batch_dims,
            lhs_rank: rank,
            rhs_rank: rank,
        },
    );
    if rank <= 2 {
        return out;
    }
    out.transpose(&matmul_output_perm(rank))
}

fn matmul_output_perm(rank: usize) -> Vec<usize> {
    let batch_ndim = rank - 2;
    let mut perm = Vec::with_capacity(rank);
    perm.push(batch_ndim);
    perm.push(batch_ndim + 1);
    perm.extend(0..batch_ndim);
    perm
}

fn matrix_transpose_perm(rank: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn reduced_shape(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    (0..shape.len())
        .filter(|axis| !axes.contains(axis))
        .map(|axis| shape[axis])
        .collect()
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
    if tensor.shape == target_shape {
        return tensor.clone();
    }

    assert!(
        tensor.shape.len() <= target_shape.len(),
        "cannot broadcast higher-rank shape {:?} to {:?}",
        tensor.shape,
        target_shape
    );

    let rank_diff = target_shape.len() - tensor.shape.len();
    let mut source_shape = Vec::with_capacity(tensor.shape.len());
    let mut dims = Vec::with_capacity(tensor.shape.len());
    for (src_axis, &src_dim) in tensor.shape.iter().enumerate() {
        let dst_axis = src_axis + rank_diff;
        let dst_dim = target_shape[dst_axis];
        assert!(
            src_dim == dst_dim || src_dim == 1,
            "cannot broadcast shape {:?} to {:?}",
            tensor.shape,
            target_shape
        );
        if src_dim == 1 && dst_dim != 1 {
            continue;
        }
        source_shape.push(src_dim);
        dims.push(dst_axis);
    }

    let source = if source_shape == tensor.shape {
        tensor.clone()
    } else {
        tensor.reshape(&source_shape)
    };
    source.broadcast_in_dim(target_shape, &dims)
}

fn broadcast_binary(a: &TracedTensor, b: &TracedTensor) -> (TracedTensor, TracedTensor) {
    if a.shape == b.shape {
        return (a.clone(), b.clone());
    }
    let target = broadcast_shape(&a.shape, &b.shape).unwrap_or_else(|| {
        panic!(
            "incompatible shapes for broadcast: {:?} and {:?}",
            a.shape, b.shape
        )
    });
    (broadcast_to(a, &target), broadcast_to(b, &target))
}
