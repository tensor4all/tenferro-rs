use super::*;

/// Validate that tensor has at least 2 dimensions.
/// Returns (m, n, batch_dims_slice).
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
/// Returns (n, batch_dims_slice).
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
///
/// Current implementation supports vector RHS only:
/// `a: (m, n, *)`, `b: (m, *)`.
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
/// For primal output shape `(*)`, cotangent must have the same shape.
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

/// Validate Hermitian/symmetric structure for batched square matrices stored
/// in column-major contiguous layout.
///
/// For complex types, checks `A[i,j] == conj(A[j,i])`.
/// For real types, checks `A[i,j] == A[j,i]`.
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
                // Hermitian check: a_ij should equal conj(a_ji)
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

/// Ensure tensor is column-major contiguous. Returns a (possibly cloned) contiguous tensor.
pub(crate) fn ensure_col_major<T: LinalgScalar>(tensor: &Tensor<T>) -> Tensor<T> {
    tensor.contiguous(MemoryOrder::ColumnMajor)
}

/// Extract the raw data slice from a tensor.
///
/// Returns an error if the tensor buffer cannot be viewed as a contiguous slice
/// (e.g., non-CPU buffer or unexpected memory layout).
pub(crate) fn extract_slice<T: LinalgScalar>(tensor: &Tensor<T>) -> Result<&[T]> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::InvalidArgument("tensor buffer is not a contiguous CPU slice".into()))
}

/// Convert an f64 constant to scalar type T.
///
/// Returns an error if the conversion is not supported by the scalar type.
pub(crate) fn scalar_from<T: LinalgScalar>(val: f64) -> Result<T> {
    T::from(val).ok_or_else(|| {
        Error::InvalidArgument(format!("cannot convert {val} to target scalar type"))
    })
}

/// Convert a tenferro_device::Error into an AutodiffError for use in AD functions.
pub(crate) fn to_ad_err(e: Error) -> chainrules_core::AutodiffError {
    chainrules_core::AutodiffError::InvalidArgument(e.to_string())
}

/// Compute batch count from batch dims (product, or 1 if empty).
pub(crate) fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

/// Build output dims: [mat_dims..., batch_dims...].
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

/// Create a Tensor from raw column-major data with the given dims.
pub(crate) fn tensor_from_data<T: LinalgScalar>(data: Vec<T>, dims: &[usize]) -> Result<Tensor<T>> {
    let strides = backend::col_major_strides(dims);
    Tensor::from_vec(data, dims, &strides, 0)
}

/// Create a Tensor from raw column-major data with the given dims.
///
/// Like [`tensor_from_data`] but only requires `Scalar`, so it works for
/// `Complex<R>` types that are not `LinalgScalar`.
pub(crate) fn tensor_from_data_scalar<T: Scalar>(
    data: Vec<T>,
    dims: &[usize],
) -> Result<Tensor<T>> {
    let strides = backend::col_major_strides(dims);
    Tensor::from_vec(data, dims, &strides, 0)
}

// ============================================================================
// matrix_exp helpers (private)
// ============================================================================

/// Pad\u{e9}\[13/13\] coefficients b\[0\]..b\[13\] (integer values as f64).
pub(crate) const PADE13_COEFFS: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];

/// Theta threshold for order-13 Pad\u{e9} (f64).
pub(crate) const THETA_13: f64 = 5.371920351148152;

/// Compute the matrix 1-norm (max column sum of absolute values).
///
/// `a` is stored column-major as a flat slice of length `n*n`.
pub(crate) fn matrix_1_norm<T: LinalgScalar>(a: &[T], n: usize) -> T::Real {
    let mut max_col_sum = <T::Real as num_traits::Zero>::zero();
    for j in 0..n {
        let mut col_sum = <T::Real as num_traits::Zero>::zero();
        for i in 0..n {
            col_sum = col_sum + a[i + j * n].abs_real();
        }
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}

/// Multiply two n x n column-major matrices using the backend.
pub(crate) fn backend_mat_mul_nn<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    prims_bridge::batched_gemm_via_prims(a, n, n, b, n)
}

/// Compute `result = alpha * a + beta * b` element-wise for flat slices.
pub(crate) fn mat_linear_combine<T: LinalgScalar>(
    alpha: T,
    a: &[T],
    beta: T,
    b: &[T],
    result: &mut [T],
) {
    for i in 0..result.len() {
        result[i] = alpha * a[i] + beta * b[i];
    }
}

/// Build an n x n identity matrix in column-major flat layout.
pub(crate) fn identity_matrix<T: LinalgScalar>(n: usize) -> Vec<T> {
    let mut eye = vec![T::zero(); n * n];
    for i in 0..n {
        eye[i + i * n] = T::one();
    }
    eye
}

/// Scale a flat matrix slice by a scalar, returning a new vector.
pub(crate) fn mat_scale<T: LinalgScalar>(a: &[T], s: T) -> Vec<T> {
    a.iter().map(|&x| x * s).collect()
}

/// Add two flat matrix slices element-wise, returning a new vector.
pub(crate) fn mat_add<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Compute matrix exponential of a single n x n column-major matrix.
///
/// Uses scaling-and-squaring with Pad\u{e9}\[13/13\] approximation.
pub(crate) fn matrix_exp_single<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    // Special case: 0x0 matrix
    if n == 0 {
        return Ok(Vec::new());
    }

    // Special case: 1x1 matrix
    if n == 1 {
        let a_f64: f64 = num_traits::NumCast::from(a[0]).ok_or_else(|| {
            Error::InvalidArgument("matrix_exp: cannot convert 1×1 element to f64".into())
        })?;
        let exp_val = a_f64.exp();
        let result_val = T::from(exp_val).ok_or_else(|| {
            Error::InvalidArgument("cannot convert exp result to target type".into())
        })?;
        return Ok(vec![result_val]);
    }

    // 1. Compute ||A||_1
    let norm_a = matrix_1_norm(a, n);
    let norm_f64: f64 = num_traits::NumCast::from(norm_a)
        .ok_or_else(|| Error::InvalidArgument("matrix_exp: cannot convert 1-norm to f64".into()))?;

    // 2. Determine scaling factor s
    let s: usize = if norm_f64 <= THETA_13 {
        0
    } else {
        (norm_f64 / THETA_13).log2().ceil().max(0.0) as usize
    };

    // 3. Scale A: a_scaled = A / 2^s
    let scale_denom = (1u64 << s.min(63)) as f64;
    let scale_inv = T::from(1.0 / scale_denom).ok_or_else(|| {
        Error::InvalidArgument("cannot convert scale factor to target type".into())
    })?;
    let a_scaled = mat_scale(a, scale_inv);

    // 4. Compute matrix powers: A2, A4, A6
    let a2 = backend_mat_mul_nn(ctx, &a_scaled, &a_scaled, n)?;
    let a4 = backend_mat_mul_nn(ctx, &a2, &a2, n)?;
    let a6 = backend_mat_mul_nn(ctx, &a4, &a2, n)?;

    // Convert Pade coefficients to type T
    let b: Vec<T> = PADE13_COEFFS
        .iter()
        .map(|&c| {
            T::from(c).ok_or_else(|| {
                Error::InvalidArgument("cannot convert Pade coefficient to target type".into())
            })
        })
        .collect::<Result<Vec<T>>>()?;

    let eye = identity_matrix::<T>(n);
    let nn = n * n;

    // 5. Compute U and V for Pade[13/13]:
    //
    //   inner_u = b[13]*A6 + b[11]*A4 + b[9]*A2
    //   U = A * (A6 * inner_u + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I)
    //
    //   inner_v = b[12]*A6 + b[10]*A4 + b[8]*A2
    //   V = A6 * inner_v + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I

    // Compute inner_u = b[13]*A6 + b[11]*A4 + b[9]*A2
    let mut inner_u = vec![T::zero(); nn];
    for i in 0..nn {
        inner_u[i] = b[13] * a6[i] + b[11] * a4[i] + b[9] * a2[i];
    }

    // a6_inner_u = A6 * inner_u
    let a6_inner_u = backend_mat_mul_nn(ctx, &a6, &inner_u, n)?;

    // u_inner = a6_inner_u + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I
    let mut u_inner = vec![T::zero(); nn];
    for i in 0..nn {
        u_inner[i] = a6_inner_u[i] + b[7] * a6[i] + b[5] * a4[i] + b[3] * a2[i] + b[1] * eye[i];
    }

    // U = A_scaled * u_inner
    let u = backend_mat_mul_nn(ctx, &a_scaled, &u_inner, n)?;

    // Compute inner_v = b[12]*A6 + b[10]*A4 + b[8]*A2
    let mut inner_v = vec![T::zero(); nn];
    for i in 0..nn {
        inner_v[i] = b[12] * a6[i] + b[10] * a4[i] + b[8] * a2[i];
    }

    // a6_inner_v = A6 * inner_v
    let a6_inner_v = backend_mat_mul_nn(ctx, &a6, &inner_v, n)?;

    // V = a6_inner_v + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    let mut v = vec![T::zero(); nn];
    for i in 0..nn {
        v[i] = a6_inner_v[i] + b[6] * a6[i] + b[4] * a4[i] + b[2] * a2[i] + b[0] * eye[i];
    }

    // 6. Solve (-U + V) * X = (U + V)  =>  X = exp(A_scaled)
    let neg_one = T::from(-1.0)
        .ok_or_else(|| Error::InvalidArgument("cannot convert -1 to target type".into()))?;
    // lhs = V - U = -U + V
    let mut lhs = vec![T::zero(); nn];
    mat_linear_combine(neg_one, &u, T::one(), &v, &mut lhs);

    // rhs = U + V
    let rhs = mat_add(&u, &v);

    // Solve lhs * X = rhs  (nrhs = n for matrix RHS)
    let mut result = vec![T::zero(); nn];
    backend::cpu::solve_slices(&lhs, &rhs, n, n, &mut result)?;

    // 7. Repeated squaring: result = result^(2^s)
    for _ in 0..s {
        result = backend_mat_mul_nn(ctx, &result, &result, n)?;
    }

    Ok(result)
}

pub(crate) fn matrix_power_single<T: LinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    n: usize,
    exponent: u64,
) -> Result<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    if exponent == 1 {
        return Ok(a.to_vec());
    }

    let mut result = identity_matrix::<T>(n);
    let mut base = a.to_vec();
    let mut power = exponent;

    while power > 0 {
        if power & 1 == 1 {
            result = backend_mat_mul_nn(ctx, &result, &base, n)?;
        }
        power >>= 1;
        if power > 0 {
            base = backend_mat_mul_nn(ctx, &base, &base, n)?;
        }
    }

    Ok(result)
}

pub(crate) fn lu_factor_impl<T: LinalgScalar, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<LuFactorExResult<T>>
where
    C: backend::TensorLinalgContextFor<T>,
{
    let (m, n, batch_dims) = validate_2d(tensor)?;
    let bc = batch_count(batch_dims);
    let k = m.min(n);
    let result = <C::Backend as backend::TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;
    let factors = pack_lu_factors(&result.l, &result.u, m, n, batch_dims)?;

    let u_input = ensure_col_major(&result.u);
    let u_data = extract_slice(&u_input)?;
    let u_offset = u_input.offset() as usize;
    let mut info = vec![0_i32; bc];

    for (batch, info_slot) in info.iter_mut().enumerate().take(bc) {
        let start = u_offset + batch * k * n;
        let u_slice = &u_data[start..start + k * n];
        for i in 0..k {
            if u_slice[i + i * k].abs_real() <= T::real_epsilon() {
                *info_slot = (i + 1) as i32;
                break;
            }
        }
    }

    Ok(LuFactorExResult {
        factors,
        pivots: result
            .pivots
            .into_iter()
            .map(|pivot| pivot as usize)
            .collect(),
        info,
    })
}

pub(crate) fn pack_lu_factors<T: LinalgScalar>(
    l: &Tensor<T>,
    u: &Tensor<T>,
    m: usize,
    n: usize,
    batch_dims: &[usize],
) -> Result<Tensor<T>> {
    let bc = batch_count(batch_dims);
    let k = m.min(n);
    let l_input = ensure_col_major(l);
    let u_input = ensure_col_major(u);
    let l_data = extract_slice(&l_input)?;
    let u_data = extract_slice(&u_input)?;
    let l_offset = l_input.offset() as usize;
    let u_offset = u_input.offset() as usize;
    let mut packed = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let l_start = l_offset + batch * m * k;
        let u_start = u_offset + batch * k * n;
        let l_slice = &l_data[l_start..l_start + m * k];
        let u_slice = &u_data[u_start..u_start + k * n];
        let packed_slice = &mut packed[batch * m * n..(batch + 1) * m * n];
        for j in 0..n {
            for i in 0..m {
                packed_slice[i + j * m] = if i > j {
                    l_slice[i + j * m]
                } else {
                    u_slice[i + j * k]
                };
            }
        }
    }

    tensor_from_data(packed, &output_dims(&[m, n], batch_dims))
}

pub(crate) fn lu_solve_impl<T: LinalgScalar, C>(
    _ctx: &mut C,
    factors: &Tensor<T>,
    pivots: &[usize],
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: backend::CpuLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    let (n, batch_dims) = validate_square(factors)?;
    let rhs = backend::tensor_helpers::validate_solve_rhs_shape(b, n, batch_dims, "lu_solve")?;
    let bc = batch_count(batch_dims);
    let expected_pivots = n * bc;
    if pivots.len() != expected_pivots {
        return Err(Error::InvalidArgument(format!(
            "lu_solve expects pivots.len() == {expected_pivots}, got {}",
            pivots.len()
        )));
    }

    let factors_input = ensure_col_major(factors);
    let rhs_input = ensure_col_major(b);
    let factors_data = extract_slice(&factors_input)?;
    let rhs_data = extract_slice(&rhs_input)?;
    let factors_offset = factors_input.offset() as usize;
    let rhs_offset = rhs_input.offset() as usize;

    let mat_size = n * n;
    let rhs_size = n * rhs.nrhs;
    let mut out = vec![T::zero(); rhs_size * bc];
    let mut lower = vec![T::zero(); mat_size];
    let mut upper = vec![T::zero(); mat_size];
    let mut permuted_rhs = vec![T::zero(); rhs_size];
    let mut tmp = vec![T::zero(); rhs_size];

    for batch in 0..bc {
        let factor_start = factors_offset + batch * mat_size;
        let rhs_start = rhs_offset + batch * rhs_size;
        let factor_slice = &factors_data[factor_start..factor_start + mat_size];
        let rhs_slice = &rhs_data[rhs_start..rhs_start + rhs_size];
        let pivot_slice = &pivots[batch * n..(batch + 1) * n];

        unpack_packed_lu_square(factor_slice, n, &mut lower, &mut upper);
        apply_lu_permutation(pivot_slice, rhs_slice, n, rhs.nrhs, &mut permuted_rhs)?;
        backend::cpu::solve_triangular_slices(&lower, &permuted_rhs, n, rhs.nrhs, false, &mut tmp)?;
        backend::cpu::solve_triangular_slices(
            &upper,
            &tmp,
            n,
            rhs.nrhs,
            true,
            &mut out[batch * rhs_size..(batch + 1) * rhs_size],
        )?;
    }

    tensor_from_data(out, &rhs.output_dims)
}

pub(crate) fn unpack_packed_lu_square<T: LinalgScalar>(
    factors: &[T],
    n: usize,
    lower: &mut [T],
    upper: &mut [T],
) {
    lower.fill(T::zero());
    upper.fill(T::zero());
    for j in 0..n {
        for i in 0..n {
            let value = factors[i + j * n];
            if i > j {
                lower[i + j * n] = value;
            } else {
                upper[i + j * n] = value;
                if i == j {
                    lower[i + j * n] = T::one();
                }
            }
        }
    }
}

pub(crate) fn apply_lu_permutation<T: LinalgScalar>(
    pivots: &[usize],
    rhs: &[T],
    n: usize,
    nrhs: usize,
    out: &mut [T],
) -> Result<()> {
    for &pivot in pivots {
        if pivot >= n {
            return Err(Error::InvalidArgument(format!(
                "lu_solve pivot index {pivot} is out of range for n={n}"
            )));
        }
    }

    for col in 0..nrhs {
        let col_offset = col * n;
        for row in 0..n {
            out[row + col_offset] = rhs[pivots[row] + col_offset];
        }
    }

    Ok(())
}

// ============================================================================
// Matrix operation helpers for AD rules
// ============================================================================

/// Transpose a column-major m×n matrix to n×m column-major.
pub(crate) fn transpose<T: LinalgScalar>(data: &[T], m: usize, n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            result[j + i * n] = data[i + j * m];
        }
    }
    result
}

/// Conjugate transpose (adjoint) of a column-major m×n matrix to n×m.
///
/// For real types this is equivalent to [`transpose`].
pub(crate) fn adjoint_transpose<T: LinalgScalar>(data: &[T], m: usize, n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            result[j + i * n] = data[i + j * m].conj();
        }
    }
    result
}

/// Scale a slice element-wise: out[i] = alpha * data[i].
pub(crate) fn scale_vec<T: LinalgScalar>(data: &[T], alpha: T) -> Vec<T> {
    data.iter().map(|&x| alpha * x).collect()
}

/// Add two slices element-wise: out[i] = a[i] + b[i].
pub(crate) fn add_vec<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Subtract two slices element-wise: out[i] = a[i] - b[i].
pub(crate) fn sub_vec<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Create identity matrix (n×n, col-major).
pub(crate) fn eye<T: LinalgScalar>(n: usize) -> Vec<T> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i + i * n] = T::one();
    }
    data
}

/// Hadamard (element-wise) product.
pub(crate) fn hadamard<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Extract lower triangular part (including diagonal) of col-major n×n.
pub(crate) fn tril<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in j..n {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Extract upper triangular part (including diagonal) of col-major n×n.
pub(crate) fn triu<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..=j {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Extract strictly lower triangular part (excluding diagonal) of col-major n×n.
pub(crate) fn tril_strict<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in (j + 1)..n {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Copyltu: Hermitianize from lower triangle.
/// M_ij = M_ij if i > j, conj(M_ji) if i < j, Re(M_ii) if i == j.
/// For real: M + tril(M,-1)^T, with diagonal halved effect.
pub(crate) fn copyltu<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            if i > j {
                result[i + j * n] = data[i + j * n];
                result[j + i * n] = data[i + j * n]; // transpose for real
            } else if i == j {
                result[i + j * n] = data[i + j * n];
            }
        }
    }
    result
}

/// phi operator for Cholesky AD: tril(X) with diagonal halved.
pub(crate) fn phi<T: LinalgScalar>(data: &[T], n: usize) -> AdResult<Vec<T>> {
    let mut result = tril(data, n);
    let half: T = scalar_from(0.5).map_err(to_ad_err)?;
    for i in 0..n {
        result[i + i * n] = result[i + i * n] * half;
    }
    Ok(result)
}

// ============================================================================
// Complex matrix helpers for eig AD rules
// ============================================================================

/// Complex type alias parameterized by real scalar.
pub(crate) type Cx<R> = num_complex::Complex<R>;

/// Extract data slice from a Tensor whose element type implements `Scalar`
/// (but not necessarily `LinalgScalar`). Used for `Tensor<Complex<R>>` in eig AD.
pub(crate) fn extract_data_scalar<T: Scalar>(tensor: &Tensor<T>) -> AdResult<Vec<T>> {
    let t = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = t.offset() as usize;
    let slice = t.buffer().as_slice().ok_or_else(|| {
        chainrules_core::AutodiffError::InvalidArgument(
            "tensor buffer is not a contiguous CPU slice".into(),
        )
    })?;
    let total_len: usize = tensor.dims().iter().product();
    Ok(slice[offset..offset + total_len].to_vec())
}

/// Complex matrix multiply: C = A * B  (all n*n, column-major flat slices).
pub(crate) fn complex_mat_mul_nn<R>(a: &[Cx<R>], b: &[Cx<R>], n: usize) -> Vec<Cx<R>>
where
    R: num_traits::Float + num_traits::NumCast,
{
    let zero = Cx::new(R::zero(), R::zero());
    let mut c = vec![zero; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut sum = zero;
            for k in 0..n {
                sum = sum + a[i + k * n] * b[k + j * n];
            }
            c[i + j * n] = sum;
        }
    }
    c
}

/// Conjugate transpose of n*n complex matrix (column-major).
pub(crate) fn complex_conj_transpose<R>(a: &[Cx<R>], n: usize) -> Vec<Cx<R>>
where
    R: num_traits::Float + num_traits::NumCast,
{
    let zero = Cx::new(R::zero(), R::zero());
    let mut result = vec![zero; n * n];
    for j in 0..n {
        for i in 0..n {
            result[i + j * n] = a[j + i * n].conj();
        }
    }
    result
}

/// Solve A X = B for X, where A and B are n*n complex matrices.
///
/// Converts the complex n*n system to a real 2n*2n system and
/// delegates to `backend::cpu::solve_slices()`.
pub(crate) fn complex_solve_nn<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[Cx<T>],
    b: &[Cx<T>],
    n: usize,
) -> AdResult<Vec<Cx<T>>>
where
    T: backend::CpuLinalgScalar,
{
    let nn = 2 * n;
    let mut a_real = vec![T::zero(); nn * nn];
    let mut b_real = vec![T::zero(); nn * nn];

    for j in 0..n {
        for i in 0..n {
            let aij = a[i + j * n];
            // Top-left: Re(A)
            a_real[i + j * nn] = aij.re;
            // Top-right: -Im(A)
            a_real[i + (j + n) * nn] = T::zero() - aij.im;
            // Bottom-left: Im(A)
            a_real[(i + n) + j * nn] = aij.im;
            // Bottom-right: Re(A)
            a_real[(i + n) + (j + n) * nn] = aij.re;

            let bij = b[i + j * n];
            b_real[i + j * nn] = bij.re;
            b_real[(i + n) + j * nn] = bij.im;
        }
    }

    let mut x_real = vec![T::zero(); nn * nn];
    backend::cpu::solve_slices(&a_real, &b_real, nn, nn, &mut x_real).map_err(to_ad_err)?;

    let zero = Cx::new(T::zero(), T::zero());
    let mut result = vec![zero; n * n];
    for j in 0..n {
        for i in 0..n {
            result[i + j * n] = Cx::new(x_real[i + j * nn], x_real[(i + n) + j * nn]);
        }
    }
    Ok(result)
}

// ============================================================================
// LinalgBackend convenience wrappers for AD code
// ============================================================================

/// Mat mul via LinalgBackend, returning Vec for convenience in AD code.
pub(crate) fn backend_mat_mul<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> AdResult<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    prims_bridge::batched_gemm_via_prims(a, m, k, b, n).map_err(to_ad_err)
}

/// Solve via LinalgBackend, returning Vec for convenience in AD code.
pub(crate) fn backend_solve<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
) -> AdResult<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    let mut x = vec![T::zero(); n * nrhs];
    backend::cpu::solve_slices(a, b, n, nrhs, &mut x).map_err(to_ad_err)?;
    Ok(x)
}

/// Solve triangular via LinalgBackend, returning Vec for convenience in AD code.
pub(crate) fn backend_solve_tri<T: LinalgScalar, C>(
    _ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
) -> AdResult<Vec<T>>
where
    T: backend::CpuLinalgScalar,
{
    let mut x = vec![T::zero(); n * nrhs];
    backend::cpu::solve_triangular_slices(a, b, n, nrhs, upper, &mut x).map_err(to_ad_err)?;
    Ok(x)
}

/// Thin SVD via LinalgBackend, returning (U, S, V) for convenience in AD code.
/// Note: returns V (not Vt) as column-major n×k for convenience in AD code.
pub(crate) fn backend_thin_svd<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>, Vec<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let k = m.min(n);
    let mut u = vec![T::zero(); m * k];
    let mut s = vec![T::zero(); k];
    let mut vt = vec![T::zero(); k * n];
    backend::cpu::thin_svd_slices(a, m, n, &mut u, &mut s, &mut vt).map_err(to_ad_err)?;
    // Convert Vt (k×n) to V (n×k) for convenience
    let v = transpose(&vt, k, n);
    Ok((u, s, v))
}

/// QR decomposition via LinalgBackend, returning (Q, R) for convenience in AD code.
pub(crate) fn backend_qr<T: LinalgScalar<Real = T> + num_traits::Float, C>(
    _ctx: &mut C,
    a: &[T],
    m: usize,
    n: usize,
) -> AdResult<(Vec<T>, Vec<T>)>
where
    T: backend::CpuLinalgScalar,
{
    let k = m.min(n);
    let mut q = vec![T::zero(); m * k];
    let mut r = vec![T::zero(); k * n];
    backend::cpu::qr_slices(a, m, n, &mut q, &mut r).map_err(to_ad_err)?;
    Ok((q, r))
}

/// phi* (adjoint of phi): phi*(X) = (X + X^T - diag(X)) / 2
/// Diagonal gets halved, off-diagonal gets symmetrized.
pub(crate) fn phi_star<T: LinalgScalar>(data: &[T], n: usize) -> AdResult<Vec<T>> {
    let half: T = scalar_from(0.5).map_err(to_ad_err)?;
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            if i == j {
                result[i + j * n] = half * data[i + j * n];
            } else {
                result[i + j * n] = half * (data[i + j * n] + data[j + i * n]);
            }
        }
    }
    Ok(result)
}

/// Extract data slice from Tensor (ensuring col-major).
pub(crate) fn extract_data<T: LinalgScalar>(tensor: &Tensor<T>) -> AdResult<(Vec<T>, usize)> {
    let t = ensure_col_major(tensor);
    let offset = t.offset() as usize;
    let slice = extract_slice(&t).map_err(to_ad_err)?;
    let total_len = tensor.dims().iter().product::<usize>();
    Ok((slice[offset..offset + total_len].to_vec(), 0))
}

// ============================================================================
// AD functions: rrule (reverse-mode, stateless)
// ============================================================================
