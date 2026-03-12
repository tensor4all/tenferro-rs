use super::*;

/// Padé[13/13] coefficients b[0]..b[13] (integer values as f64).
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

/// Theta threshold for order-13 Padé (f64).
pub(crate) const THETA_13: f64 = 5.371920351148152;

/// Compute the matrix 1-norm (max column sum of absolute values).
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
pub(crate) fn backend_mat_mul_nn<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    prims_bridge::batched_gemm_with_semiring_context(ctx, a, n, n, b, n)
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

/// Scale a flat matrix slice by a scalar.
pub(crate) fn mat_scale<T: LinalgScalar>(a: &[T], s: T) -> Vec<T> {
    a.iter().map(|&x| x * s).collect()
}

/// Add two flat matrix slices element-wise.
pub(crate) fn mat_add<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Compute matrix exponential of a single n x n column-major matrix.
pub(crate) fn matrix_exp_single<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
{
    if n == 0 {
        return Ok(Vec::new());
    }
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

    let norm_a = matrix_1_norm(a, n);
    let norm_f64: f64 = num_traits::NumCast::from(norm_a)
        .ok_or_else(|| Error::InvalidArgument("matrix_exp: cannot convert 1-norm to f64".into()))?;
    let s: usize = if norm_f64 <= THETA_13 {
        0
    } else {
        (norm_f64 / THETA_13).log2().ceil().max(0.0) as usize
    };

    let scale_denom = (1u64 << s.min(63)) as f64;
    let scale_inv = T::from(1.0 / scale_denom).ok_or_else(|| {
        Error::InvalidArgument("cannot convert scale factor to target type".into())
    })?;
    let a_scaled = mat_scale(a, scale_inv);

    let a2 = backend_mat_mul_nn(ctx, &a_scaled, &a_scaled, n)?;
    let a4 = backend_mat_mul_nn(ctx, &a2, &a2, n)?;
    let a6 = backend_mat_mul_nn(ctx, &a4, &a2, n)?;

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

    let mut inner_u = vec![T::zero(); nn];
    for i in 0..nn {
        inner_u[i] = b[13] * a6[i] + b[11] * a4[i] + b[9] * a2[i];
    }
    let a6_inner_u = backend_mat_mul_nn(ctx, &a6, &inner_u, n)?;

    let mut u_inner = vec![T::zero(); nn];
    for i in 0..nn {
        u_inner[i] = a6_inner_u[i] + b[7] * a6[i] + b[5] * a4[i] + b[3] * a2[i] + b[1] * eye[i];
    }
    let u = backend_mat_mul_nn(ctx, &a_scaled, &u_inner, n)?;

    let mut inner_v = vec![T::zero(); nn];
    for i in 0..nn {
        inner_v[i] = b[12] * a6[i] + b[10] * a4[i] + b[8] * a2[i];
    }
    let a6_inner_v = backend_mat_mul_nn(ctx, &a6, &inner_v, n)?;

    let mut v = vec![T::zero(); nn];
    for i in 0..nn {
        v[i] = a6_inner_v[i] + b[6] * a6[i] + b[4] * a4[i] + b[2] * a2[i] + b[0] * eye[i];
    }

    let neg_one = T::from(-1.0)
        .ok_or_else(|| Error::InvalidArgument("cannot convert -1 to target type".into()))?;
    let mut lhs = vec![T::zero(); nn];
    mat_linear_combine(neg_one, &u, T::one(), &v, &mut lhs);
    let rhs = mat_add(&u, &v);

    let mut result = backend::slice_bridge::solve_vec(ctx, &lhs, &rhs, n, n)?;

    for _ in 0..s {
        result = backend_mat_mul_nn(ctx, &result, &result, n)?;
    }

    Ok(result)
}

pub(crate) fn matrix_power_single<T: KernelLinalgScalar, C>(
    ctx: &mut C,
    a: &[T],
    n: usize,
    exponent: u64,
) -> Result<Vec<T>>
where
    T: KernelLinalgScalar,
    C: backend::TensorLinalgContextFor<T>,
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
