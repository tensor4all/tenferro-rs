use super::*;

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
pub(crate) fn adjoint_transpose<T: LinalgScalar>(data: &[T], m: usize, n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            result[j + i * n] = data[i + j * m].conj();
        }
    }
    result
}

/// Scale a slice element-wise: `out[i] = alpha * data[i]`.
pub(crate) fn scale_vec<T: LinalgScalar>(data: &[T], alpha: T) -> Vec<T> {
    data.iter().map(|&x| alpha * x).collect()
}

/// Add two slices element-wise.
pub(crate) fn add_vec<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Subtract two slices element-wise.
pub(crate) fn sub_vec<T: LinalgScalar>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Create identity matrix `(n × n, col-major)`.
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

/// Extract lower triangular part (including diagonal).
pub(crate) fn tril<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in j..n {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Extract upper triangular part (including diagonal).
pub(crate) fn triu<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..=j {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Extract strictly lower triangular part (excluding diagonal).
pub(crate) fn tril_strict<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in (j + 1)..n {
            result[i + j * n] = data[i + j * n];
        }
    }
    result
}

/// Hermitianize from the lower triangle.
pub(crate) fn copyltu<T: LinalgScalar>(data: &[T], n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            if i > j {
                result[i + j * n] = data[i + j * n];
                result[j + i * n] = data[i + j * n];
            } else if i == j {
                result[i + j * n] = data[i + j * n];
            }
        }
    }
    result
}

/// phi operator for Cholesky AD: `tril(X)` with diagonal halved.
pub(crate) fn phi<T: LinalgScalar>(data: &[T], n: usize) -> AdResult<Vec<T>> {
    let mut result = tril(data, n);
    let half: T = scalar_from(0.5).map_err(to_ad_err)?;
    for i in 0..n {
        result[i + i * n] = result[i + i * n] * half;
    }
    Ok(result)
}

/// phi* (adjoint of phi): `(X + X^T - diag(X)) / 2`.
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
