use super::*;

/// Complex type alias parameterized by real scalar.
pub(crate) type Cx<R> = num_complex::Complex<R>;

/// Complex matrix multiply: `C = A * B`.
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

/// Conjugate transpose of a complex column-major matrix.
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

/// Solve `A X = B` for complex square matrices.
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
            a_real[i + j * nn] = aij.re;
            a_real[i + (j + n) * nn] = T::zero() - aij.im;
            a_real[(i + n) + j * nn] = aij.im;
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
