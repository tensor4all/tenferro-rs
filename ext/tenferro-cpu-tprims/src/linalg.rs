//! tprims implementations of tenferro-linalg's injectable CPU kernels
//! (`tenferro_linalg::cpu_kernels`).
//!
//! Handled: Cholesky, triangular solve, solve, thin SVD and singular values,
//! thin QR, Hermitian eigendecomposition and eigenvalues, for single
//! (rank-2) f32/f64/c32/c64 host matrices. Outputs follow tenferro's built-in
//! kernels: Cholesky `L` with a zero upper triangle; SVD `[U, S, Vᴴ]` with
//! real non-increasing `S`; QR `[Q, R]`; eigh `[values, vectors]` with real
//! non-decreasing values. Batched inputs, other operations, and any input
//! tprims rejects (for example a matrix that is not positive definite) are
//! declined, so the built-in kernel runs and reports its own result or error.

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tenferro_cpu::provider::{CpuOperand, CpuProviderUnsupported};
use tenferro_cpu::CpuExecutionContext;
use tenferro_linalg::cpu_kernels::{CpuLinalgKernels, CpuLinalgOutcome, TriangularSolveOptions};
use tenferro_tensor::{DType, Result, Tensor, TensorView};
use tprims_blas::{Diag, Op, Side, Uplo};
use tprims_linalg::{Matrix, Vectors};

use crate::{with_exec, Elem, TprimsProvider};

fn declined<R>() -> Result<CpuLinalgOutcome<R>> {
    Ok(CpuLinalgOutcome::Unsupported(
        CpuProviderUnsupported::Layout(CpuOperand::Lhs),
    ))
}

/// A rank-2 host view of `v`, or `None` (batched, other dtype, not host).
fn matrix<'a, T: Elem>(v: &'a TensorView<'_>) -> Option<StridedView<'a, T>> {
    let t = T::view(v)?;
    if t.shape().len() != 2 {
        return None;
    }
    StridedView::new(t.host_storage().ok()?, t.shape(), t.strides(), t.offset()).ok()
}

fn tensor<T: Elem>(m: &Matrix<T>) -> Result<Tensor> {
    Tensor::from_vec_col_major(vec![m.rows(), m.cols()], m.data().to_vec())
}

/// `Vᴴ` of an `n x k` matrix, as a `k x n` column-major tensor.
fn adjoint<T: Elem>(v: &Matrix<T>) -> Result<Tensor> {
    let (n, k) = (v.rows(), v.cols());
    let mut out = Vec::with_capacity(n * k);
    for j in 0..n {
        for i in 0..k {
            out.push(v.get(j, i).conj_elem());
        }
    }
    Tensor::from_vec_col_major(vec![k, n], out)
}

/// `B` copied to a compact column-major `rows x cols` buffer (a vector is one
/// column), with the shape to return.
fn rhs<T: Elem>(v: &TensorView<'_>) -> Option<(Vec<T>, usize, usize, Vec<usize>)> {
    let t = T::view(v)?;
    let shape = t.shape().to_vec();
    let (rows, cols, strides) = match shape.len() {
        1 => (shape[0], 1, vec![t.strides()[0], shape[0].max(1) as isize]),
        2 => (shape[0], shape[1], t.strides().to_vec()),
        _ => return None,
    };
    let src = StridedView::new(t.host_storage().ok()?, &[rows, cols], &strides, t.offset()).ok()?;
    let m = Matrix::from_view(&src).ok()?;
    Some((m.data().to_vec(), rows, cols, shape))
}

macro_rules! by_dtype {
    ($dtype:expr, $f:ident($($arg:expr),*)) => {
        match $dtype {
            DType::F32 => $f::<f32>($($arg),*),
            DType::F64 => $f::<f64>($($arg),*),
            DType::C32 => $f::<Complex32>($($arg),*),
            DType::C64 => $f::<Complex64>($($arg),*),
            _ => declined(),
        }
    };
}

fn cholesky_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    input: &TensorView<'_>,
) -> Result<CpuLinalgOutcome<Tensor>> {
    let Some(a) = matrix::<T>(input) else {
        return declined();
    };
    match with_exec(ctx, |e| tprims_linalg::cholesky(e, &a)) {
        Ok(c) => Ok(CpuLinalgOutcome::Executed(tensor(c.l())?)),
        Err(_) => declined(),
    }
}

fn solve_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    a: &TensorView<'_>,
    b: &TensorView<'_>,
) -> Result<CpuLinalgOutcome<Tensor>> {
    let (Some(av), Some((mut x, rows, cols, shape))) = (matrix::<T>(a), rhs::<T>(b)) else {
        return declined();
    };
    let solved = {
        let Ok(mut xv) = StridedViewMut::new(&mut x, &[rows, cols], &[1, rows.max(1) as isize], 0)
        else {
            return declined();
        };
        with_exec(ctx, |e| tprims_linalg::solve(e, &av, &mut xv))
    };
    match solved {
        Ok(()) => Ok(CpuLinalgOutcome::Executed(Tensor::from_vec_col_major(
            shape, x,
        )?)),
        Err(_) => declined(),
    }
}

fn triangular_solve_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    o: TriangularSolveOptions,
) -> Result<CpuLinalgOutcome<Tensor>> {
    let (Some(av), Some((mut x, rows, cols, shape))) = (matrix::<T>(a), rhs::<T>(b)) else {
        return declined();
    };
    if shape.len() != 2 {
        return declined();
    }
    let side = if o.left_side { Side::Left } else { Side::Right };
    let uplo = if o.lower { Uplo::Lower } else { Uplo::Upper };
    let op = if o.transpose_a { Op::T } else { Op::N };
    let diag = if o.unit_diagonal {
        Diag::Unit
    } else {
        Diag::NonUnit
    };
    let Some(one) = tenferro_tensor::ContractionScalar::one(T::dtype())
        .ok()
        .and_then(T::scalar)
    else {
        return declined();
    };
    let solved = {
        let Ok(mut xv) = StridedViewMut::new(&mut x, &[rows, cols], &[1, rows.max(1) as isize], 0)
        else {
            return declined();
        };
        with_exec(ctx, |e| {
            tprims_blas::trsm(e, side, uplo, op, diag, one, &av, &mut xv)
        })
    };
    match solved {
        Ok(()) => Ok(CpuLinalgOutcome::Executed(Tensor::from_vec_col_major(
            shape, x,
        )?)),
        Err(_) => declined(),
    }
}

fn svd_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    input: &TensorView<'_>,
    vectors: bool,
) -> Result<CpuLinalgOutcome<Vec<Tensor>>> {
    let Some(a) = matrix::<T>(input) else {
        return declined();
    };
    let want = if vectors {
        Vectors::Thin
    } else {
        Vectors::None
    };
    let Ok(s) = with_exec(ctx, |e| tprims_linalg::svd(e, &a, want)) else {
        return declined();
    };
    let k = s.s.len();
    let values = T::real_tensor(vec![k], s.s)?;
    if !vectors {
        return Ok(CpuLinalgOutcome::Executed(vec![values]));
    }
    let (Some(u), Some(v)) = (s.u, s.v) else {
        return declined();
    };
    Ok(CpuLinalgOutcome::Executed(vec![
        tensor(&u)?,
        values,
        adjoint(&v)?,
    ]))
}

fn qr_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    input: &TensorView<'_>,
) -> Result<CpuLinalgOutcome<Vec<Tensor>>> {
    let Some(a) = matrix::<T>(input) else {
        return declined();
    };
    let Ok((q, r)) = with_exec(ctx, |e| {
        let f = tprims_linalg::qr(e, &a)?;
        Ok::<_, tprims_linalg::Error>((f.q_thin(e)?, f.r()))
    }) else {
        return declined();
    };
    Ok(CpuLinalgOutcome::Executed(vec![tensor(&q)?, tensor(&r)?]))
}

fn eigh_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    input: &TensorView<'_>,
    vectors: bool,
) -> Result<CpuLinalgOutcome<Vec<Tensor>>> {
    let Some(a) = matrix::<T>(input) else {
        return declined();
    };
    let Ok(e) = with_exec(ctx, |x| tprims_linalg::eigh(x, &a, vectors)) else {
        return declined();
    };
    let n = e.values.len();
    let values = T::real_tensor(vec![n], e.values)?;
    match (vectors, e.vectors) {
        (false, _) => Ok(CpuLinalgOutcome::Executed(vec![values])),
        (true, Some(v)) => Ok(CpuLinalgOutcome::Executed(vec![values, tensor(&v)?])),
        (true, None) => declined(),
    }
}

fn one_of(out: Result<CpuLinalgOutcome<Vec<Tensor>>>) -> Result<CpuLinalgOutcome<Tensor>> {
    Ok(match out? {
        CpuLinalgOutcome::Executed(mut v) if v.len() == 1 => {
            CpuLinalgOutcome::Executed(v.remove(0))
        }
        CpuLinalgOutcome::Executed(_) => return declined(),
        CpuLinalgOutcome::Unsupported(u) => CpuLinalgOutcome::Unsupported(u),
    })
}

impl CpuLinalgKernels for TprimsProvider {
    fn cholesky(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> Result<CpuLinalgOutcome<Tensor>> {
        by_dtype!(input.dtype(), cholesky_typed(context, &input))
    }

    fn triangular_solve(
        &self,
        context: &CpuExecutionContext<'_>,
        a: TensorView<'_>,
        b: TensorView<'_>,
        options: TriangularSolveOptions,
    ) -> Result<CpuLinalgOutcome<Tensor>> {
        if a.dtype() != b.dtype() {
            return declined();
        }
        by_dtype!(a.dtype(), triangular_solve_typed(context, &a, &b, options))
    }

    fn solve(
        &self,
        context: &CpuExecutionContext<'_>,
        a: TensorView<'_>,
        b: TensorView<'_>,
    ) -> Result<CpuLinalgOutcome<Tensor>> {
        if a.dtype() != b.dtype() {
            return declined();
        }
        by_dtype!(a.dtype(), solve_typed(context, &a, &b))
    }

    fn svd(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> Result<CpuLinalgOutcome<Vec<Tensor>>> {
        by_dtype!(input.dtype(), svd_typed(context, &input, true))
    }

    fn svd_values(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> Result<CpuLinalgOutcome<Tensor>> {
        one_of(by_dtype!(input.dtype(), svd_typed(context, &input, false)))
    }

    fn qr(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> Result<CpuLinalgOutcome<Vec<Tensor>>> {
        by_dtype!(input.dtype(), qr_typed(context, &input))
    }

    fn eigh(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> Result<CpuLinalgOutcome<Vec<Tensor>>> {
        by_dtype!(input.dtype(), eigh_typed(context, &input, true))
    }

    fn eigh_values(
        &self,
        context: &CpuExecutionContext<'_>,
        input: TensorView<'_>,
    ) -> Result<CpuLinalgOutcome<Tensor>> {
        one_of(by_dtype!(input.dtype(), eigh_typed(context, &input, false)))
    }
}
