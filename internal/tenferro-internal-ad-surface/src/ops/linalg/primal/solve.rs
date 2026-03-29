use super::common::{binary_linalg_builder, dispatch_linalg_runtime, unary_linalg_builder};
use super::*;

binary_linalg_builder!(
    SolveExBuilder,
    solve_ex,
    returns = SolveExResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::SolveEx,
    op = "solve_ex",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::solve_ex::<T, _>(ctx, builder.a, builder.b).map_err(Error::from)
);

/// Builder for `lu_solve`.
/// # Examples
///
/// ```ignore
/// let _builder = lu_solve(/* ... */);
/// ```
pub struct LuSolveBuilder<'a, T: LinalgScalar> {
    factors: &'a Tensor<T>,
    b: &'a Tensor<T>,
    pivots: Option<&'a Tensor<i32>>,
}

impl<'a, T> LuSolveBuilder<'a, T>
where
    T: LinalgRuntimeValue,
{
    /// Sets the backend pivot tensor from `lu_factor`.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivots(&pivots);
    /// ```
    pub fn pivots(mut self, pivots: &'a Tensor<i32>) -> Self {
        self.pivots = Some(pivots);
        self
    }

    /// Executes `lu_solve`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        let pivots = self.pivots.ok_or_else(|| {
            Error::Backend(tenferro_device::Error::InvalidArgument(
                "lu_solve builder requires `.pivots(&Tensor<i32>)` before `run()`".into(),
            ))
        })?;
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::LuSolve,
            "lu_solve",
            |ctx| {
                tenferro_linalg::lu_solve::<T, _>(ctx, self.factors, pivots, self.b)
                    .map_err(Error::from)
            }
        )
    }
}

/// Creates a `lu_solve` builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_solve(/* ... */);
/// ```
pub fn lu_solve<'a, T: LinalgScalar>(
    factors: &'a Tensor<T>,
    b: &'a Tensor<T>,
) -> LuSolveBuilder<'a, T> {
    LuSolveBuilder {
        factors,
        b,
        pivots: None,
    }
}

unary_linalg_builder!(
    InvBuilder,
    inv,
    returns = Tensor<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Inv,
    op = "inv",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::inv::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

unary_linalg_builder!(
    InvExBuilder,
    inv_ex,
    returns = InvExResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Inv,
    op = "inv_ex",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::inv_ex::<T, _>(ctx, builder.tensor).map_err(Error::from)
);
