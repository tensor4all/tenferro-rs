//! Lane dispatch for the factorization entries (#2552 S2b).
//!
//! Inside `CpuEngine::with_lanes` the `&mut dyn Engine` a lane receives is a borrowed engine
//! over the lane's execution bundle and buffer pool; it is not a CPU engine, so the ordinary
//! engine route (`with_linalg_backend`) does not accept it. Each `try_lane_*` function below is
//! called first by the public entry of the same operation: when the engine is a lane it runs the
//! typed body of the factorization on the lane's bundle and pool and returns `Some(result)`;
//! for any other engine it returns `None` and the entry continues on its usual route.
//!
//! Only svd, svdvals, qr, eigh, eigvalsh and eigvals run in a lane. Every other linalg
//! operation falls through to `with_linalg_backend`, which answers `Unsupported` for a lane.
//!
//! Layout: an operand without negative strides is read in place. An operand with a negative
//! stride is gathered once into a compact buffer taken from the lane's own pool and handed back
//! to that pool afterwards, so the steady-state zero-allocation property is promised for
//! non-negative-stride input only.
//!
//! Gauge: `svd_with_options` and `eigh_with_options` apply the same gauge post-processing the
//! engine route applies, to the same erased outputs, so the results are identical to it.
//!
//! The `blas` build has no typed bodies (they are `faer` bodies), so there a lane runs none of
//! these operations and every `try_lane_*` answers `None`.

#[cfg(feature = "native")]
pub(crate) use native::*;

#[cfg(not(feature = "native"))]
pub(crate) use unavailable::*;

#[cfg(not(feature = "native"))]
mod unavailable {
    use tenferro_tensor::{Engine, Tensor, TensorRead};

    use crate::extension::{EighOptions, SvdOptions};

    type Out<T> = Option<tenferro_tensor::Result<T>>;

    pub(crate) fn try_lane_svd(
        _engine: &mut dyn Engine,
        _input: TensorRead<'_>,
        _full: bool,
        _options: Option<(&'static str, SvdOptions)>,
    ) -> Out<(Tensor, Tensor, Tensor)> {
        None
    }

    pub(crate) fn try_lane_svdvals(
        _engine: &mut dyn Engine,
        _input: TensorRead<'_>,
    ) -> Out<Tensor> {
        None
    }

    pub(crate) fn try_lane_qr(
        _engine: &mut dyn Engine,
        _input: TensorRead<'_>,
    ) -> Out<(Tensor, Tensor)> {
        None
    }

    pub(crate) fn try_lane_eigh(
        _engine: &mut dyn Engine,
        _input: TensorRead<'_>,
        _options: Option<(&'static str, EighOptions)>,
    ) -> Out<(Tensor, Tensor)> {
        None
    }

    pub(crate) fn try_lane_eigvalsh(
        _engine: &mut dyn Engine,
        _input: TensorRead<'_>,
    ) -> Out<Tensor> {
        None
    }

    pub(crate) fn try_lane_eigvals(
        _engine: &mut dyn Engine,
        _input: TensorRead<'_>,
    ) -> Out<Tensor> {
        None
    }
}

#[cfg(feature = "native")]
mod native {
    use tcontract_exec::Exec;
    use tenferro_cpu::linalg_interop::BufferPool;
    use tenferro_cpu::{with_borrowed_cpu_engine, with_materialized_tensor_read};
    use tenferro_tensor::{Engine, Tensor, TensorRead, TensorView, TypedTensorView};

    use crate::error::unsupported_dtype;
    use crate::extension::{
        apply_eigh_gauge, apply_svd_gauge, validate_derivative_eps, EighOptions, SvdOptions,
    };
    use crate::LinalgScalar;

    type Out<T> = Option<tenferro_tensor::Result<T>>;

    /// Whether `engine` is a lane engine of `CpuEngine::with_lanes`.
    fn is_lane(engine: &mut dyn Engine) -> bool {
        with_borrowed_cpu_engine(engine, |_, _| ()).is_some()
    }

    /// Whether any dimension of the view has a negative stride.
    fn has_negative_stride(view: &TensorView<'_>) -> bool {
        match view {
            TensorView::F32(v) => v.strides().iter().any(|&s| s < 0),
            TensorView::F64(v) => v.strides().iter().any(|&s| s < 0),
            TensorView::I32(v) => v.strides().iter().any(|&s| s < 0),
            TensorView::I64(v) => v.strides().iter().any(|&s| s < 0),
            TensorView::Bool(v) => v.strides().iter().any(|&s| s < 0),
            TensorView::C32(v) => v.strides().iter().any(|&s| s < 0),
            TensorView::C64(v) => v.strides().iter().any(|&s| s < 0),
        }
    }

    /// Run `body` on `view` as the typed bodies require it: in place, or, for a negative stride,
    /// on a compact copy gathered into (and returned to) the lane's own pool.
    fn prepared<R>(
        exec: &mut Exec<'_>,
        pool: &mut BufferPool,
        op: &'static str,
        view: TensorView<'_>,
        body: impl FnOnce(&mut Exec<'_>, &mut BufferPool, TensorView<'_>) -> tenferro_tensor::Result<R>,
    ) -> tenferro_tensor::Result<R> {
        if !has_negative_stride(&view) {
            return body(exec, pool, view);
        }
        let dtype = view.dtype();
        with_materialized_tensor_read(
            exec,
            pool,
            op,
            TensorRead::View(view),
            |gathered, pool, exec| {
                let view = TensorRead::from_tensor(gathered)
                    .tensor_view()
                    .ok_or_else(|| unsupported_dtype(op, dtype))?;
                body(exec, pool, view)
            },
        )
    }

    /// Run the generic typed body `$body::<T>(exec, pool, &view, args..)` on the lane behind
    /// `$engine`, for the scalar type of `$input`; `None` when the engine is not a lane.
    macro_rules! on_lane {
        ($engine:expr, $input:expr, $op:expr, $body:ident($($arg:expr),*)) => {
            with_borrowed_cpu_engine($engine, |exec, pool| {
                let dtype = $input.dtype();
                let Some(view) = $input.tensor_view() else {
                    return Err(unsupported_dtype($op, dtype));
                };
                prepared(exec, pool, $op, view, |exec, pool, view| match view {
                    TensorView::F32(v) => $body::<f32>(exec, pool, &v, $($arg),*),
                    TensorView::F64(v) => $body::<f64>(exec, pool, &v, $($arg),*),
                    TensorView::C32(v) => $body::<num_complex::Complex32>(exec, pool, &v, $($arg),*),
                    TensorView::C64(v) => $body::<num_complex::Complex64>(exec, pool, &v, $($arg),*),
                    other => Err(unsupported_dtype($op, other.dtype())),
                })
            })
        };
    }

    fn svd_body<T: LinalgScalar>(
        exec: &mut Exec<'_>,
        pool: &mut BufferPool,
        view: &TypedTensorView<'_, T>,
        full: bool,
    ) -> tenferro_tensor::Result<[Tensor; 3]> {
        let (u, s, vh) = T::svd_with_exec(exec, pool, view, full)?;
        Ok([
            Tensor::from_typed(u),
            Tensor::from_typed(s),
            Tensor::from_typed(vh),
        ])
    }

    fn svdvals_body<T: LinalgScalar>(
        exec: &mut Exec<'_>,
        pool: &mut BufferPool,
        view: &TypedTensorView<'_, T>,
    ) -> tenferro_tensor::Result<Tensor> {
        T::svd_values_with_exec(exec, pool, view).map(Tensor::from_typed)
    }

    fn qr_body<T: LinalgScalar>(
        exec: &mut Exec<'_>,
        pool: &mut BufferPool,
        view: &TypedTensorView<'_, T>,
    ) -> tenferro_tensor::Result<(Tensor, Tensor)> {
        let (q, r) = T::qr_with_exec(exec, pool, view)?;
        Ok((Tensor::from_typed(q), Tensor::from_typed(r)))
    }

    fn eigh_body<T: LinalgScalar>(
        exec: &mut Exec<'_>,
        pool: &mut BufferPool,
        view: &TypedTensorView<'_, T>,
    ) -> tenferro_tensor::Result<[Tensor; 2]> {
        let (values, vectors) = T::eigh_with_exec(exec, pool, view)?;
        Ok([Tensor::from_typed(values), Tensor::from_typed(vectors)])
    }

    fn eigvalsh_body<T: LinalgScalar>(
        exec: &mut Exec<'_>,
        pool: &mut BufferPool,
        view: &TypedTensorView<'_, T>,
    ) -> tenferro_tensor::Result<Tensor> {
        T::eigh_values_with_exec(exec, pool, view).map(Tensor::from_typed)
    }

    fn eigvals_body<T: LinalgScalar>(
        exec: &mut Exec<'_>,
        pool: &mut BufferPool,
        view: &TypedTensorView<'_, T>,
    ) -> tenferro_tensor::Result<Tensor> {
        T::eig_values_with_exec(exec, pool, view).map(Tensor::from_typed)
    }

    /// SVD on a lane: `(U, S, Vh)`, thin or `full`.
    ///
    /// `options` carries the entry name for validation errors and the `SvdOptions` of the
    /// `*_with_options` entries: `derivative_eps` is validated and the gauge applied as the engine
    /// route does. `None` for the entries without options. `None` result: not a lane.
    pub(crate) fn try_lane_svd(
        engine: &mut dyn Engine,
        input: TensorRead<'_>,
        full: bool,
        options: Option<(&'static str, SvdOptions)>,
    ) -> Out<(Tensor, Tensor, Tensor)> {
        let op = if full { "svd_full" } else { "svd" };
        if !is_lane(engine) {
            return None;
        }
        if let Some(Err(error)) =
            options.map(|(entry, options)| validate_derivative_eps(entry, options.derivative_eps))
        {
            return Some(Err(error));
        }
        let lane = on_lane!(engine, input, op, svd_body(full))?;
        Some((|| {
            let mut outputs = lane?;
            if let Some((_, options)) = options {
                apply_svd_gauge(options.gauge, &mut outputs)?;
            }
            let [u, s, vh] = outputs;
            Ok((u, s, vh))
        })())
    }

    /// Singular values on a lane. `None`: not a lane.
    pub(crate) fn try_lane_svdvals(engine: &mut dyn Engine, input: TensorRead<'_>) -> Out<Tensor> {
        on_lane!(engine, input, "svd_values", svdvals_body())
    }

    /// Thin QR on a lane: `(Q, R)`. `None`: not a lane.
    pub(crate) fn try_lane_qr(
        engine: &mut dyn Engine,
        input: TensorRead<'_>,
    ) -> Out<(Tensor, Tensor)> {
        on_lane!(engine, input, "qr", qr_body())
    }

    /// Hermitian eigendecomposition on a lane: `(values, vectors)`; `options` as for
    /// [`try_lane_svd`]. `None` result: not a lane.
    pub(crate) fn try_lane_eigh(
        engine: &mut dyn Engine,
        input: TensorRead<'_>,
        options: Option<(&'static str, EighOptions)>,
    ) -> Out<(Tensor, Tensor)> {
        if !is_lane(engine) {
            return None;
        }
        if let Some(Err(error)) =
            options.map(|(entry, options)| validate_derivative_eps(entry, options.derivative_eps))
        {
            return Some(Err(error));
        }
        let lane = on_lane!(engine, input, "eigh", eigh_body())?;
        Some((|| {
            let mut outputs = lane?;
            if let Some((_, options)) = options {
                apply_eigh_gauge(options.gauge, &mut outputs)?;
            }
            let [values, vectors] = outputs;
            Ok((values, vectors))
        })())
    }

    /// Hermitian eigenvalues on a lane. `None`: not a lane.
    pub(crate) fn try_lane_eigvalsh(engine: &mut dyn Engine, input: TensorRead<'_>) -> Out<Tensor> {
        on_lane!(engine, input, "eigh_values", eigvalsh_body())
    }

    /// General eigenvalues on a lane. `None`: not a lane.
    pub(crate) fn try_lane_eigvals(engine: &mut dyn Engine, input: TensorRead<'_>) -> Out<Tensor> {
        on_lane!(engine, input, "eig_values", eigvals_body())
    }
}
