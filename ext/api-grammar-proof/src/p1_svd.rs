//! Prototype 1: complex SVD across input forms under the #2552 call grammar.
//!
//! `recv.svd(engine)` and `recv.svd_with(&opts, engine, policy)` on owned erased
//! [`Tensor`], borrowed erased [`TensorRead`], owned typed [`TypedTensor`] and
//! borrowed typed [`TypedTensorView`]. Outputs are the named [`Svd`] struct with
//! separate value and real type parameters.
//!
//! All four receivers funnel into ONE internal function ([`svd_erased`]) that
//! takes a `TensorRead`. Typed receivers erase on the way in (metadata only) and
//! run a CHECKED downcast on the way out ([`downcast_checked`]). A second,
//! direct typed route ([`svd_typed_view_direct`]) over
//! `tenferro_linalg::svd_with_exec` is built to measure what the "one
//! implementation" claim costs; see `findings/p1.md`.

use num_complex::Complex64;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::with_cpu_engine;
use tenferro_linalg::{
    svd_with_exec, LinalgScalar, SvdDriver, SvdGauge, SvdOptions as LinalgSvdOptions,
    TensorReadLinalgExt,
};
use tenferro_tensor::{Engine, Tensor, TensorRead, TensorScalar, TypedTensor, TypedTensorView};

use crate::policy::MaterializationPolicy;

/// Named SVD result: `A = u * diag(s) * vh`.
///
/// `V` is the value type, `R` the real type of the singular values. A complex
/// input has real `s`, so the two cannot share one parameter.
#[derive(Clone, Debug)]
pub struct Svd<V, R = V> {
    /// Left singular vectors.
    pub u: V,
    /// Singular values, descending, in the real scalar.
    pub s: R,
    /// Conjugate-transposed right singular vectors.
    pub vh: V,
}

impl<V, R> Svd<V, R> {
    /// Split into `(u, s, vh)`.
    pub fn into_parts(self) -> (V, R, V) {
        (self.u, self.s, self.vh)
    }
}

/// Whether the factors are thin (`k = min(m, n)`) or full (`u: m x m`, `vh: n x n`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SvdShape {
    /// `u: m x k`, `vh: k x n` (default).
    #[default]
    Thin,
    /// `u: m x m`, `vh: n x n`.
    Full,
}

/// Prototype SVD options: the existing `tenferro_linalg::SvdOptions` fields plus
/// a [`SvdShape`] enum instead of a `full_matrices: bool`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SvdOptions {
    /// Thin or full factors.
    pub shape: SvdShape,
    /// Singular-vector gauge.
    pub gauge: SvdGauge,
    /// Algorithm family (a provider may ignore it).
    pub driver: SvdDriver,
    /// AD derivative regularization; must be finite and positive.
    pub derivative_eps: f64,
}

impl Default for SvdOptions {
    fn default() -> Self {
        let base = LinalgSvdOptions::default();
        Self {
            shape: SvdShape::Thin,
            gauge: base.gauge,
            driver: base.driver,
            derivative_eps: base.derivative_eps,
        }
    }
}

impl SvdOptions {
    /// Choose thin or full factors.
    #[must_use]
    pub fn shape(mut self, shape: SvdShape) -> Self {
        self.shape = shape;
        self
    }
    /// Choose the singular-vector gauge.
    #[must_use]
    pub fn gauge(mut self, gauge: SvdGauge) -> Self {
        self.gauge = gauge;
        self
    }
    /// Choose the driver.
    #[must_use]
    pub fn driver(mut self, driver: SvdDriver) -> Self {
        self.driver = driver;
        self
    }
    /// Choose the derivative epsilon.
    #[must_use]
    pub fn derivative_eps(mut self, eps: f64) -> Self {
        self.derivative_eps = eps;
        self
    }
    fn to_linalg(self) -> LinalgSvdOptions {
        LinalgSvdOptions::default()
            .gauge(self.gauge)
            .driver(self.driver)
            .derivative_eps(self.derivative_eps)
    }
}

/// `recv.svd(engine)` / `recv.svd_with(&opts, engine, policy)` for every input form.
pub trait SvdOps {
    /// Result type: `Svd<Tensor>` for erased receivers, `Svd<TypedTensor<T>,
    /// TypedTensor<T::Real>>` for typed ones.
    type Output;

    /// All defaults (thin, raw gauge, `Allow`).
    fn svd(&self, engine: &mut dyn Engine) -> tenferro_tensor::Result<Self::Output> {
        self.svd_with(&SvdOptions::default(), engine, MaterializationPolicy::Allow)
    }

    /// Explicit options and policy.
    fn svd_with(
        &self,
        opts: &SvdOptions,
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_tensor::Result<Self::Output>;
}

fn check_policy(policy: MaterializationPolicy) -> tenferro_tensor::Result<()> {
    match policy {
        MaterializationPolicy::Allow => Ok(()),
        MaterializationPolicy::Forbid => Err(tenferro_tensor::Error::unsupported(
            "svd",
            "MaterializationPolicy::Forbid enforcement is not implemented by any provider yet \
             (prototype 4)",
        )),
    }
}

fn check_eps(eps: f64) -> tenferro_tensor::Result<()> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(tenferro_tensor::Error::invalid_argument(
            "svd_with",
            "derivative_eps",
            format!("derivative_eps must be finite and positive, got {eps}"),
        ))
    }
}

/// THE single implementation: every receiver ends here.
///
/// Full shape with a non-`Raw` gauge is refused: the current linalg API has
/// `svd_with_options_read` (thin only) and `svd_full_read` (no options), so the
/// combination cannot be expressed on top of it.
pub fn svd_erased(
    read: TensorRead<'_>,
    opts: &SvdOptions,
    engine: &mut dyn Engine,
    policy: MaterializationPolicy,
) -> tenferro_tensor::Result<Svd<Tensor>> {
    check_policy(policy)?;
    check_eps(opts.derivative_eps)?;
    let (u, s, vh) = match (opts.shape, opts.gauge) {
        (SvdShape::Thin, _) => read.svd_with_options_read(opts.to_linalg(), engine)?,
        (SvdShape::Full, SvdGauge::Raw) => read.svd_full_read(engine)?,
        (SvdShape::Full, gauge) => {
            return Err(tenferro_tensor::Error::unsupported(
                "svd_with",
                format!(
                    "SvdShape::Full cannot be combined with gauge {gauge:?}: the linalg backend \
                     has no full-matrices entry that takes SvdOptions"
                ),
            ))
        }
    };
    Ok(Svd { u, s, vh })
}

/// Checked erased-to-typed conversion of one output.
pub fn downcast_checked<T: TensorScalar>(t: Tensor) -> tenferro_tensor::Result<TypedTensor<T>> {
    let actual = t.dtype();
    T::into_typed(t)
        .map_err(|_| tenferro_tensor::Error::dtype_mismatch("svd downcast", T::dtype(), actual))
}

fn downcast_svd<T: LinalgScalar>(
    r: Svd<Tensor>,
) -> tenferro_tensor::Result<Svd<TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>>> {
    Ok(Svd {
        u: downcast_checked::<T>(r.u)?,
        s: downcast_checked::<<T as TensorScalar>::Real>(r.s)?,
        vh: downcast_checked::<T>(r.vh)?,
    })
}

impl SvdOps for Tensor {
    type Output = Svd<Tensor>;
    fn svd_with(
        &self,
        opts: &SvdOptions,
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_tensor::Result<Self::Output> {
        svd_erased(TensorRead::from_tensor(self), opts, engine, policy)
    }
}

impl SvdOps for TensorRead<'_> {
    type Output = Svd<Tensor>;
    fn svd_with(
        &self,
        opts: &SvdOptions,
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_tensor::Result<Self::Output> {
        svd_erased(self.clone(), opts, engine, policy)
    }
}

impl<T: LinalgScalar> SvdOps for TypedTensor<T> {
    type Output = Svd<TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>>;
    fn svd_with(
        &self,
        opts: &SvdOptions,
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_tensor::Result<Self::Output> {
        // Metadata-only erasure of the owner; no copy.
        downcast_svd::<T>(svd_erased(T::tensor_read(self), opts, engine, policy)?)
    }
}

impl<T: LinalgScalar> SvdOps for TypedTensorView<'_, T> {
    type Output = Svd<TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>>;
    fn svd_with(
        &self,
        opts: &SvdOptions,
        engine: &mut dyn Engine,
        policy: MaterializationPolicy,
    ) -> tenferro_tensor::Result<Self::Output> {
        let read = self.clone().into_tensor_read()?;
        downcast_svd::<T>(svd_erased(read, opts, engine, policy)?)
    }
}

/// The direct typed route: borrowed typed view straight into
/// `tenferro_linalg::svd_with_exec`, no erasure and no downcast.
///
/// Costs: takes only a `CpuEngine` (via the hidden `with_cpu_engine`), has no
/// gauge/driver/eps entry, a negative stride is `Unsupported`, and needs the
/// extra bound `LinalgScalar<RealScalar = <T as TensorScalar>::Real>` because the
/// compiler cannot see the two real-type names are equal.
pub fn svd_typed_view_direct<T>(
    view: &TypedTensorView<'_, T>,
    shape: SvdShape,
    engine: &mut dyn Engine,
    policy: MaterializationPolicy,
) -> tenferro_tensor::Result<Svd<TypedTensor<T>, TypedTensor<<T as TensorScalar>::Real>>>
where
    T: LinalgScalar<RealScalar = <T as TensorScalar>::Real>,
{
    check_policy(policy)?;
    let full = matches!(shape, SvdShape::Full);
    let (u, s, vh) = with_cpu_engine(engine, |cpu| {
        cpu.with_exec(|exec| {
            let mut buffers = BufferPool::new();
            svd_with_exec(exec, &mut buffers, view, full)
        })
    })
    .ok_or_else(|| tenferro_tensor::Error::unsupported("svd", "engine is not a CpuEngine"))???;
    Ok(Svd { u, s, vh })
}

/// Keep `Complex64` referenced for doc links in findings.
#[doc(hidden)]
pub type C64 = Complex64;
