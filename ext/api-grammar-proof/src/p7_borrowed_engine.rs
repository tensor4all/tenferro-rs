//! Prototype 7: the borrowed-`Exec` factorization entries as ordinary grammar.
//!
//! `view.svd_with(&opts, exec)` where `exec: &mut dyn Engine` is either a [`CpuEngine`] or a
//! [`BorrowedCpuEngine`] wrapping an `Exec` and a `BufferPool` the caller already holds. This
//! replaces the six `*_with_exec` free functions of tenferro-linalg (#2462): the borrowed case
//! becomes `view.svd_with(&opts, &mut BorrowedCpuEngine::new(exec, &mut pool))`, and
//! [`CpuEngine::with_lanes`](tenferro_cpu::CpuEngine) hands every concurrent lane such an engine
//! over its own share of the thread budget.
//!
//! The adapter is thin: it finds the bundle and pool behind the erased engine and calls the
//! existing typed bodies. It therefore keeps the *borrowed* input rule (a negative stride is
//! `Unsupported`, rank-N views are read in place) on **both** engines, which differs from the
//! erased engine entries (`svd_read` gathers). See `findings/p7.md`.
//!
//! ```compile_fail,E0505
//! // The borrowed engine holds the pool: the pool cannot be used (or dropped) while it lives.
//! use tenferro_api_grammar_proof::p7_borrowed_engine::BorrowedCpuEngine;
//! let mut pool = tenferro_cpu::linalg_interop::BufferPool::new();
//! let mut exec = tcontract_exec::Exec::serial();
//! let engine = BorrowedCpuEngine::new(&mut exec, &mut pool);
//! drop(pool);
//! drop(engine);
//! ```
//!
//! ```compile_fail,E0597
//! // The engine cannot outlive the bundle it borrowed.
//! use tenferro_api_grammar_proof::p7_borrowed_engine::BorrowedCpuEngine;
//! let mut pool = tenferro_cpu::linalg_interop::BufferPool::new();
//! let engine = {
//!     let mut exec = tcontract_exec::Exec::serial();
//!     BorrowedCpuEngine::new(&mut exec, &mut pool)
//! };
//! drop(engine);
//! ```

use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::{with_borrowed_cpu_engine, with_cpu_engine};
use tenferro_linalg::{
    eig_values_with_exec, eigh_values_with_exec, eigh_with_exec, qr_with_exec,
    svd_values_with_exec, svd_with_exec, LinalgScalar, SvdGauge,
};
use tenferro_tensor::{Engine, TypedTensor, TypedTensorView};

pub use crate::p1_svd::{Svd, SvdOptions, SvdShape};
pub use tenferro_cpu::{BorrowedCpuEngine, CpuEngine};

/// Named QR result: `A = q * r`.
#[derive(Clone, Debug)]
pub struct Qr<T> {
    /// Orthonormal factor.
    pub q: T,
    /// Upper-triangular factor.
    pub r: T,
}

/// Named Hermitian eigendecomposition: `A = vectors * diag(values) * vectors^H`.
#[derive(Clone, Debug)]
pub struct Eigh<V, R = V> {
    /// Eigenvalues, ascending, in the real scalar.
    pub values: R,
    /// Eigenvectors as columns.
    pub vectors: V,
}

/// Find the bundle and pool behind `engine` and run `f` on them.
///
/// A [`BorrowedCpuEngine`] lends the caller's bundle and pool. A [`CpuEngine`] lends its own
/// bundle and its own pool (`with_contraction_exec`), so the same call works on both. Any other
/// engine is refused.
fn on_engine<R>(
    engine: &mut dyn Engine,
    op: &'static str,
    f: impl FnOnce(&mut Exec<'_>, &mut BufferPool) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    let mut f = Some(f);
    if let Some(result) = with_borrowed_cpu_engine(engine, |exec, pool| {
        (f.take().expect("one branch runs"))(exec, pool)
    }) {
        return result;
    }
    if let Some(result) = with_cpu_engine(engine, |cpu| {
        cpu.with_contraction_exec(|exec, pool, _| (f.take().expect("one branch runs"))(exec, pool))
    }) {
        return result;
    }
    Err(tenferro_tensor::Error::unsupported(
        op,
        "the engine is neither a CpuEngine nor a BorrowedCpuEngine",
    ))
}

fn check_options(opts: &SvdOptions) -> tenferro_tensor::Result<()> {
    if !(opts.derivative_eps.is_finite() && opts.derivative_eps > 0.0) {
        return Err(tenferro_tensor::Error::invalid_argument(
            "svd_with",
            "derivative_eps",
            format!(
                "derivative_eps must be finite and positive, got {}",
                opts.derivative_eps
            ),
        ));
    }
    if opts.gauge != SvdGauge::Raw {
        // The borrowed typed entry has no gauge parameter (`svd_with_exec(.., full: bool)`).
        return Err(tenferro_tensor::Error::unsupported(
            "svd_with",
            format!(
                "gauge {:?} is not available on the borrowed typed route",
                opts.gauge
            ),
        ));
    }
    Ok(())
}

/// The borrowed factorizations as grammar: `recv.op(exec)` / `recv.op_with(&opts, exec)`.
pub trait BorrowedLinalg<T: LinalgScalar> {
    /// Thin SVD with default options.
    fn svd(
        &self,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Svd<TypedTensor<T>, TypedTensor<T::RealScalar>>> {
        self.svd_with(&SvdOptions::default(), exec)
    }
    /// SVD with options; thin or full by [`SvdShape`].
    fn svd_with(
        &self,
        opts: &SvdOptions,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Svd<TypedTensor<T>, TypedTensor<T::RealScalar>>>;
    /// Singular values, descending.
    fn svdvals(&self, exec: &mut dyn Engine)
        -> tenferro_tensor::Result<TypedTensor<T::RealScalar>>;
    /// Thin QR.
    fn qr(&self, exec: &mut dyn Engine) -> tenferro_tensor::Result<Qr<TypedTensor<T>>>;
    /// Hermitian eigendecomposition.
    fn eigh(
        &self,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Eigh<TypedTensor<T>, TypedTensor<T::RealScalar>>>;
    /// Hermitian eigenvalues, ascending.
    fn eigvalsh(
        &self,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>>;
    /// General eigenvalues in the complex scalar.
    fn eigvals(&self, exec: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T::Complex>>;
}

impl<T: LinalgScalar> BorrowedLinalg<T> for TypedTensorView<'_, T> {
    fn svd_with(
        &self,
        opts: &SvdOptions,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Svd<TypedTensor<T>, TypedTensor<T::RealScalar>>> {
        check_options(opts)?;
        let full = matches!(opts.shape, SvdShape::Full);
        let (u, s, vh) = on_engine(exec, "svd", |e, b| svd_with_exec(e, b, self, full))?;
        Ok(Svd { u, s, vh })
    }
    fn svdvals(
        &self,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
        on_engine(exec, "svdvals", |e, b| svd_values_with_exec(e, b, self))
    }
    fn qr(&self, exec: &mut dyn Engine) -> tenferro_tensor::Result<Qr<TypedTensor<T>>> {
        let (q, r) = on_engine(exec, "qr", |e, b| qr_with_exec(e, b, self))?;
        Ok(Qr { q, r })
    }
    fn eigh(
        &self,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Eigh<TypedTensor<T>, TypedTensor<T::RealScalar>>> {
        let (values, vectors) = on_engine(exec, "eigh", |e, b| eigh_with_exec(e, b, self))?;
        Ok(Eigh { values, vectors })
    }
    fn eigvalsh(
        &self,
        exec: &mut dyn Engine,
    ) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
        on_engine(exec, "eigvalsh", |e, b| eigh_values_with_exec(e, b, self))
    }
    fn eigvals(&self, exec: &mut dyn Engine) -> tenferro_tensor::Result<TypedTensor<T::Complex>> {
        on_engine(exec, "eigvals", |e, b| eig_values_with_exec(e, b, self))
    }
}
