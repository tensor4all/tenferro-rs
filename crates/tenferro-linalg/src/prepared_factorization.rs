use std::fmt;
use std::marker::PhantomData;

use tenferro_cpu::CpuBackend;

/// Exclusive execution scope shared by prepared factorization operations.
///
/// A session enters the backend execution domain once. Prepared SVD, QR, and
/// eigendecomposition plans can then reuse that entry without reacquiring CPU
/// resources for each matrix. The session is intentionally neither `Clone`
/// nor constructible outside [`PreparedFactorizationBackendExt`].
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::PreparedFactorizationBackendExt;
///
/// let mut backend = CpuBackend::new();
/// let entered = backend.with_prepared_factorization_session(|session| {
///     format!("{session:?}").contains("PreparedFactorizationSession")
/// });
/// assert!(entered);
/// ```
///
/// The callback-scoped session cannot escape:
///
/// ```compile_fail
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::{
///     PreparedFactorizationBackendExt, PreparedFactorizationSession,
/// };
///
/// fn escape(
///     backend: &mut CpuBackend,
/// ) -> &mut PreparedFactorizationSession<'_> {
///     backend.with_prepared_factorization_session(|session| session)
/// }
/// ```
pub struct PreparedFactorizationSession<'scope> {
    pub(crate) inner: PreparedFactorizationSessionInner,
    // Invariance prevents a session borrowed from the callback from escaping it.
    _scope: PhantomData<&'scope mut &'scope ()>,
}

impl fmt::Debug for PreparedFactorizationSession<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedFactorizationSession")
            .field("device", &"cpu")
            .finish_non_exhaustive()
    }
}

pub(crate) enum PreparedFactorizationSessionInner {
    Cpu(CpuPreparedFactorizationSession),
}

pub(crate) struct CpuPreparedFactorizationSession {
    pub(crate) backend: CpuBackend,
    #[cfg(feature = "cpu-faer")]
    pub(crate) par: faer::Par,
}

/// Backend capability for grouping prepared factorization leaf executions.
///
/// The callback cannot return the opaque session. Multiple independent calls
/// may run concurrently when their backend execution domains do not conflict.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::PreparedFactorizationBackendExt;
///
/// let mut backend = CpuBackend::new();
/// let value = backend.with_prepared_factorization_session(|_| 2 + 3);
/// assert_eq!(value, 5);
/// ```
pub trait PreparedFactorizationBackendExt: private::PreparedFactorizationDispatch {
    /// Enter one backend execution scope for one or more prepared factorizations.
    ///
    /// Panic unwinding and ordinary return both release backend execution
    /// ownership before this method returns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::PreparedFactorizationBackendExt;
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.with_prepared_factorization_session(|session| {
    ///     let _ = format!("{session:?}");
    /// });
    /// ```
    fn with_prepared_factorization_session<R: Send>(
        &mut self,
        f: impl for<'scope> FnOnce(&mut PreparedFactorizationSession<'scope>) -> R + Send,
    ) -> R {
        private::PreparedFactorizationDispatch::with_prepared_factorization_session_impl(self, f)
    }
}

impl<T: private::PreparedFactorizationDispatch> PreparedFactorizationBackendExt for T {}

pub(crate) mod private {
    use super::*;

    pub trait PreparedFactorizationDispatch {
        fn with_prepared_factorization_session_impl<R: Send>(
            &mut self,
            f: impl for<'scope> FnOnce(&mut PreparedFactorizationSession<'scope>) -> R + Send,
        ) -> R;
    }
}

impl private::PreparedFactorizationDispatch for CpuBackend {
    fn with_prepared_factorization_session_impl<R: Send>(
        &mut self,
        f: impl for<'scope> FnOnce(&mut PreparedFactorizationSession<'scope>) -> R + Send,
    ) -> R {
        let session_backend = self.clone();
        #[cfg(feature = "cpu-faer")]
        let par = self.linalg_context().faer_par();

        // A factorization session does not borrow the tensor buffer pool, so
        // CpuBackend::install avoids BackendSession's unrelated dynamic surface.
        self.install(move || {
            let mut session = PreparedFactorizationSession {
                inner: PreparedFactorizationSessionInner::Cpu(CpuPreparedFactorizationSession {
                    backend: session_backend,
                    #[cfg(feature = "cpu-faer")]
                    par,
                }),
                _scope: PhantomData,
            };
            f(&mut session)
        })
    }
}
