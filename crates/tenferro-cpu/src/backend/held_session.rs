//! Held concrete CPU session (#1945 work package U1, prototype).
//!
//! A [`CpuHeldSession`] owns its admission reservation, the engine's reusable
//! numerical resources (checked out by value, not locked for the session's
//! lifetime) and the caller-affinity guard, and installs the execution-owner
//! marker on the opening thread for its whole life. Operations build a
//! short-lived [`CpuExecSession`] view inside the method that runs them, so no
//! borrow of the session's own state is stored in the session.
//!
//! See `docs/design/held-cpu-session-1945-u1.md` for the ownership, drop, lock
//! and error contracts this implements.

use tenferro_tensor::{BackendSession, SessionEntryError};

mod phase;

pub use phase::{CpuPhase, CpuPhaseError, PhaseLane, PhaseRunError};

use super::execution_scope::{current_cpu_execution, CpuThreadExecution};
use super::{CpuBackend, CPU_BACKEND};
use crate::affinity::CallerAffinityGuard;
use crate::arbiter::{fresh_execution_owner, ExecutionOwnerGuard, ResourcePermit};
use crate::engine::EngineResourceCheckout;
use crate::gemm::GemmAnalysisCache;
use crate::provider::CpuOperationEntry;

/// Failure reported when a held session cannot restore the caller's CPU mask.
///
/// A held session's only fallible cleanup step is affinity restoration; the
/// resource checkout, the execution-owner marker and the admission reservation
/// all release infallibly.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use tenferro_cpu::CpuHeldSessionError;
///
/// let error = CpuHeldSessionError::Restore {
///     source: Box::new(std::io::Error::other("affinity restore failed")),
/// };
/// assert!(error.source().is_some());
/// ```
#[derive(Debug, thiserror::Error)]
pub enum CpuHeldSessionError {
    /// Restoring the caller's CPU affinity mask failed.
    #[error("failed to restore the caller's CPU affinity after a held session: {source}")]
    Restore {
        /// Original affinity diagnostic.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}

/// One held root CPU session, open across calls instead of one callback.
///
/// [`CpuBackend::open_session`] returns it. The session owns the CPU execution
/// admission reservation, the engine's reusable numerical resources (checked out
/// by value, not locked for the session's lifetime), the execution-owner marker
/// and the caller's narrowed CPU affinity mask until [`CpuHeldSession::close`] or
/// drop. Reusing one session across a sequence of operations amortizes entry and
/// keeps the engine's prepared-plan caches and buffer pool warm, without holding
/// any engine lock while an operation runs.
///
/// The session is `!Send + !Sync`: admission, the resource checkout, the
/// execution-owner marker and the caller's narrowed CPU mask all belong to the
/// opening thread, and a held session never migrates. Dropping a session releases
/// the same state as [`CpuHeldSession::close`], without reporting a restoration
/// failure.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{Tensor, TensorRead};
///
/// let backend = CpuBackend::with_threads(1)?;
/// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?;
/// let mut session = backend.open_session()?;
/// let y = session.with_session(|view| {
///     view.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
/// })?;
/// assert_eq!(y.as_slice::<f64>()?, &[2.0, 6.0]);
/// session.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CpuHeldSession<'b> {
    state: HeldSessionState<'b>,
}

/// Owned state of one held session.
///
/// **Field order is the release order** (design §3): the resource checkout
/// returns the engine resources first, then the execution-owner marker is
/// restored, then affinity, and only then is the arbiter reservation released.
/// Dropping a session therefore performs the documented cleanup without an
/// explicit `Drop` implementation, and [`CpuHeldSession::close`] destructures
/// the same fields to report a restoration failure.
struct HeldSessionState<'b> {
    backend: &'b CpuBackend,
    checkout: EngineResourceCheckout,
    owner: ExecutionOwnerGuard,
    affinity: CallerAffinityGuard,
    permit: ResourcePermit,
}

impl CpuBackend {
    /// Open a held root CPU session on this backend.
    ///
    /// Only the standalone-root admission shape is implemented. A held root must
    /// not escape an enclosing shared execution scope, because that scope's
    /// operation loan also protects the eager owner lock and its lifetime is not
    /// bounded by this session's, and a child handle's sessions keep using the
    /// scoped entry with their private resources.
    ///
    /// # Errors
    ///
    /// Returns [`SessionEntryError::Reentered`] when a CPU execution is already
    /// active on this thread or this handle is a child execution handle, when an
    /// execution scope is open, or when nested entry would otherwise violate CPU
    /// exclusivity. Returns [`SessionEntryError::Contended`] when another held
    /// session already owns this engine's resources. Returns
    /// [`SessionEntryError::ResourcePoisoned`] for poisoned arbiter state and
    /// [`SessionEntryError::Executor`] when the caller's CPU mask cannot be
    /// narrowed for the session.
    /// Open a held CPU session on this backend.
    ///
    /// The session owns the CPU execution admission reservation and the engine's
    /// reusable resources until it is closed, so consecutive operations reuse the
    /// same warm plan caches, buffer pool and entered execution instead of paying
    /// one session entry each. No engine lock is held while an operation runs.
    ///
    /// Open one session around a related batch of operations and close it with
    /// [`CpuHeldSession::close`]. Use the scoped
    /// [`BackendSessionHost::with_backend_session`](tenferro_tensor::BackendSessionHost::with_backend_session)
    /// instead when the work is a single callback, or when it must run inside a
    /// shared execution scope.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let backend = CpuBackend::with_threads(2)?;
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let mut session = backend.open_session()?;
    /// // One entry, two operations, one warm buffer pool and plan cache.
    /// let doubled = session.with_session(|view| {
    ///     view.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
    /// })?;
    /// let quadrupled = session.with_session(|view| {
    ///     view.add_read(
    ///         TensorRead::from_tensor(&doubled),
    ///         TensorRead::from_tensor(&doubled),
    ///     )
    /// })?;
    /// assert_eq!(quadrupled.as_slice::<f64>()?, &[4.0, 8.0]);
    /// session.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SessionEntryError::Reentered`] when a CPU execution is already
    /// active on this thread, when an execution scope is open, or when this handle
    /// is a child execution handle. Returns [`SessionEntryError::Contended`] when
    /// another held session already owns this engine's resources, and
    /// [`SessionEntryError::ResourcePoisoned`] for poisoned admission state.
    /// Returns [`SessionEntryError::Executor`] when the caller's CPU mask cannot be
    /// narrowed for the session.
    ///
    /// # Panics
    ///
    /// Never. A session that cannot be opened reports a typed error, and a session
    /// that cannot restore the caller's CPU mask reports
    /// [`CpuHeldSessionError`] from [`CpuHeldSession::close`].
    pub fn open_session(&self) -> Result<CpuHeldSession<'_>, SessionEntryError> {
        match current_cpu_execution() {
            CpuThreadExecution::Idle => {}
            CpuThreadExecution::SharedScope | CpuThreadExecution::Active => {
                return Err(SessionEntryError::Reentered {
                    backend: CPU_BACKEND,
                });
            }
        }
        if self.inherited_owner.is_some() {
            return Err(SessionEntryError::Reentered {
                backend: CPU_BACKEND,
            });
        }
        let owner = fresh_execution_owner().ok_or(SessionEntryError::Reentered {
            backend: CPU_BACKEND,
        })?;
        let permit = self.acquire_execution_permit(owner)?;
        self.adopt_held_permit(permit)
    }

    /// Turn an already-acquired root reservation into a held session.
    ///
    /// Shared by [`Self::open_session`] and by the scoped entry, so both take the
    /// same checkout, affinity and cleanup path.
    pub(super) fn adopt_held_permit(
        &self,
        permit: ResourcePermit,
    ) -> Result<CpuHeldSession<'_>, SessionEntryError> {
        let owner = permit.owner();
        let checkout =
            EngineResourceCheckout::take(&self.engine).ok_or(SessionEntryError::Contended {
                backend: CPU_BACKEND,
                message: "the CPU engine resources are already checked out by another held session"
                    .to_owned(),
            })?;
        let affinity =
            CallerAffinityGuard::enter(self.engine.domain().caller_cpus()).map_err(|source| {
                SessionEntryError::Executor {
                    backend: CPU_BACKEND,
                    source: Box::new(source),
                }
            })?;
        Ok(CpuHeldSession {
            state: HeldSessionState {
                backend: self,
                checkout,
                owner: ExecutionOwnerGuard::enter(owner),
                affinity,
                permit,
            },
        })
    }
}

impl CpuHeldSession<'_> {
    /// Run one concrete operation in this held session.
    ///
    /// The callback receives the session's concrete execution surface — the same
    /// `BackendSession` the scoped entry hands out — so every operation that takes
    /// a session (primitives, indexing, structural, reductions, einsum, linalg,
    /// borrowed reads and caller-provided output buffers) runs in this held
    /// session without opening another one. Build one short-lived view per
    /// operation: the view borrows the session's own checked-out resources and is
    /// dropped before this method returns, so no lock or cache borrow outlives the
    /// callback.
    ///
    /// Nothing in the view reaches `tenferro-ad`: a plain operation creates no
    /// eager value, trace node or gradient slot and takes no eager owner lock.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let mut session = backend.open_session()?;
    /// // Two operations reuse one session entry.
    /// let doubled = session.with_session(|view| {
    ///     view.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
    /// })?;
    /// let quadrupled = session.with_session(|view| {
    ///     view.add_read(
    ///         TensorRead::from_tensor(&doubled),
    ///         TensorRead::from_tensor(&doubled),
    ///     )
    /// })?;
    /// assert_eq!(quadrupled.as_slice::<f64>()?, &[4.0, 8.0]);
    /// session.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_session<R>(&mut self, f: impl FnOnce(&mut dyn BackendSession) -> R) -> R {
        self.with_session_cached(None, f)
    }

    /// Run one concrete operation with a caller-supplied prepared-plan cache.
    ///
    /// Same as [`Self::with_session`], except that `cache` replaces the engine's
    /// GEMM analysis cache for this callback, exactly as
    /// [`super::BackendSessionHost::with_backend_session_cached`] does.
    pub(crate) fn with_session_cached<R>(
        &mut self,
        cache: Option<&mut GemmAnalysisCache>,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> R {
        let HeldSessionState {
            backend,
            checkout,
            permit,
            ..
        } = &mut self.state;
        let entry = CpuOperationEntry::new(backend.engine.domain(), permit);
        let owner = permit.owner();
        entry.enter_owned(|context| {
            let resources = checkout.resources_mut();
            super::with_operation_session(backend, entry, Some(context), resources, cache, owner, f)
        })
    }

    /// Close this session, releasing admission and leases.
    ///
    /// Runs the documented release order explicitly so a restoration failure is
    /// reported instead of only being logged. Dropping a session instead runs the
    /// same order best-effort.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// let session = backend.open_session()?;
    /// session.close()?;
    /// // The reservation is released: the next entry is admitted immediately.
    /// let _again = backend.open_session()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuHeldSessionError::Restore`] when the caller's CPU affinity
    /// mask cannot be restored. The resources, the execution-owner marker and the
    /// admission reservation are released either way.
    pub fn close(self) -> Result<(), CpuHeldSessionError> {
        let HeldSessionState {
            backend: _,
            checkout,
            owner,
            affinity,
            permit,
        } = self.state;
        drop(checkout);
        drop(owner);
        let restored = affinity.finish();
        drop(permit);
        restored.map_err(|source| CpuHeldSessionError::Restore {
            source: Box::new(source),
        })
    }
}

#[cfg(test)]
mod tests;
