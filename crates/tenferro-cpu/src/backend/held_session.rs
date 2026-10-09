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
#[derive(Debug, thiserror::Error)]
pub(crate) enum HeldSessionError {
    /// Restoring the caller's CPU affinity mask failed.
    #[error("failed to restore the caller's CPU affinity after a held session: {source}")]
    Restore {
        /// Original affinity diagnostic.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}

/// One held root CPU session.
///
/// The session is `!Send + !Sync`: admission, the resource checkout, the
/// execution-owner marker and the caller's narrowed CPU mask all belong to the
/// opening thread, and a held session never migrates.
pub(crate) struct CpuHeldSession<'b> {
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
    // Exercised by this module's tests and by `run_backend_session_cached`
    // indirectly; U2-concrete is what exposes the held entry publicly.
    #[allow(dead_code)]
    pub(crate) fn open_session(&self) -> Result<CpuHeldSession<'_>, SessionEntryError> {
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
    /// Builds the operation view inside this method: the engine's caches and
    /// buffer pool are borrowed from the session's own checkout, wrapped in the
    /// same buffer-pool loan the scoped entry uses, and dropped before returning.
    /// `cache` lets a caller supply its own prepared-plan cache exactly as
    /// [`super::BackendSessionHost::with_backend_session_cached`] does.
    pub(crate) fn with_concrete_session<R>(
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
    /// # Errors
    ///
    /// Returns [`HeldSessionError::Restore`] when the caller's CPU affinity mask
    /// cannot be restored. The resources, the execution-owner marker and the
    /// admission reservation are released either way.
    pub(crate) fn close(self) -> Result<(), HeldSessionError> {
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
        restored.map_err(|source| HeldSessionError::Restore {
            source: Box::new(source),
        })
    }
}

#[cfg(test)]
mod tests;
