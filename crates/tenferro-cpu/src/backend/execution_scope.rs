//! Callback-lifetime admission for sequential high-level CPU operations.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use super::{CpuBackend, CpuRuntimeIdentity, CPU_BACKEND};
use crate::affinity::CallerAffinityGuard;
use crate::arbiter::{fresh_execution_owner, has_active_execution, ResourcePermit};
use crate::provider::CpuOperationEntry;
use tenferro_tensor::SessionEntryError;

struct Scope {
    identity: CpuRuntimeIdentity,
    permit: Arc<ResourcePermit>,
    operation_active: bool,
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

/// What CPU execution the current thread is inside.
///
/// An owner that serializes callers with a blocking lock (such as an eager
/// runtime's backend owner) must not wait on that lock while this thread holds
/// a CPU execution permit: another thread may hold the owner and wait for the
/// permit (#1946 F1).
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{current_cpu_execution, CpuBackend, CpuThreadExecution};
///
/// assert_eq!(current_cpu_execution(), CpuThreadExecution::Idle);
/// let backend = CpuBackend::with_threads(1)?;
/// assert_eq!(backend.install(current_cpu_execution)?, CpuThreadExecution::Active);
/// let in_scope = backend.with_execution_scope(current_cpu_execution)?;
/// assert_eq!(in_scope, CpuThreadExecution::SharedScope);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuThreadExecution {
    /// No CPU execution holds a permit on this thread.
    Idle,
    /// A shared execution scope holds a permit, and none of its operations is
    /// running: an operation or session may still be admitted under it.
    SharedScope,
    /// A CPU operation or session is running; nested entry is rejected.
    Active,
}

/// Report what CPU execution the current thread is inside, including a
/// managed Rayon worker running a session or scope callback.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{current_cpu_execution, CpuThreadExecution};
///
/// assert_eq!(current_cpu_execution(), CpuThreadExecution::Idle);
/// ```
pub fn current_cpu_execution() -> CpuThreadExecution {
    let idle_scope = SCOPE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|scope| !scope.operation_active)
    });
    if idle_scope {
        CpuThreadExecution::SharedScope
    } else if has_active_execution() {
        CpuThreadExecution::Active
    } else {
        CpuThreadExecution::Idle
    }
}

struct ScopeGuard;

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE.with(|slot| slot.borrow_mut().take());
    }
}

// The operation loan belongs to this callback thread, never a child worker.
pub(super) struct OperationGuard(PhantomData<Rc<()>>);

impl Drop for OperationGuard {
    fn drop(&mut self) {
        SCOPE.with(|slot| {
            if let Some(scope) = slot.borrow_mut().as_mut() {
                scope.operation_active = false;
            }
        });
    }
}

pub(super) enum ExecutionAdmission {
    Standalone(ResourcePermit),
    Shared(Arc<ResourcePermit>, OperationGuard),
}

impl ExecutionAdmission {
    pub(super) fn permit(&self) -> &ResourcePermit {
        match self {
            Self::Standalone(permit) => permit,
            Self::Shared(permit, _) => permit,
        }
    }
}

impl CpuBackend {
    /// Run sequential high-level CPU work in one entered execution scope.
    ///
    /// Clones of this immutable backend witness may execute ordinary tensor,
    /// eager/AD and prepared trace operations in the callback without installing
    /// the executor again. Construct the eager/traced runtime from such a clone.
    /// Each operation still owns its usual exclusive buffer/cache borrow. The
    /// scope holds the resource permit, including BLAS provider exclusion, until
    /// return or unwind. Only Tenferro-managed CPU executors are supported.
    ///
    /// Enter the scope and prepare inputs before starting a steady-state timer.
    /// This does not remove intrinsic output allocation or operation dispatch.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    ///
    /// let owner = CpuBackend::with_threads(1)?;
    /// let mut operations = owner.clone();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 3.0])?;
    /// let y = owner.with_execution_scope(|| -> Result<_, tenferro_tensor::Error> {
    ///     let y = operations.with_backend_session(|session| {
    ///         session.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
    ///     })??;
    ///     Ok(operations.with_backend_session(|session| {
    ///         session.add_read(TensorRead::from_tensor(&y), TensorRead::from_tensor(&x))
    ///     })??)
    /// })??;
    /// assert_eq!(y.as_slice::<f64>()?, &[3.0, 9.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if a scope or CPU execution is
    /// already active, or [`crate::Error::Unsupported`] for an externally managed
    /// executor. Poisoned admission state is reported as
    /// [`crate::Error::SessionEntry`]. Executor admission errors retain their
    /// typed source in [`crate::Error::BackendSource`]. A callback's return value,
    /// including its own error result, is returned unchanged inside this
    /// method's result.
    ///
    /// Sessions opened inside the callback with a different backend witness fail
    /// with [`tenferro_tensor::SessionEntryError::IncompatibleContext`], and a
    /// session opened from inside another session fails with
    /// [`tenferro_tensor::SessionEntryError::Reentered`]; neither runs its
    /// callback.
    ///
    /// # Panics
    ///
    /// A panic in the callback propagates after releasing the scope and permit.
    pub fn with_execution_scope<R>(&self, operation: impl FnOnce() -> R) -> crate::Result<R> {
        const OP: &str = "CpuBackend::with_execution_scope";
        if has_active_execution() || SCOPE.with(|slot| slot.borrow().is_some()) {
            return Err(crate::Error::runtime_state(
                OP,
                "CPU execution is already active; open the shared scope outside active scopes and backend sessions",
            ));
        }
        let owner = fresh_execution_owner().ok_or(SessionEntryError::Reentered {
            backend: CPU_BACKEND,
        })?;
        let permit = Arc::new(self.acquire_execution_permit(owner)?);
        let affinity = CallerAffinityGuard::enter(self.engine.domain().caller_cpus())
            .map_err(|error| crate::Error::backend_source(OP, error))?;
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        let result = entry.enter(entry.preferred_engine_mode(), |_| {
            SCOPE.with(|slot| {
                *slot.borrow_mut() = Some(Scope {
                    identity: self.runtime_identity.clone(),
                    permit: Arc::clone(&permit),
                    operation_active: false,
                });
            });
            let _guard = ScopeGuard;
            operation()
        });
        affinity
            .finish()
            .map_err(|error| crate::Error::backend_source(OP, error))?;
        Ok(result)
    }

    /// Admit one CPU operation or session, before any user callback runs.
    ///
    /// Inside an active execution scope this reuses the scope's permit; outside
    /// one it acquires a fresh permit, waiting in FIFO order behind other
    /// threads that hold overlapping CPU resources.
    pub(super) fn execution_admission(&self) -> Result<ExecutionAdmission, SessionEntryError> {
        let shared = SCOPE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let Some(scope) = slot.as_mut() else {
                return Ok(None);
            };
            if scope.operation_active {
                // An operation of this scope is running; nested entry falls
                // through to the reentry check below.
                return Ok(None);
            }
            if scope.identity != self.runtime_identity {
                return Err(SessionEntryError::IncompatibleContext {
                    backend: CPU_BACKEND,
                    message: "operation backend does not match the active execution scope; \
                              use a clone of the scope's backend witness"
                        .to_owned(),
                });
            }
            scope.operation_active = true;
            Ok(Some(ExecutionAdmission::Shared(
                Arc::clone(&scope.permit),
                OperationGuard(PhantomData),
            )))
        })?;
        if let Some(shared) = shared {
            return Ok(shared);
        }
        let owner = fresh_execution_owner().ok_or(SessionEntryError::Reentered {
            backend: CPU_BACKEND,
        })?;
        Ok(ExecutionAdmission::Standalone(
            self.acquire_execution_permit(owner)?,
        ))
    }
}
