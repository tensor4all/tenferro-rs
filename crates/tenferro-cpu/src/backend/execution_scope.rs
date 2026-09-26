//! Callback-lifetime admission for sequential high-level CPU operations.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use super::{CpuBackend, CpuRuntimeIdentity};
use crate::arbiter::{has_active_execution, inherited_or_new_execution_owner, ResourcePermit};
use crate::engine::CpuEngine;
use crate::provider::CpuOperationEntry;
use crate::resource_domain::CpuResourceDomain;
use crate::CpuDomainOwnership;

struct Scope {
    identity: CpuRuntimeIdentity,
    engine: Arc<CpuEngine>,
    permit: Arc<ResourcePermit>,
    operation_active: bool,
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
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

pub(crate) fn is_entered(domain: &CpuResourceDomain, permit: &ResourcePermit) -> bool {
    SCOPE.with(|slot| {
        slot.borrow().as_ref().is_some_and(|scope| {
            scope.operation_active
                && std::ptr::eq(scope.engine.domain(), domain)
                && std::ptr::eq(scope.permit.as_ref(), permit)
        })
    })
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
    /// let y = owner.with_execution_scope(|| {
    ///     let y = operations.with_backend_session(|session| {
    ///         session.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
    ///     })?;
    ///     operations.with_backend_session(|session| {
    ///         session.add_read(TensorRead::from_tensor(&y), TensorRead::from_tensor(&x))
    ///     })
    /// })??;
    /// assert_eq!(y.as_slice::<f64>()?, &[3.0, 9.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeState`] if a scope or CPU execution is
    /// already active, or [`crate::Error::Unsupported`] for an externally managed
    /// executor. Executor admission errors retain their typed source in
    /// [`crate::Error::BackendSource`]. A callback's return value, including its
    /// own error result, is returned unchanged inside this method's result.
    ///
    /// # Panics
    ///
    /// Existing infallible backend-session APIs still panic on invalid nested
    /// entry or a different backend witness. Do not enter backend operations from
    /// inside an active borrowed session/provider operation or from other workers.
    /// A panic in the callback propagates after releasing the scope and permit.
    pub fn with_execution_scope<R: Send>(
        &self,
        operation: impl FnOnce() -> R + Send,
    ) -> crate::Result<R> {
        const OP: &str = "CpuBackend::with_execution_scope";
        if has_active_execution() || SCOPE.with(|slot| slot.borrow().is_some()) {
            return Err(crate::Error::runtime_state(
                OP,
                "CPU execution is already active; open the shared scope outside active scopes and backend sessions",
            ));
        }
        if self.engine.domain().ownership() != CpuDomainOwnership::Managed {
            return Err(crate::Error::unsupported(
                OP,
                "shared execution scopes require a Tenferro-managed CPU domain; use ordinary operation entry for external domains",
            ));
        }
        let owner = inherited_or_new_execution_owner();
        let permit = Arc::new(self.acquire_execution_permit(owner));
        let entry = CpuOperationEntry::new(self.engine.domain(), &permit);
        entry
            .enter(entry.preferred_engine_mode(), |_| {
                SCOPE.with(|slot| {
                    *slot.borrow_mut() = Some(Scope {
                        identity: self.runtime_identity.clone(),
                        engine: Arc::clone(&self.engine),
                        permit: Arc::clone(&permit),
                        operation_active: false,
                    });
                });
                let _guard = ScopeGuard;
                operation()
            })
            .map_err(|error| crate::Error::backend_source(OP, error))
    }

    pub(super) fn execution_admission(&self) -> crate::Result<ExecutionAdmission> {
        let shared = SCOPE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let Some(scope) = slot.as_mut() else {
                return Ok(None);
            };
            if scope.operation_active {
                // Preserve the original backend/session nested-entry guard.
                return Ok(None);
            }
            if scope.identity != self.runtime_identity {
                return Err(crate::Error::runtime_state(
                    "CPU execution scope",
                    "operation backend does not match the scope; use a clone of its backend witness",
                ));
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
        let owner = inherited_or_new_execution_owner();
        Ok(ExecutionAdmission::Standalone(
            self.acquire_execution_permit(owner),
        ))
    }

    pub(super) fn infallible_execution_admission(&self) -> ExecutionAdmission {
        // INVARIANT: BackendSessionHost and install have existing infallible
        // callback contracts; invalid scope entry retains their panic boundary.
        self.execution_admission()
            .unwrap_or_else(|error| panic!("{error}"))
    }
}
