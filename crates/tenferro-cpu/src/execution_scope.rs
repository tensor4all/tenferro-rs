//! Dynamic extent of an explicitly entered, serial CPU workflow.
//!
//! This carries admission authority, never a pointer to borrowed backend state.
//! Ordinary operations still borrow their own buffers and reject recursive entry.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::arbiter::{ResourceOwner, ResourcePermit, BACKEND_REENTRY_PANIC};
use crate::engine::CpuEngine;
use crate::resource_domain::CpuResourceDomain;

#[derive(Debug)]
pub(crate) struct CpuWorkflowScope(pub(crate) crate::CpuBackend);

impl tenferro_tensor::BackendExecutionScope for CpuWorkflowScope {
    fn run(&self, task: Box<dyn FnOnce() + Send + '_>) {
        self.0.with_execution_scope(task);
    }
}

struct Scope {
    engine: Arc<CpuEngine>,
    permit: Arc<ResourcePermit>,
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
    static IN_OPERATION: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn idle_owner() -> Option<ResourceOwner> {
    if IN_OPERATION.get() {
        return None;
    }
    SCOPE.with(|slot| slot.borrow().as_ref().map(|scope| scope.permit.owner()))
}

pub(crate) fn shared_permit(domain: &CpuResourceDomain) -> Option<ResourcePermit> {
    SCOPE.with(|slot| {
        slot.borrow().as_ref().map(|scope| {
            assert!(
                !IN_OPERATION.get() && std::ptr::eq(scope.engine.domain(), domain),
                "{BACKEND_REENTRY_PANIC}: workflow domain mismatch or recursive operation"
            );
            ResourcePermit::shared(Arc::clone(&scope.permit))
        })
    })
}

pub(crate) fn matches(domain: &CpuResourceDomain, owner: ResourceOwner) -> bool {
    SCOPE.with(|slot| {
        slot.borrow().as_ref().is_some_and(|scope| {
            std::ptr::eq(scope.engine.domain(), domain) && scope.permit.owner() == owner
        })
    })
}

pub(crate) fn operation<R>(f: impl FnOnce() -> R) -> R {
    assert!(!IN_OPERATION.replace(true), "{BACKEND_REENTRY_PANIC}");
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_OPERATION.set(false);
        }
    }
    let _restore = Restore;
    f()
}

pub(crate) fn run<R>(
    engine: Arc<CpuEngine>,
    permit: Arc<ResourcePermit>,
    f: impl FnOnce() -> R,
) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPE.with(|slot| *slot.borrow_mut() = None);
        }
    }
    SCOPE.with(|slot| {
        assert!(slot.borrow().is_none(), "{BACKEND_REENTRY_PANIC}");
        *slot.borrow_mut() = Some(Scope { engine, permit });
    });
    let _restore = Restore;
    f()
}
