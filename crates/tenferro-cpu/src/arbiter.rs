use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use thiserror::Error;

use crate::CpuSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResourceOwner(u64);

impl ResourceOwner {
    fn fresh() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_OWNER.fetch_add(1, Ordering::Relaxed))
    }
}

thread_local! {
    static THREAD_OWNER: ResourceOwner = ResourceOwner::fresh();
    static EXECUTION_OWNER: Cell<Option<ResourceOwner>> = const { Cell::new(None) };
    static WORKER_EXECUTION_SCOPE: RefCell<Option<Arc<ExecutionScopeState>>> = const { RefCell::new(None) };
}

pub(crate) const BACKEND_REENTRY_PANIC: &str =
    "CpuBackend cannot be re-entered while another CPU backend execution is active on this thread or managed Rayon scope";

pub(crate) fn inherited_or_new_execution_owner() -> ResourceOwner {
    if WORKER_EXECUTION_SCOPE.with(|scope| {
        scope
            .borrow()
            .as_ref()
            .is_some_and(|scope| scope.has_active_owner())
    }) {
        panic!("{BACKEND_REENTRY_PANIC}");
    }
    if EXECUTION_OWNER.with(Cell::get).is_some() {
        panic!("{BACKEND_REENTRY_PANIC}");
    }
    ResourceOwner::fresh()
}

pub(crate) fn current_execution_owner() -> Option<ResourceOwner> {
    EXECUTION_OWNER.with(Cell::get)
}

pub(crate) fn with_execution_owner<R>(owner: ResourceOwner, op: impl FnOnce() -> R) -> R {
    struct RestoreOwner(Option<ResourceOwner>);

    impl Drop for RestoreOwner {
        fn drop(&mut self) {
            EXECUTION_OWNER.set(self.0);
        }
    }

    let previous = EXECUTION_OWNER.replace(Some(owner));
    let _restore = RestoreOwner(previous);
    op()
}

pub(crate) fn register_worker_execution_scope(scope: Arc<ExecutionScopeState>) {
    WORKER_EXECUTION_SCOPE.with(|slot| {
        let mut slot = slot.borrow_mut();
        // INVARIANT: each owned Rayon worker is a dedicated thread whose start
        // hook runs once; replacing a live scope would mix context ownership.
        debug_assert!(slot.is_none());
        *slot = Some(scope);
    });
}

pub(crate) fn worker_execution_scope_matches(scope: &Arc<ExecutionScopeState>) -> bool {
    WORKER_EXECUTION_SCOPE.with(|current| {
        current
            .borrow()
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, scope))
    })
}

#[cfg(test)]
pub(crate) fn worker_execution_scope_registered() -> bool {
    WORKER_EXECUTION_SCOPE.with(|scope| scope.borrow().is_some())
}

#[derive(Debug, Default)]
pub(crate) struct ExecutionScopeState {
    active: Mutex<ExecutionOwnerState>,
}

#[derive(Debug, Default)]
struct ExecutionOwnerState {
    owner: Option<ResourceOwner>,
    depth: usize,
}

impl ExecutionScopeState {
    pub(crate) fn enter(&self, owner: ResourceOwner) -> ExecutionScopeGuard<'_> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match active.owner {
            None => {
                active.owner = Some(owner);
                active.depth = 1;
            }
            Some(current) if current == owner => active.depth += 1,
            Some(current) => panic!(
                "CPU execution owner invariant violated: active {current:?}, requested {owner:?}"
            ),
        }
        ExecutionScopeGuard { scope: self, owner }
    }

    fn has_active_owner(&self) -> bool {
        self.active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .owner
            .is_some()
    }
}

pub(crate) struct ExecutionScopeGuard<'a> {
    scope: &'a ExecutionScopeState,
    owner: ResourceOwner,
}

impl Drop for ExecutionScopeGuard<'_> {
    fn drop(&mut self) {
        let mut active = self
            .scope
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(active.owner, Some(self.owner));
        active.depth -= 1;
        if active.depth == 0 {
            active.owner = None;
        }
    }
}

#[cfg(test)]
fn request_owner() -> ResourceOwner {
    EXECUTION_OWNER
        .with(Cell::get)
        .unwrap_or_else(|| THREAD_OWNER.with(|owner| *owner))
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub(crate) enum ResourceArbiterError {
    #[error("CPU resource arbiter state is poisoned")]
    StatePoisoned,
    #[error("CPU resource arbiter request IDs are exhausted")]
    RequestIdExhausted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResourceRequest {
    CpuSet(CpuSet),
    ProviderExclusive,
}

impl ResourceRequest {
    fn conflicts_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ProviderExclusive, _) | (_, Self::ProviderExclusive) => true,
            (Self::CpuSet(left), Self::CpuSet(right)) => left.overlaps(right),
        }
    }
}

#[derive(Debug)]
struct Waiter {
    id: u64,
    request: ResourceRequest,
    owner: ResourceOwner,
}

#[derive(Debug)]
struct ActiveRequest {
    id: u64,
    request: ResourceRequest,
    owner: ResourceOwner,
}

#[derive(Debug, Default)]
struct ArbiterState {
    next_request_id: u64,
    waiters: VecDeque<Waiter>,
    // INVARIANT: active admission only needs conflict scans and id removal. A
    // retained Vec avoids the per-permit node allocation of a tree map.
    active: Vec<ActiveRequest>,
}

#[derive(Debug, Default)]
struct ArbiterInner {
    state: Mutex<ArbiterState>,
    changed: Condvar,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ResourceArbiter {
    inner: Arc<ArbiterInner>,
}

impl ResourceArbiter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn global() -> Self {
        static GLOBAL: OnceLock<ResourceArbiter> = OnceLock::new();
        GLOBAL.get_or_init(Self::new).clone()
    }

    #[cfg(test)]
    pub(crate) fn acquire(&self, cpus: CpuSet) -> Result<ResourcePermit, ResourceArbiterError> {
        self.acquire_request(ResourceRequest::CpuSet(cpus), request_owner())
    }

    #[cfg(test)]
    pub(crate) fn try_acquire(
        &self,
        cpus: CpuSet,
    ) -> Result<Option<ResourcePermit>, ResourceArbiterError> {
        self.try_acquire_request(ResourceRequest::CpuSet(cpus))
    }

    #[cfg(test)]
    pub(crate) fn acquire_provider_exclusive(
        &self,
    ) -> Result<ResourcePermit, ResourceArbiterError> {
        self.acquire_request(ResourceRequest::ProviderExclusive, request_owner())
    }

    #[cfg(test)]
    pub(crate) fn try_acquire_provider_exclusive(
        &self,
    ) -> Result<Option<ResourcePermit>, ResourceArbiterError> {
        self.try_acquire_request(ResourceRequest::ProviderExclusive)
    }

    pub(crate) fn acquire_recovering(&self, cpus: CpuSet, owner: ResourceOwner) -> ResourcePermit {
        self.acquire_request_recovering(ResourceRequest::CpuSet(cpus), owner)
    }

    pub(crate) fn acquire_provider_exclusive_recovering(
        &self,
        owner: ResourceOwner,
    ) -> ResourcePermit {
        self.acquire_request_recovering(ResourceRequest::ProviderExclusive, owner)
    }

    fn acquire_request_recovering(
        &self,
        request: ResourceRequest,
        owner: ResourceOwner,
    ) -> ResourcePermit {
        loop {
            match self.acquire_request(request.clone(), owner) {
                Ok(permit) => return permit,
                Err(ResourceArbiterError::StatePoisoned) => {
                    self.inner.state.clear_poison();
                }
                Err(ResourceArbiterError::RequestIdExhausted) => {
                    let mut state = self
                        .inner
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    while !state.active.is_empty() || !state.waiters.is_empty() {
                        state = self
                            .inner
                            .changed
                            .wait(state)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                    }
                    state.next_request_id = 0;
                }
            }
        }
    }

    fn acquire_request(
        &self,
        request: ResourceRequest,
        owner: ResourceOwner,
    ) -> Result<ResourcePermit, ResourceArbiterError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResourceArbiterError::StatePoisoned)?;
        let id = state.next_request_id;
        state.next_request_id = state
            .next_request_id
            .checked_add(1)
            .ok_or(ResourceArbiterError::RequestIdExhausted)?;
        state.waiters.push_back(Waiter { id, request, owner });
        // Skip the broadcast when we are the only waiter: no other thread is
        // blocked on this arbiter's condvar, so the futex wake is pure overhead.
        // The grant/release paths still notify when there is someone to wake.
        if state.waiters.len() > 1 {
            self.inner.changed.notify_all();
        }

        loop {
            let Some(position) = state.waiters.iter().position(|waiter| waiter.id == id) else {
                return Err(ResourceArbiterError::StatePoisoned);
            };
            let request = &state.waiters[position].request;
            let reentrant = state.active.iter().any(|active| active.owner == owner);
            let active_compatible = state
                .active
                .iter()
                .all(|active| active.owner == owner || !active.request.conflicts_with(request));
            let older_compatible = reentrant
                || state
                    .waiters
                    .iter()
                    .take(position)
                    .all(|older| !older.request.conflicts_with(request));
            if active_compatible && older_compatible {
                let Some(waiter) = state.waiters.remove(position) else {
                    return Err(ResourceArbiterError::StatePoisoned);
                };
                state.active.push(ActiveRequest {
                    id,
                    request: waiter.request,
                    owner: waiter.owner,
                });
                return Ok(ResourcePermit {
                    inner: Arc::clone(&self.inner),
                    id,
                    owner,
                    reentrant,
                });
            }

            state = match self.inner.changed.wait(state) {
                Ok(state) => state,
                Err(poisoned) => {
                    let mut state = poisoned.into_inner();
                    state.waiters.retain(|waiter| waiter.id != id);
                    self.inner.changed.notify_all();
                    return Err(ResourceArbiterError::StatePoisoned);
                }
            };
        }
    }

    #[cfg(test)]
    fn try_acquire_request(
        &self,
        request: ResourceRequest,
    ) -> Result<Option<ResourcePermit>, ResourceArbiterError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResourceArbiterError::StatePoisoned)?;
        let owner = request_owner();
        let reentrant = state.active.iter().any(|active| active.owner == owner);
        let conflicts_with_active = state
            .active
            .iter()
            .any(|active| active.owner != owner && active.request.conflicts_with(&request));
        let bypasses_waiter = !reentrant
            && state
                .waiters
                .iter()
                .any(|waiter| waiter.request.conflicts_with(&request));
        if conflicts_with_active || bypasses_waiter {
            return Ok(None);
        }
        let id = state.next_request_id;
        state.next_request_id = state
            .next_request_id
            .checked_add(1)
            .ok_or(ResourceArbiterError::RequestIdExhausted)?;
        state.active.push(ActiveRequest { id, request, owner });
        Ok(Some(ResourcePermit {
            inner: Arc::clone(&self.inner),
            id,
            owner,
            reentrant,
        }))
    }

    #[cfg(test)]
    fn wait_for_waiter_count_for_test(
        &self,
        expected: usize,
        timeout: std::time::Duration,
    ) -> bool {
        let Ok(mut state) = self.inner.state.lock() else {
            return false;
        };
        let deadline = std::time::Instant::now() + timeout;
        while state.waiters.len() < expected {
            let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return false;
            };
            let Ok((next, wait)) = self.inner.changed.wait_timeout(state, remaining) else {
                return false;
            };
            state = next;
            if wait.timed_out() && state.waiters.len() < expected {
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    fn poison_for_test(&self) {
        let inner = Arc::clone(&self.inner);
        let _ = std::panic::catch_unwind(move || {
            let _state = inner.state.lock().unwrap();
            panic!("forced arbiter poisoning");
        });
    }
}

pub(crate) struct ResourcePermit {
    inner: Arc<ArbiterInner>,
    id: u64,
    owner: ResourceOwner,
    reentrant: bool,
}

impl ResourcePermit {
    pub(crate) fn is_reentrant(&self) -> bool {
        self.reentrant
    }

    pub(crate) fn owner(&self) -> ResourceOwner {
        self.owner
    }
}

impl fmt::Debug for ResourcePermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourcePermit")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Drop for ResourcePermit {
    fn drop(&mut self) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(position) = state.active.iter().position(|active| active.id == self.id) {
            state.active.swap_remove(position);
        }
        // Only broadcast when a waiter could be released; with none waiting the
        // futex wake is pure overhead on the hot drop path.
        if !state.waiters.is_empty() {
            self.inner.changed.notify_all();
        }
    }
}

#[cfg(test)]
mod tests;
