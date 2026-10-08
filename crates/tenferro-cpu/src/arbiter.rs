use std::cell::Cell;
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
}

pub(crate) fn has_active_execution() -> bool {
    EXECUTION_OWNER.with(Cell::get).is_some()
}

/// Return a fresh owner for a top-level CPU execution, or `None` when another
/// CPU backend execution is already active on this thread. Nested entry could
/// violate CPU exclusivity, so callers report it as a typed reentry error
/// before running any user callback.
pub(crate) fn fresh_execution_owner() -> Option<ResourceOwner> {
    (!has_active_execution()).then(ResourceOwner::fresh)
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
    #[error("CPU resources are busy; a Rayon worker must not wait for an owner")]
    Contended,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResourceRequest {
    CpuSet(CpuSet),
}

impl ResourceRequest {
    fn conflicts_with(&self, other: &Self) -> bool {
        match (self, other) {
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
    #[cfg(test)]
    recovery_waiters: std::sync::atomic::AtomicUsize,
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

    /// Acquire a descendant entry: only while `owner` still holds an active
    /// request, never waiting and never taking a fresh owner.
    pub(crate) fn try_acquire_reentrant(
        &self,
        cpus: CpuSet,
        owner: ResourceOwner,
    ) -> Result<Option<ResourcePermit>, ResourceArbiterError> {
        self.try_acquire_request_with_owner(ResourceRequest::CpuSet(cpus), owner, true)
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

    /// Wait in FIFO order for `cpus`. Contention with another owner is waited
    /// out; only poisoned arbiter state is reported.
    pub(crate) fn acquire_waiting(
        &self,
        cpus: CpuSet,
        owner: ResourceOwner,
    ) -> Result<ResourcePermit, ResourceArbiterError> {
        self.acquire_request_waiting(ResourceRequest::CpuSet(cpus), owner)
    }

    fn acquire_request_waiting(
        &self,
        request: ResourceRequest,
        owner: ResourceOwner,
    ) -> Result<ResourcePermit, ResourceArbiterError> {
        // A foreign/global pool child may be joined by the permit's owner.
        // Rayon does not expose ancestry: never park a worker behind that owner.
        if rayon::current_thread_index().is_some() {
            return self
                .try_acquire_request_with_owner(request, owner, false)?
                .ok_or(ResourceArbiterError::Contended);
        }
        loop {
            match self.acquire_request(request.clone(), owner) {
                Ok(permit) => return Ok(permit),
                // Poison means a thread panicked inside arbiter bookkeeping, so
                // the active/waiter lists cannot be trusted; report it.
                Err(
                    error @ (ResourceArbiterError::StatePoisoned | ResourceArbiterError::Contended),
                ) => {
                    return Err(error);
                }
                // Exhaustion is waitable: once every permit and waiter drains,
                // request ids restart from zero.
                Err(ResourceArbiterError::RequestIdExhausted) => {
                    let mut state = self
                        .inner
                        .state
                        .lock()
                        .map_err(|_| ResourceArbiterError::StatePoisoned)?;
                    while !state.active.is_empty() || !state.waiters.is_empty() {
                        // Mark the park while holding the mutex, immediately
                        // before wait releases it into the condvar: a drop can
                        // then only notify after this thread is actually parked
                        // (the drop cannot acquire the mutex first).
                        #[cfg(test)]
                        self.inner
                            .recovery_waiters
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        state = self
                            .inner
                            .changed
                            .wait(state)
                            .map_err(|_| ResourceArbiterError::StatePoisoned)?;
                        #[cfg(test)]
                        self.inner
                            .recovery_waiters
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
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
        // Skip the broadcast when we are the only queued waiter: no other
        // queued waiter can be blocked on the condvar. (Exhaustion-recovery
        // waiters are intentionally uncovered here; the unconditional drop
        // broadcast below wakes them.)
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
                    kind: ResourcePermitKind::Arbitrated {
                        inner: Arc::clone(&self.inner),
                        id,
                    },
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
        self.try_acquire_request_with_owner(request, request_owner(), false)
    }

    fn try_acquire_request_with_owner(
        &self,
        request: ResourceRequest,
        owner: ResourceOwner,
        require_reentrant: bool,
    ) -> Result<Option<ResourcePermit>, ResourceArbiterError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResourceArbiterError::StatePoisoned)?;
        let reentrant = state.active.iter().any(|active| active.owner == owner);
        // A descendant entry must join an existing request of the same owner; it
        // must never be admitted as a fresh independent owner, which the atomic
        // check here prevents even if the issuing request is released meanwhile.
        if require_reentrant && !reentrant {
            return Ok(None);
        }
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
            kind: ResourcePermitKind::Arbitrated {
                inner: Arc::clone(&self.inner),
                id,
            },
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
        // Poll instead of waiting on the production condvar: the acquire
        // fast path legitimately skips the broadcast when a waiter is the
        // only one, so a helper without a waiter-list entry would otherwise
        // wait the full timeout.
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let count = match self.inner.state.lock() {
                Ok(state) => state.waiters.len(),
                Err(_) => return false,
            };
            if count >= expected {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[cfg(test)]
    pub(crate) fn poison_for_test(&self) {
        let inner = Arc::clone(&self.inner);
        let _ = std::panic::catch_unwind(move || {
            let _state = inner.state.lock().unwrap();
            panic!("forced arbiter poisoning");
        });
    }
}

enum ResourcePermitKind {
    Arbitrated { inner: Arc<ArbiterInner>, id: u64 },
}

pub(crate) struct ResourcePermit {
    kind: ResourcePermitKind,
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
        let mut permit = f.debug_struct("ResourcePermit");
        match &self.kind {
            ResourcePermitKind::Arbitrated { id, .. } => permit.field("id", id),
        };
        permit.finish_non_exhaustive()
    }
}

impl Drop for ResourcePermit {
    fn drop(&mut self) {
        match &self.kind {
            ResourcePermitKind::Arbitrated { inner, id } => {
                let mut state = inner
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(position) = state.active.iter().position(|active| active.id == *id) {
                    state.active.swap_remove(position);
                }
                // Always broadcast: the request-id-exhaustion recovery loop parks on
                // the condvar WITHOUT a waiter-list entry (until active and waiters
                // are both empty), so the waiter list cannot tell whether a thread is
                // parked. Skipping here would risk stranding that recovery waiter.
                inner.changed.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests;
