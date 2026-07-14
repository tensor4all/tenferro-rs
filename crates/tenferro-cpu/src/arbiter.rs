use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
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

pub(crate) fn inherited_or_new_execution_owner() -> ResourceOwner {
    EXECUTION_OWNER
        .with(Cell::get)
        .unwrap_or_else(ResourceOwner::fresh)
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
    request: ResourceRequest,
    owner: ResourceOwner,
}

#[derive(Debug, Default)]
struct ArbiterState {
    next_request_id: u64,
    waiters: VecDeque<Waiter>,
    active: BTreeMap<u64, ActiveRequest>,
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
        self.acquire_request(ResourceRequest::CpuSet(cpus))
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
        self.acquire_request(ResourceRequest::ProviderExclusive)
    }

    #[cfg(test)]
    pub(crate) fn try_acquire_provider_exclusive(
        &self,
    ) -> Result<Option<ResourcePermit>, ResourceArbiterError> {
        self.try_acquire_request(ResourceRequest::ProviderExclusive)
    }

    pub(crate) fn acquire_recovering(&self, cpus: CpuSet) -> ResourcePermit {
        self.acquire_request_recovering(ResourceRequest::CpuSet(cpus))
    }

    pub(crate) fn acquire_provider_exclusive_recovering(&self) -> ResourcePermit {
        self.acquire_request_recovering(ResourceRequest::ProviderExclusive)
    }

    fn acquire_request_recovering(&self, request: ResourceRequest) -> ResourcePermit {
        loop {
            match self.acquire_request(request.clone()) {
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
    ) -> Result<ResourcePermit, ResourceArbiterError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResourceArbiterError::StatePoisoned)?;
        let id = state.next_request_id;
        let owner = request_owner();
        state.next_request_id = state
            .next_request_id
            .checked_add(1)
            .ok_or(ResourceArbiterError::RequestIdExhausted)?;
        state.waiters.push_back(Waiter { id, request, owner });
        self.inner.changed.notify_all();

        loop {
            let Some(position) = state.waiters.iter().position(|waiter| waiter.id == id) else {
                return Err(ResourceArbiterError::StatePoisoned);
            };
            let request = &state.waiters[position].request;
            let reentrant = state.active.values().any(|active| active.owner == owner);
            let active_compatible = state
                .active
                .values()
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
                state.active.insert(
                    id,
                    ActiveRequest {
                        request: waiter.request,
                        owner: waiter.owner,
                    },
                );
                return Ok(ResourcePermit {
                    inner: Arc::clone(&self.inner),
                    id,
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
        let reentrant = state.active.values().any(|active| active.owner == owner);
        let conflicts_with_active = state
            .active
            .values()
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
        state.active.insert(id, ActiveRequest { request, owner });
        Ok(Some(ResourcePermit {
            inner: Arc::clone(&self.inner),
            id,
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
    reentrant: bool,
}

impl ResourcePermit {
    pub(crate) fn is_reentrant(&self) -> bool {
        self.reentrant
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
        state.active.remove(&self.id);
        self.inner.changed.notify_all();
    }
}

#[cfg(test)]
mod tests;
