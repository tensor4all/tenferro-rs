#![allow(dead_code)]
// P4-C0 deliberately lands the complete generic cache before P4-C1 wires a
// semantic `PreparedProgram` key into `RuntimeState`. The cache is exercised by
// runtime::tests::cache under all-targets until the production owner is added.

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use lru::LruCache;

use super::cache_owner::{FrozenCacheOwner, FrozenCacheOwnerKind};
use super::{
    CacheInFlightBehavior, CacheOwnerFailure, CacheStats, PreparationKeySummary, PrepareError,
    RuntimeCacheError, RuntimeStateError, SpecializationRequirements,
};

const PREPARED_CACHE_LOCK: &str = "prepared-cache.state";
const DEFAULT_MAX_PREPARED_ENTRIES: usize = 128;
const DEFAULT_MAX_RETAINED_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_IN_FLIGHT_ENTRIES: usize = 16;
const DEFAULT_MAX_QUEUED_DISTINCT_KEYS: usize = 64;

thread_local! {
    static PREPARATION_STACK: RefCell<Vec<StackFrame>> = const { RefCell::new(Vec::new()) };
}

/// Bounded prepared-plan cache limits.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::PreparedPlanCacheLimits;
///
/// assert_eq!(PreparedPlanCacheLimits::default().max_entries.get(), 128);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedPlanCacheLimits {
    /// Maximum retained terminal entries.
    pub max_entries: NonZeroUsize,
    /// Maximum retained logical bytes.
    pub max_retained_bytes: NonZeroUsize,
    /// Maximum distinct keys that may actively prepare at once.
    pub max_in_flight_entries: NonZeroUsize,
    /// Maximum distinct queued keys waiting for an in-flight slot.
    pub max_queued_distinct_keys: NonZeroUsize,
}

impl Default for PreparedPlanCacheLimits {
    fn default() -> Self {
        Self {
            max_entries: NonZeroUsize::new(DEFAULT_MAX_PREPARED_ENTRIES)
                .expect("default entry limit is nonzero"),
            max_retained_bytes: NonZeroUsize::new(DEFAULT_MAX_RETAINED_BYTES)
                .expect("default byte limit is nonzero"),
            max_in_flight_entries: NonZeroUsize::new(DEFAULT_MAX_IN_FLIGHT_ENTRIES)
                .expect("default in-flight limit is nonzero"),
            max_queued_distinct_keys: NonZeroUsize::new(DEFAULT_MAX_QUEUED_DISTINCT_KEYS)
                .expect("default queue limit is nonzero"),
        }
    }
}

/// Prepared-plan cache statistics.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::PreparedPlanCacheStats;
///
/// assert_eq!(PreparedPlanCacheStats::default().entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparedPlanCacheStats {
    /// Retained terminal entries.
    pub entries: usize,
    /// Logical bytes retained by entries, active attempts, and queued records.
    pub retained_bytes: usize,
    /// Ready or redirect cache hits.
    pub hits: u64,
    /// Distinct producer starts.
    pub misses: u64,
    /// Waits on an existing active or queued key.
    pub waits: u64,
    /// Hits on retained deterministic failures.
    pub negative_hits: u64,
    /// Producer callbacks invoked.
    pub preparations: u64,
    /// Retained entries evicted by entry or byte limits.
    pub evictions: u64,
    /// Redirect terminals returned.
    pub redirects: u64,
    /// Active distinct producer count.
    pub in_flight: usize,
    /// Highest observed active producer count.
    pub peak_in_flight: usize,
    /// Distinct queued keys.
    pub queued_distinct_keys: usize,
    /// Capacity refusals.
    pub capacity_refusals: u64,
}

/// Aggregate runtime cache statistics.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::RuntimeCacheStats;
///
/// assert_eq!(RuntimeCacheStats::default().prepared_plans.entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeCacheStats {
    /// Runtime-owned prepared-plan cache statistics.
    pub prepared_plans: PreparedPlanCacheStats,
    /// Direct engine cache-owner statistics.
    pub engines: CacheStats,
    /// Extension module cache-owner statistics.
    pub extensions: CacheStats,
}

pub(crate) trait PreparedCacheKey: Clone + Debug + Send + Sync + 'static {
    type Shared: Debug + Send + Sync + 'static;

    fn compact_digest(&self) -> u128;
    fn exact_eq(&self, other: &Self) -> bool;
    fn retained_bytes(&self) -> Option<usize>;
    fn summary(&self) -> PreparationKeySummary;
    fn shared_retention(&self) -> Option<SharedRetention<Self::Shared>>;
}

pub(crate) trait PreparedValue: Debug + Send + Sync + 'static {
    fn retained_bytes(&self) -> Option<usize>;
}

#[derive(Debug)]
pub(crate) struct SharedRetention<S: Debug + Send + Sync + 'static> {
    pub(crate) value: Arc<S>,
    pub(crate) retained_bytes: Option<usize>,
}

impl<S: Debug + Send + Sync + 'static> Clone for SharedRetention<S> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            retained_bytes: self.retained_bytes,
        }
    }
}

#[derive(Debug)]
pub(crate) enum CacheProduced<K: PreparedCacheKey, V: PreparedValue> {
    Ready {
        value: Arc<V>,
        shared: Option<SharedRetention<K::Shared>>,
    },
    Redirect {
        requirements: SpecializationRequirements,
        shared: Option<SharedRetention<K::Shared>>,
    },
    FailedDeterministic {
        error: Arc<PrepareError>,
        shared: Option<SharedRetention<K::Shared>>,
    },
    FailedTransient(Arc<PrepareError>),
}

#[derive(Debug)]
pub(crate) enum CacheLookup<K: PreparedCacheKey, V: PreparedValue> {
    Ready(Arc<V>),
    Redirect {
        requirements: SpecializationRequirements,
        shared: Option<Arc<K::Shared>>,
    },
    FailedDeterministic(Arc<PrepareError>),
    FailedTransient(Arc<PrepareError>),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct AttemptId(u128);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct EntryId(u128);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct QueueId(u128);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct SharedRetentionId(NonZeroUsize);

impl SharedRetentionId {
    fn of<S: Debug + Send + Sync + 'static>(value: &Arc<S>) -> Self {
        Self(NonZeroUsize::new(Arc::as_ptr(value) as usize).expect("Arc data pointer is non-null"))
    }
}

#[derive(Debug)]
struct CacheGenerationToken;

#[derive(Debug)]
struct CacheRevisionToken;

#[derive(Debug, thiserror::Error)]
enum CacheInternalError {
    #[error("prepared-cache retained-byte accounting overflow")]
    AccountingOverflow,
    #[error("prepared-cache shared retention pointer collision")]
    SharedPointerCollision,
}

#[derive(Debug)]
enum CacheTerminal<K: PreparedCacheKey, V: PreparedValue> {
    Ready(Arc<V>),
    Redirect {
        requirements: SpecializationRequirements,
        shared: Option<Arc<K::Shared>>,
    },
    FailedDeterministic(Arc<PrepareError>),
    FailedTransient(Arc<PrepareError>),
}

impl<K: PreparedCacheKey, V: PreparedValue> Clone for CacheTerminal<K, V> {
    fn clone(&self) -> Self {
        match self {
            Self::Ready(value) => Self::Ready(Arc::clone(value)),
            Self::Redirect {
                requirements,
                shared,
            } => Self::Redirect {
                requirements: requirements.clone(),
                shared: shared.as_ref().map(Arc::clone),
            },
            Self::FailedDeterministic(error) => Self::FailedDeterministic(Arc::clone(error)),
            Self::FailedTransient(error) => Self::FailedTransient(Arc::clone(error)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RetainedTerminalKind {
    Ready,
    Redirect,
    FailedDeterministic,
    FailedTransient,
}

impl RetainedTerminalKind {
    fn from_terminal<K: PreparedCacheKey, V: PreparedValue>(
        terminal: &CacheTerminal<K, V>,
    ) -> Self {
        match terminal {
            CacheTerminal::Ready(_) => Self::Ready,
            CacheTerminal::Redirect { .. } => Self::Redirect,
            CacheTerminal::FailedDeterministic(_) => Self::FailedDeterministic,
            CacheTerminal::FailedTransient(_) => Self::FailedTransient,
        }
    }
}

#[derive(Debug)]
enum PreparedEntryState<K: PreparedCacheKey, V: PreparedValue> {
    Preparing {
        attempt_id: AttemptId,
        waiting_readers: usize,
    },
    Retained(CacheTerminal<K, V>),
    Ephemeral {
        terminal: CacheTerminal<K, V>,
        remaining_readers: usize,
    },
}

#[derive(Debug)]
struct EntryRecord<K: PreparedCacheKey, V: PreparedValue> {
    id: EntryId,
    digest: u128,
    key: Arc<K>,
    state: PreparedEntryState<K, V>,
    generation: Arc<CacheGenerationToken>,
    retained_bytes: usize,
    shared_id: Option<SharedRetentionId>,
}

#[derive(Debug)]
struct ActiveAttempt<K: PreparedCacheKey> {
    id: AttemptId,
    entry_id: Option<EntryId>,
    digest: u128,
    key: Arc<K>,
    generation: Arc<CacheGenerationToken>,
    key_bytes: usize,
    signature_bytes: usize,
    attempt_metadata_bytes: usize,
    visible_entry_bytes: usize,
}

impl<K: PreparedCacheKey> ActiveAttempt<K> {
    fn retained_bytes_without_visible_entry(&self) -> usize {
        self.key_bytes
            .saturating_add(self.signature_bytes)
            .saturating_add(self.attempt_metadata_bytes)
    }
}

#[derive(Debug)]
struct QueueRecord<K: PreparedCacheKey> {
    id: QueueId,
    digest: u128,
    key: Arc<K>,
    generation: Arc<CacheGenerationToken>,
    key_bytes: usize,
    signature_bytes: usize,
    metadata_bytes: usize,
    tickets: usize,
}

impl<K: PreparedCacheKey> QueueRecord<K> {
    fn retained_bytes(&self) -> usize {
        self.key_bytes
            .saturating_add(self.signature_bytes)
            .saturating_add(self.metadata_bytes)
    }
}

#[derive(Debug)]
struct SharedCharge<S: Debug + Send + Sync + 'static> {
    value: Arc<S>,
    retained_bytes: usize,
    references: usize,
}

#[derive(Debug)]
struct CacheState<K: PreparedCacheKey, V: PreparedValue> {
    entry_buckets: HashMap<u128, Vec<EntryId>>,
    entries: HashMap<EntryId, EntryRecord<K, V>>,
    lru: LruCache<EntryId, ()>,
    queue_buckets: HashMap<u128, Vec<QueueId>>,
    queue_order: VecDeque<QueueId>,
    queue_records: HashMap<QueueId, QueueRecord<K>>,
    attempts: HashMap<AttemptId, ActiveAttempt<K>>,
    shared: HashMap<SharedRetentionId, SharedCharge<K::Shared>>,
    generation: Arc<CacheGenerationToken>,
    revision: Arc<CacheRevisionToken>,
    next_attempt: u128,
    next_entry: u128,
    next_queue: u128,
    limits: PreparedPlanCacheLimits,
    retained_bytes: usize,
    in_flight: usize,
    stats: PreparedPlanCacheStats,
}

impl<K: PreparedCacheKey, V: PreparedValue> CacheState<K, V> {
    fn new(limits: PreparedPlanCacheLimits) -> Self {
        Self {
            entry_buckets: HashMap::new(),
            entries: HashMap::new(),
            lru: LruCache::unbounded(),
            queue_buckets: HashMap::new(),
            queue_order: VecDeque::new(),
            queue_records: HashMap::new(),
            attempts: HashMap::new(),
            shared: HashMap::new(),
            generation: Arc::new(CacheGenerationToken),
            revision: Arc::new(CacheRevisionToken),
            next_attempt: 0,
            next_entry: 0,
            next_queue: 0,
            limits,
            retained_bytes: 0,
            in_flight: 0,
            stats: PreparedPlanCacheStats::default(),
        }
    }

    fn snapshot_stats(&self) -> PreparedPlanCacheStats {
        let mut stats = self.stats;
        stats.entries = self.lru.len();
        stats.retained_bytes = self.retained_bytes;
        stats.in_flight = self.in_flight;
        stats.queued_distinct_keys = self.queue_records.len();
        stats
    }

    fn bump_revision(&mut self) {
        self.revision = Arc::new(CacheRevisionToken);
    }
}

pub(crate) struct PreparedPlanCache<K: PreparedCacheKey, V: PreparedValue> {
    state: Mutex<CacheState<K, V>>,
    changed: Condvar,
}

impl<K: PreparedCacheKey, V: PreparedValue> PreparedPlanCache<K, V> {
    pub(crate) fn new(limits: PreparedPlanCacheLimits) -> Self {
        Self {
            state: Mutex::new(CacheState::new(limits)),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn get_or_prepare(
        &self,
        key: K,
        behavior: CacheInFlightBehavior,
        signature_bytes: usize,
        producer: impl FnOnce() -> CacheProduced<K, V>,
    ) -> Result<CacheLookup<K, V>, Arc<PrepareError>> {
        self.get_or_prepare_inner(key, behavior, signature_bytes, producer, || {}, || {})
    }

    #[cfg(test)]
    pub(crate) fn get_or_prepare_with_entry_wait_hooks_for_test(
        &self,
        key: K,
        behavior: CacheInFlightBehavior,
        signature_bytes: usize,
        producer: impl FnOnce() -> CacheProduced<K, V>,
        before_entry_wait: impl FnMut(),
        before_condvar_wait: impl FnMut(),
    ) -> Result<CacheLookup<K, V>, Arc<PrepareError>> {
        self.get_or_prepare_inner(
            key,
            behavior,
            signature_bytes,
            producer,
            before_entry_wait,
            before_condvar_wait,
        )
    }

    fn get_or_prepare_inner(
        &self,
        key: K,
        behavior: CacheInFlightBehavior,
        signature_bytes: usize,
        producer: impl FnOnce() -> CacheProduced<K, V>,
        mut before_entry_wait: impl FnMut(),
        mut before_condvar_wait: impl FnMut(),
    ) -> Result<CacheLookup<K, V>, Arc<PrepareError>> {
        let summary = key.summary();
        check_preparation_stack::<K>(summary)?;
        let digest = key.compact_digest();
        let mut producer = Some(producer);
        loop {
            match self.probe(&key, digest)? {
                ProbeResult::Entry(entry_id) => match self.handle_entry(entry_id)? {
                    EntryAction::Return(lookup) => return Ok(lookup),
                    EntryAction::Wait(waiter) => {
                        before_entry_wait();
                        match waiter.wait_once(&mut before_condvar_wait)? {
                            WaitOutcome::RetryProbe => continue,
                            WaitOutcome::Return(lookup) => return Ok(lookup),
                        }
                    }
                    EntryAction::ReturnRetry => continue,
                },
                ProbeResult::Queue(queue_id) => {
                    let ticket = self.add_queue_ticket(queue_id)?;
                    match ticket.wait_for_turn()? {
                        QueueOutcome::RetryProbe => continue,
                        QueueOutcome::Produce(guard) => {
                            return self.run_producer(
                                summary,
                                guard,
                                producer
                                    .take()
                                    .expect("producer is consumed by only one cache miss"),
                            );
                        }
                    }
                }
                ProbeResult::NoMatch {
                    revision,
                    generation,
                } => {
                    match self.reserve_miss(
                        key.clone(),
                        digest,
                        behavior,
                        signature_bytes,
                        revision,
                        generation,
                    )? {
                        MissReservation::Produce(guard) => {
                            return self.run_producer(
                                summary,
                                guard,
                                producer
                                    .take()
                                    .expect("producer is consumed by only one cache miss"),
                            );
                        }
                        MissReservation::Wait(ticket) => match ticket.wait_for_turn()? {
                            QueueOutcome::RetryProbe => continue,
                            QueueOutcome::Produce(guard) => {
                                return self.run_producer(
                                    summary,
                                    guard,
                                    producer
                                        .take()
                                        .expect("producer is consumed by only one cache miss"),
                                );
                            }
                        },
                    }
                }
            }
        }
    }

    pub(crate) fn limits(&self) -> Result<PreparedPlanCacheLimits, RuntimeStateError> {
        self.state
            .lock()
            .map(|state| state.limits)
            .map_err(|_| poisoned_state_error())
    }

    pub(crate) fn set_limits(
        &self,
        limits: PreparedPlanCacheLimits,
    ) -> Result<(), RuntimeStateError> {
        let mut state = self.state.lock().map_err(|_| poisoned_state_error())?;
        state.limits = limits;
        evict_to_limits(&mut state);
        state.bump_revision();
        drop(state);
        self.changed.notify_all();
        Ok(())
    }

    pub(crate) fn stats(&self) -> Result<PreparedPlanCacheStats, RuntimeStateError> {
        self.state
            .lock()
            .map(|state| state.snapshot_stats())
            .map_err(|_| poisoned_state_error())
    }

    pub(crate) fn clear(&self) -> Result<(), RuntimeStateError> {
        let mut state = self.state.lock().map_err(|_| poisoned_state_error())?;
        clear_state(&mut state);
        drop(state);
        self.changed.notify_all();
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn ready_charge_breakdown_for_test(
        &self,
        key: &K,
        value: &V,
        has_shared: bool,
    ) -> Option<RetainedChargeBreakdown> {
        let key_payload = key.retained_bytes()?;
        let value_payload = value.retained_bytes()?;
        let entry_record = size_of::<EntryRecord<K, V>>();
        let compact_bucket_record = size_of::<EntryId>();
        let lru_record = size_of::<EntryId>().checked_add(size_of::<()>())?;
        let shared_record = if has_shared {
            size_of::<SharedRetentionId>().checked_add(size_of::<SharedCharge<K::Shared>>())?
        } else {
            0
        };
        let total = checked_sum([
            key_payload,
            value_payload,
            entry_record,
            compact_bucket_record,
            lru_record,
            shared_record,
        ])?;
        Some(RetainedChargeBreakdown {
            key_payload,
            value_payload,
            entry_record,
            compact_bucket_record,
            lru_record,
            shared_record,
            total,
        })
    }

    fn run_producer(
        &self,
        summary: PreparationKeySummary,
        guard: ProducerGuard<'_, K, V>,
        producer: impl FnOnce() -> CacheProduced<K, V>,
    ) -> Result<CacheLookup<K, V>, Arc<PrepareError>> {
        let stack_guard = StackGuard::push::<K>(summary);
        let produced = producer();
        drop(stack_guard);
        guard.publish(produced)
    }

    fn probe(&self, key: &K, digest: u128) -> Result<ProbeResult, Arc<PrepareError>> {
        loop {
            let (revision, generation, entry_ids, queue_ids) = {
                let state = self.lock_state()?;
                (
                    Arc::clone(&state.revision),
                    Arc::clone(&state.generation),
                    state
                        .entry_buckets
                        .get(&digest)
                        .cloned()
                        .unwrap_or_default(),
                    state
                        .queue_buckets
                        .get(&digest)
                        .cloned()
                        .unwrap_or_default(),
                )
            };

            for entry_id in entry_ids {
                let Some(candidate_key) =
                    self.entry_key_if_generation_current(entry_id, digest, &revision, &generation)?
                else {
                    continue;
                };
                if key.exact_eq(candidate_key.as_ref()) {
                    let state = self.lock_state()?;
                    if !Arc::ptr_eq(&revision, &state.revision)
                        || !Arc::ptr_eq(&generation, &state.generation)
                    {
                        continue;
                    }
                    if state
                        .entries
                        .get(&entry_id)
                        .is_some_and(|entry| entry.digest == digest)
                    {
                        return Ok(ProbeResult::Entry(entry_id));
                    }
                    continue;
                }
            }

            for queue_id in queue_ids {
                let Some(candidate_key) =
                    self.queue_key_if_generation_current(queue_id, digest, &revision, &generation)?
                else {
                    continue;
                };
                if key.exact_eq(candidate_key.as_ref()) {
                    let state = self.lock_state()?;
                    if !Arc::ptr_eq(&revision, &state.revision)
                        || !Arc::ptr_eq(&generation, &state.generation)
                    {
                        continue;
                    }
                    if state
                        .queue_records
                        .get(&queue_id)
                        .is_some_and(|record| record.digest == digest)
                    {
                        return Ok(ProbeResult::Queue(queue_id));
                    }
                    continue;
                }
            }

            let state = self.lock_state()?;
            if !Arc::ptr_eq(&revision, &state.revision)
                || !Arc::ptr_eq(&generation, &state.generation)
            {
                continue;
            }
            return Ok(ProbeResult::NoMatch {
                revision: Arc::clone(&state.revision),
                generation: Arc::clone(&state.generation),
            });
        }
    }

    fn entry_key_if_generation_current(
        &self,
        entry_id: EntryId,
        digest: u128,
        revision: &Arc<CacheRevisionToken>,
        generation: &Arc<CacheGenerationToken>,
    ) -> Result<Option<Arc<K>>, Arc<PrepareError>> {
        let state = self.lock_state()?;
        if !Arc::ptr_eq(revision, &state.revision) || !Arc::ptr_eq(generation, &state.generation) {
            return Ok(None);
        }
        Ok(state
            .entries
            .get(&entry_id)
            .filter(|entry| entry.digest == digest)
            .map(|entry| Arc::clone(&entry.key)))
    }

    fn queue_key_if_generation_current(
        &self,
        queue_id: QueueId,
        digest: u128,
        revision: &Arc<CacheRevisionToken>,
        generation: &Arc<CacheGenerationToken>,
    ) -> Result<Option<Arc<K>>, Arc<PrepareError>> {
        let state = self.lock_state()?;
        if !Arc::ptr_eq(revision, &state.revision) || !Arc::ptr_eq(generation, &state.generation) {
            return Ok(None);
        }
        Ok(state
            .queue_records
            .get(&queue_id)
            .filter(|record| record.digest == digest)
            .map(|record| Arc::clone(&record.key)))
    }

    fn handle_entry(&self, entry_id: EntryId) -> Result<EntryAction<'_, K, V>, Arc<PrepareError>> {
        let mut state = self.lock_state()?;
        if let Some((lookup, terminal_kind)) = state.entries.get(&entry_id).and_then(|entry| {
            if let PreparedEntryState::Retained(terminal) = &entry.state {
                let lookup = lookup_from_terminal(terminal, entry.shared_id, &state.shared);
                Some((lookup, RetainedTerminalKind::from_terminal(terminal)))
            } else {
                None
            }
        }) {
            match terminal_kind {
                RetainedTerminalKind::FailedDeterministic => {
                    state.stats.negative_hits = state.stats.negative_hits.saturating_add(1);
                }
                RetainedTerminalKind::Redirect => {
                    state.stats.hits = state.stats.hits.saturating_add(1);
                    state.stats.redirects = state.stats.redirects.saturating_add(1);
                }
                RetainedTerminalKind::Ready => {
                    state.stats.hits = state.stats.hits.saturating_add(1);
                }
                RetainedTerminalKind::FailedTransient => {}
            }
            let _ = state.lru.get(&entry_id);
            return Ok(EntryAction::Return(lookup));
        }

        let Some(entry) = state.entries.get_mut(&entry_id) else {
            return Ok(EntryAction::ReturnRetry);
        };
        match &mut entry.state {
            PreparedEntryState::Retained(_) => Ok(EntryAction::ReturnRetry),
            PreparedEntryState::Preparing {
                attempt_id,
                waiting_readers,
            } => {
                let attempt_id = *attempt_id;
                *waiting_readers = waiting_readers
                    .checked_add(1)
                    .ok_or_else(accounting_prepare_error)?;
                let _ = entry;
                state.stats.waits = state.stats.waits.saturating_add(1);
                Ok(EntryAction::Wait(EntryWaitGuard {
                    cache: self,
                    entry_id,
                    attempt_id,
                    armed: true,
                }))
            }
            PreparedEntryState::Ephemeral { .. } => Ok(EntryAction::ReturnRetry),
        }
    }

    fn add_queue_ticket(
        &self,
        queue_id: QueueId,
    ) -> Result<QueueTicketGuard<'_, K, V>, Arc<PrepareError>> {
        let mut state = self.lock_state()?;
        let Some(record) = state.queue_records.get_mut(&queue_id) else {
            return Ok(QueueTicketGuard {
                cache: self,
                queue_id,
                armed: false,
            });
        };
        record.tickets = record
            .tickets
            .checked_add(1)
            .ok_or_else(accounting_prepare_error)?;
        state.stats.waits = state.stats.waits.saturating_add(1);
        Ok(QueueTicketGuard {
            cache: self,
            queue_id,
            armed: true,
        })
    }

    fn reserve_miss(
        &self,
        key: K,
        digest: u128,
        behavior: CacheInFlightBehavior,
        signature_bytes: usize,
        revision: Arc<CacheRevisionToken>,
        generation: Arc<CacheGenerationToken>,
    ) -> Result<MissReservation<'_, K, V>, Arc<PrepareError>> {
        let mut state = self.lock_state()?;
        if !Arc::ptr_eq(&revision, &state.revision) || !Arc::ptr_eq(&generation, &state.generation)
        {
            return Ok(MissReservation::Wait(QueueTicketGuard {
                cache: self,
                queue_id: QueueId(0),
                armed: false,
            }));
        }
        if state.in_flight < state.limits.max_in_flight_entries.get()
            && state.queue_order.is_empty()
        {
            let guard = start_attempt(
                &mut state,
                self,
                Arc::new(key),
                digest,
                signature_bytes,
                None,
            )?;
            return Ok(MissReservation::Produce(guard));
        }
        if behavior == CacheInFlightBehavior::Refuse {
            state.stats.capacity_refusals = state.stats.capacity_refusals.saturating_add(1);
            return Err(capacity_error(&state));
        }
        if state.queue_records.len() >= state.limits.max_queued_distinct_keys.get() {
            state.stats.capacity_refusals = state.stats.capacity_refusals.saturating_add(1);
            return Err(capacity_error(&state));
        }
        let queue_id = allocate_queue_id(&mut state);
        let key = Arc::new(key);
        let key_bytes = key.retained_bytes().ok_or_else(accounting_prepare_error)?;
        let metadata_bytes = checked_sum([
            size_of::<QueueRecord<K>>(),
            size_of::<QueueId>(),
            size_of::<QueueId>(),
        ])
        .ok_or_else(accounting_prepare_error)?;
        let retained_bytes = checked_sum([key_bytes, signature_bytes, metadata_bytes])
            .ok_or_else(accounting_prepare_error)?;
        let record = QueueRecord {
            id: queue_id,
            digest,
            key,
            generation: Arc::clone(&state.generation),
            key_bytes,
            signature_bytes,
            metadata_bytes,
            tickets: 1,
        };
        state
            .queue_buckets
            .entry(digest)
            .or_default()
            .push(queue_id);
        state.queue_order.push_back(queue_id);
        state.queue_records.insert(queue_id, record);
        state.retained_bytes = state
            .retained_bytes
            .checked_add(retained_bytes)
            .ok_or_else(accounting_prepare_error)?;
        state.stats.waits = state.stats.waits.saturating_add(1);
        state.bump_revision();
        self.changed.notify_all();
        Ok(MissReservation::Wait(QueueTicketGuard {
            cache: self,
            queue_id,
            armed: true,
        }))
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, CacheState<K, V>>, Arc<PrepareError>> {
        self.state.lock().map_err(|_| {
            Arc::new(PrepareError::CacheState {
                source: poisoned_state_error(),
            })
        })
    }

    fn recover_state(&self) -> MutexGuard<'_, CacheState<K, V>> {
        match self.state.lock() {
            Ok(state) => state,
            Err(error) => {
                // INVARIANT: Guard Drop cannot return Result, so it must finish guard-owned
                // accounting and notify waiters; the poison bit remains set, making later
                // Result-returning cache APIs report `prepared-cache.state`.
                error.into_inner()
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn poison_state_for_test(&self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => panic!("cache state should not already be poisoned in test setup"),
            };
            panic!("poison prepared cache state for test");
        }));
    }
}

#[derive(Debug)]
enum ProbeResult {
    Entry(EntryId),
    Queue(QueueId),
    NoMatch {
        revision: Arc<CacheRevisionToken>,
        generation: Arc<CacheGenerationToken>,
    },
}

enum EntryAction<'a, K: PreparedCacheKey, V: PreparedValue> {
    Return(CacheLookup<K, V>),
    Wait(EntryWaitGuard<'a, K, V>),
    ReturnRetry,
}

enum MissReservation<'a, K: PreparedCacheKey, V: PreparedValue> {
    Produce(ProducerGuard<'a, K, V>),
    Wait(QueueTicketGuard<'a, K, V>),
}

enum WaitOutcome<K: PreparedCacheKey, V: PreparedValue> {
    RetryProbe,
    Return(CacheLookup<K, V>),
}

enum QueueOutcome<'a, K: PreparedCacheKey, V: PreparedValue> {
    RetryProbe,
    Produce(ProducerGuard<'a, K, V>),
}

struct ProducerGuard<'a, K: PreparedCacheKey, V: PreparedValue> {
    cache: &'a PreparedPlanCache<K, V>,
    attempt_id: AttemptId,
    entry_id: EntryId,
    published: bool,
}

impl<K: PreparedCacheKey, V: PreparedValue> ProducerGuard<'_, K, V> {
    fn publish(
        mut self,
        produced: CacheProduced<K, V>,
    ) -> Result<CacheLookup<K, V>, Arc<PrepareError>> {
        let mut state = self.cache.lock_state()?;
        let Some(attempt) = state.attempts.remove(&self.attempt_id) else {
            self.published = true;
            return Ok(lookup_from_produced(produced));
        };
        debug_assert_eq!(attempt.id, self.attempt_id);
        if let Some(active_entry_id) = attempt.entry_id {
            debug_assert_eq!(active_entry_id, self.entry_id);
            if let Some(entry) = state.entries.get(&active_entry_id) {
                debug_assert_eq!(attempt.digest, entry.digest);
            }
        }
        finish_attempt_accounting(&mut state, &attempt);
        let Some(entry_id) = attempt.entry_id else {
            state.bump_revision();
            self.published = true;
            drop(state);
            self.cache.changed.notify_all();
            return Ok(lookup_from_produced(produced));
        };
        let Some(entry) = state.entries.get(&entry_id) else {
            state.bump_revision();
            self.published = true;
            drop(state);
            self.cache.changed.notify_all();
            return Ok(lookup_from_produced(produced));
        };
        debug_assert_eq!(entry.id, entry_id);
        debug_assert!(Arc::ptr_eq(&entry.generation, &attempt.generation));
        debug_assert_eq!(entry.retained_bytes, attempt.visible_entry_bytes);
        let PreparedEntryState::Preparing {
            attempt_id,
            waiting_readers,
        } = &entry.state
        else {
            state.bump_revision();
            self.published = true;
            drop(state);
            self.cache.changed.notify_all();
            return Ok(lookup_from_produced(produced));
        };
        if *attempt_id != self.attempt_id {
            state.bump_revision();
            self.published = true;
            drop(state);
            self.cache.changed.notify_all();
            return Ok(lookup_from_produced(produced));
        }
        let waiting_readers = *waiting_readers;

        if !Arc::ptr_eq(&attempt.generation, &state.generation) {
            remove_entry(&mut state, entry_id, false);
            state.bump_revision();
            self.published = true;
            drop(state);
            self.cache.changed.notify_all();
            return Ok(lookup_from_produced(produced));
        }

        let publication = build_publication(attempt.key.as_ref(), produced);
        let producer_lookup = publication.lookup();
        match publication.retention {
            PublicationRetention::Transient | PublicationRetention::Uncacheable => {
                publish_ephemeral_or_remove(
                    &mut state,
                    entry_id,
                    publication.terminal,
                    waiting_readers,
                );
            }
            PublicationRetention::Retain {
                entry_bytes,
                shared,
            } => {
                let shared_new_bytes = shared
                    .as_ref()
                    .map(|shared| shared_new_charge(&state, shared))
                    .transpose()?
                    .unwrap_or(0);
                let total_new_entry = entry_bytes
                    .checked_add(shared_new_bytes)
                    .ok_or_else(accounting_prepare_error)?;
                if total_new_entry > state.limits.max_retained_bytes.get() {
                    publish_ephemeral_or_remove(
                        &mut state,
                        entry_id,
                        publication.terminal,
                        waiting_readers,
                    );
                } else {
                    let shared_id = if let Some(shared) = shared {
                        Some(retain_shared(&mut state, shared)?)
                    } else {
                        None
                    };
                    retain_entry(
                        &mut state,
                        entry_id,
                        publication.terminal,
                        entry_bytes,
                        shared_id,
                    )?;
                    evict_to_limits(&mut state);
                }
            }
        }
        state.bump_revision();
        self.published = true;
        drop(state);
        self.cache.changed.notify_all();
        Ok(producer_lookup)
    }
}

impl<K: PreparedCacheKey, V: PreparedValue> Drop for ProducerGuard<'_, K, V> {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        let mut state = self.cache.recover_state();
        if let Some(attempt) = state.attempts.remove(&self.attempt_id) {
            finish_attempt_accounting(&mut state, &attempt);
            if let Some(entry_id) = attempt.entry_id {
                remove_entry(&mut state, entry_id, false);
            }
            state.bump_revision();
        } else {
            remove_entry(&mut state, self.entry_id, false);
            state.bump_revision();
        }
        drop(state);
        self.cache.changed.notify_all();
    }
}

struct EntryWaitGuard<'a, K: PreparedCacheKey, V: PreparedValue> {
    cache: &'a PreparedPlanCache<K, V>,
    entry_id: EntryId,
    attempt_id: AttemptId,
    armed: bool,
}

impl<K: PreparedCacheKey, V: PreparedValue> EntryWaitGuard<'_, K, V> {
    fn wait_once(
        mut self,
        before_condvar_wait: &mut impl FnMut(),
    ) -> Result<WaitOutcome<K, V>, Arc<PrepareError>> {
        let mut state = self.cache.lock_state()?;
        while self.same_attempt_is_preparing(&state) {
            before_condvar_wait();
            state = match self.cache.changed.wait(state) {
                Ok(state) => state,
                Err(error) => {
                    drop(error.into_inner());
                    return Err(Arc::new(PrepareError::CacheState {
                        source: poisoned_state_error(),
                    }));
                }
            };
        }
        let outcome = self.complete_with_locked_state(&mut state);
        drop(state);
        if matches!(outcome, WaitOutcome::RetryProbe) {
            self.cache.changed.notify_all();
        }
        Ok(outcome)
    }

    fn same_attempt_is_preparing(&self, state: &CacheState<K, V>) -> bool {
        matches!(
            state.entries.get(&self.entry_id),
            Some(EntryRecord {
                state: PreparedEntryState::Preparing { attempt_id, .. },
                ..
            }) if *attempt_id == self.attempt_id
        )
    }

    fn complete_with_locked_state(&mut self, state: &mut CacheState<K, V>) -> WaitOutcome<K, V> {
        if !self.armed {
            return WaitOutcome::RetryProbe;
        }
        let Some(entry) = state.entries.get_mut(&self.entry_id) else {
            self.armed = false;
            return WaitOutcome::RetryProbe;
        };
        match &mut entry.state {
            PreparedEntryState::Preparing {
                attempt_id,
                waiting_readers,
            } if *attempt_id == self.attempt_id => {
                *waiting_readers = waiting_readers.saturating_sub(1);
                self.armed = false;
                WaitOutcome::RetryProbe
            }
            PreparedEntryState::Retained(_) => {
                self.armed = false;
                WaitOutcome::RetryProbe
            }
            PreparedEntryState::Ephemeral {
                terminal,
                remaining_readers,
            } => {
                let lookup = lookup_from_terminal(terminal, entry.shared_id, &state.shared);
                *remaining_readers = remaining_readers.saturating_sub(1);
                let remove = *remaining_readers == 0;
                self.armed = false;
                if remove {
                    remove_entry(state, self.entry_id, false);
                    state.bump_revision();
                }
                WaitOutcome::Return(lookup)
            }
            _ => {
                self.armed = false;
                WaitOutcome::RetryProbe
            }
        }
    }
}

impl<K: PreparedCacheKey, V: PreparedValue> Drop for EntryWaitGuard<'_, K, V> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self.cache.recover_state();
        match state.entries.get_mut(&self.entry_id) {
            Some(EntryRecord {
                state:
                    PreparedEntryState::Preparing {
                        attempt_id,
                        waiting_readers,
                    },
                ..
            }) if *attempt_id == self.attempt_id => {
                *waiting_readers = waiting_readers.saturating_sub(1);
            }
            Some(EntryRecord {
                state:
                    PreparedEntryState::Ephemeral {
                        remaining_readers, ..
                    },
                ..
            }) => {
                *remaining_readers = remaining_readers.saturating_sub(1);
                if *remaining_readers == 0 {
                    remove_entry(&mut state, self.entry_id, false);
                    state.bump_revision();
                }
            }
            _ => {}
        }
        drop(state);
        self.cache.changed.notify_all();
    }
}

struct QueueTicketGuard<'a, K: PreparedCacheKey, V: PreparedValue> {
    cache: &'a PreparedPlanCache<K, V>,
    queue_id: QueueId,
    armed: bool,
}

impl<'a, K: PreparedCacheKey, V: PreparedValue> QueueTicketGuard<'a, K, V> {
    fn wait_for_turn(mut self) -> Result<QueueOutcome<'a, K, V>, Arc<PrepareError>> {
        if !self.armed {
            return Ok(QueueOutcome::RetryProbe);
        }
        let mut state = self.cache.lock_state()?;
        loop {
            let Some(record) = state.queue_records.get(&self.queue_id) else {
                self.armed = false;
                return Ok(QueueOutcome::RetryProbe);
            };
            debug_assert_eq!(record.id, self.queue_id);
            debug_assert!(Arc::ptr_eq(&record.generation, &state.generation));
            let at_front = state.queue_order.front().copied() == Some(self.queue_id);
            if at_front && state.in_flight < state.limits.max_in_flight_entries.get() {
                let key = Arc::clone(&record.key);
                let digest = record.digest;
                let signature_bytes = record.signature_bytes;
                let guard = start_attempt(
                    &mut state,
                    self.cache,
                    key,
                    digest,
                    signature_bytes,
                    Some(self.queue_id),
                )?;
                self.armed = false;
                return Ok(QueueOutcome::Produce(guard));
            }
            state = match self.cache.changed.wait(state) {
                Ok(state) => state,
                Err(error) => {
                    drop(error.into_inner());
                    return Err(Arc::new(PrepareError::CacheState {
                        source: poisoned_state_error(),
                    }));
                }
            };
        }
    }
}

impl<K: PreparedCacheKey, V: PreparedValue> Drop for QueueTicketGuard<'_, K, V> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self.cache.recover_state();
        if let Some(record) = state.queue_records.get_mut(&self.queue_id) {
            record.tickets = record.tickets.saturating_sub(1);
            if record.tickets == 0 {
                remove_queue_record(&mut state, self.queue_id);
                state.bump_revision();
            }
        }
        drop(state);
        self.cache.changed.notify_all();
    }
}

#[derive(Clone, Copy, Debug)]
struct StackFrame {
    type_id: TypeId,
    summary: PreparationKeySummary,
}

struct StackGuard;

impl StackGuard {
    fn push<K: PreparedCacheKey>(summary: PreparationKeySummary) -> Self {
        PREPARATION_STACK.with(|stack| {
            stack.borrow_mut().push(StackFrame {
                type_id: TypeId::of::<K>(),
                summary,
            });
        });
        Self
    }
}

impl Drop for StackGuard {
    fn drop(&mut self) {
        PREPARATION_STACK.with(|stack| {
            let popped = stack.borrow_mut().pop();
            debug_assert!(popped.is_some());
        });
    }
}

pub(crate) struct RuntimeCacheSet<K: PreparedCacheKey, V: PreparedValue> {
    prepared: PreparedPlanCache<K, V>,
}

impl<K: PreparedCacheKey, V: PreparedValue> RuntimeCacheSet<K, V> {
    pub(crate) fn new(limits: PreparedPlanCacheLimits) -> Self {
        Self {
            prepared: PreparedPlanCache::new(limits),
        }
    }

    pub(crate) fn cache_stats(
        &self,
        owners: &[FrozenCacheOwner],
    ) -> Result<RuntimeCacheStats, RuntimeCacheError> {
        let mut runtime_error = None;
        let prepared = match self.prepared.stats() {
            Ok(stats) => stats,
            Err(error) => {
                runtime_error = Some(error);
                PreparedPlanCacheStats::default()
            }
        };
        let mut engines = CacheStats::default();
        let mut extensions = CacheStats::default();
        let mut failures = Vec::new();
        for owner in owners {
            match owner.owner.cache_stats() {
                Ok(stats) => match owner.kind {
                    FrozenCacheOwnerKind::Engine => saturating_add_cache_stats(&mut engines, stats),
                    FrozenCacheOwnerKind::Extension => {
                        saturating_add_cache_stats(&mut extensions, stats);
                    }
                },
                Err(source) => failures.push(CacheOwnerFailure {
                    owner: owner.id.clone(),
                    source,
                }),
            }
        }
        if runtime_error.is_none() && failures.is_empty() {
            Ok(RuntimeCacheStats {
                prepared_plans: prepared,
                engines,
                extensions,
            })
        } else {
            Err(RuntimeCacheError::Aggregate {
                runtime: runtime_error,
                owners: failures.into_boxed_slice(),
            })
        }
    }

    pub(crate) fn clear_caches(
        &self,
        owners: &[FrozenCacheOwner],
    ) -> Result<(), RuntimeCacheError> {
        let runtime_error = self.prepared.clear().err();
        let mut failures = Vec::new();
        for owner in owners {
            if let Err(source) = owner.owner.clear_caches() {
                failures.push(CacheOwnerFailure {
                    owner: owner.id.clone(),
                    source,
                });
            }
        }
        if runtime_error.is_none() && failures.is_empty() {
            Ok(())
        } else {
            Err(RuntimeCacheError::Aggregate {
                runtime: runtime_error,
                owners: failures.into_boxed_slice(),
            })
        }
    }

    #[allow(dead_code)]
    pub(crate) fn prepared(&self) -> &PreparedPlanCache<K, V> {
        &self.prepared
    }
}

#[cfg(test)]
pub(crate) struct RetainedChargeBreakdown {
    pub key_payload: usize,
    pub value_payload: usize,
    pub entry_record: usize,
    pub compact_bucket_record: usize,
    pub lru_record: usize,
    pub shared_record: usize,
    pub total: usize,
}

#[derive(Debug)]
struct Publication<K: PreparedCacheKey, V: PreparedValue> {
    terminal: CacheTerminal<K, V>,
    lookup: CacheLookup<K, V>,
    retention: PublicationRetention<K>,
}

impl<K: PreparedCacheKey, V: PreparedValue> Publication<K, V> {
    fn lookup(&self) -> CacheLookup<K, V> {
        match &self.lookup {
            CacheLookup::Ready(value) => CacheLookup::Ready(Arc::clone(value)),
            CacheLookup::Redirect {
                requirements,
                shared,
            } => CacheLookup::Redirect {
                requirements: requirements.clone(),
                shared: shared.as_ref().map(Arc::clone),
            },
            CacheLookup::FailedDeterministic(error) => {
                CacheLookup::FailedDeterministic(Arc::clone(error))
            }
            CacheLookup::FailedTransient(error) => CacheLookup::FailedTransient(Arc::clone(error)),
        }
    }
}

#[derive(Debug)]
enum PublicationRetention<K: PreparedCacheKey> {
    Retain {
        entry_bytes: usize,
        shared: Option<SharedRetention<K::Shared>>,
    },
    Transient,
    Uncacheable,
}

fn build_publication<K: PreparedCacheKey, V: PreparedValue>(
    key: &K,
    produced: CacheProduced<K, V>,
) -> Publication<K, V> {
    match produced {
        CacheProduced::Ready { value, shared } => {
            let effective_shared = shared.or_else(|| key.shared_retention());
            let terminal = CacheTerminal::Ready(Arc::clone(&value));
            let lookup = CacheLookup::Ready(value);
            let retention = ready_entry_bytes::<K, V>(
                key,
                match &terminal {
                    CacheTerminal::Ready(value) => value,
                    _ => unreachable!(),
                },
            )
            .map(|entry_bytes| PublicationRetention::Retain {
                entry_bytes,
                shared: effective_shared,
            })
            .unwrap_or(PublicationRetention::Uncacheable);
            Publication {
                terminal,
                lookup,
                retention,
            }
        }
        CacheProduced::Redirect {
            requirements,
            shared,
        } => {
            let effective_shared = shared.or_else(|| key.shared_retention());
            let shared_arc = effective_shared
                .as_ref()
                .map(|shared| Arc::clone(&shared.value));
            let terminal = CacheTerminal::Redirect {
                requirements: requirements.clone(),
                shared: shared_arc.clone(),
            };
            let lookup = CacheLookup::Redirect {
                requirements: requirements.clone(),
                shared: shared_arc,
            };
            let retention = redirect_entry_bytes::<K, V>(key, &requirements)
                .map(|entry_bytes| PublicationRetention::Retain {
                    entry_bytes,
                    shared: effective_shared,
                })
                .unwrap_or(PublicationRetention::Uncacheable);
            Publication {
                terminal,
                lookup,
                retention,
            }
        }
        CacheProduced::FailedDeterministic { error, shared } => {
            let effective_shared = shared.or_else(|| key.shared_retention());
            let terminal = CacheTerminal::FailedDeterministic(Arc::clone(&error));
            let lookup = CacheLookup::FailedDeterministic(error);
            let retention = deterministic_error_entry_bytes::<K, V>(key)
                .map(|entry_bytes| PublicationRetention::Retain {
                    entry_bytes,
                    shared: effective_shared,
                })
                .unwrap_or(PublicationRetention::Uncacheable);
            Publication {
                terminal,
                lookup,
                retention,
            }
        }
        CacheProduced::FailedTransient(error) => Publication {
            terminal: CacheTerminal::FailedTransient(Arc::clone(&error)),
            lookup: CacheLookup::FailedTransient(error),
            retention: PublicationRetention::Transient,
        },
    }
}

fn start_attempt<'a, K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    cache: &'a PreparedPlanCache<K, V>,
    key: Arc<K>,
    digest: u128,
    signature_bytes: usize,
    promoted_queue: Option<QueueId>,
) -> Result<ProducerGuard<'a, K, V>, Arc<PrepareError>> {
    if let Some(queue_id) = promoted_queue {
        remove_queue_record(state, queue_id);
    }
    let key_bytes = key.retained_bytes().ok_or_else(accounting_prepare_error)?;
    let attempt_metadata_bytes = size_of::<ActiveAttempt<K>>();
    let visible_entry_bytes = checked_sum([size_of::<EntryRecord<K, V>>(), size_of::<EntryId>()])
        .ok_or_else(accounting_prepare_error)?;
    let active_bytes = checked_sum([
        key_bytes,
        signature_bytes,
        attempt_metadata_bytes,
        visible_entry_bytes,
    ])
    .ok_or_else(accounting_prepare_error)?;
    let attempt_id = allocate_attempt_id(state);
    let entry_id = allocate_entry_id(state);
    let attempt = ActiveAttempt {
        id: attempt_id,
        entry_id: Some(entry_id),
        digest,
        key: Arc::clone(&key),
        generation: Arc::clone(&state.generation),
        key_bytes,
        signature_bytes,
        attempt_metadata_bytes,
        visible_entry_bytes,
    };
    let entry = EntryRecord {
        id: entry_id,
        digest,
        key,
        state: PreparedEntryState::Preparing {
            attempt_id,
            waiting_readers: 0,
        },
        generation: Arc::clone(&state.generation),
        retained_bytes: visible_entry_bytes,
        shared_id: None,
    };
    state.attempts.insert(attempt_id, attempt);
    state.entries.insert(entry_id, entry);
    state
        .entry_buckets
        .entry(digest)
        .or_default()
        .push(entry_id);
    state.retained_bytes = state
        .retained_bytes
        .checked_add(active_bytes)
        .ok_or_else(accounting_prepare_error)?;
    state.in_flight = state
        .in_flight
        .checked_add(1)
        .ok_or_else(accounting_prepare_error)?;
    state.stats.preparations = state.stats.preparations.saturating_add(1);
    state.stats.misses = state.stats.misses.saturating_add(1);
    state.stats.peak_in_flight = state.stats.peak_in_flight.max(state.in_flight);
    state.bump_revision();
    cache.changed.notify_all();
    Ok(ProducerGuard {
        cache,
        attempt_id,
        entry_id,
        published: false,
    })
}

fn retain_entry<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    entry_id: EntryId,
    terminal: CacheTerminal<K, V>,
    entry_bytes: usize,
    shared_id: Option<SharedRetentionId>,
) -> Result<(), Arc<PrepareError>> {
    let Some(entry) = state.entries.get_mut(&entry_id) else {
        return Ok(());
    };
    let old_bytes = entry.retained_bytes;
    entry.state = PreparedEntryState::Retained(terminal);
    entry.retained_bytes = entry_bytes;
    entry.shared_id = shared_id;
    state.retained_bytes = state.retained_bytes.saturating_sub(old_bytes);
    state.retained_bytes = state
        .retained_bytes
        .checked_add(entry_bytes)
        .ok_or_else(accounting_prepare_error)?;
    state.lru.put(entry_id, ());
    Ok(())
}

fn publish_ephemeral_or_remove<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    entry_id: EntryId,
    terminal: CacheTerminal<K, V>,
    waiting_readers: usize,
) {
    if waiting_readers == 0 {
        remove_entry(state, entry_id, false);
        return;
    }
    let Some(digest) = state.entries.get(&entry_id).map(|entry| entry.digest) else {
        return;
    };
    remove_entry_index(state, digest, entry_id);
    if let Some(entry) = state.entries.get_mut(&entry_id) {
        state.retained_bytes = state.retained_bytes.saturating_sub(entry.retained_bytes);
        let ephemeral_bytes = size_of::<EntryRecord<K, V>>();
        entry.retained_bytes = ephemeral_bytes;
        entry.shared_id = None;
        entry.state = PreparedEntryState::Ephemeral {
            terminal,
            remaining_readers: waiting_readers,
        };
        state.retained_bytes = state.retained_bytes.saturating_add(ephemeral_bytes);
    }
}

fn evict_to_limits<K: PreparedCacheKey, V: PreparedValue>(state: &mut CacheState<K, V>) {
    while state.lru.len() > state.limits.max_entries.get()
        || state.retained_bytes > state.limits.max_retained_bytes.get()
    {
        let Some((entry_id, ())) = state.lru.pop_lru() else {
            break;
        };
        remove_entry(state, entry_id, true);
        state.stats.evictions = state.stats.evictions.saturating_add(1);
    }
}

fn remove_entry<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    entry_id: EntryId,
    lru_already_removed: bool,
) {
    let Some(entry) = state.entries.remove(&entry_id) else {
        return;
    };
    remove_entry_index(state, entry.digest, entry_id);
    if !lru_already_removed {
        let _ = state.lru.pop(&entry_id);
    }
    if let Some(shared_id) = entry.shared_id {
        release_shared(state, shared_id);
    }
    state.retained_bytes = state.retained_bytes.saturating_sub(entry.retained_bytes);
}

fn remove_entry_index<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    digest: u128,
    entry_id: EntryId,
) {
    if let Some(bucket) = state.entry_buckets.get_mut(&digest) {
        bucket.retain(|id| *id != entry_id);
        if bucket.is_empty() {
            state.entry_buckets.remove(&digest);
        }
    }
}

fn remove_queue_record<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    queue_id: QueueId,
) {
    let Some(record) = state.queue_records.remove(&queue_id) else {
        return;
    };
    if let Some(bucket) = state.queue_buckets.get_mut(&record.digest) {
        bucket.retain(|id| *id != queue_id);
        if bucket.is_empty() {
            state.queue_buckets.remove(&record.digest);
        }
    }
    state.queue_order.retain(|id| *id != queue_id);
    state.retained_bytes = state.retained_bytes.saturating_sub(record.retained_bytes());
}

fn clear_state<K: PreparedCacheKey, V: PreparedValue>(state: &mut CacheState<K, V>) {
    let mut retained_bytes = 0usize;
    for attempt in state.attempts.values_mut() {
        attempt.entry_id = None;
        retained_bytes =
            retained_bytes.saturating_add(attempt.retained_bytes_without_visible_entry());
    }
    state.entries.clear();
    state.entry_buckets.clear();
    state.lru.clear();
    state.queue_records.clear();
    state.queue_buckets.clear();
    state.queue_order.clear();
    state.shared.clear();
    state.retained_bytes = retained_bytes;
    state.generation = Arc::new(CacheGenerationToken);
    state.bump_revision();
}

fn finish_attempt_accounting<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    attempt: &ActiveAttempt<K>,
) {
    state.retained_bytes = state
        .retained_bytes
        .saturating_sub(attempt.retained_bytes_without_visible_entry());
    state.in_flight = state.in_flight.saturating_sub(1);
}

fn retain_shared<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    shared: SharedRetention<K::Shared>,
) -> Result<SharedRetentionId, Arc<PrepareError>> {
    let id = SharedRetentionId::of(&shared.value);
    if let Some(charge) = state.shared.get_mut(&id) {
        if !Arc::ptr_eq(&charge.value, &shared.value) {
            return Err(Arc::new(PrepareError::Engine {
                source: Arc::new(CacheInternalError::SharedPointerCollision),
            }));
        }
        charge.references = charge
            .references
            .checked_add(1)
            .ok_or_else(accounting_prepare_error)?;
        return Ok(id);
    }
    let retained_bytes = shared_new_charge(state, &shared)?;
    state.retained_bytes = state
        .retained_bytes
        .checked_add(retained_bytes)
        .ok_or_else(accounting_prepare_error)?;
    state.shared.insert(
        id,
        SharedCharge {
            value: shared.value,
            retained_bytes,
            references: 1,
        },
    );
    Ok(id)
}

fn shared_new_charge<K: PreparedCacheKey, V: PreparedValue>(
    state: &CacheState<K, V>,
    shared: &SharedRetention<K::Shared>,
) -> Result<usize, Arc<PrepareError>> {
    let id = SharedRetentionId::of(&shared.value);
    if let Some(charge) = state.shared.get(&id) {
        if Arc::ptr_eq(&charge.value, &shared.value) {
            return Ok(0);
        }
        return Err(Arc::new(PrepareError::Engine {
            source: Arc::new(CacheInternalError::SharedPointerCollision),
        }));
    }
    checked_sum([
        shared.retained_bytes.ok_or_else(accounting_prepare_error)?,
        size_of::<SharedRetentionId>(),
        size_of::<SharedCharge<K::Shared>>(),
    ])
    .ok_or_else(accounting_prepare_error)
}

fn release_shared<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
    shared_id: SharedRetentionId,
) {
    let Some(charge) = state.shared.get_mut(&shared_id) else {
        return;
    };
    charge.references = charge.references.saturating_sub(1);
    if charge.references == 0 {
        let charge = state
            .shared
            .remove(&shared_id)
            .expect("shared charge exists");
        state.retained_bytes = state.retained_bytes.saturating_sub(charge.retained_bytes);
    }
}

fn lookup_from_terminal<K: PreparedCacheKey, V: PreparedValue>(
    terminal: &CacheTerminal<K, V>,
    shared_id: Option<SharedRetentionId>,
    shared_table: &HashMap<SharedRetentionId, SharedCharge<K::Shared>>,
) -> CacheLookup<K, V> {
    match terminal {
        CacheTerminal::Ready(value) => CacheLookup::Ready(Arc::clone(value)),
        CacheTerminal::Redirect {
            requirements,
            shared,
        } => CacheLookup::Redirect {
            requirements: requirements.clone(),
            shared: shared.as_ref().map(Arc::clone).or_else(|| {
                shared_id.and_then(|id| shared_table.get(&id).map(|s| Arc::clone(&s.value)))
            }),
        },
        CacheTerminal::FailedDeterministic(error) => {
            CacheLookup::FailedDeterministic(Arc::clone(error))
        }
        CacheTerminal::FailedTransient(error) => CacheLookup::FailedTransient(Arc::clone(error)),
    }
}

fn lookup_from_produced<K: PreparedCacheKey, V: PreparedValue>(
    produced: CacheProduced<K, V>,
) -> CacheLookup<K, V> {
    match produced {
        CacheProduced::Ready { value, .. } => CacheLookup::Ready(value),
        CacheProduced::Redirect {
            requirements,
            shared,
        } => CacheLookup::Redirect {
            requirements,
            shared: shared.map(|shared| shared.value),
        },
        CacheProduced::FailedDeterministic { error, .. } => CacheLookup::FailedDeterministic(error),
        CacheProduced::FailedTransient(error) => CacheLookup::FailedTransient(error),
    }
}

fn ready_entry_bytes<K: PreparedCacheKey, V: PreparedValue>(key: &K, value: &V) -> Option<usize> {
    checked_sum([
        key.retained_bytes()?,
        value.retained_bytes()?,
        size_of::<EntryRecord<K, V>>(),
        size_of::<EntryId>(),
        size_of::<EntryId>().checked_add(size_of::<()>())?,
    ])
}

fn redirect_entry_bytes<K: PreparedCacheKey, V: PreparedValue>(
    key: &K,
    requirements: &SpecializationRequirements,
) -> Option<usize> {
    checked_sum([
        key.retained_bytes()?,
        specialization_requirements_retained_bytes(requirements)?,
        size_of::<EntryRecord<K, V>>(),
        size_of::<EntryId>(),
        size_of::<EntryId>().checked_add(size_of::<()>())?,
    ])
}

fn deterministic_error_entry_bytes<K: PreparedCacheKey, V: PreparedValue>(
    key: &K,
) -> Option<usize> {
    checked_sum([
        key.retained_bytes()?,
        size_of::<PrepareError>(),
        size_of::<EntryRecord<K, V>>(),
        size_of::<EntryId>(),
        size_of::<EntryId>().checked_add(size_of::<()>())?,
    ])
}

fn specialization_requirements_retained_bytes(
    requirements: &SpecializationRequirements,
) -> Option<usize> {
    checked_sum([
        size_of::<SpecializationRequirements>(),
        requirements
            .inputs()
            .len()
            .checked_mul(size_of::<super::InputSpecializationRequirements>())?,
    ])
}

fn allocate_attempt_id<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
) -> AttemptId {
    loop {
        state.next_attempt = next_nonzero_id(state.next_attempt);
        let id = AttemptId(state.next_attempt);
        if !state.attempts.contains_key(&id) {
            return id;
        }
    }
}

fn allocate_entry_id<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
) -> EntryId {
    loop {
        state.next_entry = next_nonzero_id(state.next_entry);
        let id = EntryId(state.next_entry);
        if !state.entries.contains_key(&id) {
            return id;
        }
    }
}

fn allocate_queue_id<K: PreparedCacheKey, V: PreparedValue>(
    state: &mut CacheState<K, V>,
) -> QueueId {
    loop {
        state.next_queue = next_nonzero_id(state.next_queue);
        let id = QueueId(state.next_queue);
        if !state.queue_records.contains_key(&id) {
            return id;
        }
    }
}

fn next_nonzero_id(current: u128) -> u128 {
    let next = current.wrapping_add(1);
    if next == 0 {
        1
    } else {
        next
    }
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
}

fn check_preparation_stack<K: PreparedCacheKey>(
    requested: PreparationKeySummary,
) -> Result<(), Arc<PrepareError>> {
    PREPARATION_STACK.with(|stack| {
        let stack = stack.borrow();
        let Some(parent) = stack.last().copied() else {
            return Ok(());
        };
        if parent.type_id == TypeId::of::<K>() && parent.summary == requested {
            Err(Arc::new(PrepareError::PreparationCycle { key: requested }))
        } else {
            Err(Arc::new(PrepareError::NestedPreparationUnsupported {
                parent: parent.summary,
                requested,
            }))
        }
    })
}

fn poisoned_state_error() -> RuntimeStateError {
    RuntimeStateError::Poisoned {
        lock: PREPARED_CACHE_LOCK,
    }
}

fn accounting_prepare_error() -> Arc<PrepareError> {
    Arc::new(PrepareError::Engine {
        source: Arc::new(CacheInternalError::AccountingOverflow),
    })
}

fn capacity_error<K: PreparedCacheKey, V: PreparedValue>(
    state: &CacheState<K, V>,
) -> Arc<PrepareError> {
    Arc::new(PrepareError::CacheInFlightCapacityExceeded {
        in_flight: state.in_flight,
        queued_distinct_keys: state.queue_records.len(),
    })
}

fn saturating_add_cache_stats(total: &mut CacheStats, value: CacheStats) {
    total.entries = total.entries.saturating_add(value.entries);
    total.retained_bytes = total.retained_bytes.saturating_add(value.retained_bytes);
    total.hits = total.hits.saturating_add(value.hits);
    total.misses = total.misses.saturating_add(value.misses);
    total.evictions = total.evictions.saturating_add(value.evictions);
    total.clears = total.clears.saturating_add(value.clears);
}
