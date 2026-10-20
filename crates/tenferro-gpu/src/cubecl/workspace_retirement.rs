//! Deferred retirement of vendor device workspaces.
//!
//! A retired workspace returns its CubeCL handle to the shared pool, so the
//! handle may only be released once the vendor work that used it has completed
//! on its stream. `Workspace::drop` used to synchronize that stream, which
//! drains the pipeline: `gpu/tensornetwork` trace execution spent 54% of each
//! call with the device idle between kernels because plan-cache eviction
//! synchronized once per evicted workspace.
//!
//! Retirement instead records a CUDA event on the workspace's stream and
//! releases the handle once the event reports completion. The event is the
//! completion witness that the stream barrier provided before, so a block is
//! still never returned to the pool while vendor work may reference it.

use std::collections::VecDeque;

use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUevent, CUevent_flags, CUstream};

use super::runtime::CudaRuntimeState;

/// In-flight retirements allowed before retirement falls back to a barrier.
///
/// The queue exists to avoid draining the stream on eviction, not to buffer an
/// unbounded number of workspaces. Retirements normally resolve within the next
/// contraction, so the depth stays in the single digits.
pub(crate) const DEFAULT_WORKSPACE_RETIREMENT_CAPACITY: usize = 16;

/// One workspace waiting for its stream to reach a recorded event.
#[derive(Debug)]
struct RetiredWorkspace {
    event: CUevent,
    handle: cubecl_runtime::server::Handle,
    stream: u64,
}

/// Counters for deferred workspace retirement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceRetirementStats {
    /// Workspaces handed to the queue instead of a stream barrier.
    pub deferred: u64,
    /// Queued workspaces whose handle has been released.
    pub released: u64,
    /// Retirements that fell back to synchronizing the stream.
    pub barrier_fallbacks: u64,
    /// Handles leaked because no completion could be proven.
    pub leaked: u64,
    /// Retirements currently waiting for completion.
    pub in_flight: usize,
    /// High-water mark of `in_flight`.
    pub max_in_flight: usize,
}

impl WorkspaceRetirementStats {
    fn record_deferred(&mut self) {
        self.deferred += 1;
        self.in_flight += 1;
        self.max_in_flight = self.max_in_flight.max(self.in_flight);
    }
}

/// Bounded queue of workspaces awaiting stream completion.
#[derive(Debug)]
pub(crate) struct WorkspaceRetirementQueue {
    entries: VecDeque<RetiredWorkspace>,
    capacity: usize,
    stats: WorkspaceRetirementStats,
}

impl Default for WorkspaceRetirementQueue {
    fn default() -> Self {
        Self::new(DEFAULT_WORKSPACE_RETIREMENT_CAPACITY)
    }
}

impl WorkspaceRetirementQueue {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            stats: WorkspaceRetirementStats::default(),
        }
    }

    pub(crate) fn stats(&self) -> WorkspaceRetirementStats {
        self.stats
    }

    /// Device bytes of the workspaces still waiting for their stream.
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.entries.iter().map(|entry| entry.handle.size()).sum()
    }

    /// Defer `handle` until the work already enqueued on `stream` completes.
    ///
    /// Falls back to a stream barrier when no event can be recorded, when the
    /// queue is at capacity, or when a previous retirement could not be
    /// resolved. A handle is never released without a completion witness: if
    /// even the barrier fails, the handle is leaked.
    pub(crate) fn retire(
        &mut self,
        runtime: &CudaRuntimeState,
        stream: u64,
        handle: cubecl_runtime::server::Handle,
    ) {
        self.drain(runtime);
        if self.entries.len() >= self.capacity {
            self.stats.barrier_fallbacks += 1;
            self.barrier_oldest(runtime);
        }
        match self.record_event(runtime, stream) {
            Some(event) => {
                self.entries.push_back(RetiredWorkspace {
                    event,
                    handle,
                    stream,
                });
                self.stats.record_deferred();
            }
            None => {
                self.stats.barrier_fallbacks += 1;
                self.barrier(runtime, stream, handle);
            }
        }
    }

    /// Release every queued workspace whose event has completed.
    pub(crate) fn drain(&mut self, runtime: &CudaRuntimeState) {
        let mut index = 0;
        while index < self.entries.len() {
            let completed = {
                let entry = &self.entries[index];
                unsafe { cuda_result::event::query(entry.event) }.is_ok()
            };
            if completed {
                let entry = self
                    .entries
                    .remove(index)
                    .expect("index is within the retirement queue");
                self.release(runtime, entry);
            } else {
                index += 1;
            }
        }
    }

    /// Resolve every queued retirement before returning, for explicit barriers
    /// and teardown. Waits on each recorded event rather than on the stream, so
    /// work enqueued after the retirement point stays asynchronous.
    pub(crate) fn drain_blocking(&mut self, runtime: &CudaRuntimeState) {
        while let Some(entry) = self.entries.pop_front() {
            self.wait_for(runtime, entry);
        }
    }

    fn record_event(&self, runtime: &CudaRuntimeState, stream: u64) -> Option<CUevent> {
        if runtime
            .set_current_cuda_context("cutensor_workspace_retire")
            .is_err()
        {
            return None;
        }
        let event = cuda_result::event::create(CUevent_flags::CU_EVENT_DISABLE_TIMING).ok()?;
        // SAFETY: `event` was just created and `stream` is a live CUDA stream
        // owned by this runtime; recording binds the event to that stream.
        if unsafe { cuda_result::event::record(event, stream as CUstream) }.is_err() {
            // SAFETY: the event is not recorded anywhere and is destroyed once.
            let _ = unsafe { cuda_result::event::destroy(event) };
            return None;
        }
        Some(event)
    }

    fn barrier_oldest(&mut self, runtime: &CudaRuntimeState) {
        if let Some(entry) = self.entries.pop_front() {
            self.wait_for(runtime, entry);
        }
    }

    /// Block until this retirement's recorded point is reached, then release.
    fn wait_for(&mut self, runtime: &CudaRuntimeState, entry: RetiredWorkspace) {
        let waited = runtime
            .set_current_cuda_context("cutensor_workspace_wait")
            .is_ok()
            // SAFETY: `entry.event` was recorded on `entry.stream` and is
            // destroyed exactly once below.
            && unsafe { cuda_result::event::synchronize(entry.event) }.is_ok();
        if !waited
            && runtime
                .synchronize_raw_stream(entry.stream, "cutensor_workspace_wait")
                .is_err()
        {
            self.stats.leaked += 1;
            // SAFETY: a leaked event is never recorded or queried again.
            let _ = unsafe { cuda_result::event::destroy(entry.event) };
            std::mem::forget(entry.handle);
            self.stats.in_flight = self.stats.in_flight.saturating_sub(1);
            return;
        }
        self.release(runtime, entry);
    }

    /// Release a retirement whose completion has already been observed.
    fn release(&mut self, _runtime: &CudaRuntimeState, entry: RetiredWorkspace) {
        // SAFETY: the caller observed completion (event query, event wait, or a
        // stream barrier) and the event is destroyed exactly once here.
        let _ = unsafe { cuda_result::event::destroy(entry.event) };
        drop(entry.handle);
        self.stats.released += 1;
        self.stats.in_flight = self.stats.in_flight.saturating_sub(1);
    }

    fn barrier(
        &mut self,
        runtime: &CudaRuntimeState,
        stream: u64,
        handle: cubecl_runtime::server::Handle,
    ) {
        if runtime
            .synchronize_raw_stream(stream, "cutensor_workspace_drop")
            .is_err()
        {
            self.stats.leaked += 1;
            std::mem::forget(handle);
            return;
        }
        drop(handle);
        self.stats.released += 1;
    }
}
