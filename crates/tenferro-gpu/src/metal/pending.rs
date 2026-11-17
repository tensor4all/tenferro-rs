//! Pending device work per shared allocation.
//!
//! One in-order command queue means a submission's completion implies every earlier
//! submission on that queue has completed. Recording only the **newest** submission that
//! named an allocation is therefore enough to answer "may the CPU touch these bytes yet",
//! and it is the whole rule: no per-resource event, no fence call, no second bookkeeping
//! path.
//!
//! Both directions of design §5 fall out of it, because a submission is recorded whether
//! it reads or writes the allocation: a CPU read waits for the newest touch (which is at
//! least as new as every pending write), and a CPU write waits for it too (which is at
//! least as new as every pending read).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use t4a_metal_runtime::Completion;
use tenferro_tensor::HostAccessError;

/// The newest submission that named one allocation, read or write.
#[derive(Debug)]
pub(crate) struct PendingTouches {
    /// The newest submission, once one has been committed on this allocation, or a poison
    /// marker when a commit failed after the runtime may already have submitted it.
    last: Mutex<LastTouch>,
    /// Domain-wide wait count, shared with the context's statistics.
    waits: Arc<AtomicU64>,
}

/// What the host may assume about the newest submission on one allocation.
#[derive(Clone, Debug)]
enum LastTouch {
    /// No submission has named the allocation.
    Idle,
    /// The newest submitted completion; waiting on it covers every earlier submission.
    Submitted(Completion),
    /// A commit failed after the runtime may have submitted its command buffer and the
    /// runtime could not be synchronised afterwards. The completion that would prove the
    /// work finished is unknown, so host access must fail rather than wait on a stale one.
    Poisoned(String),
}

impl PendingTouches {
    /// Bookkeeping for one allocation, counting waits in the domain's counter.
    pub(crate) fn new(waits: Arc<AtomicU64>) -> Self {
        Self {
            last: Mutex::new(LastTouch::Idle),
            waits,
        }
    }

    /// Record `completion` as the newest submission that named this allocation.
    ///
    /// The engine calls this once per named allocation after a successful commit, under
    /// the same runtime lock that serialised the commit, so submissions are recorded in
    /// the order they were committed. Recording also clears a poison marker: a later
    /// completion on the in-order queue proves the poisoned submission finished.
    pub(crate) fn record(&self, completion: &Completion) {
        *self.lock() = LastTouch::Submitted(completion.clone());
    }

    /// Refuse later host access because a submitted completion is unknown.
    ///
    /// The engine calls this only when `commit` failed after the runtime may have submitted
    /// the command buffer and a following `synchronize` also failed, so neither the
    /// completion nor an idle queue can be demonstrated. A poison marker is replaced by the
    /// next successful `record`, whose completion covers every earlier submission on the
    /// in-order queue.
    pub(crate) fn poison(&self, reason: impl Into<String>) {
        *self.lock() = LastTouch::Poisoned(reason.into());
    }

    /// Wait for every submission that has named this allocation.
    ///
    /// The domain's wait counter counts only a wait that had a submission to wait for, so
    /// `MetalStats::waits` reports real waits: an empty allocation that was never touched
    /// does not inflate it, and a test can use the counter as evidence that a launch was
    /// recorded on the allocation it names.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::BackendFailure`] when the submission completed with an error or
    /// did not complete within the runtime's timeout, or when a failed commit left the
    /// newest submitted completion unknown (a poisoned allocation).
    pub(crate) fn wait(&self) -> Result<(), HostAccessError> {
        // Clone out of the lock and drop it before waiting: `Completion::wait` polls the
        // GPU and may block for the runtime's timeout, and a submission on another thread
        // records its completion here.
        let pending = self.lock().clone();
        match pending {
            LastTouch::Idle => Ok(()),
            LastTouch::Submitted(completion) => {
                self.waits.fetch_add(1, Ordering::Relaxed);
                completion
                    .wait()
                    .map_err(|error| HostAccessError::BackendFailure {
                        message: error.to_string(),
                    })
            }
            LastTouch::Poisoned(reason) => Err(HostAccessError::BackendFailure { message: reason }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LastTouch> {
        // A panicking holder could leave the value half-updated, but the value is one
        // `LastTouch` whose only invariant is "newest wins", so a poisoned lock is still
        // readable: recover it rather than stranding every later wait.
        self.last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
