//! Phase lease: lend the session's CPU pool to a downstream phase scheduler.
//!
//! See `docs/design/held-cpu-session-1945-phase-lease.md`. A phase drives one lane
//! per worker of the context's own pool through `ThreadPool::broadcast`, keeps one
//! owner-inheriting child session open per lane for the whole lane, joins every
//! lane, and supports cooperative cancellation. It never creates a pool, never
//! admits a second root execution, and never leaves a peer stranded.

use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

use tenferro_tensor::{BackendSession, BackendSessionHost, SessionEntryError};

use super::HeldSessionState;
use crate::arbiter::{ResourceOwner, ResourcePermit};
use crate::backend::{CpuBackend, CPU_BACKEND};
use crate::engine::EngineResources;
use crate::provider::CpuOperationEntry;

/// Why a phase could not be started.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use tenferro_cpu::CpuPhaseError;
///
/// let error = CpuPhaseError::TargetPoolCaller { backend: "CpuBackend" };
/// assert!(error.to_string().contains("CpuBackend"));
/// assert!(error.source().is_none());
/// ```
#[derive(Debug, Clone, thiserror::Error)]
pub enum CpuPhaseError {
    /// The phase was opened from a worker of the pool it would broadcast to.
    ///
    /// This is the phase's driver-placement policy, not a limitation of the
    /// broadcast itself: a target-pool driver would run the inline lane on a worker
    /// that is not the session's opening thread, and would have to re-enter a child
    /// session on a thread that already carries the root's execution.
    #[error(
        "{backend}: a CPU phase cannot be driven from a worker of its own pool; open it on a \
         thread outside that pool"
    )]
    TargetPoolCaller {
        /// Backend that rejected the phase.
        backend: &'static str,
    },
}

/// Why [`CpuPhase::run`] did not complete successfully.
///
/// # Examples
///
/// ```
/// use std::error::Error;
/// use tenferro_cpu::PhaseRunError;
/// use tenferro_tensor::SessionEntryError;
///
/// let error: PhaseRunError<SessionEntryError> =
///     PhaseRunError::Session(SessionEntryError::Reentered {
///         backend: "CpuBackend",
///     });
/// assert!(error.to_string().contains("CpuBackend"));
/// assert!(std::error::Error::source(&error).is_some());
/// ```
#[derive(Debug, thiserror::Error)]
pub enum PhaseRunError<E> {
    /// A lane callback returned an error. Every other lane was joined first, and
    /// one lane error is reported rather than a chronologically first one.
    #[error("CPU phase lane failed: {0}")]
    Lane(#[source] E),
    /// A lane could not enter its child session or could not clean up. This
    /// replaces a lane error, exactly as a restoration failure replaces a callback
    /// value on the scoped entry.
    #[error("CPU phase session failed: {0}")]
    Session(#[source] SessionEntryError),
}

/// One lane of a running phase.
///
/// A lane is bound to one worker of the context's pool and to one child session
/// that stays open for the whole lane, so its scratch and prepared plans are reused
/// across every work item the callback pulls. The lane exposes the operation
/// surface only: it cannot open a root session or another phase.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{Tensor, TensorRead};
///
/// let backend = CpuBackend::with_threads(1)?;
/// let x = Tensor::from_vec_col_major(vec![1], vec![7.0_f64])?;
/// let mut session = backend.open_session()?;
/// session.phase(|phase| -> Result<(), Box<dyn std::error::Error>> {
///     phase.run(|_index, lane| -> Result<(), tenferro_tensor::Error> {
///         let value = lane
///             .session()
///             .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))?;
///         assert_eq!(value.as_slice::<f64>()?, &[14.0]);
///         Ok(())
///     })?;
///     Ok(())
/// })?;
/// session.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PhaseLane<'lane> {
    session: &'lane mut dyn BackendSession,
    cancel: &'lane AtomicBool,
}

impl fmt::Debug for PhaseLane<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PhaseLane")
            .field("cancelled", &self.cancelled())
            .finish_non_exhaustive()
    }
}

impl PhaseLane<'_> {
    /// The worker-local execution surface for this lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// let x = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let mut session = backend.open_session()?;
    /// session.phase(|phase| -> Result<(), Box<dyn std::error::Error>> {
    ///     phase.run(|_index, lane| {
    ///         let doubled = lane.session().add_read(
    ///             TensorRead::from_tensor(&x),
    ///             TensorRead::from_tensor(&x),
    ///         )?;
    ///         assert_eq!(doubled.as_slice::<f64>()?, &[4.0]);
    ///         Ok::<(), tenferro_tensor::Error>(())
    ///     })?;
    ///     Ok(())
    /// })?;
    /// session.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn session(&mut self) -> &mut dyn BackendSession {
        self.session
    }

    /// Whether some lane already failed or unwound.
    ///
    /// A lane callback must return cooperatively once this is set and must never
    /// wait indefinitely for a peer: the broadcast can only join callbacks that
    /// terminate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// let mut session = backend.open_session()?;
    /// let observed = std::sync::atomic::AtomicBool::new(false);
    /// session.phase(|phase| -> Result<(), Box<dyn std::error::Error>> {
    ///     phase.run(|_index, lane| {
    ///         observed.store(lane.cancelled(), std::sync::atomic::Ordering::Relaxed);
    ///         Ok::<(), std::convert::Infallible>(())
    ///     })?;
    ///     Ok(())
    /// })?;
    /// assert!(
    ///     !observed.load(std::sync::atomic::Ordering::Relaxed),
    ///     "a clean phase reports no cancellation"
    /// );
    /// session.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// A phase lease over one held CPU session.
///
/// Created by [`CpuHeldSession::phase`](super::CpuHeldSession::phase), which borrows
/// the session mutably so the caller's own numerical execution stays parked for the
/// phase's duration.
pub struct CpuPhase<'h> {
    backend: &'h CpuBackend,
    permit: &'h ResourcePermit,
    owner: ResourceOwner,
    child: CpuBackend,
    pool: Option<&'h rayon::ThreadPool>,
    lanes: usize,
    /// The inline lane runs under the caller-affinity guard and the owner marker the
    /// root session installed on its **opening thread**. Moving a phase to another
    /// thread would run that lane outside the selected CPU set, so the lease is
    /// bound to the thread that opened it.
    _opening_thread: PhantomData<Rc<()>>,
}

impl fmt::Debug for CpuPhase<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuPhase")
            .field("lanes", &self.lanes)
            .field("pooled", &self.pool.is_some())
            .finish_non_exhaustive()
    }
}

impl CpuPhase<'_> {
    /// The number of lanes this phase drives.
    ///
    /// One for a context without an inner execution pool (including every
    /// one-worker context), otherwise the context's thread budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// let mut session = backend.open_session()?;
    /// session.phase(|phase| {
    ///     assert_eq!(phase.lanes(), 1);
    ///     Ok::<(), std::convert::Infallible>(())
    /// })?;
    /// session.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Run `lane` on every lane and return after all of them have finished.
    ///
    /// Each lane keeps one owner-inheriting child session open for the whole call,
    /// so the callback reuses that child's private scratch across every work item it
    /// pulls. Lanes are joined on the success, error and panic paths; on the first
    /// lane error or unwind the other lanes observe
    /// [`PhaseLane::cancelled`] and stop pulling work.
    ///
    /// The phase reports success only after joining. Retaining lane outputs privately
    /// and publishing them only after a successful run is the callback's
    /// responsibility: this method does not roll back caller-owned outputs or shared
    /// state.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let backend = CpuBackend::with_threads(2)?;
    /// let x = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// let mut session = backend.open_session()?;
    /// let results = std::sync::Mutex::new(Vec::new());
    /// session.phase(|phase| -> Result<(), Box<dyn std::error::Error>> {
    ///     phase.run(|index, lane| {
    ///         if lane.cancelled() {
    ///             return Ok::<(), tenferro_tensor::Error>(());
    ///         }
    ///         let value = lane.session().add_read(
    ///             TensorRead::from_tensor(&x),
    ///             TensorRead::from_tensor(&x),
    ///         )?;
    ///         results
    ///             .lock()
    ///             .expect("lane results lock")
    ///             .push((index, value.as_slice::<f64>()?[0]));
    ///         Ok(())
    ///     })?;
    ///     Ok(())
    /// })?;
    /// let results = results.into_inner().expect("lane results");
    /// assert!(!results.is_empty(), "at least one lane ran");
    /// assert!(results.iter().all(|(_, value)| *value == 6.0));
    /// session.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`PhaseRunError::Session`] when a lane could not enter its child
    /// session or could not be cleaned up, and [`PhaseRunError::Lane`] with the
    /// callback's own error otherwise. Both are reported only after every lane has
    /// finished.
    ///
    /// # Panics
    ///
    /// A panic inside a lane callback is not converted into an error: every lane is
    /// joined first and one panic is then resumed on the calling thread.
    pub fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut PhaseLane<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), PhaseRunError<E>> {
        if self.lanes <= 1 {
            return self.run_single_lane(&lane);
        }
        let pool = match self.pool {
            Some(pool) => pool,
            None => return self.run_single_lane(&lane),
        };
        // A worker of this pool can never drive the blocking broadcast below.
        if pool.current_thread_index().is_some() {
            return Err(PhaseRunError::Session(SessionEntryError::Reentered {
                backend: CPU_BACKEND,
            }));
        }
        let cancel = AtomicBool::new(false);
        let child = &self.child;
        let cancel_ref = &cancel;
        let lane_ref = &lane;
        let outcomes =
            pool.broadcast(|context| run_worker_lane(child, cancel_ref, context.index(), lane_ref));
        reduce_outcomes(outcomes)
    }

    fn run_single_lane<E: Send>(
        &mut self,
        lane: &(impl Fn(usize, &mut PhaseLane<'_>) -> Result<(), E> + Sync),
    ) -> Result<(), PhaseRunError<E>> {
        // A single lane has no peer that could set the stop signal, so it is not
        // checked here; the flag stays part of the common lane surface and is always
        // false for this lane.
        let cancel = AtomicBool::new(false);
        // The pool has no inner execution region here, so this lane runs inline under
        // the admission the root session already holds. It gets the same private
        // scratch and panic isolation a worker lane gets, without a second admission
        // and without relaxing same-thread child rejection.
        let mut resources = EngineResources::for_child_execution(self.buffer_limit());
        let entry = CpuOperationEntry::new(self.backend.engine.domain(), self.permit);
        let owner = self.owner;
        let backend = self.backend;
        let outcome = match entry.enter_owned(|context| {
            crate::backend::with_operation_session(
                backend,
                entry,
                Some(context),
                &mut resources,
                None,
                owner,
                |session| {
                    let mut lane_handle = PhaseLane {
                        session,
                        cancel: &cancel,
                    };
                    lane(0, &mut lane_handle)
                },
            )
        }) {
            Ok(()) => None,
            Err(error) => Some(PhaseRunError::Lane(error)),
        };
        reduce_outcomes([outcome])
    }

    fn buffer_limit(&self) -> usize {
        self.backend.shared.buffer_limit.load(Ordering::Relaxed)
    }
}

/// Marks a failed lane so its peers stop pulling work, including while unwinding.
struct CancelOnUnwind<'a> {
    cancel: &'a AtomicBool,
}

impl Drop for CancelOnUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }
}

fn run_worker_lane<E>(
    child: &CpuBackend,
    cancel: &AtomicBool,
    index: usize,
    lane: &(impl Fn(usize, &mut PhaseLane<'_>) -> Result<(), E> + Sync),
) -> Option<PhaseRunError<E>> {
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let mut child = child.clone();
    let outcome = {
        // Also arms cancellation while a worker unwinds, so peers stop pulling work
        // even when this lane fails by panicking.
        let _unwind = CancelOnUnwind { cancel };
        child.with_backend_session(|session| {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            let mut lane_handle = PhaseLane { session, cancel };
            let outcome = {
                // Signal a callback failure *before* the child session is cleaned up,
                // so peers stop pulling work during the cleanup window rather than
                // after it.
                let _callback_unwind = CancelOnUnwind { cancel };
                lane(index, &mut lane_handle)
            };
            if outcome.is_err() {
                cancel.store(true, Ordering::Relaxed);
            }
            outcome
        })
    };
    fold_lane_outcome(outcome, cancel)
}

fn fold_lane_outcome<E>(
    outcome: Result<Result<(), E>, SessionEntryError>,
    cancel: &AtomicBool,
) -> Option<PhaseRunError<E>> {
    match outcome {
        Ok(Ok(())) => None,
        Ok(Err(error)) => {
            cancel.store(true, Ordering::Relaxed);
            Some(PhaseRunError::Lane(error))
        }
        Err(error) => {
            cancel.store(true, Ordering::Relaxed);
            Some(PhaseRunError::Session(error))
        }
    }
}

/// A session (entry or cleanup) failure replaces a lane error, as a restoration
/// failure replaces a callback value on the scoped entry.
fn reduce_outcomes<E>(
    outcomes: impl IntoIterator<Item = Option<PhaseRunError<E>>>,
) -> Result<(), PhaseRunError<E>> {
    let mut session_error = None;
    let mut lane_error = None;
    for outcome in outcomes {
        match outcome {
            Some(PhaseRunError::Session(error)) => {
                session_error.get_or_insert(error);
            }
            Some(PhaseRunError::Lane(error)) => {
                lane_error.get_or_insert(error);
            }
            None => {}
        }
    }
    if let Some(error) = session_error {
        return Err(PhaseRunError::Session(error));
    }
    if let Some(error) = lane_error {
        return Err(PhaseRunError::Lane(error));
    }
    Ok(())
}

impl super::CpuHeldSession<'_> {
    /// Open a phase lease that lends this session's CPU pool to a downstream
    /// scheduler.
    ///
    /// The phase borrows the session mutably, so the caller's own numerical execution
    /// is parked for its duration, and it returns after every lane has been joined. It
    /// drives the context's own pool — never a pool created for the phase — with one
    /// lane per worker, and each lane keeps one owner-inheriting child session open
    /// with its own private scratch. A context without an inner execution pool,
    /// including every one-worker context, drives a single lane inline on the caller.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{Tensor, TensorRead};
    ///
    /// let backend = CpuBackend::with_threads(2)?;
    /// let x = Tensor::from_vec_col_major(vec![1], vec![5.0_f64])?;
    /// let mut session = backend.open_session()?;
    /// let lane_count = std::sync::atomic::AtomicUsize::new(0);
    /// session.phase(|phase| -> Result<(), Box<dyn std::error::Error>> {
    ///     assert_eq!(phase.lanes(), 2);
    ///     phase.run(|_index, lane| {
    ///         lane_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ///         let value = lane.session().add_read(
    ///             TensorRead::from_tensor(&x),
    ///             TensorRead::from_tensor(&x),
    ///         )?;
    ///         assert_eq!(value.as_slice::<f64>()?, &[10.0]);
    ///         Ok::<(), tenferro_tensor::Error>(())
    ///     })?;
    ///     Ok(())
    /// })?;
    /// assert_eq!(lane_count.load(std::sync::atomic::Ordering::Relaxed), 2);
    /// session.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`CpuPhaseError::TargetPoolCaller`] when the calling thread is a worker
    /// of the pool the phase would broadcast to, where the blocking broadcast can never
    /// be driven. The check happens before any work starts.
    ///
    /// # Panics
    ///
    /// Never. A lane panic is resumed on the calling thread after every lane has been
    /// joined, which is the caller's panic, not this method's.
    pub fn phase<R>(&mut self, f: impl FnOnce(&mut CpuPhase<'_>) -> R) -> Result<R, CpuPhaseError> {
        let HeldSessionState {
            backend, permit, ..
        } = &mut self.state;
        let owner = permit.owner();
        let domain = backend.engine.domain();
        let budget = domain.thread_budget().get();
        let pool = if budget > 1 {
            domain.rayon_pool()
        } else {
            None
        };
        if let Some(pool) = pool {
            if pool.current_thread_index().is_some() {
                return Err(CpuPhaseError::TargetPoolCaller {
                    backend: CPU_BACKEND,
                });
            }
        }
        let mut phase = CpuPhase {
            backend,
            permit,
            owner,
            child: backend.with_inherited_owner(owner),
            pool,
            lanes: budget.max(1),
            _opening_thread: PhantomData,
        };
        Ok(f(&mut phase))
    }
}

#[cfg(test)]
mod tests;
