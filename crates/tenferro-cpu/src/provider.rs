//! The CPU execution-context seam shared with the lower numerical libraries.
//!
//! tenferro owns tensor semantics, placement and resource lifetime; cpueinsum,
//! tcontract and tlinalg own numerical execution, lane selection and scheduling.
//! [`CpuExecutionContext`] is the one narrow, owner-scoped interop seam through
//! which those libraries receive the caller's selected pool and budget. It
//! exposes immutable domain facts and never lets a callee install a callback
//! into a pool.

use core::fmt;
use std::num::NonZeroUsize;

use tenferro_tensor::{Tensor, TensorRead};

use crate::buffer_pool::BufferPool;
use crate::contraction::ExecutionResources;

/// Parallel scheduling mode selected for one CPU operation.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::ParallelMode;
/// assert_ne!(ParallelMode::Sequential, ParallelMode::Inner);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelMode {
    /// The operation runs inline on the caller's thread with no fan-out.
    Sequential,
    /// The operation may use the selected pool's inner parallel region, bounded
    /// by [`CpuExecutionContext::thread_budget`].
    Inner,
}

/// Borrowed execution policy of one CPU engine for one operation.
///
/// The context exposes the engine's pool, budget and batch policy to the lower
/// numerical libraries. Callees may read the pool and budget; they cannot
/// install the caller's continuation into the pool or submit through tenferro.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuExecutionContext;
/// # fn inspect(context: &CpuExecutionContext<'_>) {
/// assert!(context.thread_budget().get() >= 1);
/// # }
/// ```
#[derive(Clone, Copy)]
pub struct CpuExecutionContext<'a> {
    resources: &'a ExecutionResources,
    batch_policy: crate::CpuBatchPolicy,
}

impl fmt::Debug for CpuExecutionContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuExecutionContext")
            .field("thread_budget", &self.thread_budget())
            .field("parallel_mode", &self.parallel_mode())
            .finish_non_exhaustive()
    }
}

impl<'a> CpuExecutionContext<'a> {
    pub(crate) fn new(
        resources: &'a ExecutionResources,
        batch_policy: crate::CpuBatchPolicy,
    ) -> Self {
        Self {
            resources,
            batch_policy,
        }
    }

    pub(crate) fn uses_inner_parallelism(self) -> bool {
        self.thread_budget().get() > 1 && self.resources.rayon_pool().is_some()
    }

    /// Return the non-zero maximum participating-thread budget: the engine's
    /// pool size, or one without a pool.
    pub fn thread_budget(&self) -> NonZeroUsize {
        self.resources.thread_budget()
    }

    /// Return the scheduling mode of this context: [`ParallelMode::Inner`] on
    /// a pool with more than one worker, [`ParallelMode::Sequential`] otherwise.
    pub fn parallel_mode(&self) -> ParallelMode {
        if self.uses_inner_parallelism() {
            ParallelMode::Inner
        } else {
            ParallelMode::Sequential
        }
    }

    /// The effective batch policy for batched work in this context.
    ///
    /// It is the engine default unless an enclosing
    /// [`crate::with_batch_policy`] scope overrode it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuBatchStrategy, CpuEngine};
    ///
    /// let mut engine = CpuEngine::serial();
    /// let strategy = engine.with_linalg_pool(|context, _| Ok(context.batch_policy().strategy))?;
    /// assert_eq!(strategy, CpuBatchStrategy::Auto);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn batch_policy(&self) -> crate::CpuBatchPolicy {
        self.batch_policy
    }

    /// Return the faer policy selected by this operation context.
    ///
    /// This hidden public method is the engine-scoped extension contract used
    /// by operation-family crates such as `tenferro-linalg`, so sibling crates
    /// do not derive a second CPU threading policy. A `Par::rayon` value must
    /// be used inside [`CpuExecutionContext::rayon_pool`]'s `install`.
    #[cfg(feature = "native")]
    #[doc(hidden)]
    pub fn faer_parallelism(self) -> faer::Par {
        if self.uses_inner_parallelism() {
            faer::Par::rayon(self.thread_budget().get())
        } else {
            faer::Par::Seq
        }
    }

    /// The Rayon pool this context's inner parallel region runs on.
    ///
    /// `Some` exactly when the context owns an inner region. A callee running
    /// its own kernels on the pool must use at most
    /// [`CpuExecutionContext::thread_budget`] threads, which can be smaller than
    /// the pool, and runs in place on a worker of that pool.
    pub fn rayon_pool(&self) -> Option<&'a rayon::ThreadPool> {
        if self.uses_inner_parallelism() {
            self.resources.rayon_pool()
        } else {
            None
        }
    }

    /// Borrow the retained lower-library execution resources for one call.
    ///
    /// This is the engine-scoped interop seam for the extracted numerical
    /// providers (`tlinalg`, `cpueinsum`): they receive one `Exec` that reuses
    /// the engine's persisted pool wrapper and workspace, instead of a
    /// per-call `Pool::borrow`. The shared SPMD gate is preserved: a context
    /// that does not own an inner region lends a serial-with-workspace token.
    ///
    /// # Errors
    ///
    /// Returns the CPU backend error when the retained execution resources
    /// cannot produce a token, for example an invalid thread budget.
    #[doc(hidden)]
    pub fn contraction_exec(&self) -> crate::Result<cpueinsum::Exec<'a>> {
        let budget = if self.uses_inner_parallelism() {
            self.thread_budget().get()
        } else {
            1
        };
        Ok(self
            .resources
            .exec(budget)?
            .with_batch_policy(self.batch_policy))
    }

    /// Effective native-kernel degree inside this admitted CPU context.
    ///
    /// A sequential context and a context without a pool use one thread.
    #[doc(hidden)]
    pub fn native_thread_count(&self) -> usize {
        if self.uses_inner_parallelism() {
            self.thread_budget().get()
        } else {
            1
        }
    }

    /// Execution policy for a lower strided-rs call made from this context.
    pub(crate) fn strided_exec_context(&self) -> strided_kernel::ExecContext {
        if self.uses_inner_parallelism() {
            // An operation-local thread limit for strided's replay policy; the
            // pool itself is the already-admitted context pool installed by
            // `with_native_parallelism`.
            match strided_kernel::ExecContext::max_threads(self.thread_budget().get()) {
                Ok(context) => context,
                // INVARIANT: the budget is a NonZeroUsize, so this is positive.
                Err(_) => unreachable!("CpuExecutionContext has a non-zero thread budget"),
            }
        } else {
            strided_kernel::ExecContext::serial()
        }
    }

    /// Run a lower-library numerical operation with this context's parallelism.
    ///
    /// The engine's pool is installed for the duration of the operation only,
    /// and only when the engine has more than one worker; a single-worker or
    /// serial engine runs the operation on the calling thread under a
    /// sequential policy.
    pub(crate) fn with_native_parallelism<R: Send>(
        &self,
        operation: impl FnOnce() -> R + Send,
    ) -> R {
        match (self.uses_inner_parallelism(), self.rayon_pool()) {
            // strided decides each operation's width against the engine's pool and
            // enters it only when the operation fans out, so small operations
            // stay on the calling thread (#2206).
            (true, Some(pool)) => {
                strided_basic::with_execution_pool(pool, self.thread_budget(), || {
                    strided_kernel::with_execution_policy(
                        strided_kernel::ExecutionPolicy::Rayon {
                            max_threads: self.thread_budget(),
                        },
                        operation,
                    )
                })
            }
            _ => strided_kernel::with_execution_policy(
                strided_kernel::ExecutionPolicy::Sequential,
                operation,
            ),
        }
    }

    /// Run a faer computation with this context's parallelism.
    ///
    /// faer's `Par::rayon` schedules on the pool of the calling thread, so a
    /// parallel faer call must run inside the engine's pool; a serial context
    /// runs on the caller with `Par::Seq`.
    pub(crate) fn with_faer_pool<R: Send>(
        &self,
        operation: impl FnOnce(faer::Par) -> R + Send,
    ) -> R {
        let par = self.faer_parallelism();
        match (par, self.rayon_pool()) {
            (faer::Par::Rayon(_), Some(pool)) if pool.current_thread_index().is_none() => {
                pool.install(|| operation(par))
            }
            _ => operation(par),
        }
    }

    /// Materialize a borrowed tensor view for one scoped operation and reclaim
    /// its temporary host buffer before returning.
    ///
    /// Owned tensor inputs are borrowed directly. View inputs are materialized
    /// from `buffers`, passed to `operation`, and returned to the same pool on
    /// both success and ordinary error.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::RuntimeState`] when the view is not
    /// accessible from CPU host memory, propagates typed view-materialization
    /// errors, and otherwise returns the error produced by `operation`.
    #[doc(hidden)]
    pub fn with_materialized_tensor_read<R>(
        &self,
        buffers: &mut BufferPool,
        op: &'static str,
        input: TensorRead<'_>,
        operation: impl FnOnce(&Tensor, &mut BufferPool) -> tenferro_tensor::Result<R>,
    ) -> tenferro_tensor::Result<R> {
        match input {
            TensorRead::Tensor(tensor) => operation(tensor, buffers),
            TensorRead::View(view) => {
                let materialized = self.with_native_parallelism(|| {
                    crate::materialize_tensor_read(buffers, op, TensorRead::View(view))
                })?;
                let result = operation(&materialized, buffers);
                crate::backend::reclaim_tensor(buffers, materialized);
                result
            }
        }
    }

    /// Reshape a compact tensor while retaining the current execution proof.
    ///
    /// Small inputs and contexts that cannot parallelize keep the serial host
    /// copy. A large input in a parallel-capable context
    /// materializes through the same pooled strided copy the session's
    /// `reshape_read` uses, sharing `structural::reshape_uses_pool` and its
    /// threshold rather than a second copy of either. The helper does not enter
    /// an executor; the pooled branch takes a buffer from the caller's pool.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when the input and output
    /// shapes have different element counts or the requested layout is invalid.
    #[doc(hidden)]
    pub fn reshape_tensor(
        &self,
        buffers: &mut BufferPool,
        input: &Tensor,
        shape: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        if crate::structural::reshape_uses_pool(input, self.uses_inner_parallelism())? {
            return self.with_native_parallelism(|| {
                crate::structural::reshape_read_with_pool(
                    buffers,
                    TensorRead::from_tensor(input),
                    shape,
                )
            });
        }
        crate::structural::reshape(input, shape)
    }
}
