//! The CPU execution-context seam shared with the lower numerical libraries.
//!
//! tenferro owns tensor semantics, placement and resource lifetime; cpueinsum,
//! tprims and tlinalg own numerical execution, lane selection and scheduling.
//! [`CpuExecutionContext`] is the one narrow, owner-scoped interop seam through
//! which those libraries receive the caller's selected pool and budget. It
//! exposes immutable domain facts and never lets a callee install a callback
//! into a pool.

use core::fmt;
use std::num::NonZeroUsize;

use tenferro_tensor::{Tensor, TensorRead};

use crate::arbiter::{with_execution_owner, ResourcePermit};
use crate::buffer_pool::BufferPool;
use crate::resource_domain::CpuResourceDomain;
use crate::{CpuDomainId, CpuSet};

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

/// Borrowed execution policy for an already-admitted CPU operation.
///
/// The context exposes immutable domain facts while keeping the resource lease
/// private. Callees may read the selected pool and budget; they cannot install
/// the caller's continuation into a pool or submit through tenferro.
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
    domain: &'a CpuResourceDomain,
    parallel_mode: ParallelMode,
}

impl fmt::Debug for CpuExecutionContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuExecutionContext")
            .field("domain_id", &self.domain_id())
            .field("cpus", &self.cpus())
            .field("thread_budget", &self.thread_budget())
            .field("parallel_mode", &self.parallel_mode())
            .finish_non_exhaustive()
    }
}

impl<'a> CpuExecutionContext<'a> {
    fn entered(domain: &'a CpuResourceDomain, parallel_mode: ParallelMode) -> Self {
        Self {
            domain,
            parallel_mode,
        }
    }

    fn uses_inner_parallelism(self) -> bool {
        self.parallel_mode == ParallelMode::Inner
            && self.thread_budget().get() > 1
            && self.domain.rayon_pool().is_some()
    }

    /// Return the stable identity of the selected CPU resource domain.
    pub fn domain_id(&self) -> CpuDomainId {
        self.domain.id()
    }

    /// Return the selected domain's declared logical CPU set.
    pub fn cpus(&self) -> &CpuSet {
        self.domain.cpus()
    }

    /// Return the non-zero maximum participating-thread budget.
    pub fn thread_budget(&self) -> NonZeroUsize {
        self.domain.thread_budget()
    }

    /// Return the scheduling mode selected for this entered operation.
    pub fn parallel_mode(&self) -> ParallelMode {
        self.parallel_mode
    }

    /// Return the faer policy selected by this operation context.
    ///
    /// This hidden public method is the owner-scoped extension contract used by
    /// operation-family crates such as `tenferro-linalg`, so sibling crates do
    /// not derive a second CPU threading policy.
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
            self.domain.rayon_pool()
        } else {
            None
        }
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
    /// The selected pool is installed for the duration of the operation only:
    /// tenferro confines one lower-library numerical call, and never the
    /// caller's continuation.
    pub(crate) fn with_native_parallelism<R: Send>(
        &self,
        operation: impl FnOnce() -> R + Send,
    ) -> R {
        let policy = if self.uses_inner_parallelism() {
            strided_kernel::ExecutionPolicy::Rayon {
                max_threads: self.thread_budget(),
            }
        } else {
            strided_kernel::ExecutionPolicy::Sequential
        };
        let run = || strided_kernel::with_execution_policy(policy, operation);
        match self.rayon_pool() {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }

    /// Materialize a borrowed tensor view for one scoped operation and reclaim
    /// its temporary host buffer before returning.
    ///
    /// Owned tensor inputs are borrowed directly. View inputs are materialized
    /// from `buffers`, passed to `operation`, and returned to the same pool on
    /// both success and ordinary error. The receiver is an unforgeable proof
    /// that the caller is already inside the selected CPU execution domain.
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
    /// This metadata-only helper does not enter an executor or borrow scratch.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when the input and output
    /// shapes have different element counts or the requested layout is invalid.
    #[doc(hidden)]
    pub fn reshape_tensor(
        &self,
        input: &Tensor,
        shape: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        crate::structural::reshape(input, shape)
    }
}

/// Crate-private unentered capability for one CPU operation.
///
/// This is the only type that owns the resource permit. A
/// [`CpuExecutionContext`] is constructed only from an admitted entry.
#[derive(Clone, Copy)]
pub(crate) struct CpuOperationEntry<'a> {
    domain: &'a CpuResourceDomain,
    permit: &'a ResourcePermit,
}

impl<'a> CpuOperationEntry<'a> {
    pub(crate) fn new(domain: &'a CpuResourceDomain, permit: &'a ResourcePermit) -> Self {
        Self { domain, permit }
    }

    pub(crate) fn domain_id(self) -> CpuDomainId {
        self.domain.id()
    }

    /// Borrow the admitted resources without dispatching the caller's work.
    pub(crate) fn enter<R>(
        self,
        parallel_mode: ParallelMode,
        operation: impl FnOnce(&CpuExecutionContext<'_>) -> R,
    ) -> R {
        with_execution_owner(self.permit.owner(), || {
            let context = CpuExecutionContext::entered(self.domain, parallel_mode);
            operation(&context)
        })
    }

    /// Enter this domain for a whole backend session.
    pub(crate) fn enter_managed_session<R>(
        self,
        operation: impl FnOnce(CpuExecutionContext<'a>) -> R,
    ) -> Result<R, tenferro_tensor::SessionEntryError> {
        let mode = self.preferred_engine_mode();
        Ok(self.enter(mode, |_| {
            operation(CpuExecutionContext::entered(self.domain, mode))
        }))
    }

    pub(crate) fn preferred_engine_mode(self) -> ParallelMode {
        if self.domain.thread_budget().get() > 1 && self.domain.rayon_pool().is_some() {
            ParallelMode::Inner
        } else {
            ParallelMode::Sequential
        }
    }

    pub(crate) fn thread_budget(self) -> NonZeroUsize {
        self.domain.thread_budget()
    }
}
