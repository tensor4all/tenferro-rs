//! Object-safe CPU contraction provider contracts.
//!
//! Providers synchronously write into engine-owned outputs. Request
//! constructors are crate-private because only the CPU engine may attest that
//! tensor metadata and reachable ranges have already been validated.

use core::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use tenferro_tensor::backend::GroupedGemmJob;
use tenferro_tensor::{
    DType, DotGeneralAccumulation, Tensor, TensorRead, TensorView, TensorViewMut, TensorWrite,
};

use crate::arbiter::{with_execution_owner, ResourcePermit};
use crate::backend::CpuBackendKind;
use crate::buffer_pool::BufferPool;
use crate::domain_executor::{indexed_jobs, install_scoped};
#[cfg(feature = "cpu-blas")]
use crate::provider_capability::builtin_blas_execution_capabilities;
#[cfg(not(feature = "cpu-blas"))]
use crate::provider_capability::serial_capabilities;
use crate::provider_capability::{engine_worker_capabilities, CpuProviderExecutionCapabilities};
use crate::resource_domain::CpuResourceDomain;
use crate::{
    CpuDomainExecutorError, CpuDomainId, CpuInnerParallelism, CpuPlacementGuarantee, CpuSet,
};

/// Operand named by a provider capability reason.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuOperand;
/// assert_eq!(CpuOperand::Lhs, CpuOperand::Lhs);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuOperand {
    /// Left input operand.
    Lhs,
    /// Right input operand.
    Rhs,
    /// Writable output operand.
    Output,
}

/// Allocation-free reason that a provider cannot execute a validated request.
///
/// An unsupported outcome must be reported before the provider mutates the
/// output.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuProviderUnsupported;
/// assert_eq!(
///     CpuProviderUnsupported::RuntimeUnavailable,
///     CpuProviderUnsupported::RuntimeUnavailable,
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CpuProviderUnsupported {
    /// The provider does not implement the scalar dtype.
    DType(DType),
    /// The provider does not implement the input ranks.
    Rank {
        /// Left input rank.
        lhs: usize,
        /// Right input rank.
        rhs: usize,
    },
    /// The provider does not implement an operand layout.
    Layout(CpuOperand),
    /// The provider cannot implement the requested conjugation.
    Conjugation,
    /// The provider cannot implement the requested alpha/beta update.
    Accumulation,
    /// The provider cannot implement a strided batch.
    StridedBatch,
    /// The provider cannot implement grouped GEMM.
    Grouped,
    /// The optional provider runtime is not available in this process.
    RuntimeUnavailable,
}

/// Result of attempting a validated request through one provider slot.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::{CpuProviderOutcome, CpuProviderUnsupported};
/// let outcome = CpuProviderOutcome::Unsupported(
///     CpuProviderUnsupported::RuntimeUnavailable,
/// );
/// assert!(matches!(outcome, CpuProviderOutcome::Unsupported(_)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum CpuProviderOutcome {
    /// The provider fully executed the request.
    Executed,
    /// The provider did not mutate the output and resolution may continue.
    Unsupported(CpuProviderUnsupported),
}

/// Parallel scheduling mode selected for one CPU operation.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::ParallelMode;
/// assert_ne!(ParallelMode::Sequential, ParallelMode::Outer);
/// assert_ne!(ParallelMode::Outer, ParallelMode::Inner);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelMode {
    /// Neither the engine nor the provider may fan out this operation.
    Sequential,
    /// The engine owns outer fan-out and delegates Sequential child contexts.
    /// Providers do not receive an Outer context.
    Outer,
    /// One provider kernel may use the selected executor's inner region.
    Inner,
}

/// Borrowed execution policy for an already-entered CPU operation.
///
/// The context exposes immutable domain facts while keeping the resource lease,
/// executor object, and the checked executor-entry boundary private. Providers
/// cannot install or submit work through this value.
///
/// # Examples
///
/// Providers inspect this value inside a trait method:
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
            .field("placement_guarantee", &self.placement_guarantee())
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

    /// Return the stable identity of the selected CPU resource domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuExecutionContext;
    /// # fn inspect(context: &CpuExecutionContext<'_>) {
    /// let _domain_id = context.domain_id();
    /// # }
    /// ```
    pub fn domain_id(&self) -> CpuDomainId {
        self.domain.id()
    }

    /// Return the selected domain's declared logical CPU set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuExecutionContext;
    /// # fn inspect(context: &CpuExecutionContext<'_>) {
    /// assert!(!context.cpus().is_empty());
    /// # }
    /// ```
    pub fn cpus(&self) -> &CpuSet {
        self.domain.cpus()
    }

    /// Return the non-zero maximum participating-thread budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuExecutionContext;
    /// # fn inspect(context: &CpuExecutionContext<'_>) {
    /// assert!(context.thread_budget().get() >= 1);
    /// # }
    /// ```
    pub fn thread_budget(&self) -> NonZeroUsize {
        self.domain.thread_budget()
    }

    /// Return the strength of the selected domain's placement guarantee.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuExecutionContext;
    /// # fn inspect(context: &CpuExecutionContext<'_>) {
    /// let _guarantee = context.placement_guarantee();
    /// # }
    /// ```
    pub fn placement_guarantee(&self) -> CpuPlacementGuarantee {
        self.domain.placement_guarantee()
    }

    /// Return the engine-selected scheduling mode for this entered provider call.
    ///
    /// Provider calls observe Sequential or Inner. Outer scheduling creates a
    /// separate Sequential context inside every submitted child.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuExecutionContext, ParallelMode};
    /// # fn inspect(context: &CpuExecutionContext<'_>) {
    /// assert!(matches!(
    ///     context.parallel_mode(),
    ///     ParallelMode::Sequential | ParallelMode::Inner
    /// ));
    /// # }
    /// ```
    pub fn parallel_mode(&self) -> ParallelMode {
        self.parallel_mode
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
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{StridedSliceSpec, TensorRead, TensorView, TypedTensor};
    ///
    /// let mut backend = CpuBackend::with_threads(1)?;
    /// let input = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// backend.with_linalg_pool(|context, buffers| {
    ///     let view = input
    ///         .as_view()
    ///         .try_slice(&[StridedSliceSpec::reverse()])?;
    ///     context.with_materialized_tensor_read(
    ///         buffers,
    ///         "example",
    ///         TensorRead::from_view(TensorView::F64(view)),
    ///         |materialized, _| {
    ///             assert_eq!(materialized.as_slice::<f64>().unwrap(), &[2.0, 1.0]);
    ///             Ok(())
    ///         },
    ///     )
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
                reclaim_tensor(buffers, materialized);
                result
            }
        }
    }

    /// Reshape a compact tensor while retaining the current execution proof.
    ///
    /// This metadata-only helper does not enter an executor or borrow another
    /// scratch pool.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] when the input and output
    /// shapes have different element counts or the requested layout is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut backend = CpuBackend::with_threads(1)?;
    /// let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// backend.with_linalg_pool(|context, _| {
    ///     let output = context.reshape_tensor(&input, &[2, 1])?;
    ///     assert_eq!(output.shape(), &[2, 1]);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[doc(hidden)]
    pub fn reshape_tensor(
        &self,
        input: &Tensor,
        shape: &[usize],
    ) -> tenferro_tensor::Result<Tensor> {
        crate::structural::reshape(input, shape)
    }

    /// Return the faer policy selected by this operation context.
    ///
    /// This hidden public method is the owner-scoped extension contract used by
    /// operation-family crates such as `tenferro-linalg`. Keeping the mapping
    /// here prevents sibling crates from deriving a second CPU threading
    /// policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuExecutionContext;
    /// # fn inspect(context: &CpuExecutionContext<'_>) {
    /// let _policy = context.faer_parallelism();
    /// # }
    /// ```
    #[cfg(feature = "cpu-faer")]
    #[doc(hidden)]
    pub fn faer_parallelism(self) -> faer::Par {
        match (
            self.parallel_mode,
            self.domain.executor_capabilities().inner_parallelism,
        ) {
            (ParallelMode::Inner, CpuInnerParallelism::Rayon) if self.thread_budget().get() > 1 => {
                faer::Par::rayon(self.thread_budget().get())
            }
            _ => faer::Par::Seq,
        }
    }

    pub(crate) fn strided_exec_context(&self) -> strided_kernel::ExecContext {
        match (
            self.parallel_mode,
            self.domain.executor_capabilities().inner_parallelism,
        ) {
            (ParallelMode::Inner, CpuInnerParallelism::Rayon) if self.thread_budget().get() > 1 => {
                // This is only an operation-local thread limit for strided's
                // replay policy. The Rayon pool itself is the already-entered
                // CpuContext pool installed by `with_native_parallelism`.
                match strided_kernel::ExecContext::max_threads(self.thread_budget().get()) {
                    Ok(context) => context,
                    // INVARIANT: CpuExecutionContext stores a NonZeroUsize
                    // thread budget, and this branch passes that positive value.
                    Err(_) => unreachable!("CpuExecutionContext has a non-zero thread budget"),
                }
            }
            _ => strided_kernel::ExecContext::serial(),
        }
    }

    pub(crate) fn with_native_parallelism<R>(&self, operation: impl FnOnce() -> R) -> R {
        let policy = match (
            self.parallel_mode,
            self.domain.executor_capabilities().inner_parallelism,
        ) {
            (ParallelMode::Inner, CpuInnerParallelism::Rayon) if self.thread_budget().get() > 1 => {
                strided_kernel::ExecutionPolicy::Rayon {
                    max_threads: self.thread_budget(),
                }
            }
            _ => strided_kernel::ExecutionPolicy::Sequential,
        };
        strided_kernel::with_execution_policy(policy, operation)
    }
}

fn reclaim_tensor(buffers: &mut BufferPool, tensor: Tensor) {
    match tensor {
        Tensor::F32(tensor) => crate::backend::reclaim_typed(buffers, tensor),
        Tensor::F64(tensor) => crate::backend::reclaim_typed(buffers, tensor),
        Tensor::I32(tensor) => crate::backend::reclaim_typed(buffers, tensor),
        Tensor::I64(tensor) => crate::backend::reclaim_typed(buffers, tensor),
        Tensor::Bool(tensor) => crate::backend::reclaim_typed(buffers, tensor),
        Tensor::C32(tensor) => crate::backend::reclaim_typed(buffers, tensor),
        Tensor::C64(tensor) => crate::backend::reclaim_typed(buffers, tensor),
    }
}

/// Crate-private unentered capability for one CPU operation.
///
/// This is the only type that owns the resource permit and may cross the
/// selected domain executor boundary. A [`CpuExecutionContext`] is constructed
/// only inside an installed job or an outer child job.
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

    pub(crate) fn enter<R: Send>(
        self,
        parallel_mode: ParallelMode,
        operation: impl FnOnce(&CpuExecutionContext<'_>) -> R + Send,
    ) -> Result<R, CpuDomainExecutorError> {
        if parallel_mode == ParallelMode::Outer {
            return Err(CpuDomainExecutorError::Scheduling {
                message: "CPU executor install requires Sequential or Inner mode, got Outer"
                    .to_owned(),
            });
        }
        let owner = self.permit.owner();
        with_execution_owner(owner, || {
            install_scoped(self.domain.executor().as_ref(), || {
                with_execution_owner(owner, || {
                    let context = CpuExecutionContext::entered(self.domain, parallel_mode);
                    operation(&context)
                })
            })
        })
    }

    pub(crate) fn enter_or_reuse<R: Send>(
        self,
        entered: Option<&CpuExecutionContext<'_>>,
        parallel_mode: ParallelMode,
        operation: impl FnOnce(&CpuExecutionContext<'_>) -> R + Send,
    ) -> Result<R, CpuDomainExecutorError> {
        let Some(entered) = entered else {
            return self.enter(parallel_mode, operation);
        };
        if parallel_mode == ParallelMode::Outer {
            return Err(CpuDomainExecutorError::Scheduling {
                message: "entered CPU session requires Sequential or Inner mode, got Outer"
                    .to_owned(),
            });
        }
        if entered.domain_id() != self.domain.id() {
            return Err(CpuDomainExecutorError::Scheduling {
                message: format!(
                    "entered CPU session domain {:?} does not match operation domain {:?}",
                    entered.domain_id(),
                    self.domain.id()
                ),
            });
        }
        let owner = self.permit.owner();
        Ok(with_execution_owner(owner, || {
            let context = CpuExecutionContext::entered(self.domain, parallel_mode);
            operation(&context)
        }))
    }

    pub(crate) fn supports_infallible_session_entry(self) -> bool {
        self.domain.ownership() == crate::CpuDomainOwnership::Managed
    }

    pub(crate) fn enter_managed_session<R: Send>(
        self,
        operation: impl FnOnce(CpuExecutionContext<'a>) -> R + Send,
    ) -> R {
        assert!(
            self.supports_infallible_session_entry(),
            "managed session entry requires a Tenferro-managed CPU domain"
        );
        let mode = self.preferred_engine_mode();
        self.enter(mode, |_| {
            operation(CpuExecutionContext::entered(self.domain, mode))
        })
        .unwrap_or_else(|error| {
            panic!("Tenferro-managed CPU executor violated synchronous install contract: {error}")
        })
    }

    pub(crate) fn submit_outer(
        self,
        len: usize,
        operation: impl Fn(usize, &CpuExecutionContext<'_>) -> Result<(), CpuDomainExecutorError> + Sync,
    ) -> Result<(), CpuDomainExecutorError> {
        if !self.supports_outer() {
            return Err(CpuDomainExecutorError::Scheduling {
                message: format!(
                    "CPU domain {:?} does not support Outer mode",
                    self.domain.id()
                ),
            });
        }
        let owner = self.permit.owner();
        let lane_count = len.min(self.domain.thread_budget().get());
        let jobs = indexed_jobs(lane_count, |lane| {
            // INVARIANT: valid lanes partition `0..len` by residue modulo the
            // nonzero `lane_count`, so every logical job runs exactly once
            // while the executor can schedule at most the domain budget.
            let mut index = lane;
            while index < len {
                with_execution_owner(owner, || {
                    let context =
                        CpuExecutionContext::entered(self.domain, ParallelMode::Sequential);
                    operation(index, &context)
                })?;
                let Some(next) = index.checked_add(lane_count) else {
                    break;
                };
                index = next;
            }
            Ok(())
        });
        with_execution_owner(owner, || self.domain.executor().submit(&jobs))?;
        if let Some(index) = jobs.invalid_index_attempt() {
            return Err(CpuDomainExecutorError::Scheduling {
                message: format!(
                    "executor requested scoped CPU lane index {index}, but the submission has {lane_count} lanes for {len} logical jobs"
                ),
            });
        }
        Ok(())
    }

    pub(crate) fn preferred_engine_mode(self) -> ParallelMode {
        if self.domain.thread_budget().get() > 1
            && self.domain.executor_capabilities().inner_parallelism == CpuInnerParallelism::Rayon
        {
            ParallelMode::Inner
        } else {
            ParallelMode::Sequential
        }
    }

    pub(crate) fn preferred_provider_mode(
        self,
        accepts: impl Fn(ParallelMode) -> bool,
    ) -> Result<ParallelMode, crate::CpuProviderDomainError> {
        if self.domain.thread_budget().get() == 1 {
            return if accepts(ParallelMode::Sequential) {
                Ok(ParallelMode::Sequential)
            } else {
                Err(crate::CpuProviderDomainError::ParallelModeNotSupported {
                    mode: ParallelMode::Sequential,
                })
            };
        }
        if accepts(ParallelMode::Inner) {
            return Ok(ParallelMode::Inner);
        }
        if accepts(ParallelMode::Sequential) {
            return Ok(ParallelMode::Sequential);
        }
        Err(crate::CpuProviderDomainError::ParallelModeNotSupported {
            mode: ParallelMode::Inner,
        })
    }

    pub(crate) fn preferred_linalg_mode(self, kind: CpuBackendKind) -> ParallelMode {
        if self.domain.thread_budget().get() == 1 {
            return ParallelMode::Sequential;
        }
        match kind {
            CpuBackendKind::Faer => self.preferred_engine_mode(),
            // The linalg operation-family provider has not yet moved onto the
            // provider capability traits. Preserve its existing ownership
            // policy until that trait boundary is introduced.
            CpuBackendKind::Blas => ParallelMode::Inner,
        }
    }

    pub(crate) fn provider_default_compatibility_mode(self) -> ParallelMode {
        if self.domain.thread_budget().get() == 1 {
            ParallelMode::Sequential
        } else {
            ParallelMode::Inner
        }
    }

    pub(crate) fn thread_budget(self) -> NonZeroUsize {
        self.domain.thread_budget()
    }

    pub(crate) fn supports_outer(self) -> bool {
        self.domain.thread_budget().get() > 1
            && self.domain.executor_capabilities().outer_parallelism
    }
}

/// Checked element offset and strides for a batched matrix operand.
///
/// # Examples
///
/// Providers receive this descriptor from a validated request:
///
/// ```
/// use tenferro_cpu::provider::CpuGemmRequest;
/// # fn inspect(request: &CpuGemmRequest<'_, '_, '_>) {
/// assert!(request.lhs_layout().row_stride() != 0 || request.rows() <= 1);
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuBatchedMatrixLayout {
    offset: isize,
    row_stride: isize,
    column_stride: isize,
    batch_stride: isize,
}

impl CpuBatchedMatrixLayout {
    #[allow(dead_code)]
    pub(crate) fn new(
        offset: isize,
        row_stride: isize,
        column_stride: isize,
        batch_stride: isize,
    ) -> Self {
        Self {
            offset,
            row_stride,
            column_stride,
            batch_stride,
        }
    }

    /// Return the checked base element offset.
    pub fn offset(self) -> isize {
        self.offset
    }

    /// Return the row element stride.
    pub fn row_stride(self) -> isize {
        self.row_stride
    }

    /// Return the column element stride.
    pub fn column_stride(self) -> isize {
        self.column_stride
    }

    /// Return the batch element stride.
    pub fn batch_stride(self) -> isize {
        self.batch_stride
    }
}

/// Validated borrowed GEMM request.
///
/// A batch count of one is a single GEMM. A larger batch count is a strided
/// batched GEMM.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuGemmRequest;
/// # fn inspect(request: &CpuGemmRequest<'_, '_, '_>) {
/// assert!(request.batch_count() >= 1);
/// # }
/// ```
#[derive(Debug)]
pub struct CpuGemmRequest<'request, 'input, 'output> {
    lhs: &'request TensorRead<'input>,
    rhs: &'request TensorRead<'input>,
    output: &'request mut TensorWrite<'output>,
    rows: usize,
    columns: usize,
    contracted: usize,
    batch_count: usize,
    lhs_layout: CpuBatchedMatrixLayout,
    rhs_layout: CpuBatchedMatrixLayout,
    output_layout: CpuBatchedMatrixLayout,
    accumulation: DotGeneralAccumulation,
}

pub(crate) struct CpuGemmRequestParts<'request, 'input, 'output> {
    pub(crate) lhs: &'request TensorRead<'input>,
    pub(crate) rhs: &'request TensorRead<'input>,
    pub(crate) output: &'request mut TensorWrite<'output>,
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) contracted: usize,
    pub(crate) batch_count: usize,
    pub(crate) lhs_layout: CpuBatchedMatrixLayout,
    pub(crate) rhs_layout: CpuBatchedMatrixLayout,
    pub(crate) output_layout: CpuBatchedMatrixLayout,
    pub(crate) accumulation: DotGeneralAccumulation,
}

impl<'request, 'input, 'output> CpuGemmRequest<'request, 'input, 'output> {
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn new(
        lhs: &'request TensorRead<'input>,
        rhs: &'request TensorRead<'input>,
        output: &'request mut TensorWrite<'output>,
        rows: usize,
        columns: usize,
        contracted: usize,
        batch_count: usize,
        lhs_layout: CpuBatchedMatrixLayout,
        rhs_layout: CpuBatchedMatrixLayout,
        output_layout: CpuBatchedMatrixLayout,
        accumulation: DotGeneralAccumulation,
    ) -> Self {
        Self {
            lhs,
            rhs,
            output,
            rows,
            columns,
            contracted,
            batch_count,
            lhs_layout,
            rhs_layout,
            output_layout,
            accumulation,
        }
    }

    /// Return the borrowed left input.
    pub fn lhs(&self) -> &TensorRead<'input> {
        self.lhs
    }

    /// Return the borrowed right input.
    pub fn rhs(&self) -> &TensorRead<'input> {
        self.rhs
    }

    /// Reborrow the writable output for the duration of the current call.
    pub fn output(&mut self) -> &mut TensorWrite<'output> {
        self.output
    }

    /// Return the number of output rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return the number of output columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Return the contracted dimension.
    pub fn contracted(&self) -> usize {
        self.contracted
    }

    /// Return the number of matrices in this strided batch.
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Return the left input matrix layout.
    pub fn lhs_layout(&self) -> CpuBatchedMatrixLayout {
        self.lhs_layout
    }

    /// Return the right input matrix layout.
    pub fn rhs_layout(&self) -> CpuBatchedMatrixLayout {
        self.rhs_layout
    }

    /// Return the output matrix layout.
    pub fn output_layout(&self) -> CpuBatchedMatrixLayout {
        self.output_layout
    }

    /// Return conjugation and alpha/beta update semantics.
    pub fn accumulation(&self) -> DotGeneralAccumulation {
        self.accumulation
    }

    pub(crate) fn into_parts(self) -> CpuGemmRequestParts<'request, 'input, 'output> {
        CpuGemmRequestParts {
            lhs: self.lhs,
            rhs: self.rhs,
            output: self.output,
            rows: self.rows,
            columns: self.columns,
            contracted: self.contracted,
            batch_count: self.batch_count,
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
            output_layout: self.output_layout,
            accumulation: self.accumulation,
        }
    }
}

/// Validated borrowed grouped-GEMM request.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuGroupedGemmRequest;
/// # fn inspect(request: &CpuGroupedGemmRequest<'_, '_, '_>) {
/// assert_eq!(request.jobs().len(), request.jobs().iter().count());
/// # }
/// ```
#[derive(Debug)]
pub struct CpuGroupedGemmRequest<'request, 'input, 'output> {
    lhs: &'request TensorRead<'input>,
    rhs: &'request TensorRead<'input>,
    output: &'request mut TensorWrite<'output>,
    jobs: &'request [GroupedGemmJob],
    accumulation: DotGeneralAccumulation,
}

impl<'request, 'input, 'output> CpuGroupedGemmRequest<'request, 'input, 'output> {
    #[allow(dead_code)]
    pub(crate) fn new(
        lhs: &'request TensorRead<'input>,
        rhs: &'request TensorRead<'input>,
        output: &'request mut TensorWrite<'output>,
        jobs: &'request [GroupedGemmJob],
        accumulation: DotGeneralAccumulation,
    ) -> Self {
        Self {
            lhs,
            rhs,
            output,
            jobs,
            accumulation,
        }
    }

    /// Return the borrowed left input.
    pub fn lhs(&self) -> &TensorRead<'input> {
        self.lhs
    }

    /// Return the borrowed right input.
    pub fn rhs(&self) -> &TensorRead<'input> {
        self.rhs
    }

    /// Reborrow the writable output.
    pub fn output(&mut self) -> &mut TensorWrite<'output> {
        self.output
    }

    /// Return ordered, pairwise-disjoint validated jobs.
    pub fn jobs(&self) -> &[GroupedGemmJob] {
        self.jobs
    }

    /// Return shared conjugation and alpha/beta semantics.
    pub fn accumulation(&self) -> DotGeneralAccumulation {
        self.accumulation
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        &'request TensorRead<'input>,
        &'request TensorRead<'input>,
        &'request mut TensorWrite<'output>,
        &'request [GroupedGemmJob],
        DotGeneralAccumulation,
    ) {
        (
            self.lhs,
            self.rhs,
            self.output,
            self.jobs,
            self.accumulation,
        )
    }
}

/// Engine-requested layout materialization.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuLayoutTransformIntent;
/// assert_eq!(
///     CpuLayoutTransformIntent::CanonicalColumnMajor,
///     CpuLayoutTransformIntent::CanonicalColumnMajor,
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuLayoutTransformIntent {
    /// Materialize a compact canonical column-major tensor.
    CanonicalColumnMajor,
}

/// Validated borrowed layout-transform request.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::{CpuLayoutTransformIntent, CpuLayoutTransformRequest};
/// # fn inspect(request: &CpuLayoutTransformRequest<'_, '_, '_>) {
/// assert_eq!(request.intent(), CpuLayoutTransformIntent::CanonicalColumnMajor);
/// # }
/// ```
#[derive(Debug)]
pub struct CpuLayoutTransformRequest<'request, 'input, 'output> {
    input: &'request TensorRead<'input>,
    output: &'request mut TensorWrite<'output>,
    intent: CpuLayoutTransformIntent,
    conjugate: bool,
}

impl<'request, 'input, 'output> CpuLayoutTransformRequest<'request, 'input, 'output> {
    #[allow(dead_code)]
    pub(crate) fn new(
        input: &'request TensorRead<'input>,
        output: &'request mut TensorWrite<'output>,
        intent: CpuLayoutTransformIntent,
        conjugate: bool,
    ) -> Self {
        Self {
            input,
            output,
            intent,
            conjugate,
        }
    }

    /// Return the borrowed input.
    pub fn input(&self) -> &TensorRead<'input> {
        self.input
    }

    /// Reborrow the writable output.
    pub fn output(&mut self) -> &mut TensorWrite<'output> {
        self.output
    }

    /// Return the requested materialization intent.
    pub fn intent(&self) -> CpuLayoutTransformIntent {
        self.intent
    }

    /// Return whether materialization must conjugate each input element.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::provider::CpuLayoutTransformRequest;
    /// # fn inspect(request: &CpuLayoutTransformRequest<'_, '_, '_>) {
    /// let _must_conjugate = request.conjugate();
    /// # }
    /// ```
    pub fn conjugate(&self) -> bool {
        self.conjugate
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        &'request TensorRead<'input>,
        &'request mut TensorWrite<'output>,
        CpuLayoutTransformIntent,
        bool,
    ) {
        (self.input, self.output, self.intent, self.conjugate)
    }
}

/// Validated ordered contraction-role groups.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuDotGeneralRequest;
/// # fn inspect(request: &CpuDotGeneralRequest<'_, '_, '_>) {
/// let _pairs = request.axes().contracting_pairs().count();
/// # }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CpuContractionAxes<'a> {
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_contracting: &'a [usize],
    rhs_contracting: &'a [usize],
    lhs_batch: &'a [usize],
    rhs_batch: &'a [usize],
    lhs_role_mask: Option<u64>,
    rhs_role_mask: Option<u64>,
}

impl<'a> CpuContractionAxes<'a> {
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn new(
        lhs_rank: usize,
        rhs_rank: usize,
        lhs_contracting: &'a [usize],
        rhs_contracting: &'a [usize],
        lhs_batch: &'a [usize],
        rhs_batch: &'a [usize],
        lhs_role_mask: Option<u64>,
        rhs_role_mask: Option<u64>,
    ) -> Self {
        Self {
            lhs_rank,
            rhs_rank,
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
            lhs_role_mask,
            rhs_role_mask,
        }
    }

    /// Return ordered contracting-axis pairs.
    pub fn contracting_pairs(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        self.lhs_contracting
            .iter()
            .copied()
            .zip(self.rhs_contracting.iter().copied())
    }

    /// Return ordered batch-axis pairs.
    pub fn batch_pairs(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        self.lhs_batch
            .iter()
            .copied()
            .zip(self.rhs_batch.iter().copied())
    }

    /// Return left axes that are neither contracting nor batch axes.
    pub fn lhs_free_axes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.lhs_rank).filter(move |&axis| !self.lhs_axis_has_role(axis))
    }

    /// Return right axes that are neither contracting nor batch axes.
    pub fn rhs_free_axes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.rhs_rank).filter(move |&axis| !self.rhs_axis_has_role(axis))
    }

    fn lhs_axis_has_role(&self, axis: usize) -> bool {
        self.lhs_role_mask.map_or_else(
            || self.lhs_contracting.contains(&axis) || self.lhs_batch.contains(&axis),
            |mask| mask & (1_u64 << axis) != 0,
        )
    }

    fn rhs_axis_has_role(&self, axis: usize) -> bool {
        self.rhs_role_mask.map_or_else(
            || self.rhs_contracting.contains(&axis) || self.rhs_batch.contains(&axis),
            |mask| mask & (1_u64 << axis) != 0,
        )
    }
}

/// Validated borrowed semantic binary `dot_general` request.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuDotGeneralRequest;
/// # fn inspect(request: &CpuDotGeneralRequest<'_, '_, '_>) {
/// let _ = request.accumulation();
/// # }
/// ```
#[derive(Debug)]
pub struct CpuDotGeneralRequest<'request, 'input, 'output> {
    lhs: &'request TensorRead<'input>,
    rhs: &'request TensorRead<'input>,
    output: &'request mut TensorWrite<'output>,
    axes: CpuContractionAxes<'request>,
    accumulation: DotGeneralAccumulation,
}

impl<'request, 'input, 'output> CpuDotGeneralRequest<'request, 'input, 'output> {
    #[allow(dead_code)]
    pub(crate) fn new(
        lhs: &'request TensorRead<'input>,
        rhs: &'request TensorRead<'input>,
        output: &'request mut TensorWrite<'output>,
        axes: CpuContractionAxes<'request>,
        accumulation: DotGeneralAccumulation,
    ) -> Self {
        Self {
            lhs,
            rhs,
            output,
            axes,
            accumulation,
        }
    }

    /// Return the borrowed left input.
    pub fn lhs(&self) -> &TensorRead<'input> {
        self.lhs
    }

    /// Return the borrowed right input.
    pub fn rhs(&self) -> &TensorRead<'input> {
        self.rhs
    }

    /// Reborrow the writable output.
    pub fn output(&mut self) -> &mut TensorWrite<'output> {
        self.output
    }

    /// Return the validated ordered axis groups.
    pub fn axes(&self) -> &CpuContractionAxes<'request> {
        &self.axes
    }

    /// Return conjugation and alpha/beta update semantics.
    pub fn accumulation(&self) -> DotGeneralAccumulation {
        self.accumulation
    }

    /// Consume the request and return the validated operand borrows.
    ///
    /// External general-contraction providers use this when they need
    /// simultaneous immutable access to both inputs and mutable access to the
    /// output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::provider::CpuDotGeneralRequest;
    ///
    /// fn consume_request(request: CpuDotGeneralRequest<'_, '_, '_>) {
    ///     let (_lhs, _rhs, _output, _axes, _accumulation) = request.into_parts();
    /// }
    /// ```
    pub fn into_parts(
        self,
    ) -> (
        &'request TensorRead<'input>,
        &'request TensorRead<'input>,
        &'request mut TensorWrite<'output>,
        CpuContractionAxes<'request>,
        DotGeneralAccumulation,
    ) {
        (
            self.lhs,
            self.rhs,
            self.output,
            self.axes,
            self.accumulation,
        )
    }
}

/// Provider for validated GEMM-family requests.
///
/// # Examples
///
/// Trait objects are supported directly:
///
/// ```
/// use tenferro_cpu::provider::CpuGemmProvider;
/// # fn accepts_provider(_: &dyn CpuGemmProvider) {}
/// ```
pub trait CpuGemmProvider: fmt::Debug + Send + Sync + 'static {
    /// Return immutable count, placement, and fan-out capabilities.
    ///
    /// This declaration must describe controls actually applied and restored
    /// by the provider adapter around each call. Merely discovering a runtime
    /// symbol is insufficient. A provider bundle samples this method exactly
    /// once during construction and keeps that snapshot for its lifetime; the
    /// returned contract must therefore remain valid for the provider object.
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities;

    /// Execute one validated GEMM.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendSource`] or
    /// [`tenferro_tensor::Error::BackendFailure`] when the provider runtime
    /// fails. A detected inconsistency in engine-attested request metadata is
    /// returned as [`tenferro_tensor::Error::Validation`]. Unsupported
    /// capabilities use [`CpuProviderOutcome::Unsupported`] instead.
    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;

    /// Execute one validated strided-batched GEMM.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendSource`] or
    /// [`tenferro_tensor::Error::BackendFailure`] when the provider runtime
    /// fails. A detected inconsistency in engine-attested request metadata is
    /// returned as [`tenferro_tensor::Error::Validation`]. Unsupported
    /// capabilities use [`CpuProviderOutcome::Unsupported`] instead.
    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;

    /// Execute one validated grouped GEMM.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendSource`] or
    /// [`tenferro_tensor::Error::BackendFailure`] when the provider runtime
    /// fails. A detected inconsistency in engine-attested request metadata is
    /// returned as [`tenferro_tensor::Error::Validation`]. Unsupported
    /// capabilities use [`CpuProviderOutcome::Unsupported`] instead.
    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}

/// Provider for engine-owned tensor materialization.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuLayoutTransformProvider;
/// # fn accepts_provider(_: &dyn CpuLayoutTransformProvider) {}
/// ```
pub trait CpuLayoutTransformProvider: fmt::Debug + Send + Sync + 'static {
    /// Return immutable count, placement, and fan-out capabilities.
    ///
    /// A provider bundle samples this method exactly once during construction
    /// and uses the stored descriptor for all validation and dispatch.
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities;

    /// Materialize one validated input into a preallocated output.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendSource`] or
    /// [`tenferro_tensor::Error::BackendFailure`] when execution fails. A
    /// detected inconsistency in engine-attested layout or range metadata is
    /// returned as [`tenferro_tensor::Error::Validation`]. Unsupported layouts
    /// use [`CpuProviderOutcome::Unsupported`] instead.
    fn materialize(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}

/// Provider for complete validated binary `dot_general` requests.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuGeneralContractionProvider;
/// # fn accepts_provider(_: &dyn CpuGeneralContractionProvider) {}
/// ```
pub trait CpuGeneralContractionProvider: fmt::Debug + Send + Sync + 'static {
    /// Return immutable count, placement, and fan-out capabilities.
    ///
    /// A provider bundle samples this method exactly once during construction
    /// and uses the stored descriptor for all validation and dispatch.
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities;

    /// Execute one complete semantic contraction.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendSource`] or
    /// [`tenferro_tensor::Error::BackendFailure`] when the provider runtime
    /// fails. A detected inconsistency in engine-attested axes or range
    /// metadata is returned as [`tenferro_tensor::Error::Validation`].
    /// Unsupported contractions use [`CpuProviderOutcome::Unsupported`]
    /// instead.
    fn dot_general(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}

/// Built-in faer GEMM provider.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::{CpuGemmProvider, FaerGemmProvider};
/// let provider: &dyn CpuGemmProvider = &FaerGemmProvider;
/// let _ = provider;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct FaerGemmProvider;

impl CpuGemmProvider for FaerGemmProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        engine_worker_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        #[cfg(feature = "cpu-faer")]
        {
            crate::gemm::execute_faer_gemm_request(context, request)
        }
        #[cfg(not(feature = "cpu-faer"))]
        {
            let _ = (context, request);
            Ok(CpuProviderOutcome::Unsupported(
                CpuProviderUnsupported::RuntimeUnavailable,
            ))
        }
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        #[cfg(feature = "cpu-faer")]
        {
            crate::gemm::execute_faer_grouped_request(context, request)
        }
        #[cfg(not(feature = "cpu-faer"))]
        {
            let _ = (context, request);
            Ok(CpuProviderOutcome::Unsupported(
                CpuProviderUnsupported::RuntimeUnavailable,
            ))
        }
    }
}

/// Built-in BLAS GEMM provider.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::{BlasGemmProvider, CpuGemmProvider};
/// let provider: &dyn CpuGemmProvider = &BlasGemmProvider;
/// let _ = provider;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct BlasGemmProvider;

impl CpuGemmProvider for BlasGemmProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        #[cfg(feature = "cpu-blas")]
        {
            builtin_blas_execution_capabilities()
        }
        #[cfg(not(feature = "cpu-blas"))]
        {
            // This build returns RuntimeUnavailable without invoking BLAS.
            serial_capabilities()
        }
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        #[cfg(feature = "cpu-blas")]
        {
            crate::gemm::execute_blas_gemm_request(context, request)
        }
        #[cfg(not(feature = "cpu-blas"))]
        {
            let _ = (context, request);
            Ok(CpuProviderOutcome::Unsupported(
                CpuProviderUnsupported::RuntimeUnavailable,
            ))
        }
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        #[cfg(feature = "cpu-blas")]
        {
            crate::gemm::execute_blas_grouped_request(context, request)
        }
        #[cfg(not(feature = "cpu-blas"))]
        {
            let _ = (context, request);
            Ok(CpuProviderOutcome::Unsupported(
                CpuProviderUnsupported::RuntimeUnavailable,
            ))
        }
    }
}

/// Built-in strided layout materialization provider.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::{CpuLayoutTransformProvider, StridedLayoutTransformProvider};
/// let provider: &dyn CpuLayoutTransformProvider = &StridedLayoutTransformProvider;
/// let _ = provider;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct StridedLayoutTransformProvider;

impl CpuLayoutTransformProvider for StridedLayoutTransformProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        engine_worker_capabilities()
    }

    fn materialize(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        context.with_native_parallelism(|| materialize_strided_layout(request))
    }
}

fn materialize_strided_layout(
    request: CpuLayoutTransformRequest<'_, '_, '_>,
) -> tenferro_tensor::Result<CpuProviderOutcome> {
    let (input, output, _intent, conjugate) = request.into_parts();
    if conjugate {
        macro_rules! dispatch_conjugated {
            ($owned:ident, $view:ident) => {
                match (input, &mut *output) {
                    (
                        TensorRead::Tensor(Tensor::$owned(input)),
                        TensorWrite::Tensor(Tensor::$owned(output)),
                    ) => {
                        let input = input.as_view();
                        let mut output = output.as_view_mut();
                        crate::structural::typed_conjugate_view_into(
                            &input,
                            &mut output,
                            "cpu layout materialization",
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::View(TensorView::$view(input)),
                        TensorWrite::Tensor(Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        crate::structural::typed_conjugate_view_into(
                            input,
                            &mut output,
                            "cpu layout materialization",
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::Tensor(Tensor::$owned(input)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        let input = input.as_view();
                        crate::structural::typed_conjugate_view_into(
                            &input,
                            output,
                            "cpu layout materialization",
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::View(TensorView::$view(input)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        crate::structural::typed_conjugate_view_into(
                            input,
                            output,
                            "cpu layout materialization",
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    _ => {}
                }
            };
        }
        dispatch_conjugated!(F32, F32);
        dispatch_conjugated!(F64, F64);
        dispatch_conjugated!(C32, C32);
        dispatch_conjugated!(C64, C64);
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(input.dtype()),
        ));
    }
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            match (input, &mut *output) {
                (
                    TensorRead::Tensor(Tensor::$owned(input)),
                    TensorWrite::Tensor(Tensor::$owned(output)),
                ) => {
                    let input = input.as_view();
                    let mut output = output.as_view_mut();
                    crate::structural::typed_copy_view_into(
                        &input,
                        &mut output,
                        "cpu layout materialization",
                    )?;
                    return Ok(CpuProviderOutcome::Executed);
                }
                (
                    TensorRead::View(TensorView::$view(input)),
                    TensorWrite::Tensor(Tensor::$owned(output)),
                ) => {
                    let mut output = output.as_view_mut();
                    crate::structural::typed_copy_view_into(
                        input,
                        &mut output,
                        "cpu layout materialization",
                    )?;
                    return Ok(CpuProviderOutcome::Executed);
                }
                (
                    TensorRead::Tensor(Tensor::$owned(input)),
                    TensorWrite::View(TensorViewMut::$view(output)),
                ) => {
                    let input = input.as_view();
                    crate::structural::typed_copy_view_into(
                        &input,
                        output,
                        "cpu layout materialization",
                    )?;
                    return Ok(CpuProviderOutcome::Executed);
                }
                (
                    TensorRead::View(TensorView::$view(input)),
                    TensorWrite::View(TensorViewMut::$view(output)),
                ) => {
                    crate::structural::typed_copy_view_into(
                        input,
                        output,
                        "cpu layout materialization",
                    )?;
                    return Ok(CpuProviderOutcome::Executed);
                }
                _ => {}
            }
        };
    }
    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(I32, I32);
    dispatch!(I64, I64);
    dispatch!(Bool, Bool);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::DType(input.dtype()),
    ))
}

pub(crate) fn builtin_gemm_provider(kind: CpuBackendKind) -> Arc<dyn CpuGemmProvider> {
    match kind {
        CpuBackendKind::Faer => Arc::new(FaerGemmProvider),
        CpuBackendKind::Blas => Arc::new(BlasGemmProvider),
    }
}

pub(crate) fn builtin_layout_provider() -> Arc<dyn CpuLayoutTransformProvider> {
    Arc::new(StridedLayoutTransformProvider)
}

#[cfg(test)]
pub(crate) mod tests;
