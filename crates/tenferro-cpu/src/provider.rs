//! Object-safe CPU contraction provider contracts.
//!
//! Providers synchronously write into engine-owned outputs. Request
//! constructors are crate-private because only the CPU engine may attest that
//! tensor metadata and reachable ranges have already been validated.

use core::fmt;
use core::num::NonZeroUsize;
use std::sync::Arc;

use tenferro_tensor::backend::GroupedGemmJob;
use tenferro_tensor::{
    DType, DotGeneralAccumulation, Tensor, TensorRead, TensorView, TensorViewMut, TensorWrite,
};

use crate::backend::CpuBackendKind;
use crate::CpuContext;

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

/// Kernel-level parallelism permitted for one provider invocation.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuKernelParallelism;
/// assert_ne!(
///     CpuKernelParallelism::Sequential,
///     CpuKernelParallelism::Inner,
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuKernelParallelism {
    /// The provider must execute this request sequentially.
    Sequential,
    /// The provider may use the engine's configured inner parallelism.
    Inner,
}

/// Read-only execution policy visible to CPU providers.
///
/// The context deliberately exposes no pool, session, permit, or installation
/// API.
///
/// # Examples
///
/// Providers inspect this value inside a trait method:
///
/// ```
/// use tenferro_cpu::provider::CpuProviderContext;
/// # fn inspect(context: &CpuProviderContext<'_>) {
/// assert!(context.thread_budget() >= 1);
/// # }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CpuProviderContext<'a> {
    context: &'a CpuContext,
    kernel_parallelism: CpuKernelParallelism,
}

impl<'a> CpuProviderContext<'a> {
    #[allow(dead_code)]
    pub(crate) fn new(context: &'a CpuContext, kernel_parallelism: CpuKernelParallelism) -> Self {
        Self {
            context,
            kernel_parallelism,
        }
    }

    /// Return the validated engine thread budget.
    pub fn thread_budget(&self) -> usize {
        self.context.num_threads()
    }

    pub(crate) fn nonzero_thread_budget(&self) -> NonZeroUsize {
        self.context.nonzero_thread_budget()
    }

    /// Return whether this invocation may use inner kernel parallelism.
    pub fn kernel_parallelism(&self) -> CpuKernelParallelism {
        self.kernel_parallelism
    }

    #[allow(dead_code)]
    pub(crate) fn cpu_context(&self) -> &CpuContext {
        self.context
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
    /// use tenferro_cpu::provider::{
    ///     CpuLayoutTransformProvider, CpuLayoutTransformRequest, CpuProviderContext,
    ///     CpuProviderOutcome, CpuProviderUnsupported,
    /// };
    ///
    /// #[derive(Debug)]
    /// struct InspectingProvider;
    ///
    /// impl CpuLayoutTransformProvider for InspectingProvider {
    ///     fn materialize(
    ///         &self,
    ///         _context: &CpuProviderContext<'_>,
    ///         request: CpuLayoutTransformRequest<'_, '_, '_>,
    ///     ) -> tenferro_tensor::Result<CpuProviderOutcome> {
    ///         let _must_conjugate = request.conjugate();
    ///         Ok(CpuProviderOutcome::Unsupported(
    ///             CpuProviderUnsupported::RuntimeUnavailable,
    ///         ))
    ///     }
    /// }
    ///
    /// let _provider: &dyn CpuLayoutTransformProvider = &InspectingProvider;
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

    #[cfg(feature = "cpu-tblis-provider")]
    pub(crate) fn into_parts(
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
    /// Execute one validated GEMM.
    fn gemm(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;

    /// Execute one validated strided-batched GEMM.
    fn strided_batched_gemm(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;

    /// Execute one validated grouped GEMM.
    fn grouped_gemm(
        &self,
        context: &CpuProviderContext<'_>,
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
    /// Materialize one validated input into a preallocated output.
    fn materialize(
        &self,
        context: &CpuProviderContext<'_>,
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
    /// Execute one complete semantic contraction.
    fn dot_general(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}

/// Built-in TBLIS general-contraction provider.
///
/// Without a TBLIS feature this provider reports
/// [`CpuProviderUnsupported::RuntimeUnavailable`] without modifying output.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::{
///     CpuGeneralContractionProvider, TblisGeneralContractionProvider,
/// };
/// let provider: &dyn CpuGeneralContractionProvider = &TblisGeneralContractionProvider;
/// let _ = provider;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct TblisGeneralContractionProvider;

impl CpuGeneralContractionProvider for TblisGeneralContractionProvider {
    fn dot_general(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        #[cfg(feature = "cpu-tblis-provider")]
        {
            crate::gemm::execute_tblis_general_request(context, request)
        }
        #[cfg(not(feature = "cpu-tblis-provider"))]
        {
            let _ = (context, request);
            Ok(CpuProviderOutcome::Unsupported(
                CpuProviderUnsupported::RuntimeUnavailable,
            ))
        }
    }
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
    fn gemm(
        &self,
        context: &CpuProviderContext<'_>,
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
        context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuProviderContext<'_>,
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
    fn gemm(
        &self,
        context: &CpuProviderContext<'_>,
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
        context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuProviderContext<'_>,
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
    fn materialize(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        with_layout_execution_policy(context, || materialize_strided_layout(request))
    }
}

fn with_layout_execution_policy<R: Send>(
    context: &CpuProviderContext<'_>,
    operation: impl FnOnce() -> R + Send,
) -> R {
    match context.kernel_parallelism() {
        CpuKernelParallelism::Sequential => strided_kernel::with_execution_policy(
            strided_kernel::ExecutionPolicy::Sequential,
            operation,
        ),
        CpuKernelParallelism::Inner => {
            let max_threads = context.nonzero_thread_budget();
            context.cpu_context().install(|| {
                strided_kernel::with_execution_policy(
                    strided_kernel::ExecutionPolicy::Rayon { max_threads },
                    operation,
                )
            })
        }
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

/// Dispatch one prevalidated 1-by-1 GEMM for the provider allocation probe.
///
/// This is test instrumentation rather than an application-facing API. It is
/// public only because Cargo integration tests link the library without
/// `cfg(test)`.
#[doc(hidden)]
pub fn __dispatch_gemm_for_allocation_probe<'input, 'output>(
    provider: &dyn CpuGemmProvider,
    context: &CpuContext,
    lhs: &TensorRead<'input>,
    rhs: &TensorRead<'input>,
    output: &mut TensorWrite<'output>,
    accumulation: DotGeneralAccumulation,
) -> tenferro_tensor::Result<CpuProviderOutcome> {
    const SHAPE: &[usize] = &[1, 1];
    if lhs.shape() != SHAPE || rhs.shape() != SHAPE || output.shape() != SHAPE {
        return Err(tenferro_tensor::Error::invalid_argument(
            "provider allocation probe",
            "shape",
            "the allocation probe accepts only 1-by-1 tensors",
        ));
    }
    if lhs.dtype() != rhs.dtype() || lhs.dtype() != output.dtype() {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            "provider allocation probe",
            lhs.dtype(),
            output.dtype(),
        ));
    }

    let provider_context = CpuProviderContext::new(context, CpuKernelParallelism::Sequential);
    let lhs_layout = CpuBatchedMatrixLayout::new(lhs.offset(), 1, 1, 1);
    let rhs_layout = CpuBatchedMatrixLayout::new(rhs.offset(), 1, 1, 1);
    let output_layout = CpuBatchedMatrixLayout::new(output.offset(), 1, 1, 1);
    let request = CpuGemmRequest::new(
        lhs,
        rhs,
        output,
        1,
        1,
        1,
        1,
        lhs_layout,
        rhs_layout,
        output_layout,
        accumulation,
    );
    provider.gemm(&provider_context, request)
}

#[cfg(test)]
mod tests;
