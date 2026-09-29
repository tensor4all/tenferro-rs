use tenferro_tensor::{
    DType, DotGeneralAccumulation, DotGeneralConfig, ShapeMismatch, Tensor, TensorRead, TensorView,
    TensorViewMut, TensorWrite, TypedTensor, ValidationError,
};

use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;
use std::any::{Any, TypeId};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::backend::CpuBackendKind;
use crate::batch_policy::strategy_unavailable;
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::provider::{
    builtin_gemm_provider, builtin_layout_provider, CpuContractionAxes, CpuDotGeneralRequest,
    CpuExecutionContext, CpuGemmProvider, CpuGeneralContractionProvider, CpuGroupedGemmRequest,
    CpuLayoutTransformIntent, CpuLayoutTransformProvider, CpuLayoutTransformRequest,
    CpuOperationEntry, CpuProviderOutcome, CpuProviderUnsupported, CpuUninitGemmProvider,
};
use crate::CpuBatchStrategy;
use crate::{
    gemm::GemmAnalysisCache, CpuDomainExecutorError, CpuDomainId, CpuProviderDomainError, Error,
    ParallelMode, PooledUninitOutput, Result,
};

const OP: &str = "dot_general";

/// Policy applied when the configured general-contraction provider reports a
/// typed capability miss.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::GeneralContractionPolicy;
/// assert_ne!(
///     GeneralContractionPolicy::Preferred,
///     GeneralContractionPolicy::Required,
/// );
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GeneralContractionPolicy {
    /// Continue to the configured layout-plus-GEMM path.
    #[default]
    Preferred,
    /// Convert a capability miss into a structured unsupported error.
    Required,
}

#[derive(Debug)]
pub(crate) struct DotGeneralRuntime {
    pub(crate) general: Option<Arc<dyn CpuGeneralContractionProvider>>,
    pub(crate) gemm: Arc<dyn CpuGemmProvider>,
    pub(crate) layout: Arc<dyn CpuLayoutTransformProvider>,
    general_capabilities: Option<crate::CpuProviderExecutionCapabilities>,
    gemm_capabilities: crate::CpuProviderExecutionCapabilities,
    layout_capabilities: crate::CpuProviderExecutionCapabilities,
    pub(crate) general_policy: GeneralContractionPolicy,
    grouped_scheduling: GroupedGemmScheduling,
    capability_policy: ProviderCapabilityPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedGemmScheduling {
    ProviderOwned,
    EngineOuter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProviderCapabilityPolicy {
    Strict,
    ProviderDefaultCompatibility,
}

const GROUPED_JOB_STATE_BITS: usize = 2;
const GROUPED_JOBS_PER_STATE_WORD: usize = usize::BITS as usize / GROUPED_JOB_STATE_BITS;
const GROUPED_INLINE_STATE_WORDS: usize = 4;
#[cfg(test)]
const GROUPED_INLINE_JOB_CAPACITY: usize = GROUPED_INLINE_STATE_WORDS * GROUPED_JOBS_PER_STATE_WORD;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
enum GroupedJobState {
    Unclaimed = 0,
    Running = 1,
    Complete = 2,
    Reserved = 3,
}

impl GroupedJobState {
    fn from_bits(bits: usize) -> Self {
        match bits {
            0 => Self::Unclaimed,
            1 => Self::Running,
            2 => Self::Complete,
            _ => Self::Reserved,
        }
    }
}

// INVARIANT: the public safe executor boundary can independently duplicate or
// omit any grouped job, so sound post-submit auditing requires O(job_count)
// state with at least UNCLAIMED/RUNNING/COMPLETE. Packing two bits per job into
// four inline AtomicUsize words covers 2 * usize::BITS jobs without allocation;
// only larger groups spill. Whole-word CAS updates preserve neighboring states.
struct PackedJobStates {
    words: SmallVec<[AtomicUsize; GROUPED_INLINE_STATE_WORDS]>,
    len: usize,
}

impl PackedJobStates {
    fn new(len: usize) -> Self {
        let word_count = len.div_ceil(GROUPED_JOBS_PER_STATE_WORD);
        let mut words = SmallVec::new();
        words.resize_with(word_count, || AtomicUsize::new(0));
        Self { words, len }
    }

    fn position(index: usize) -> (usize, usize) {
        let word = index / GROUPED_JOBS_PER_STATE_WORD;
        let shift = (index % GROUPED_JOBS_PER_STATE_WORD) * GROUPED_JOB_STATE_BITS;
        (word, shift)
    }

    fn state(&self, index: usize) -> GroupedJobState {
        let (word, shift) = Self::position(index);
        let bits = (self.words[word].load(Ordering::Acquire) >> shift) & 0b11;
        GroupedJobState::from_bits(bits)
    }

    fn try_claim(&self, index: usize) -> std::result::Result<(), GroupedJobState> {
        let (word, shift) = Self::position(index);
        let word = &self.words[word];
        let mask = 0b11usize << shift;
        let running = (GroupedJobState::Running as usize) << shift;
        let mut observed = word.load(Ordering::Acquire);
        loop {
            let state = GroupedJobState::from_bits((observed & mask) >> shift);
            if state != GroupedJobState::Unclaimed {
                return Err(state);
            }
            let updated = (observed & !mask) | running;
            match word.compare_exchange_weak(observed, updated, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Ok(()),
                Err(current) => observed = current,
            }
        }
    }

    fn complete(&self, index: usize) -> bool {
        let (word, shift) = Self::position(index);
        let word = &self.words[word];
        let mask = 0b11usize << shift;
        let complete = (GroupedJobState::Complete as usize) << shift;
        let mut observed = word.load(Ordering::Acquire);
        loop {
            if GroupedJobState::from_bits((observed & mask) >> shift) != GroupedJobState::Running {
                return false;
            }
            let updated = (observed & !mask) | complete;
            match word.compare_exchange_weak(observed, updated, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(current) => observed = current,
            }
        }
    }

    fn first_incomplete(&self) -> Option<(usize, GroupedJobState)> {
        (0..self.len)
            .map(|index| (index, self.state(index)))
            .find(|(_, state)| *state != GroupedJobState::Complete)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn word_count(&self) -> usize {
        self.words.len()
    }

    #[cfg(test)]
    fn spilled(&self) -> bool {
        self.words.spilled()
    }
}

fn standard_grouped_scheduling(kind: CpuBackendKind) -> GroupedGemmScheduling {
    match kind {
        CpuBackendKind::Faer => GroupedGemmScheduling::EngineOuter,
        CpuBackendKind::Blas => GroupedGemmScheduling::ProviderOwned,
    }
}

#[derive(Debug)]
pub(crate) struct CpuProviderBundleInner {
    pub(crate) dot_general: DotGeneralRuntime,
    /// Typed slots of operation-family crates, at most one per type.
    pub(crate) extensions: ProviderExtensions,
}

/// Provider objects installed by operation-family crates that tenferro-cpu
/// does not name (for example tenferro-linalg's kernel trait), keyed by
/// their Rust type.
#[derive(Clone, Default)]
pub(crate) struct ProviderExtensions(Vec<(TypeId, Arc<dyn Any + Send + Sync>)>);

impl ProviderExtensions {
    fn insert<E: Any + Send + Sync>(&mut self, extension: Arc<E>) {
        let id = TypeId::of::<E>();
        self.0.retain(|(key, _)| *key != id);
        self.0.push((id, extension));
    }

    fn get<E: Any + Send + Sync>(&self) -> Option<Arc<E>> {
        let id = TypeId::of::<E>();
        self.0
            .iter()
            .find(|(key, _)| *key == id)
            .and_then(|(_, extension)| Arc::clone(extension).downcast::<E>().ok())
    }
}

impl std::fmt::Debug for ProviderExtensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderExtensions")
            .field("count", &self.0.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CpuProviderDomainContract {
    CooperativeCpuSet,
    CallerManaged,
}

/// Immutable direct provider slots installed on a CPU backend.
///
/// Clones share the same slot identity and may safely share compatible
/// analysis-cache entries.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuBackendKind, CpuProviderBundle};
/// let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled()).build()?;
/// let cloned = bundle.clone();
/// assert!(bundle.shares_identity_with(&cloned));
/// # Ok::<(), tenferro_cpu::CpuProviderBundleBuildError>(())
/// ```
#[derive(Clone, Debug)]
pub struct CpuProviderBundle {
    inner: Arc<CpuProviderBundleInner>,
}

impl CpuProviderBundle {
    pub(crate) fn standard(kind: CpuBackendKind, provider_default_compatibility: bool) -> Self {
        let gemm = builtin_gemm_provider(kind);
        let layout = builtin_layout_provider();
        let gemm_capabilities = gemm.execution_capabilities();
        let layout_capabilities = layout.execution_capabilities();
        Self {
            inner: Arc::new(CpuProviderBundleInner {
                dot_general: DotGeneralRuntime {
                    general: None,
                    gemm,
                    layout,
                    general_capabilities: None,
                    gemm_capabilities,
                    layout_capabilities,
                    general_policy: GeneralContractionPolicy::Preferred,
                    grouped_scheduling: standard_grouped_scheduling(kind),
                    capability_policy: if provider_default_compatibility {
                        ProviderCapabilityPolicy::ProviderDefaultCompatibility
                    } else {
                        ProviderCapabilityPolicy::Strict
                    },
                },
                extensions: ProviderExtensions::default(),
            }),
        }
    }

    /// Start a bundle builder with the standard providers for `kind`.
    pub fn builder(kind: CpuBackendKind) -> CpuProviderBundleBuilder {
        CpuProviderBundleBuilder {
            gemm: Some(builtin_gemm_provider(kind)),
            layout: Some(builtin_layout_provider()),
            general: None,
            general_policy: GeneralContractionPolicy::Preferred,
            grouped_scheduling: standard_grouped_scheduling(kind),
            capability_policy: ProviderCapabilityPolicy::Strict,
            extensions: ProviderExtensions::default(),
        }
    }

    /// Start an empty custom builder.
    pub fn custom_builder() -> CpuProviderBundleBuilder {
        CpuProviderBundleBuilder {
            gemm: None,
            layout: None,
            general: None,
            general_policy: GeneralContractionPolicy::Preferred,
            grouped_scheduling: GroupedGemmScheduling::ProviderOwned,
            capability_policy: ProviderCapabilityPolicy::Strict,
            extensions: ProviderExtensions::default(),
        }
    }

    /// The extension of type `E` installed with
    /// [`CpuProviderBundleBuilder::extension`], if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackendKind, CpuProviderBundle};
    /// #[derive(Debug)]
    /// struct MyKernels;
    /// let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
    ///     .extension(Arc::new(MyKernels))
    ///     .build()?;
    /// assert!(bundle.extension::<MyKernels>().is_some());
    /// # Ok::<(), tenferro_cpu::CpuProviderBundleBuildError>(())
    /// ```
    pub fn extension<E: std::any::Any + Send + Sync>(&self) -> Option<Arc<E>> {
        self.inner.extensions.get::<E>()
    }

    /// Return whether two handles share one immutable provider identity.
    pub fn shares_identity_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn inner(&self) -> &Arc<CpuProviderBundleInner> {
        &self.inner
    }

    pub(crate) fn dot_general(&self) -> &DotGeneralRuntime {
        &self.inner.dot_general
    }

    pub(crate) fn validate_for_domain(
        &self,
        domain_id: CpuDomainId,
        thread_budget: std::num::NonZeroUsize,
        contract: CpuProviderDomainContract,
    ) -> std::result::Result<(), CpuProviderBundleInstallError> {
        let runtime = self.dot_general();
        let validate = |provider, capabilities| {
            let result = match contract {
                CpuProviderDomainContract::CooperativeCpuSet => {
                    crate::provider_capability::validate_provider_for_domain(
                        capabilities,
                        thread_budget,
                    )
                }
                CpuProviderDomainContract::CallerManaged => {
                    crate::provider_capability::validate_provider_for_caller_managed_domain(
                        capabilities,
                        thread_budget,
                    )
                }
            };
            result.map_err(|source| CpuProviderBundleInstallError::IncompatibleDomain {
                domain_id,
                provider,
                source,
            })
        };

        if let Some(capabilities) = runtime.general_capabilities {
            validate(CpuProviderSlot::GeneralContraction, capabilities)?;
        }
        validate(CpuProviderSlot::Gemm, runtime.gemm_capabilities)?;
        validate(
            CpuProviderSlot::LayoutTransform,
            runtime.layout_capabilities,
        )?;

        let selected_mode = if thread_budget.get() == 1 {
            ParallelMode::Sequential
        } else if runtime.accepts_dot_general_mode(ParallelMode::Inner) {
            ParallelMode::Inner
        } else {
            ParallelMode::Sequential
        };
        for (provider, capabilities) in [
            (CpuProviderSlot::Gemm, runtime.gemm_capabilities),
            (
                CpuProviderSlot::LayoutTransform,
                runtime.layout_capabilities,
            ),
        ] {
            if !capabilities.accepts_mode(selected_mode) {
                return Err(CpuProviderBundleInstallError::IncompatibleDomain {
                    domain_id,
                    provider,
                    source: CpuProviderDomainError::ParallelModeNotSupported {
                        mode: selected_mode,
                    },
                });
            }
        }
        if let Some(capabilities) = runtime.general_capabilities {
            if !capabilities.accepts_mode(selected_mode) {
                return Err(CpuProviderBundleInstallError::IncompatibleDomain {
                    domain_id,
                    provider: CpuProviderSlot::GeneralContraction,
                    source: CpuProviderDomainError::ParallelModeNotSupported {
                        mode: selected_mode,
                    },
                });
            }
        }
        if runtime.grouped_scheduling == GroupedGemmScheduling::EngineOuter
            && !runtime.gemm_capabilities.accepts_mode(ParallelMode::Outer)
        {
            return Err(CpuProviderBundleInstallError::IncompatibleDomain {
                domain_id,
                provider: CpuProviderSlot::Gemm,
                source: CpuProviderDomainError::ParallelModeNotSupported {
                    mode: ParallelMode::Outer,
                },
            });
        }
        Ok(())
    }

    pub(crate) fn preflight_dot_general(&self, entry: &CpuOperationEntry<'_>) -> Result<()> {
        self.inner
            .dot_general
            .dot_general_mode(entry, None)
            .map(|_| ())
            .map_err(|error| Error::backend_source(OP, error))
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_dot_general_into(
        &self,
        entry: &CpuOperationEntry<'_>,
        buffers: &mut BufferPool,
        cache: &mut GemmAnalysisCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        output: TensorWrite<'_>,
    ) -> Result<()> {
        self.execute_dot_general_into_scoped(
            entry,
            None,
            buffers,
            cache,
            cache_slot,
            lhs,
            rhs,
            config,
            accumulation,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_dot_general_into_scoped(
        &self,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
        buffers: &mut BufferPool,
        cache: &mut GemmAnalysisCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        output: TensorWrite<'_>,
    ) -> Result<()> {
        self.inner.dot_general.execute_into(
            &self.inner,
            entry,
            entered,
            buffers,
            cache,
            cache_slot,
            lhs,
            rhs,
            config,
            accumulation,
            output,
        )
    }

    #[cfg(test)]
    pub(crate) fn execute_grouped_gemm(
        &self,
        entry: &CpuOperationEntry<'_>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::backend::GroupedGemmConfig<'_>,
        output: TensorWrite<'_>,
    ) -> Result<()> {
        self.execute_grouped_gemm_scoped(entry, None, lhs, rhs, config, output)
    }

    pub(crate) fn execute_grouped_gemm_scoped(
        &self,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::backend::GroupedGemmConfig<'_>,
        output: TensorWrite<'_>,
    ) -> Result<()> {
        self.inner
            .dot_general
            .execute_grouped(entry, entered, lhs, rhs, config, output)
    }
}

/// `out = alpha * op(lhs) * op(rhs) + beta * out` for a contraction whose axes
/// are all batch axes, through the strided elementwise kernels.
fn execute_all_batch_elementwise(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    output: TensorWrite<'_>,
) -> Result<()> {
    let exec = context.strided_exec_context();
    // Output axes are the batch axes in configuration order; align each
    // operand to that order with a metadata-only transpose.
    let lhs_view = TensorRead::from_view(transposed_read_view(lhs, &config.lhs_batch_dims)?);
    let rhs_view = TensorRead::from_view(transposed_read_view(rhs, &config.rhs_batch_dims)?);
    let lhs_conj = if accumulation.lhs_conj {
        Some(crate::elementwise::conj_read_with_pool(
            buffers,
            &exec,
            lhs_view.clone(),
        )?)
    } else {
        None
    };
    let rhs_conj = if accumulation.rhs_conj {
        Some(crate::elementwise::conj_read_with_pool(
            buffers,
            &exec,
            rhs_view.clone(),
        )?)
    } else {
        None
    };
    let factors = [
        lhs_conj.as_ref().map_or(lhs_view, TensorRead::from_tensor),
        rhs_conj.as_ref().map_or(rhs_view, TensorRead::from_tensor),
    ];
    let overwrite = DotGeneralAccumulation::overwrite(lhs.dtype())?;
    let (result, product) =
        if accumulation.alpha == overwrite.alpha && accumulation.beta == overwrite.beta {
            // Overwrite: the product is written straight into `output`.
            let result = tenferro_internal_cpu_kernels::elementwise_read_into_with_context(
                tenferro_tensor::ElementwiseReadOp::Multiply,
                &factors,
                output,
                &exec,
                |inputs, out| {
                    crate::backend::elementwise_read_into_fallback_with_pool(
                        buffers,
                        &exec,
                        tenferro_tensor::ElementwiseReadOp::Multiply,
                        inputs,
                        out,
                    )
                },
            );
            (result, None)
        } else {
            let [lhs_factor, rhs_factor] = factors;
            let product =
                crate::elementwise::mul_read_with_pool(buffers, &exec, lhs_factor, rhs_factor)?;
            let result = crate::blas1::axpby_read_into_accum(
                context,
                buffers,
                accumulation.alpha,
                TensorRead::from_tensor(&product),
                accumulation.beta,
                output,
            );
            (result, Some(product))
        };
    for temporary in [lhs_conj, rhs_conj, product].into_iter().flatten() {
        crate::backend::reclaim_tensor(buffers, temporary);
    }
    result
}

/// A canonical GEMM operand: the caller's compact view or a packed copy.
// INVARIANT: two stack-local values per canonical contraction; boxing the view
// would add a heap allocation to the path this type exists to make cheaper.
#[allow(clippy::large_enum_variant)]
enum CanonicalOperand<'input> {
    Borrowed(TensorRead<'input>),
    Packed(Tensor),
}

impl CanonicalOperand<'_> {
    fn read(&self) -> TensorRead<'_> {
        match self {
            Self::Borrowed(read) => read.clone(),
            Self::Packed(tensor) => TensorRead::from_tensor(tensor),
        }
    }

    fn reclaim(self, buffers: &mut BufferPool) {
        if let Self::Packed(tensor) = self {
            crate::backend::reclaim_tensor(buffers, tensor);
        }
    }
}

/// Whether every operand axis is a batch axis (an elementwise product).
fn is_all_batch_contraction(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &DotGeneralConfig,
) -> bool {
    lhs.shape().len() == config.lhs_batch_dims.len()
        && rhs.shape().len() == config.rhs_batch_dims.len()
}

/// Number of batch items of a contraction: the product of its batch extents.
fn dot_batch_items(lhs: &TensorRead<'_>, config: &DotGeneralConfig) -> Result<usize> {
    config
        .lhs_batch_dims
        .iter()
        .try_fold(1usize, |items, &axis| {
            lhs.shape()
                .get(axis)
                .and_then(|&extent| items.checked_mul(extent))
        })
        .ok_or_else(|| {
            Error::invalid_argument(OP, "lhs", "batch axes are out of range or overflow usize")
        })
}

/// The batch policy in force for an operation: the entered session context's
/// (which carries any scoped override) or the operation entry's.
fn effective_batch_policy(
    entry: &CpuOperationEntry<'_>,
    entered: Option<&CpuExecutionContext<'_>>,
) -> crate::CpuBatchPolicy {
    entered.map_or_else(|| entry.batch_policy(), CpuExecutionContext::batch_policy)
}

/// Translate a resolved strategy into the provider's vendor-batch control.
fn vendor_batch_for(
    policy: crate::CpuBatchPolicy,
    strategy: CpuBatchStrategy,
) -> crate::provider::CpuVendorBatch {
    match strategy {
        CpuBatchStrategy::WholeBatchVendor => crate::provider::CpuVendorBatch::Required,
        CpuBatchStrategy::Auto => crate::provider::CpuVendorBatch::Allowed {
            max_item_dim: policy.thresholds().vendor_batch_max_item_dim(),
        },
        _ => crate::provider::CpuVendorBatch::Forbidden,
    }
}

fn unsupported_provider_error(capability: &'static str, reason: CpuProviderUnsupported) -> Error {
    Error::unsupported(
        OP,
        format!("configured CPU {capability} provider reported unsupported: {reason:?}"),
    )
}

impl DotGeneralRuntime {
    fn accepts_dot_general_mode(&self, mode: crate::ParallelMode) -> bool {
        self.general_capabilities
            .is_none_or(|capabilities| capabilities.accepts_mode(mode))
            && self.gemm_capabilities.accepts_mode(mode)
            && self.layout_capabilities.accepts_mode(mode)
    }

    fn validate_strict_capability(
        &self,
        capabilities: crate::CpuProviderExecutionCapabilities,
        thread_budget: usize,
    ) -> std::result::Result<(), CpuProviderDomainError> {
        if self.capability_policy == ProviderCapabilityPolicy::ProviderDefaultCompatibility {
            return Ok(());
        }
        if capabilities.thread_count == crate::CpuThreadCountControl::GlobalOrUncontrolled {
            return Err(CpuProviderDomainError::ThreadCountNotEnforceable {
                thread_budget,
                control: capabilities.thread_count,
            });
        }
        Ok(())
    }

    fn dot_general_mode(
        &self,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
    ) -> std::result::Result<ParallelMode, CpuProviderDomainError> {
        // A contraction reached from a lane of outer fan-out (for example an
        // algorithm's per-item work) contributes every provider it can reach to
        // the lane's nesting check, whatever the capability policy.
        if entered.is_some_and(CpuExecutionContext::is_outer_fan_out_lane) {
            crate::provider::check_outer_fan_out_delegates(
                self.general_capabilities
                    .iter()
                    .chain([&self.gemm_capabilities, &self.layout_capabilities]),
            )?;
            return Ok(ParallelMode::Sequential);
        }
        if self.capability_policy == ProviderCapabilityPolicy::ProviderDefaultCompatibility {
            return Ok(entry.provider_default_compatibility_mode());
        }
        let thread_budget = entry.thread_budget().get();
        if let Some(capabilities) = self.general_capabilities {
            self.validate_strict_capability(capabilities, thread_budget)?;
        }
        self.validate_strict_capability(self.gemm_capabilities, thread_budget)?;
        self.validate_strict_capability(self.layout_capabilities, thread_budget)?;
        entry.preferred_provider_mode(|mode| self.accepts_dot_general_mode(mode))
    }

    /// Apply a forced batch strategy to a strided-batched contraction's mode
    /// before any output write.
    fn batched_dot_mode(
        &self,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
        mode: ParallelMode,
        items: usize,
    ) -> Result<ParallelMode> {
        if items <= 1 {
            return Ok(mode);
        }
        let strategy = effective_batch_policy(entry, entered).strategy();
        match strategy {
            CpuBatchStrategy::Sequential => {
                if self.accepts_dot_general_mode(ParallelMode::Sequential) {
                    Ok(ParallelMode::Sequential)
                } else {
                    Err(strategy_unavailable(
                        OP,
                        strategy,
                        "a contraction provider does not accept sequential calls",
                    ))
                }
            }
            // Forced outer lanes split the batch inside the entered Inner
            // context; one thread or an unsplittable layout is a typed error
            // raised where the plan is known.
            CpuBatchStrategy::OuterParallel if entry.thread_budget().get() > 1 => {
                Ok(ParallelMode::Inner)
            }
            CpuBatchStrategy::OuterParallel => Err(strategy_unavailable(
                OP,
                strategy,
                "the selected CPU domain has one thread, so there are no outer lanes",
            )),
            _ => Ok(mode),
        }
    }

    fn grouped_mode(
        &self,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
    ) -> std::result::Result<ParallelMode, CpuProviderDomainError> {
        if entered.is_some_and(CpuExecutionContext::is_outer_fan_out_lane) {
            crate::provider::check_outer_fan_out_delegates([&self.gemm_capabilities])?;
            return Ok(ParallelMode::Sequential);
        }
        if self.capability_policy == ProviderCapabilityPolicy::ProviderDefaultCompatibility {
            return Ok(entry.provider_default_compatibility_mode());
        }
        self.validate_strict_capability(self.gemm_capabilities, entry.thread_budget().get())?;
        entry.preferred_provider_mode(|mode| self.gemm_capabilities.accepts_mode(mode))
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_into(
        &self,
        bundle_identity: &Arc<CpuProviderBundleInner>,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
        buffers: &mut BufferPool,
        cache: &mut GemmAnalysisCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        output: TensorWrite<'_>,
    ) -> Result<()> {
        let validated = validate_dot_general(&lhs, &rhs, &output, config, accumulation)?;
        let mode = self
            .dot_general_mode(entry, entered)
            .map_err(|error| Error::backend_source(OP, error))?;
        let mode = self.batched_dot_mode(entry, entered, mode, dot_batch_items(&lhs, config)?)?;
        cache.bind_provider_bundle(bundle_identity);
        entry
            .enter_or_reuse(entered, mode, |provider_context| {
                self.execute_into_validated(
                    provider_context,
                    validated,
                    buffers,
                    cache,
                    cache_slot,
                    lhs,
                    rhs,
                    config,
                    accumulation,
                    output,
                )
            })
            .map_err(|error| Error::backend_source(OP, error))?
    }

    // INVARIANT: these arguments are distinct borrowed components of one
    // validated dispatch; grouping them would duplicate validation-owned
    // metadata or add a request allocation to the hot path.
    #[allow(clippy::too_many_arguments)]
    fn execute_into_validated(
        &self,
        provider_context: &CpuExecutionContext<'_>,
        validated: ValidatedDotGeneral<'_>,
        buffers: &mut BufferPool,
        cache: &mut GemmAnalysisCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        mut output: TensorWrite<'_>,
    ) -> Result<()> {
        if let Some(general) = &self.general {
            let request = validated.request(&lhs, &rhs, &mut output, accumulation);
            match general.dot_general(provider_context, request)? {
                CpuProviderOutcome::Executed => return Ok(()),
                CpuProviderOutcome::Unsupported(reason) => {
                    if self.general_policy == GeneralContractionPolicy::Required {
                        return Err(unsupported_provider_error(
                            "required general-contraction",
                            reason,
                        ));
                    }
                }
            }
        }

        // A contraction in which every axis is a batch axis is an elementwise
        // product; classify it before GEMM lowering instead of running one 1x1
        // GEMM per element.
        if is_all_batch_contraction(&lhs, &rhs, config) {
            return execute_all_batch_elementwise(
                provider_context,
                buffers,
                &lhs,
                &rhs,
                config,
                accumulation,
                output,
            );
        }

        if let Some(plan) =
            crate::gemm::prepare_provider_gemm(cache, cache_slot, &lhs, &rhs, &output, config)?
        {
            match execute_gemm_plan(
                self.gemm.as_ref(),
                provider_context,
                plan,
                &lhs,
                &rhs,
                accumulation,
                &mut output,
            )? {
                CpuProviderOutcome::Executed => return Ok(()),
                CpuProviderOutcome::Unsupported(reason)
                    if !canonical_gemm_fallback_supported(reason) =>
                {
                    return Err(unsupported_provider_error("GEMM", reason));
                }
                CpuProviderOutcome::Unsupported(_) => {}
            }
        }

        self.execute_canonical_gemm(
            provider_context,
            buffers,
            cache,
            cache_slot,
            &lhs,
            &rhs,
            config,
            accumulation,
            &mut output,
        )
    }

    /// Materialize both operands into the canonical GEMM layout, run
    /// `execute` on them, then return the temporaries to the pool.
    #[allow(clippy::too_many_arguments)]
    fn with_canonical_operands<R>(
        &self,
        provider_context: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        lhs: &TensorRead<'_>,
        rhs: &TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        execute: impl FnOnce(
            &TensorRead<'_>,
            &TensorRead<'_>,
            &DotGeneralConfig,
            DotGeneralAccumulation,
        ) -> Result<R>,
    ) -> Result<R> {
        let (lhs_perm, rhs_perm, canonical_config) =
            crate::gemm::canonical_gemm_layout(config, lhs.shape().len(), rhs.shape().len());
        let lhs_canonical = self.canonical_operand(
            provider_context,
            buffers,
            lhs,
            &lhs_perm,
            accumulation.lhs_conj,
        )?;
        let rhs_canonical = match self.canonical_operand(
            provider_context,
            buffers,
            rhs,
            &rhs_perm,
            accumulation.rhs_conj,
        ) {
            Ok(operand) => operand,
            Err(error) => {
                lhs_canonical.reclaim(buffers);
                return Err(error);
            }
        };
        let result = execute(
            &lhs_canonical.read(),
            &rhs_canonical.read(),
            &canonical_config,
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                ..accumulation
            },
        );
        lhs_canonical.reclaim(buffers);
        rhs_canonical.reclaim(buffers);
        result
    }

    /// An operand in canonical GEMM layout: borrowed when the permuted view is
    /// already compact column-major and needs no conjugation, packed otherwise.
    fn canonical_operand<'input>(
        &self,
        provider_context: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TensorRead<'input>,
        permutation: &[usize],
        conjugate: bool,
    ) -> Result<CanonicalOperand<'input>> {
        if !conjugate {
            let permuted = TensorRead::from_view(transposed_read_view(input, permutation)?);
            if permuted.is_col_major_contiguous()? {
                return Ok(CanonicalOperand::Borrowed(permuted));
            }
        }
        materialize_canonical_operand(
            self.layout.as_ref(),
            provider_context,
            buffers,
            input,
            permutation,
            conjugate,
        )
        .map(CanonicalOperand::Packed)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_canonical_gemm(
        &self,
        provider_context: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        cache: &mut GemmAnalysisCache,
        cache_slot: Option<usize>,
        lhs: &TensorRead<'_>,
        rhs: &TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        output: &mut TensorWrite<'_>,
    ) -> Result<()> {
        self.with_canonical_operands(
            provider_context,
            buffers,
            lhs,
            rhs,
            config,
            accumulation,
            |lhs, rhs, canonical_config, canonical_accumulation| {
                // The canonical operands' grouping is known from the axis
                // counts; only an unusual layout needs the general analysis.
                let plan = match crate::gemm::canonical_provider_gemm_plan_into(
                    lhs,
                    rhs,
                    output,
                    canonical_config,
                )? {
                    Some(plan) => Some(plan),
                    None => crate::gemm::prepare_provider_gemm_canonical(
                        cache,
                        cache_slot,
                        lhs,
                        rhs,
                        output,
                        canonical_config,
                    )?,
                };
                let Some(plan) = plan else {
                    return Err(Error::unsupported(
                        OP,
                        "configured CPU layout-plus-GEMM path cannot represent the canonical contraction",
                    ));
                };
                match execute_gemm_plan(
                    self.gemm.as_ref(),
                    provider_context,
                    plan,
                    lhs,
                    rhs,
                    canonical_accumulation,
                    output,
                )? {
                    CpuProviderOutcome::Executed => Ok(()),
                    CpuProviderOutcome::Unsupported(reason) => {
                        Err(unsupported_provider_error("GEMM", reason))
                    }
                }
            },
        )
    }

    /// Execute a `beta == 0` allocated dot into uninitialized pooled bytes.
    ///
    /// Returns [`CpuProviderOutcome::Executed`] after every destination
    /// element is initialized, or [`CpuProviderOutcome::Unsupported`] when the
    /// GEMM provider cannot execute the planned contraction into
    /// uninitialized storage (the caller discards the checkout and retries on
    /// the zeroed path). Errors propagate; a provider error may follow a
    /// partial write, so it is never silently retried.
    ///
    /// The direct GEMM plan is tried first and the canonical packing fallback
    /// second, both into the uninitialized destination; operand packing draws
    /// on `buffers` while the destination is a separate checkout. An
    /// all-batch contraction is left to the zeroed path, which runs it
    /// elementwise instead of as per-element GEMMs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_dot_into_uninit(
        &self,
        bundle_identity: &Arc<CpuProviderBundleInner>,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
        buffers: &mut BufferPool,
        cache: &mut GemmAnalysisCache,
        cache_slot: Option<usize>,
        lhs: &TensorRead<'_>,
        rhs: &TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        output_shape: &[usize],
        output_bytes: &mut [MaybeUninit<u8>],
    ) -> Result<CpuProviderOutcome> {
        let Some(witness) = self.gemm.uninit_provider() else {
            return Err(Error::unsupported(
                OP,
                "configured CPU GEMM provider does not expose the uninitialized-output contract",
            ));
        };
        let mode = self
            .dot_general_mode(entry, entered)
            .map_err(|error| Error::backend_source(OP, error))?;
        let items = dot_batch_items(lhs, config)?;
        let mode = self.batched_dot_mode(entry, entered, mode, items)?;
        // The uninitialized-output request has no vendor-batch control; a forced
        // whole-batch vendor call goes through the zeroed path, which has one.
        if items > 1
            && effective_batch_policy(entry, entered).strategy()
                == CpuBatchStrategy::WholeBatchVendor
        {
            return Ok(CpuProviderOutcome::Unsupported(
                CpuProviderUnsupported::StridedBatch,
            ));
        }
        if is_all_batch_contraction(lhs, rhs, config) {
            return Ok(CpuProviderOutcome::Unsupported(
                CpuProviderUnsupported::Layout(crate::provider::CpuOperand::Output),
            ));
        }
        cache.bind_provider_bundle(bundle_identity);
        entry
            .enter_or_reuse(entered, mode, |provider_context| {
                if let Some(plan) = crate::gemm::prepare_provider_gemm_into_uninit(
                    cache,
                    cache_slot,
                    lhs,
                    rhs,
                    output_shape,
                    config,
                )? {
                    match execute_gemm_plan_into_uninit(
                        witness,
                        provider_context,
                        plan,
                        lhs,
                        rhs,
                        accumulation,
                        &mut *output_bytes,
                    )? {
                        CpuProviderOutcome::Executed => return Ok(CpuProviderOutcome::Executed),
                        CpuProviderOutcome::Unsupported(reason)
                            if !canonical_gemm_fallback_supported(reason) =>
                        {
                            return Ok(CpuProviderOutcome::Unsupported(reason));
                        }
                        // Unsupported leaves the destination untouched.
                        CpuProviderOutcome::Unsupported(_) => {}
                    }
                }
                self.with_canonical_operands(
                    provider_context,
                    buffers,
                    lhs,
                    rhs,
                    config,
                    accumulation,
                    |lhs, rhs, canonical_config, canonical_accumulation| {
                        let plan = match crate::gemm::canonical_provider_gemm_plan_uninit(
                            lhs,
                            rhs,
                            output_shape,
                            canonical_config,
                        )? {
                            Some(plan) => Some(plan),
                            None => crate::gemm::prepare_provider_gemm_canonical_into_uninit(
                                cache,
                                cache_slot,
                                lhs,
                                rhs,
                                output_shape,
                                canonical_config,
                            )?,
                        };
                        let Some(plan) = plan else {
                            return Ok(CpuProviderOutcome::Unsupported(
                                CpuProviderUnsupported::Layout(crate::provider::CpuOperand::Output),
                            ));
                        };
                        execute_gemm_plan_into_uninit(
                            witness,
                            provider_context,
                            plan,
                            lhs,
                            rhs,
                            canonical_accumulation,
                            output_bytes,
                        )
                    },
                )
            })
            .map_err(|error| Error::backend_source(OP, error))?
    }

    #[allow(clippy::redundant_closure)]
    fn execute_grouped(
        &self,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::backend::GroupedGemmConfig<'_>,
        mut output: TensorWrite<'_>,
    ) -> Result<()> {
        tenferro_tensor::backend::validate_grouped_gemm(
            &lhs,
            &rhs,
            &output,
            config,
            "grouped_gemm",
        )?;
        let policy = effective_batch_policy(entry, entered);
        let jobs = config.jobs().len();
        // The policy governs batches; a single job is a plain GEMM.
        let strategy = if jobs > 1 {
            policy.strategy()
        } else {
            CpuBatchStrategy::Auto
        };
        let fan_out = match strategy {
            CpuBatchStrategy::Auto
                if self.grouped_scheduling == GroupedGemmScheduling::EngineOuter =>
            {
                match entered {
                    None => (entry.supports_outer()
                        && policy
                            .thresholds()
                            .fans_out(jobs, jobs.min(entry.thread_budget().get())))
                    .then_some(crate::provider::CpuOuterFanOut::Executor(*entry)),
                    // Inside a session the lanes share the entered pool, so
                    // only enough estimated work per lane pays for the split.
                    Some(context) if context.can_fan_out_lanes() => {
                        auto_grouped_lane_count(
                            policy.thresholds(),
                            config.jobs(),
                            context.thread_budget().get(),
                        )
                        .map(|_| crate::provider::CpuOuterFanOut::Lanes(*context))
                    }
                    Some(_) => None,
                }
            }
            CpuBatchStrategy::OuterParallel => Some(match entered {
                None if entry.supports_outer() => crate::provider::CpuOuterFanOut::Executor(*entry),
                Some(context) if context.can_fan_out_lanes() => {
                    crate::provider::CpuOuterFanOut::Lanes(*context)
                }
                _ => {
                    return Err(strategy_unavailable(
                        "grouped_gemm",
                        strategy,
                        "the selected CPU domain cannot fan out (one thread, or no outer-capable executor)",
                    ))
                }
            }),
            _ => None,
        };
        if let Some(fan_out) = fan_out {
            // Reject an independent-runtime GEMM before any lane runs.
            let checked = crate::provider::check_outer_fan_out_delegates([&self.gemm_capabilities])
                .map_err(|error| Error::backend_source("grouped_gemm", error))?;
            // The outer-scheduled grouped path only carries the four floating and complex
            // presets; the table is kept in one macro so its per-dtype invocation is one line
            // rather than the full argument list, and the definition is covered once.
            // Lanes of an entered context run one contiguous job chunk each:
            // one task per job made 1024 4^3 jobs 4x slower at 4 threads than
            // one thread. The executor keeps one index per job and schedules
            // them itself.
            let chunks = match fan_out {
                crate::provider::CpuOuterFanOut::Lanes(context) => auto_grouped_lane_count(
                    policy.thresholds(),
                    config.jobs(),
                    context.thread_budget().get(),
                )
                .unwrap_or_else(|| jobs.min(context.thread_budget().get())),
                crate::provider::CpuOuterFanOut::Executor(_) => jobs,
            };
            macro_rules! outer_typed {
                ($variant:ident, $storage:expr, $base:expr) => {
                    execute_grouped_outer_typed(
                        self.gemm.as_ref(),
                        checked,
                        fan_out,
                        chunks,
                        &lhs,
                        &rhs,
                        config,
                        $storage,
                        $base,
                        |view| TensorViewMut::$variant(view),
                    )
                };
            }

            return match &mut output {
                TensorWrite::Tensor(tensor) => match tensor.dtype() {
                    DType::F32 => {
                        outer_typed!(F32, dot_write_operand::<f32>(tensor)?.host_data_mut()?, 0)
                    }
                    DType::F64 => {
                        outer_typed!(F64, dot_write_operand::<f64>(tensor)?.host_data_mut()?, 0)
                    }
                    DType::C32 => outer_typed!(
                        C32,
                        dot_write_operand::<Complex32>(tensor)?.host_data_mut()?,
                        0
                    ),
                    DType::C64 => outer_typed!(
                        C64,
                        dot_write_operand::<Complex64>(tensor)?.host_data_mut()?,
                        0
                    ),
                    _ => Err(unsupported_provider_error(
                        "grouped-GEMM",
                        CpuProviderUnsupported::DType(tensor.dtype()),
                    )),
                },
                TensorWrite::View(TensorViewMut::F32(output)) => {
                    let base = output.offset();
                    outer_typed!(F32, output.host_storage_mut()?, base)
                }
                TensorWrite::View(TensorViewMut::F64(output)) => {
                    let base = output.offset();
                    outer_typed!(F64, output.host_storage_mut()?, base)
                }
                TensorWrite::View(TensorViewMut::C32(output)) => {
                    let base = output.offset();
                    outer_typed!(C32, output.host_storage_mut()?, base)
                }
                TensorWrite::View(TensorViewMut::C64(output)) => {
                    let base = output.offset();
                    outer_typed!(C64, output.host_storage_mut()?, base)
                }
                _ => Err(unsupported_provider_error(
                    "grouped-GEMM",
                    CpuProviderUnsupported::DType(output.dtype()),
                )),
            };
        }
        let mut mode = self
            .grouped_mode(entry, entered)
            .map_err(|error| Error::backend_source("grouped_gemm", error))?;
        if strategy == CpuBatchStrategy::Sequential {
            if !self
                .gemm_capabilities
                .accepts_mode(ParallelMode::Sequential)
            {
                return Err(strategy_unavailable(
                    "grouped_gemm",
                    strategy,
                    "the GEMM provider does not accept sequential calls",
                ));
            }
            mode = ParallelMode::Sequential;
        }
        let vendor_batch = vendor_batch_for(policy, strategy);
        entry
            .enter_or_reuse(entered, mode, |provider_context| {
                let request = CpuGroupedGemmRequest::new(
                    &lhs,
                    &rhs,
                    &mut output,
                    config.jobs(),
                    config.accumulation(),
                )
                .with_vendor_batch(vendor_batch);
                match self.gemm.grouped_gemm(provider_context, request)? {
                    CpuProviderOutcome::Executed => Ok(()),
                    CpuProviderOutcome::Unsupported(reason) => {
                        Err(unsupported_provider_error("grouped-GEMM", reason))
                    }
                }
            })
            .map_err(|error| Error::backend_source("grouped_gemm", error))?
    }
}

fn execute_gemm_plan(
    provider: &dyn CpuGemmProvider,
    context: &CpuExecutionContext<'_>,
    plan: crate::gemm::ProviderGemmPlan,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    accumulation: DotGeneralAccumulation,
    output: &mut TensorWrite<'_>,
) -> Result<CpuProviderOutcome> {
    let batch_count = plan.batch_count();
    let policy = context.batch_policy();
    let strategy = if batch_count > 1 {
        policy.strategy()
    } else {
        CpuBatchStrategy::Auto
    };
    // A strided batch keeps per-item GEMM unless the whole-batch vendor call
    // is requested: on OpenBLAS 0.3.32 at one thread `cblas_dgemm_batch` was
    // 1.8x slower at 8^3 and 3.8x at 16^3 items (the `strided_batch_route`
    // bench), so the grouped cutoff does not transfer to strided batches.
    let vendor_batch = match strategy {
        CpuBatchStrategy::Auto if batch_count > 1 => crate::provider::CpuVendorBatch::Forbidden,
        _ => vendor_batch_for(policy, strategy),
    };
    if batch_count > 1
        && matches!(
            strategy,
            CpuBatchStrategy::Auto | CpuBatchStrategy::OuterParallel
        )
    {
        if let Some(outcome) =
            try_execute_gemm_plan_on_lanes(provider, context, plan, lhs, rhs, accumulation, output)?
        {
            return Ok(outcome);
        }
        if strategy == CpuBatchStrategy::OuterParallel {
            return Err(forced_lanes_unavailable());
        }
    }
    let request = plan
        .request(lhs, rhs, output, accumulation)
        .with_vendor_batch(vendor_batch);
    let outcome = if batch_count == 1 {
        provider.gemm(context, request)?
    } else {
        provider.strided_batched_gemm(context, request)?
    };
    Ok(outcome)
}

/// The lanes a strided batch runs on, or `None` for one provider call.
///
/// `Auto` asks the policy's lane cost model
/// ([`crate::CpuBatchThresholds::auto_lanes`]); `OuterParallel` forces one lane
/// per thread, capped by the batch. The context must be able to fan out.
/// The typed error for a forced `OuterParallel` strided batch that cannot be
/// split: the context cannot fan out, the output items do not occupy disjoint
/// increasing ranges, or the provider may not run inside tenferro lanes.
fn forced_lanes_unavailable() -> Error {
    strategy_unavailable(
        OP,
        CpuBatchStrategy::OuterParallel,
        "this strided batch cannot be split over outer lanes (one thread or a nested lane, \
         overlapping or reversed output items, or a provider that runs its own threads)",
    )
}

fn strided_batch_lanes(
    context: &CpuExecutionContext<'_>,
    plan: crate::gemm::ProviderGemmPlan,
) -> Option<usize> {
    let batch = plan.batch_count();
    if batch <= 1 || !context.can_fan_out_lanes() {
        return None;
    }
    let threads = context.thread_budget().get();
    let policy = context.batch_policy();
    match policy.strategy() {
        CpuBatchStrategy::Auto => {
            let thresholds = policy.thresholds();
            let item_ns = thresholds.lane_item_ns(plan.rows(), plan.columns(), plan.contracted());
            thresholds.auto_lanes(batch, item_ns.saturating_mul(batch), threads)
        }
        CpuBatchStrategy::OuterParallel => Some(threads.min(batch)).filter(|&lanes| lanes >= 2),
        _ => None,
    }
}

/// The number of outer lanes `Auto` uses for grouped jobs inside an entered
/// context, or `None` when fewer than two lanes would each receive enough
/// estimated work. Each lane runs a contiguous chunk of at least one job, so
/// lanes never exceed the job count.
fn auto_grouped_lane_count(
    thresholds: crate::CpuBatchThresholds,
    jobs: &[tenferro_tensor::backend::GroupedGemmJob],
    threads: usize,
) -> Option<usize> {
    let total_ns = jobs.iter().fold(0usize, |total, job| {
        total.saturating_add(thresholds.lane_item_ns(job.rows(), job.cols(), job.contracted()))
    });
    thresholds.auto_lanes(jobs.len(), total_ns, threads)
}

/// Run a strided batch as one contiguous chunk of items per outer lane when
/// `Auto` may fan out: the context owns more than one Rayon thread, the lane
/// cost model ([`strided_batch_lanes`]) and the policy thresholds allow it, the provider
/// may run inside a lane, and the output items occupy disjoint increasing
/// ranges. Returns `None` to keep the single provider call.
fn try_execute_gemm_plan_on_lanes(
    provider: &dyn CpuGemmProvider,
    context: &CpuExecutionContext<'_>,
    plan: crate::gemm::ProviderGemmPlan,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    accumulation: DotGeneralAccumulation,
    output: &mut TensorWrite<'_>,
) -> Result<Option<CpuProviderOutcome>> {
    let Some(lanes) = strided_batch_lanes(context, plan) else {
        return Ok(None);
    };
    if crate::provider::check_outer_fan_out_delegates([&provider.execution_capabilities()]).is_err()
    {
        return Ok(None);
    }
    let Some(item_span) = output_item_span(plan) else {
        return Ok(None);
    };
    macro_rules! typed {
        ($ty:ty, $variant:ident, $storage:expr) => {
            execute_gemm_chunks_on_lanes::<$ty>(
                provider,
                context,
                plan,
                lhs,
                rhs,
                accumulation,
                $storage,
                item_span,
                lanes,
                |view| TensorViewMut::$variant(view),
            )
        };
    }
    match output {
        TensorWrite::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => typed!(f32, F32, dot_write_operand::<f32>(tensor)?.host_data_mut()?),
            DType::F64 => typed!(f64, F64, dot_write_operand::<f64>(tensor)?.host_data_mut()?),
            DType::C32 => typed!(
                Complex32,
                C32,
                dot_write_operand::<Complex32>(tensor)?.host_data_mut()?
            ),
            DType::C64 => typed!(
                Complex64,
                C64,
                dot_write_operand::<Complex64>(tensor)?.host_data_mut()?
            ),
            _ => Ok(None),
        },
        TensorWrite::View(TensorViewMut::F32(view)) => typed!(f32, F32, view.host_storage_mut()?),
        TensorWrite::View(TensorViewMut::F64(view)) => typed!(f64, F64, view.host_storage_mut()?),
        TensorWrite::View(TensorViewMut::C32(view)) => {
            typed!(Complex32, C32, view.host_storage_mut()?)
        }
        TensorWrite::View(TensorViewMut::C64(view)) => {
            typed!(Complex64, C64, view.host_storage_mut()?)
        }
        TensorWrite::View(_) => Ok(None),
    }
}

/// The element span of one output item, when consecutive items occupy
/// disjoint increasing ranges (positive strides and `span <= batch stride`).
fn output_item_span(plan: crate::gemm::ProviderGemmPlan) -> Option<usize> {
    let layout = plan.output_layout();
    let positive = |stride: isize| usize::try_from(stride).ok().filter(|&stride| stride > 0);
    let (row, column, batch) = (
        positive(layout.row_stride())?,
        positive(layout.column_stride())?,
        positive(layout.batch_stride())?,
    );
    let span = plan
        .rows()
        .checked_sub(1)?
        .checked_mul(row)?
        .checked_add(plan.columns().checked_sub(1)?.checked_mul(column)?)?
        .checked_add(1)?;
    (span <= batch).then_some(span)
}

#[allow(clippy::too_many_arguments)]
fn execute_gemm_chunks_on_lanes<T>(
    provider: &dyn CpuGemmProvider,
    context: &CpuExecutionContext<'_>,
    plan: crate::gemm::ProviderGemmPlan,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    accumulation: DotGeneralAccumulation,
    storage: &mut [T],
    item_span: usize,
    lanes: usize,
    wrap: for<'a> fn(tenferro_tensor::TypedTensorViewMut<'a, T>) -> TensorViewMut<'a>,
) -> Result<Option<CpuProviderOutcome>>
where
    T: Send + Sync + 'static,
{
    let batch = plan.batch_count();
    let layout = plan.output_layout();
    let (Ok(first), Ok(batch_stride)) = (
        usize::try_from(layout.offset()),
        usize::try_from(layout.batch_stride()),
    ) else {
        return Ok(None);
    };
    // Split the output storage into one disjoint slice per chunk of items.
    let mut chunks = Vec::with_capacity(lanes);
    let mut rest = storage;
    let mut cursor = 0usize;
    let mut start = 0usize;
    for lane in 0..lanes {
        let len = batch / lanes + usize::from(lane < batch % lanes);
        let (Some(begin), Some(end)) = (
            start
                .checked_mul(batch_stride)
                .and_then(|value| value.checked_add(first)),
            (start + len - 1)
                .checked_mul(batch_stride)
                .and_then(|value| value.checked_add(first))
                .and_then(|value| value.checked_add(item_span)),
        ) else {
            return Ok(None);
        };
        if end - cursor > rest.len() {
            return Ok(None);
        }
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(begin - cursor);
        let (chunk, tail) = tail.split_at_mut(end - begin);
        rest = tail;
        cursor = end;
        let Some(chunk_plan) = plan.batch_chunk(start, len, 0) else {
            return Ok(None);
        };
        let shape = [plan.rows(), plan.columns(), len];
        let strides = [
            layout.row_stride(),
            layout.column_stride(),
            layout.batch_stride(),
        ];
        let view = tenferro_tensor::TypedTensorViewMut::from_slice(shape, strides, 0, chunk)?;
        chunks.push((chunk_plan, view));
        start += len;
    }

    let outcomes = std::sync::Mutex::new(Vec::with_capacity(lanes));
    context.with_outer_lanes(chunks, |(chunk_plan, view), lane| {
        let mut chunk_output = TensorWrite::from_view(wrap(view));
        let request = chunk_plan
            .request(lhs, rhs, &mut chunk_output, accumulation)
            .with_vendor_batch(crate::provider::CpuVendorBatch::Forbidden);
        let outcome = provider.strided_batched_gemm(lane, request);
        outcomes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(outcome);
    });
    let outcomes = outcomes
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut unsupported = None;
    for outcome in outcomes {
        match outcome? {
            CpuProviderOutcome::Executed => {}
            CpuProviderOutcome::Unsupported(reason) => unsupported = Some(reason),
        }
    }
    match unsupported {
        None => Ok(Some(CpuProviderOutcome::Executed)),
        // A declining lane wrote nothing, but its siblings may have: an
        // overwrite is redone in full by the caller's fallback, while an
        // accumulation cannot be retried without double counting.
        Some(reason) if accumulation_is_overwrite(accumulation)? => {
            Ok(Some(CpuProviderOutcome::Unsupported(reason)))
        }
        Some(reason) => Err(unsupported_provider_error("GEMM", reason)),
    }
}

fn accumulation_is_overwrite(accumulation: DotGeneralAccumulation) -> Result<bool> {
    let overwrite = DotGeneralAccumulation::overwrite(accumulation.alpha.dtype())?;
    Ok(accumulation.alpha == overwrite.alpha && accumulation.beta == overwrite.beta)
}

fn execute_gemm_plan_into_uninit(
    witness: &dyn CpuUninitGemmProvider,
    context: &CpuExecutionContext<'_>,
    plan: crate::gemm::ProviderGemmPlan,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    accumulation: DotGeneralAccumulation,
    output_bytes: &mut [MaybeUninit<u8>],
) -> Result<CpuProviderOutcome> {
    // An allocated batch takes the same Auto lane split as a caller-owned
    // destination; without it eager and allocating calls stayed serial (#1898).
    let strategy = context.batch_policy().strategy();
    if plan.batch_count() > 1
        && matches!(
            strategy,
            CpuBatchStrategy::Auto | CpuBatchStrategy::OuterParallel
        )
    {
        if let Some(outcome) = try_execute_gemm_plan_into_uninit_on_lanes(
            witness,
            context,
            plan,
            lhs,
            rhs,
            accumulation,
            output_bytes,
        )? {
            return Ok(outcome);
        }
        if strategy == CpuBatchStrategy::OuterParallel {
            return Err(forced_lanes_unavailable());
        }
    }
    let request = plan.uninit_request(lhs, rhs, accumulation);
    // SAFETY: the witness is structural proof the provider asserted the
    // full-overwrite contract via `unsafe impl`; the caller guarantees
    // beta == 0, so every destination element is written before `Executed`
    // and never read.
    unsafe { witness.gemm_into_uninit(context, request, output_bytes) }
}

/// Run an allocated strided batch as one contiguous chunk of items per outer
/// lane, under the same gate as [`try_execute_gemm_plan_on_lanes`]. Each lane
/// fully overwrites its own disjoint byte range. Returns `None` to keep the
/// single provider call.
fn try_execute_gemm_plan_into_uninit_on_lanes(
    witness: &dyn CpuUninitGemmProvider,
    context: &CpuExecutionContext<'_>,
    plan: crate::gemm::ProviderGemmPlan,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    accumulation: DotGeneralAccumulation,
    output_bytes: &mut [MaybeUninit<u8>],
) -> Result<Option<CpuProviderOutcome>> {
    let batch = plan.batch_count();
    let Some(lanes) = strided_batch_lanes(context, plan) else {
        return Ok(None);
    };
    if crate::provider::check_outer_fan_out_delegates([&witness.execution_capabilities()]).is_err()
    {
        return Ok(None);
    }
    let element_size = match lhs.dtype() {
        DType::F32 => std::mem::size_of::<f32>(),
        DType::F64 => std::mem::size_of::<f64>(),
        DType::C32 => std::mem::size_of::<Complex32>(),
        DType::C64 => std::mem::size_of::<Complex64>(),
        _ => return Ok(None),
    };
    let Some(item_span) = output_item_span(plan) else {
        return Ok(None);
    };
    let layout = plan.output_layout();
    let (Ok(first), Ok(batch_stride)) = (
        usize::try_from(layout.offset()),
        usize::try_from(layout.batch_stride()),
    ) else {
        return Ok(None);
    };
    // Split the destination bytes into one disjoint slice per chunk of items.
    let mut chunks = Vec::with_capacity(lanes);
    let mut rest = output_bytes;
    let mut cursor = 0usize;
    let mut start = 0usize;
    for lane in 0..lanes {
        let len = batch / lanes + usize::from(lane < batch % lanes);
        let (Some(begin), Some(end)) = (
            start
                .checked_mul(batch_stride)
                .and_then(|value| value.checked_add(first))
                .and_then(|value| value.checked_mul(element_size)),
            (start + len - 1)
                .checked_mul(batch_stride)
                .and_then(|value| value.checked_add(first))
                .and_then(|value| value.checked_add(item_span))
                .and_then(|value| value.checked_mul(element_size)),
        ) else {
            return Ok(None);
        };
        if end - cursor > rest.len() {
            return Ok(None);
        }
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(begin - cursor);
        let (chunk, tail) = tail.split_at_mut(end - begin);
        rest = tail;
        cursor = end;
        let Some(chunk_plan) = plan.batch_chunk(start, len, 0) else {
            return Ok(None);
        };
        chunks.push((chunk_plan, chunk));
        start += len;
    }

    let outcomes = std::sync::Mutex::new(Vec::with_capacity(lanes));
    context.with_outer_lanes(chunks, |(chunk_plan, chunk), lane| {
        let request = chunk_plan.uninit_request(lhs, rhs, accumulation);
        // SAFETY: as in `execute_gemm_plan_into_uninit`; each chunk is a
        // disjoint slice covering exactly the items of `chunk_plan`, whose
        // output layout starts at offset 0 within that slice.
        let outcome = unsafe { witness.gemm_into_uninit(lane, request, chunk) };
        outcomes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(outcome);
    });
    let outcomes = outcomes
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut unsupported = None;
    for outcome in outcomes {
        match outcome? {
            CpuProviderOutcome::Executed => {}
            CpuProviderOutcome::Unsupported(reason) => unsupported = Some(reason),
        }
    }
    // A declining lane leaves its chunk uninitialized; the caller discards an
    // unsupported uninitialized checkout, so partial writes are never observed.
    Ok(Some(match unsupported {
        None => CpuProviderOutcome::Executed,
        Some(reason) => CpuProviderOutcome::Unsupported(reason),
    }))
}

fn canonical_gemm_fallback_supported(reason: CpuProviderUnsupported) -> bool {
    matches!(
        reason,
        CpuProviderUnsupported::Layout(crate::provider::CpuOperand::Lhs)
            | CpuProviderUnsupported::Layout(crate::provider::CpuOperand::Rhs)
            | CpuProviderUnsupported::Conjugation
    )
}

fn transposed_read_view<'input>(
    input: &TensorRead<'input>,
    permutation: &[usize],
) -> Result<TensorView<'input>> {
    Ok(match input.clone().tensor_view() {
        TensorView::F32(view) => TensorView::F32(view.transpose_view(permutation)?),
        TensorView::F64(view) => TensorView::F64(view.transpose_view(permutation)?),
        TensorView::I32(view) => TensorView::I32(view.transpose_view(permutation)?),
        TensorView::I64(view) => TensorView::I64(view.transpose_view(permutation)?),
        TensorView::Bool(view) => TensorView::Bool(view.transpose_view(permutation)?),
        TensorView::C32(view) => TensorView::C32(view.transpose_view(permutation)?),
        TensorView::C64(view) => TensorView::C64(view.transpose_view(permutation)?),
    })
}

fn pooled_zero_tensor<T>(buffers: &mut BufferPool, shape: Vec<usize>) -> Result<TypedTensor<T>>
where
    T: PoolScalar + Clone + 'static,
{
    let element_count =
        tenferro_tensor::validate::checked_shape_product(OP, "canonical operand", &shape)?;
    TypedTensor::from_vec_col_major(shape, T::pool_acquire_zeroed(buffers, element_count))
}

fn allocate_canonical_operand(
    buffers: &mut BufferPool,
    dtype: DType,
    shape: Vec<usize>,
) -> Result<Tensor> {
    match dtype {
        DType::F32 => pooled_zero_tensor(buffers, shape).map(Tensor::from_typed::<f32>),
        DType::F64 => pooled_zero_tensor(buffers, shape).map(Tensor::from_typed::<f64>),
        DType::C32 => {
            pooled_zero_tensor(buffers, shape).map(Tensor::from_typed::<num_complex::Complex32>)
        }
        DType::C64 => {
            pooled_zero_tensor(buffers, shape).map(Tensor::from_typed::<num_complex::Complex64>)
        }
        dtype => Err(Error::unsupported_dtype(
            OP,
            dtype,
            crate::cpu_contraction_unsupported_dtype_message(dtype),
        )),
    }
}

/// The typed tensor behind a write adapter's tensor, or the refusal this provider reports.
fn dot_write_operand<T: tenferro_tensor::TensorScalar>(
    tensor: &mut Tensor,
) -> crate::Result<&mut TypedTensor<T>> {
    let dtype = tensor.dtype();
    tensor.as_typed_mut::<T>().ok_or_else(|| {
        unsupported_provider_error("grouped-GEMM", CpuProviderUnsupported::DType(dtype))
    })
}

/// The typed tensor behind a read or write adapter's tensor, or the refusal this module reports.
fn validated_operand<'a, T: tenferro_tensor::TensorScalar>(
    tensor: &'a Tensor,
    op: &'static str,
    message: &'static str,
) -> crate::Result<&'a TypedTensor<T>> {
    tensor
        .as_typed::<T>()
        .ok_or_else(|| crate::Error::unsupported_dtype(op, tensor.dtype(), message))
}

fn materialize_canonical_operand(
    provider: &dyn CpuLayoutTransformProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TensorRead<'_>,
    permutation: &[usize],
    conjugate: bool,
) -> Result<Tensor> {
    let input_view = transposed_read_view(input, permutation)?;
    let dtype = input_view.dtype();
    let input = TensorRead::from_view(input_view);
    if let Some(witness) = provider.uninit_provider() {
        let mut output = UninitTensor::acquire(buffers, dtype, input.shape().to_vec())?;
        let outcome = {
            let output_bytes = output.as_uninit_bytes_mut();
            // SAFETY: `witness` is structural proof the provider asserted the
            // full-overwrite contract via `unsafe impl`; `Executed` means
            // every element of `output_bytes` was written by
            // `materialize_into_uninit` (never read).
            unsafe {
                witness.materialize_into_uninit(
                    context,
                    &input,
                    CpuLayoutTransformIntent::CanonicalColumnMajor,
                    conjugate,
                    output_bytes,
                )
            }
        };
        match outcome {
            Ok(CpuProviderOutcome::Executed) => {
                // SAFETY: the unsafe provider contract guarantees the
                // destination is fully initialized before `Executed`.
                return unsafe { output.assume_init() };
            }
            Ok(CpuProviderOutcome::Unsupported(_)) => {
                // Discard the uninit checkout (drop frees via
                // `pool_discard_uninit`) and fall back to the zeroed path.
            }
            Err(error) => return Err(error),
        }
    }
    let shape = input.shape().to_vec();
    materialize_canonical_operand_zeroed(provider, context, buffers, &input, shape, conjugate)
}

fn materialize_canonical_operand_zeroed(
    provider: &dyn CpuLayoutTransformProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TensorRead<'_>,
    shape: Vec<usize>,
    conjugate: bool,
) -> Result<Tensor> {
    let mut output = allocate_canonical_operand(buffers, input.dtype(), shape)?;
    let outcome = {
        let mut output_write = TensorWrite::from_tensor(&mut output);
        let request = CpuLayoutTransformRequest::new(
            input,
            &mut output_write,
            CpuLayoutTransformIntent::CanonicalColumnMajor,
            conjugate,
        );
        provider.materialize(context, request)
    };
    match outcome {
        Ok(CpuProviderOutcome::Executed) => Ok(output),
        Ok(CpuProviderOutcome::Unsupported(reason)) => {
            crate::backend::reclaim_tensor(buffers, output);
            Err(unsupported_provider_error("layout-transform", reason))
        }
        Err(error) => {
            crate::backend::reclaim_tensor(buffers, output);
            Err(error)
        }
    }
}

/// Dtype-dispatched pooled full-overwrite destination for the uninitialized
/// dot paths.
///
/// The destination travels only as `MaybeUninit` bytes until an unsafe
/// `assume_init` completes the handoff; no `TensorWrite` is ever fabricated
/// over uninitialized storage.
pub(crate) enum UninitTensor {
    F32(PooledUninitOutput<f32>),
    F64(PooledUninitOutput<f64>),
    C32(PooledUninitOutput<Complex32>),
    C64(PooledUninitOutput<Complex64>),
}

impl UninitTensor {
    pub(crate) fn acquire(buffers: &BufferPool, dtype: DType, shape: Vec<usize>) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(Self::F32(PooledUninitOutput::new(buffers, shape)?)),
            DType::F64 => Ok(Self::F64(PooledUninitOutput::new(buffers, shape)?)),
            DType::C32 => Ok(Self::C32(PooledUninitOutput::new(buffers, shape)?)),
            DType::C64 => Ok(Self::C64(PooledUninitOutput::new(buffers, shape)?)),
            dtype => Err(Error::unsupported_dtype(
                OP,
                dtype,
                crate::cpu_contraction_unsupported_dtype_message(dtype),
            )),
        }
    }

    pub(crate) fn as_uninit_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        match self {
            Self::F32(output) => output.as_uninit_bytes_mut(),
            Self::F64(output) => output.as_uninit_bytes_mut(),
            Self::C32(output) => output.as_uninit_bytes_mut(),
            Self::C64(output) => output.as_uninit_bytes_mut(),
        }
    }

    /// # Safety
    ///
    /// Every logical destination element must have been initialized by the
    /// completed unsafe provider call before this handoff; otherwise reading
    /// or dropping the returned tensor is undefined behavior.
    pub(crate) unsafe fn assume_init(self) -> Result<Tensor> {
        // SAFETY: the caller proves every logical destination element was
        // written before `Executed` by the unsafe provider impl.
        unsafe {
            match self {
                Self::F32(output) => output.assume_init().map(Tensor::from_typed::<f32>),
                Self::F64(output) => output.assume_init().map(Tensor::from_typed::<f64>),
                Self::C32(output) => output
                    .assume_init()
                    .map(Tensor::from_typed::<num_complex::Complex32>),
                Self::C64(output) => output
                    .assume_init()
                    .map(Tensor::from_typed::<num_complex::Complex64>),
            }
        }
    }
}

fn checked_grouped_output_range(
    output_base: usize,
    output_len: usize,
    job: &tenferro_tensor::backend::GroupedGemmJob,
) -> Result<std::ops::Range<usize>> {
    let len = job.rows().checked_mul(job.cols()).ok_or_else(|| {
        Error::invalid_argument(
            "grouped_gemm",
            "jobs",
            "grouped-GEMM output span overflows usize",
        )
    })?;
    let start = output_base.checked_add(job.out_offset()).ok_or_else(|| {
        Error::invalid_argument(
            "grouped_gemm",
            "jobs",
            "grouped-GEMM output offset overflows usize",
        )
    })?;
    let end = start.checked_add(len).ok_or_else(|| {
        Error::invalid_argument(
            "grouped_gemm",
            "jobs",
            "grouped-GEMM output end overflows usize",
        )
    })?;
    if end > output_len {
        return Err(Error::invalid_argument(
            "grouped_gemm",
            "jobs",
            "grouped-GEMM output range exceeds host storage",
        ));
    }
    Ok(start..end)
}

/// Check every job's output range against the storage and report whether the
/// nonempty jobs start at strictly increasing offsets.
///
/// With increasing starts, the grouped validator's pairwise disjointness makes
/// every contiguous run of jobs end before the next run starts: a job `i`
/// before a nonempty job `b` has `start_i < start_b`, so disjointness forces
/// `end_i <= start_b`. Contiguous job chunks then own disjoint storage ranges.
fn grouped_output_starts_increase(
    jobs: &[tenferro_tensor::backend::GroupedGemmJob],
    output_base: usize,
    output_len: usize,
) -> Result<bool> {
    let mut previous = None;
    let mut increasing = true;
    for job in jobs {
        let range = checked_grouped_output_range(output_base, output_len, job)?;
        if !range.is_empty() {
            increasing &= previous.is_none_or(|start| start < range.start);
            previous = Some(range.start);
        }
    }
    Ok(increasing)
}

/// The storage range a contiguous run of validated jobs writes, and the jobs
/// with their output offsets rebased to that range.
fn grouped_chunk(
    jobs: &[tenferro_tensor::backend::GroupedGemmJob],
    output_base: usize,
    output_len: usize,
) -> Result<(
    std::ops::Range<usize>,
    SmallVec<[tenferro_tensor::backend::GroupedGemmJob; 1]>,
)> {
    let mut union: Option<std::ops::Range<usize>> = None;
    for job in jobs {
        let range = checked_grouped_output_range(output_base, output_len, job)?;
        if !range.is_empty() {
            union = Some(union.map_or(range.clone(), |union| {
                union.start.min(range.start)..union.end.max(range.end)
            }));
        }
    }
    let union = union.unwrap_or(0..0);
    let rebased = jobs
        .iter()
        .map(|job| {
            let start = output_base + job.out_offset();
            // An empty job writes nothing; any in-range offset serves.
            let out_offset = if job.rows() == 0 || job.cols() == 0 {
                0
            } else {
                start - union.start
            };
            tenferro_tensor::backend::GroupedGemmJob::new(
                out_offset,
                job.lhs_offset(),
                job.rhs_offset(),
                job.rows(),
                job.contracted(),
                job.cols(),
            )
        })
        .collect();
    Ok((union, rebased))
}

// INVARIANT: provider, context, tensor views, grouped metadata, and output
// storage are independent borrowed parts of one already-validated request.
#[allow(clippy::too_many_arguments)]
fn execute_grouped_outer_typed<T>(
    provider: &dyn CpuGemmProvider,
    checked: crate::provider::OuterFanOutChecked,
    fan_out: crate::provider::CpuOuterFanOut<'_>,
    chunks: usize,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &tenferro_tensor::backend::GroupedGemmConfig<'_>,
    output_storage: &mut [T],
    output_base: isize,
    wrap_output: for<'a> fn(tenferro_tensor::TypedTensorViewMut<'a, T>) -> TensorViewMut<'a>,
) -> Result<()>
where
    T: Send + Sync + 'static,
{
    const NO_DUPLICATE: usize = usize::MAX;

    let output_base = usize::try_from(output_base).map_err(|_| {
        Error::invalid_argument(
            "grouped_gemm",
            "output",
            "grouped-GEMM output base offset is negative",
        )
    })?;
    let output_storage_len = output_storage.len();
    // Every unit is one provider call: a call per job cost about 0.4 us of
    // request setup against 0.1 us per job inside one call, so lanes of an
    // entered context take contiguous chunks of jobs. Chunks need increasing
    // output starts to own disjoint ranges; otherwise every job is a unit.
    // Only this O(jobs) scan runs before the fan-out: ranges and rebased jobs
    // are built inside each unit, because serial per-job setup cost as much as
    // 4^3 GEMMs themselves.
    let jobs = config.jobs();
    let job_count = jobs.len();
    let unit_count = if chunks < job_count
        && grouped_output_starts_increase(jobs, output_base, output_storage_len)?
    {
        chunks.max(1)
    } else {
        for job in jobs {
            checked_grouped_output_range(output_base, output_storage_len, job)?;
        }
        job_count
    };
    let unit_jobs =
        |unit: usize| unit * job_count / unit_count..(unit + 1) * job_count / unit_count;

    let output_address = output_storage.as_mut_ptr() as usize;
    let operation_error = std::sync::Mutex::new(None);
    let failed = std::sync::atomic::AtomicBool::new(false);
    let unit_states = PackedJobStates::new(unit_count);
    let duplicate_index = AtomicUsize::new(NO_DUPLICATE);
    fan_out
        .submit(checked, unit_count, |index, provider_context| {
        if unit_states.try_claim(index).is_err() {
            let _ = duplicate_index.compare_exchange(
                NO_DUPLICATE,
                index,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            return Err(CpuDomainExecutorError::Scheduling {
                message: format!(
                    "executor invoked grouped-GEMM duplicate index {index}; every index must run exactly once"
                ),
            });
        }

        // A relaxed flag read keeps the error mutex off the per-unit path,
        // so concurrent lanes do not bounce its cache line.
        if !failed.load(Ordering::Relaxed) {
            let result = (|| -> Result<()> {
                let (range, rebased) =
                    grouped_chunk(&jobs[unit_jobs(index)], output_base, output_storage_len)?;
                let start = range.start;
                let len = range.len();
                // INVARIANT: `grouped_chunk` built this unit range from the
                // checked in-bounds job ranges of this allocation, and unit
                // ranges are pairwise disjoint: single-job units by the common
                // grouped validator, chunks of several jobs by increasing
                // output starts (`grouped_output_starts_increase`). The packed atomic claim changed this unit
                // from UNCLAIMED to RUNNING without clobbering neighboring
                // states, so even a contract-violating safe executor cannot
                // send a second invocation of this index to the provider.
                // SAFETY: `start..start + len` is in this allocation, and the
                // atomic claim permits exactly one invocation of each unit to
                // construct its mutable slice over a range no other unit uses.
                let output_slice = unsafe {
                    std::slice::from_raw_parts_mut((output_address as *mut T).add(start), len)
                };
                let output_view =
                    tenferro_tensor::TypedTensorViewMut::from_slice([len], [1], 0, output_slice)?;
                let mut output = TensorWrite::from_view(wrap_output(output_view));
                let request = CpuGroupedGemmRequest::new(
                    lhs,
                    rhs,
                    &mut output,
                    &rebased,
                    config.accumulation(),
                );
                match provider.grouped_gemm(provider_context, request)? {
                    CpuProviderOutcome::Executed => Ok(()),
                    CpuProviderOutcome::Unsupported(reason) => {
                        Err(unsupported_provider_error("grouped-GEMM", reason))
                    }
                }
            })();
            if let Err(error) = result {
                failed.store(true, Ordering::Relaxed);
                *operation_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
            }
        }
        let _ = unit_states.complete(index);
        Ok(())
    })
        .map_err(|error| Error::backend_source("grouped_gemm", error))?;
    let duplicate = duplicate_index.load(Ordering::Acquire);
    if duplicate != NO_DUPLICATE {
        return Err(Error::backend_source(
            "grouped_gemm",
            CpuDomainExecutorError::Scheduling {
                message: format!(
                    "executor invoked grouped-GEMM duplicate index {duplicate}; every index must run exactly once"
                ),
            },
        ));
    }
    if let Some((index, state)) = unit_states.first_incomplete() {
        let detail = if state == GroupedJobState::Unclaimed {
            format!("executor omitted grouped-GEMM missing index {index}")
        } else {
            format!("executor did not complete grouped-GEMM index {index}")
        };
        return Err(Error::backend_source(
            "grouped_gemm",
            CpuDomainExecutorError::Scheduling { message: detail },
        ));
    }
    match operation_error.into_inner() {
        Ok(Some(error)) => Err(error),
        Err(poisoned) => poisoned.into_inner().map_or(Ok(()), Err),
        Ok(None) => Ok(()),
    }
}

/// Error returned when a custom CPU provider bundle omits mandatory slots.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuProviderBundle;
/// assert!(CpuProviderBundle::custom_builder().build().is_err());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("missing mandatory CPU provider slots: GEMM={gemm}, layout={layout}")]
pub struct CpuProviderBundleBuildError {
    gemm: bool,
    layout: bool,
}

/// Provider slot that failed construction-time domain validation.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuProviderSlot;
/// assert_ne!(CpuProviderSlot::Gemm, CpuProviderSlot::LayoutTransform);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuProviderSlot {
    /// GEMM, strided-batched GEMM, and grouped-GEMM provider.
    Gemm,
    /// Layout materialization provider.
    LayoutTransform,
    /// Optional complete general-contraction provider.
    GeneralContraction,
}

/// Failure to install a CPU provider bundle for the backend's domains.
///
/// Phase 2 reserves this typed surface for construction-time domain/provider
/// validation. Provider capability classification populates concrete
/// incompatibilities without adding a second installation API.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuProviderBundleInstallError;
/// # fn diagnostic(error: &CpuProviderBundleInstallError) -> String {
/// error.to_string()
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CpuProviderBundleInstallError {
    /// A provider capability cannot satisfy one selected resource domain.
    #[error(
        "CPU provider bundle slot {provider:?} is incompatible with domain {domain_id:?}: {source}"
    )]
    IncompatibleDomain {
        /// Domain rejected by construction-time validation.
        domain_id: tenferro_tensor::CpuDomainId,
        /// Provider slot rejected by the domain contract.
        provider: CpuProviderSlot,
        /// Typed count, placement, or parallel-mode incompatibility.
        #[source]
        source: CpuProviderDomainError,
    },
}

/// Construction-time builder for immutable CPU provider slots.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuBackendKind, CpuProviderBundle};
/// let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled()).build()?;
/// assert!(bundle.shares_identity_with(&bundle.clone()));
/// # Ok::<(), tenferro_cpu::CpuProviderBundleBuildError>(())
/// ```
#[derive(Debug)]
pub struct CpuProviderBundleBuilder {
    gemm: Option<Arc<dyn CpuGemmProvider>>,
    layout: Option<Arc<dyn CpuLayoutTransformProvider>>,
    general: Option<Arc<dyn CpuGeneralContractionProvider>>,
    general_policy: GeneralContractionPolicy,
    grouped_scheduling: GroupedGemmScheduling,
    capability_policy: ProviderCapabilityPolicy,
    extensions: ProviderExtensions,
}

impl CpuProviderBundleBuilder {
    #[cfg(test)]
    pub(crate) fn provider_default_compatibility(mut self) -> Self {
        self.capability_policy = ProviderCapabilityPolicy::ProviderDefaultCompatibility;
        self
    }

    /// Replace the GEMM-family provider slot.
    pub fn gemm_provider(mut self, provider: Arc<dyn CpuGemmProvider>) -> Self {
        self.gemm = Some(provider);
        self.grouped_scheduling = GroupedGemmScheduling::ProviderOwned;
        self
    }

    /// Permit the engine to fan out grouped GEMM into concurrent single-job calls.
    ///
    /// The installed GEMM provider must be safe for concurrent calls and must
    /// honor [`crate::provider::ParallelMode::Sequential`] without creating inner
    /// workers. Custom providers remain provider-owned unless this capability
    /// is selected explicitly.
    pub fn engine_outer_grouped_gemm(mut self) -> Self {
        self.grouped_scheduling = GroupedGemmScheduling::EngineOuter;
        self
    }

    /// Replace the layout-materialization provider slot.
    pub fn layout_transform_provider(
        mut self,
        provider: Arc<dyn CpuLayoutTransformProvider>,
    ) -> Self {
        self.layout = Some(provider);
        self
    }

    /// Install a preferred general-contraction provider.
    pub fn prefer_general_contraction_provider(
        mut self,
        provider: Arc<dyn CpuGeneralContractionProvider>,
    ) -> Self {
        self.general = Some(provider);
        self.general_policy = GeneralContractionPolicy::Preferred;
        self
    }

    /// Install a required general-contraction provider.
    pub fn require_general_contraction_provider(
        mut self,
        provider: Arc<dyn CpuGeneralContractionProvider>,
    ) -> Self {
        self.general = Some(provider);
        self.general_policy = GeneralContractionPolicy::Required;
        self
    }

    /// Install a provider object of an operation-family crate, keyed by its
    /// type `E`; a later install of the same type replaces the earlier one.
    ///
    /// tenferro-cpu does not interpret extensions. The crate that defines `E`
    /// looks it up through [`CpuProviderBundle::extension`] (or
    /// [`crate::CpuExecSession::provider_extension`] inside a session) and
    /// owns its contract, including how it uses the provider's
    /// [`crate::CpuExecutionContext`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_cpu::{CpuBackendKind, CpuProviderBundle};
    /// let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
    ///     .extension(Arc::new(42_u32))
    ///     .build()?;
    /// assert_eq!(bundle.extension::<u32>().as_deref(), Some(&42));
    /// # Ok::<(), tenferro_cpu::CpuProviderBundleBuildError>(())
    /// ```
    pub fn extension<E: std::any::Any + Send + Sync>(mut self, extension: Arc<E>) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// Validate the mandatory slots and freeze the bundle identity.
    ///
    /// # Errors
    ///
    /// Returns [`CpuProviderBundleBuildError`] when GEMM or layout is absent.
    pub fn build(self) -> std::result::Result<CpuProviderBundle, CpuProviderBundleBuildError> {
        let missing = CpuProviderBundleBuildError {
            gemm: self.gemm.is_none(),
            layout: self.layout.is_none(),
        };
        let (Some(gemm), Some(layout)) = (self.gemm, self.layout) else {
            return Err(missing);
        };
        let general_capabilities = self
            .general
            .as_ref()
            .map(|provider| provider.execution_capabilities());
        let gemm_capabilities = gemm.execution_capabilities();
        let layout_capabilities = layout.execution_capabilities();
        Ok(CpuProviderBundle {
            inner: Arc::new(CpuProviderBundleInner {
                dot_general: DotGeneralRuntime {
                    general: self.general,
                    gemm,
                    layout,
                    general_capabilities,
                    gemm_capabilities,
                    layout_capabilities,
                    general_policy: self.general_policy,
                    grouped_scheduling: self.grouped_scheduling,
                    capability_policy: self.capability_policy,
                },
                extensions: self.extensions,
            }),
        })
    }
}

fn validate_axis_ranges(axes: &[usize], rank: usize) -> Result<()> {
    for &axis in axes {
        if axis >= rank {
            return Err(Error::axis_out_of_bounds(OP, axis, rank));
        }
    }
    Ok(())
}

fn role_mask(axes: &[usize], rank: usize, role: &'static str) -> Result<Option<u64>> {
    if rank > 64 {
        for (position, &axis) in axes.iter().enumerate() {
            if axes[..position].contains(&axis) {
                return Err(Error::duplicate_axis(OP, axis, role));
            }
        }
        return Ok(None);
    }

    let mut mask = 0_u64;
    for &axis in axes {
        let bit = 1_u64 << axis;
        if mask & bit != 0 {
            return Err(Error::duplicate_axis(OP, axis, role));
        }
        mask |= bit;
    }
    Ok(Some(mask))
}

fn validate_disjoint(
    first: &[usize],
    first_mask: Option<u64>,
    first_role: &'static str,
    second: &[usize],
    second_mask: Option<u64>,
    second_role: &'static str,
) -> Result<()> {
    let overlap = match (first_mask, second_mask) {
        (Some(first), Some(second)) => first & second,
        _ => 0,
    };
    let conflict = if overlap != 0 || first_mask.is_none() {
        first.iter().copied().find(|axis| second.contains(axis))
    } else {
        None
    };
    if let Some(axis) = conflict {
        return Err(Error::validation(
            OP,
            ValidationError::AxisRoleConflict {
                axis,
                first_role,
                second_role,
            },
        ));
    }
    Ok(())
}

pub(crate) fn validate_axis_groups<'a>(
    lhs_rank: usize,
    rhs_rank: usize,
    config: &'a DotGeneralConfig,
) -> Result<CpuContractionAxes<'a>> {
    validate_axis_ranges(&config.lhs_contracting_dims, lhs_rank)?;
    validate_axis_ranges(&config.rhs_contracting_dims, rhs_rank)?;
    validate_axis_ranges(&config.lhs_batch_dims, lhs_rank)?;
    validate_axis_ranges(&config.rhs_batch_dims, rhs_rank)?;

    let lhs_contracting_mask = role_mask(
        &config.lhs_contracting_dims,
        lhs_rank,
        "lhs_contracting_dims",
    )?;
    let rhs_contracting_mask = role_mask(
        &config.rhs_contracting_dims,
        rhs_rank,
        "rhs_contracting_dims",
    )?;
    let lhs_batch_mask = role_mask(&config.lhs_batch_dims, lhs_rank, "lhs_batch_dims")?;
    let rhs_batch_mask = role_mask(&config.rhs_batch_dims, rhs_rank, "rhs_batch_dims")?;

    validate_disjoint(
        &config.lhs_contracting_dims,
        lhs_contracting_mask,
        "lhs contracting",
        &config.lhs_batch_dims,
        lhs_batch_mask,
        "lhs batch",
    )?;
    validate_disjoint(
        &config.rhs_contracting_dims,
        rhs_contracting_mask,
        "rhs contracting",
        &config.rhs_batch_dims,
        rhs_batch_mask,
        "rhs batch",
    )?;

    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "dot_general_config",
            format!(
                "lhs/rhs contracting dim counts differ ({} vs {})",
                config.lhs_contracting_dims.len(),
                config.rhs_contracting_dims.len(),
            ),
        ));
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "dot_general_config",
            format!(
                "lhs/rhs batch dim counts differ ({} vs {})",
                config.lhs_batch_dims.len(),
                config.rhs_batch_dims.len(),
            ),
        ));
    }

    Ok(CpuContractionAxes::new(
        lhs_rank,
        rhs_rank,
        &config.lhs_contracting_dims,
        &config.rhs_contracting_dims,
        &config.lhs_batch_dims,
        &config.rhs_batch_dims,
        lhs_contracting_mask.zip(lhs_batch_mask).map(|(a, b)| a | b),
        rhs_contracting_mask.zip(rhs_batch_mask).map(|(a, b)| a | b),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidatedDotGeneral<'a> {
    axes: CpuContractionAxes<'a>,
    #[cfg(test)]
    output_element_count: usize,
}

impl<'a> ValidatedDotGeneral<'a> {
    #[cfg(test)]
    pub(crate) fn axes(&self) -> &CpuContractionAxes<'a> {
        &self.axes
    }

    #[cfg(test)]
    pub(crate) fn output_element_count(&self) -> usize {
        self.output_element_count
    }

    pub(crate) fn request<'request, 'input, 'output>(
        &'request self,
        lhs: &'request TensorRead<'input>,
        rhs: &'request TensorRead<'input>,
        output: &'request mut TensorWrite<'output>,
        accumulation: DotGeneralAccumulation,
    ) -> CpuDotGeneralRequest<'request, 'input, 'output>
    where
        'a: 'request,
    {
        CpuDotGeneralRequest::new(lhs, rhs, output, self.axes, accumulation)
    }
}

fn validate_paired_extents(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    axes: &CpuContractionAxes<'_>,
) -> Result<()> {
    for (lhs_axis, rhs_axis) in axes.contracting_pairs().chain(axes.batch_pairs()) {
        if lhs.shape()[lhs_axis] != rhs.shape()[rhs_axis] {
            return Err(Error::validation(
                OP,
                ShapeMismatch::ContractedDimensions {
                    lhs_axis,
                    lhs_size: lhs.shape()[lhs_axis],
                    rhs_axis,
                    rhs_size: rhs.shape()[rhs_axis],
                }
                .into(),
            ));
        }
    }
    Ok(())
}

fn expected_output_shape(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    axes: &CpuContractionAxes<'_>,
) -> Vec<usize> {
    axes.lhs_free_axes()
        .map(|axis| lhs.shape()[axis])
        .chain(axes.rhs_free_axes().map(|axis| rhs.shape()[axis]))
        .chain(
            axes.batch_pairs()
                .map(|(lhs_axis, _)| lhs.shape()[lhs_axis]),
        )
        .collect()
}

fn output_shape_matches(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    output: &TensorWrite<'_>,
    axes: &CpuContractionAxes<'_>,
) -> Result<()> {
    let expected_rank =
        axes.lhs_free_axes().count() + axes.rhs_free_axes().count() + axes.batch_pairs().len();
    let mut actual = output.shape().iter().copied();
    let matches = output.shape().len() == expected_rank
        && axes
            .lhs_free_axes()
            .map(|axis| lhs.shape()[axis])
            .chain(axes.rhs_free_axes().map(|axis| rhs.shape()[axis]))
            .chain(
                axes.batch_pairs()
                    .map(|(lhs_axis, _)| lhs.shape()[lhs_axis]),
            )
            .all(|expected| actual.next() == Some(expected));
    if matches {
        return Ok(());
    }

    Err(Error::validation(
        OP,
        ShapeMismatch::ExpectedActual {
            expected: expected_output_shape(lhs, rhs, axes).into(),
            actual: output.shape().to_vec().into(),
        }
        .into(),
    ))
}

fn layout_overflow() -> Error {
    Error::validation(OP, ValidationError::IntegerOverflow)
}

pub(crate) fn validate_layout_metadata(
    role: &'static str,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    storage_len: usize,
) -> Result<usize> {
    if shape.len() != strides.len() {
        return Err(Error::validation(
            OP,
            ValidationError::RankMismatch {
                expected: shape.len(),
                actual: strides.len(),
            },
        ));
    }
    let element_count = tenferro_tensor::validate::checked_shape_product(OP, role, shape)?;

    if shape.contains(&0) {
        let offset = usize::try_from(offset).map_err(|_| {
            Error::invalid_argument(OP, role, "minimum reachable offset is negative")
        })?;
        if offset > storage_len {
            return Err(Error::validation(OP, ValidationError::ViewOutOfBounds));
        }
        return Ok(element_count);
    }

    let mut minimum = offset;
    let mut maximum = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let steps = isize::try_from(extent - 1).map_err(|_| layout_overflow())?;
        let end = stride.checked_mul(steps).ok_or_else(layout_overflow)?;
        let (axis_minimum, axis_maximum) = if end < 0 { (end, 0) } else { (0, end) };
        minimum = minimum
            .checked_add(axis_minimum)
            .ok_or_else(layout_overflow)?;
        maximum = maximum
            .checked_add(axis_maximum)
            .ok_or_else(layout_overflow)?;
    }
    let minimum = usize::try_from(minimum)
        .map_err(|_| Error::invalid_argument(OP, role, "minimum reachable offset is negative"))?;
    let maximum = usize::try_from(maximum)
        .map_err(|_| Error::invalid_argument(OP, role, "maximum reachable offset is negative"))?;
    if minimum > maximum || maximum >= storage_len {
        return Err(Error::validation(OP, ValidationError::ViewOutOfBounds));
    }
    Ok(element_count)
}

macro_rules! validate_owned_layout {
    ($tensor:expr, $role:expr) => {{
        let tensor = $tensor;
        if tensor.backend_buffer().is_some() {
            return Err(crate::cpu_backend_buffer_error(OP));
        }
        // INVARIANT: an owned tensor's layout is compact column-major at offset
        // zero by construction, so the only reachable-range fact to check is
        // that its storage holds every logical element.
        let storage_len = tensor.host_data()?.len();
        let element_count =
            tenferro_tensor::validate::checked_shape_product(OP, $role, tensor.shape())?;
        if element_count > storage_len {
            return Err(Error::validation(OP, ValidationError::ViewOutOfBounds));
        }
        Ok(element_count)
    }};
}

macro_rules! validate_read_view_layout {
    ($view:expr, $role:expr) => {{
        let view = $view;
        let storage_len = view.host_storage()?.len();
        validate_layout_metadata(
            $role,
            view.shape(),
            view.strides(),
            view.offset(),
            storage_len,
        )
    }};
}

macro_rules! validate_write_view_layout {
    ($view:expr, $role:expr) => {{
        let view = $view;
        let storage_len = view.host_storage()?.len();
        validate_layout_metadata(
            $role,
            view.shape(),
            view.strides(),
            view.offset(),
            storage_len,
        )
    }};
}

/// The owned-operand layout table shared by the read and write validators.
///
/// Each arm reads the typed tensor its own dtype guard already selected, so
/// `validated_operand`'s refusal cannot fire from inside an arm. Keeping the table in one
/// macro means both validators share one covered definition instead of two hand-written
/// tables whose per-dtype lines differ only by the operation name and message.
macro_rules! validate_owned_layout_table {
    ($tensor:expr, $op:expr, $message:expr, $role:expr) => {
        match $tensor.dtype() {
            // A caller-owned payload is not a runtime operand.
            DType::External(type_id) => Err(crate::Error::unsupported_dtype(
                $op,
                DType::External(type_id),
                $message,
            )),
            DType::F32 => {
                validate_owned_layout!(validated_operand::<f32>($tensor, $op, $message)?, $role)
            }
            DType::F64 => {
                validate_owned_layout!(validated_operand::<f64>($tensor, $op, $message)?, $role)
            }
            DType::I32 => {
                validate_owned_layout!(validated_operand::<i32>($tensor, $op, $message)?, $role)
            }
            DType::I64 => {
                validate_owned_layout!(validated_operand::<i64>($tensor, $op, $message)?, $role)
            }
            DType::Bool => {
                validate_owned_layout!(validated_operand::<bool>($tensor, $op, $message)?, $role)
            }
            DType::C32 => validate_owned_layout!(
                validated_operand::<Complex32>($tensor, $op, $message)?,
                $role
            ),
            DType::C64 => validate_owned_layout!(
                validated_operand::<Complex64>($tensor, $op, $message)?,
                $role
            ),
        }
    };
}

fn validate_read_layout(tensor: &TensorRead<'_>, role: &'static str) -> Result<usize> {
    match tensor {
        TensorRead::Tensor(tensor) => validate_owned_layout_table!(
            tensor,
            "validate_read_layout",
            "an externally defined payload is not a runtime operand",
            role
        ),
        TensorRead::View(view) => match view {
            TensorView::F32(view) => validate_read_view_layout!(view, role),
            TensorView::F64(view) => validate_read_view_layout!(view, role),
            TensorView::I32(view) => validate_read_view_layout!(view, role),
            TensorView::I64(view) => validate_read_view_layout!(view, role),
            TensorView::Bool(view) => validate_read_view_layout!(view, role),
            TensorView::C32(view) => validate_read_view_layout!(view, role),
            TensorView::C64(view) => validate_read_view_layout!(view, role),
        },
    }
}

fn validate_write_layout(tensor: &TensorWrite<'_>, role: &'static str) -> Result<usize> {
    match tensor {
        TensorWrite::Tensor(tensor) => validate_owned_layout_table!(
            tensor,
            "validate_write_layout",
            "an externally defined payload is not a runtime destination",
            role
        ),
        TensorWrite::View(view) => match view {
            TensorViewMut::F32(view) => validate_write_view_layout!(view, role),
            TensorViewMut::F64(view) => validate_write_view_layout!(view, role),
            TensorViewMut::I32(view) => validate_write_view_layout!(view, role),
            TensorViewMut::I64(view) => validate_write_view_layout!(view, role),
            TensorViewMut::Bool(view) => validate_write_view_layout!(view, role),
            TensorViewMut::C32(view) => validate_write_view_layout!(view, role),
            TensorViewMut::C64(view) => validate_write_view_layout!(view, role),
        },
    }
}

pub(crate) fn validate_dot_general<'a>(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    output: &TensorWrite<'_>,
    config: &'a DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
) -> Result<ValidatedDotGeneral<'a>> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::dtype_mismatch(OP, lhs.dtype(), rhs.dtype()));
    }
    if output.dtype() != lhs.dtype() {
        return Err(Error::dtype_mismatch(OP, output.dtype(), lhs.dtype()));
    }
    if accumulation.alpha.dtype() != lhs.dtype() {
        return Err(Error::dtype_mismatch(
            OP,
            lhs.dtype(),
            accumulation.alpha.dtype(),
        ));
    }
    if accumulation.beta.dtype() != lhs.dtype() {
        return Err(Error::dtype_mismatch(
            OP,
            lhs.dtype(),
            accumulation.beta.dtype(),
        ));
    }

    crate::structural::validate_cpu_host_placement(OP, "lhs", read_placement(lhs))?;
    crate::structural::validate_cpu_host_placement(OP, "rhs", read_placement(rhs))?;
    crate::structural::validate_cpu_host_placement(OP, "output", write_placement(output))?;
    validate_read_layout(lhs, "lhs")?;
    validate_read_layout(rhs, "rhs")?;
    // The element count only feeds a test accessor; the validation is the point.
    let _output_element_count = validate_write_layout(output, "output")?;

    let axes = validate_axis_groups(lhs.shape().len(), rhs.shape().len(), config)?;
    validate_paired_extents(lhs, rhs, &axes)?;
    output_shape_matches(lhs, rhs, output, &axes)?;

    Ok(ValidatedDotGeneral {
        axes,
        #[cfg(test)]
        output_element_count: _output_element_count,
    })
}

fn read_placement<'a>(tensor: &'a TensorRead<'_>) -> &'a tenferro_tensor::Placement {
    match tensor {
        TensorRead::Tensor(tensor) => tensor.placement(),
        TensorRead::View(view) => match view {
            tenferro_tensor::TensorView::F32(view) => view.placement(),
            tenferro_tensor::TensorView::F64(view) => view.placement(),
            tenferro_tensor::TensorView::I32(view) => view.placement(),
            tenferro_tensor::TensorView::I64(view) => view.placement(),
            tenferro_tensor::TensorView::Bool(view) => view.placement(),
            tenferro_tensor::TensorView::C32(view) => view.placement(),
            tenferro_tensor::TensorView::C64(view) => view.placement(),
        },
    }
}

fn write_placement<'a>(tensor: &'a TensorWrite<'_>) -> &'a tenferro_tensor::Placement {
    match tensor {
        TensorWrite::Tensor(tensor) => tensor.placement(),
        TensorWrite::View(view) => match view {
            tenferro_tensor::TensorViewMut::F32(view) => view.placement(),
            tenferro_tensor::TensorViewMut::F64(view) => view.placement(),
            tenferro_tensor::TensorViewMut::I32(view) => view.placement(),
            tenferro_tensor::TensorViewMut::I64(view) => view.placement(),
            tenferro_tensor::TensorViewMut::Bool(view) => view.placement(),
            tenferro_tensor::TensorViewMut::C32(view) => view.placement(),
            tenferro_tensor::TensorViewMut::C64(view) => view.placement(),
        },
    }
}

#[cfg(test)]
mod tests;
