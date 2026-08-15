use tenferro_tensor::{
    DType, DotGeneralAccumulation, DotGeneralConfig, ShapeMismatch, Tensor, TensorRead, TensorView,
    TensorViewMut, TensorWrite, TypedTensor, ValidationError,
};

use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::backend::CpuBackendKind;
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::provider::{
    builtin_gemm_provider, builtin_layout_provider, CpuContractionAxes, CpuDotGeneralRequest,
    CpuExecutionContext, CpuGemmProvider, CpuGeneralContractionProvider, CpuGroupedGemmRequest,
    CpuLayoutTransformIntent, CpuLayoutTransformProvider, CpuLayoutTransformRequest,
    CpuOperationEntry, CpuProviderOutcome, CpuProviderUnsupported, CpuUninitGemmProvider,
};
use crate::{
    gemm::GemmAnalysisCache, CpuDomainExecutorError, CpuDomainId, CpuPlacementGuarantee,
    CpuProviderDomainError, CpuSet, Error, ParallelMode, PooledUninitOutput, Result,
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
        }
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
        placement_guarantee: CpuPlacementGuarantee,
        domain_cpus: &CpuSet,
        process_allowed_cpus: &CpuSet,
    ) -> std::result::Result<(), CpuProviderBundleInstallError> {
        let runtime = self.dot_general();
        let validate = |provider, capabilities| {
            crate::provider_capability::validate_provider_for_domain(
                capabilities,
                thread_budget,
                placement_guarantee,
                domain_cpus,
                process_allowed_cpus,
            )
            .map_err(|source| CpuProviderBundleInstallError::IncompatibleDomain {
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
            .dot_general_mode(entry)
            .map(|_| ())
            .map_err(|error| Error::backend_source(OP, error))
    }

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
    ) -> std::result::Result<ParallelMode, CpuProviderDomainError> {
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

    fn grouped_mode(
        &self,
        entry: &CpuOperationEntry<'_>,
    ) -> std::result::Result<ParallelMode, CpuProviderDomainError> {
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
            .dot_general_mode(entry)
            .map_err(|error| Error::backend_source(OP, error))?;
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
        let (lhs_perm, rhs_perm, canonical_config) =
            crate::gemm::canonical_gemm_layout(config, lhs.shape().len(), rhs.shape().len());
        let lhs_canonical = materialize_canonical_operand(
            self.layout.as_ref(),
            provider_context,
            buffers,
            lhs,
            &lhs_perm,
            accumulation.lhs_conj,
        )?;
        let rhs_canonical = match materialize_canonical_operand(
            self.layout.as_ref(),
            provider_context,
            buffers,
            rhs,
            &rhs_perm,
            accumulation.rhs_conj,
        ) {
            Ok(tensor) => tensor,
            Err(error) => {
                reclaim_temporary(buffers, lhs_canonical);
                return Err(error);
            }
        };

        let result = {
            let lhs = TensorRead::from_tensor(&lhs_canonical);
            let rhs = TensorRead::from_tensor(&rhs_canonical);
            let canonical_accumulation = DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                ..accumulation
            };
            match crate::gemm::prepare_provider_gemm_canonical(
                cache,
                cache_slot,
                &lhs,
                &rhs,
                output,
                &canonical_config,
            ) {
                Ok(Some(plan)) => match execute_gemm_plan(
                    self.gemm.as_ref(),
                    provider_context,
                    plan,
                    &lhs,
                    &rhs,
                    canonical_accumulation,
                    output,
                ) {
                    Ok(CpuProviderOutcome::Executed) => Ok(()),
                    Ok(CpuProviderOutcome::Unsupported(reason)) => {
                        Err(unsupported_provider_error("GEMM", reason))
                    }
                    Err(error) => Err(error),
                },
                Ok(None) => Err(Error::unsupported(
                    OP,
                    "configured CPU layout-plus-GEMM path cannot represent the canonical contraction",
                )),
                Err(error) => Err(error),
            }
        };
        reclaim_temporary(buffers, lhs_canonical);
        reclaim_temporary(buffers, rhs_canonical);
        result
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
    /// Only the direct GEMM plan is attempted here: the uninit checkout holds
    /// the scratch pool exclusively, so the canonical fallback (which
    /// materializes operands from the pool) is left to the zeroed path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_dot_into_uninit(
        &self,
        bundle_identity: &Arc<CpuProviderBundleInner>,
        entry: &CpuOperationEntry<'_>,
        entered: Option<&CpuExecutionContext<'_>>,
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
            .dot_general_mode(entry)
            .map_err(|error| Error::backend_source(OP, error))?;
        cache.bind_provider_bundle(bundle_identity);
        entry
            .enter_or_reuse(entered, mode, |provider_context| {
                let Some(plan) = crate::gemm::prepare_provider_gemm_into_uninit(
                    cache,
                    cache_slot,
                    lhs,
                    rhs,
                    output_shape,
                    config,
                )?
                else {
                    // No direct plan; the canonical path needs the scratch
                    // pool, which is exclusively held by the uninit checkout.
                    // The caller falls back to the zeroed path.
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
                    accumulation,
                    output_bytes,
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
        if entered.is_none()
            && self.grouped_scheduling == GroupedGemmScheduling::EngineOuter
            && entry.supports_outer()
            && config.jobs().len() > 1
        {
            if !self
                .gemm_capabilities
                .accepts_mode(crate::ParallelMode::Outer)
            {
                return Err(Error::backend_source(
                    "grouped_gemm",
                    crate::CpuProviderDomainError::ParallelModeNotSupported {
                        mode: crate::ParallelMode::Outer,
                    },
                ));
            }
            return match &mut output {
                TensorWrite::Tensor(Tensor::F32(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    entry,
                    &lhs,
                    &rhs,
                    config,
                    output.host_data_mut()?,
                    0,
                    |view| TensorViewMut::F32(view),
                ),
                TensorWrite::Tensor(Tensor::F64(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    entry,
                    &lhs,
                    &rhs,
                    config,
                    output.host_data_mut()?,
                    0,
                    |view| TensorViewMut::F64(view),
                ),
                TensorWrite::Tensor(Tensor::C32(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    entry,
                    &lhs,
                    &rhs,
                    config,
                    output.host_data_mut()?,
                    0,
                    |view| TensorViewMut::C32(view),
                ),
                TensorWrite::Tensor(Tensor::C64(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    entry,
                    &lhs,
                    &rhs,
                    config,
                    output.host_data_mut()?,
                    0,
                    |view| TensorViewMut::C64(view),
                ),
                TensorWrite::View(TensorViewMut::F32(output)) => {
                    let base = output.offset();
                    execute_grouped_outer_typed(
                        self.gemm.as_ref(),
                        entry,
                        &lhs,
                        &rhs,
                        config,
                        output.host_storage_mut()?,
                        base,
                        |view| TensorViewMut::F32(view),
                    )
                }
                TensorWrite::View(TensorViewMut::F64(output)) => {
                    let base = output.offset();
                    execute_grouped_outer_typed(
                        self.gemm.as_ref(),
                        entry,
                        &lhs,
                        &rhs,
                        config,
                        output.host_storage_mut()?,
                        base,
                        |view| TensorViewMut::F64(view),
                    )
                }
                TensorWrite::View(TensorViewMut::C32(output)) => {
                    let base = output.offset();
                    execute_grouped_outer_typed(
                        self.gemm.as_ref(),
                        entry,
                        &lhs,
                        &rhs,
                        config,
                        output.host_storage_mut()?,
                        base,
                        |view| TensorViewMut::C32(view),
                    )
                }
                TensorWrite::View(TensorViewMut::C64(output)) => {
                    let base = output.offset();
                    execute_grouped_outer_typed(
                        self.gemm.as_ref(),
                        entry,
                        &lhs,
                        &rhs,
                        config,
                        output.host_storage_mut()?,
                        base,
                        |view| TensorViewMut::C64(view),
                    )
                }
                _ => Err(unsupported_provider_error(
                    "grouped-GEMM",
                    CpuProviderUnsupported::DType(output.dtype()),
                )),
            };
        }
        let mode = self
            .grouped_mode(entry)
            .map_err(|error| Error::backend_source("grouped_gemm", error))?;
        entry
            .enter_or_reuse(entered, mode, |provider_context| {
                let request = CpuGroupedGemmRequest::new(
                    &lhs,
                    &rhs,
                    &mut output,
                    config.jobs(),
                    config.accumulation(),
                );
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
    let request = plan.request(lhs, rhs, output, accumulation);
    let outcome = if batch_count == 1 {
        provider.gemm(context, request)?
    } else {
        provider.strided_batched_gemm(context, request)?
    };
    Ok(outcome)
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
    let request = plan.uninit_request(lhs, rhs, accumulation);
    // SAFETY: the witness is structural proof the provider asserted the
    // full-overwrite contract via `unsafe impl`; the caller guarantees
    // beta == 0, so every destination element is written before `Executed`
    // and never read.
    unsafe { witness.gemm_into_uninit(context, request, output_bytes) }
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
        DType::F32 => pooled_zero_tensor(buffers, shape).map(Tensor::F32),
        DType::F64 => pooled_zero_tensor(buffers, shape).map(Tensor::F64),
        DType::C32 => pooled_zero_tensor(buffers, shape).map(Tensor::C32),
        DType::C64 => pooled_zero_tensor(buffers, shape).map(Tensor::C64),
        dtype => Err(Error::unsupported_dtype(
            OP,
            dtype,
            crate::cpu_contraction_unsupported_dtype_message(dtype),
        )),
    }
}

fn reclaim_temporary(buffers: &mut BufferPool, tensor: Tensor) {
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
    let shape = input_view.shape().to_vec();
    let input = TensorRead::from_view(input_view);
    if let Some(witness) = provider.uninit_provider() {
        let mut output = UninitTensor::acquire(buffers, dtype, shape.clone())?;
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
            reclaim_temporary(buffers, output);
            Err(unsupported_provider_error("layout-transform", reason))
        }
        Err(error) => {
            reclaim_temporary(buffers, output);
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
pub(crate) enum UninitTensor<'pool> {
    F32(PooledUninitOutput<'pool, f32>),
    F64(PooledUninitOutput<'pool, f64>),
    C32(PooledUninitOutput<'pool, Complex32>),
    C64(PooledUninitOutput<'pool, Complex64>),
}

impl<'pool> UninitTensor<'pool> {
    pub(crate) fn acquire(
        buffers: &'pool mut BufferPool,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<Self> {
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
                Self::F32(output) => output.assume_init().map(Tensor::F32),
                Self::F64(output) => output.assume_init().map(Tensor::F64),
                Self::C32(output) => output.assume_init().map(Tensor::C32),
                Self::C64(output) => output.assume_init().map(Tensor::C64),
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

// INVARIANT: provider, context, tensor views, grouped metadata, and output
// storage are independent borrowed parts of one already-validated request.
#[allow(clippy::too_many_arguments)]
fn execute_grouped_outer_typed<T>(
    provider: &dyn CpuGemmProvider,
    entry: &CpuOperationEntry<'_>,
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
    for job in config.jobs() {
        checked_grouped_output_range(output_base, output_storage_len, job)?;
    }

    let output_address = output_storage.as_mut_ptr() as usize;
    let operation_error = std::sync::Mutex::new(None);
    let job_states = PackedJobStates::new(config.jobs().len());
    let duplicate_index = AtomicUsize::new(NO_DUPLICATE);
    entry
        .submit_outer(config.jobs().len(), |index, provider_context| {
        if job_states.try_claim(index).is_err() {
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

        let already_failed = operation_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some();
        if !already_failed {
            let job = &config.jobs()[index];
            let result = (|| -> Result<()> {
                let range = checked_grouped_output_range(output_base, output_storage_len, job)?;
                let len = range.len();
                let start = range.start;
                // INVARIANT: the immutable job, output base, and allocation
                // length are identical to preflight. The shared checked helper
                // therefore reconstructs the same in-bounds range inside this
                // worker; the common grouped validator also proved distinct
                // job ranges disjoint. Before reaching this point, the
                // packed atomic claim changed this job from UNCLAIMED to
                // RUNNING without clobbering neighboring states, so even a
                // contract-violating safe executor cannot send a second
                // invocation of this index to the provider.
                // SAFETY: `start..start + len` is in this allocation. Distinct
                // jobs have disjoint validated ranges, and the atomic claim
                // permits exactly one invocation of each job to construct its
                // mutable slice.
                let output_slice = unsafe {
                    std::slice::from_raw_parts_mut((output_address as *mut T).add(start), len)
                };
                let output_view =
                    tenferro_tensor::TypedTensorViewMut::from_slice([len], [1], 0, output_slice)?;
                let mut output = TensorWrite::from_view(wrap_output(output_view));
                let job = tenferro_tensor::backend::GroupedGemmJob::new(
                    0,
                    job.lhs_offset(),
                    job.rhs_offset(),
                    job.rows(),
                    job.contracted(),
                    job.cols(),
                );
                let request = CpuGroupedGemmRequest::new(
                    lhs,
                    rhs,
                    &mut output,
                    std::slice::from_ref(&job),
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
                *operation_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
            }
        }
        let _ = job_states.complete(index);
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
    if let Some((index, state)) = job_states.first_incomplete() {
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
}

impl CpuProviderBundleBuilder {
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
    output_element_count: usize,
}

impl<'a> ValidatedDotGeneral<'a> {
    pub(crate) fn axes(&self) -> &CpuContractionAxes<'a> {
        &self.axes
    }

    pub(crate) fn output_element_count(&self) -> usize {
        self.output_element_count
    }

    #[allow(dead_code)]
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
        let storage_len = tensor.host_data()?.len();
        validate_layout_metadata(
            $role,
            tensor.shape(),
            tensor.layout().strides(),
            tensor.layout().offset(),
            storage_len,
        )
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

fn validate_read_layout(tensor: &TensorRead<'_>, role: &'static str) -> Result<usize> {
    match tensor {
        TensorRead::Tensor(tensor) => match tensor {
            Tensor::F32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::F64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::Bool(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C64(tensor) => validate_owned_layout!(tensor, role),
        },
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
        TensorWrite::Tensor(tensor) => match tensor {
            Tensor::F32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::F64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::Bool(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C64(tensor) => validate_owned_layout!(tensor, role),
        },
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
    let output_element_count = validate_write_layout(output, "output")?;

    let axes = validate_axis_groups(lhs.shape().len(), rhs.shape().len(), config)?;
    validate_paired_extents(lhs, rhs, &axes)?;
    output_shape_matches(lhs, rhs, output, &axes)?;

    Ok(ValidatedDotGeneral {
        axes,
        output_element_count,
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
