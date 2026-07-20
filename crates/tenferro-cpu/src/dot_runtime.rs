use tenferro_tensor::{
    Buffer, DType, DotGeneralAccumulation, DotGeneralConfig, ShapeMismatch, Tensor, TensorRead,
    TensorView, TensorViewMut, TensorWrite, TypedTensor, ValidationError,
};

use rayon::prelude::*;
use std::sync::Arc;

use crate::backend::CpuBackendKind;
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::provider::{
    builtin_gemm_provider, builtin_layout_provider, CpuContractionAxes, CpuDotGeneralRequest,
    CpuGemmProvider, CpuGeneralContractionProvider, CpuGroupedGemmRequest, CpuKernelParallelism,
    CpuLayoutTransformIntent, CpuLayoutTransformProvider, CpuLayoutTransformRequest,
    CpuProviderContext, CpuProviderOutcome, CpuProviderUnsupported,
};
use crate::{gemm::GemmAnalysisCache, CpuContext, Error, Result};

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
    pub(crate) general_policy: GeneralContractionPolicy,
    grouped_scheduling: GroupedGemmScheduling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupedGemmScheduling {
    ProviderOwned,
    EngineOuter,
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
    pub(crate) fn standard(kind: CpuBackendKind) -> Self {
        Self {
            inner: Arc::new(CpuProviderBundleInner {
                dot_general: DotGeneralRuntime {
                    general: None,
                    gemm: builtin_gemm_provider(kind),
                    layout: builtin_layout_provider(),
                    general_policy: GeneralContractionPolicy::Preferred,
                    grouped_scheduling: standard_grouped_scheduling(kind),
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

    pub(crate) fn execute_dot_general_into(
        &self,
        context: &CpuContext,
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
            context,
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
        context: &CpuContext,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &tenferro_tensor::backend::GroupedGemmConfig<'_>,
        output: TensorWrite<'_>,
    ) -> Result<()> {
        self.inner
            .dot_general
            .execute_grouped(context, lhs, rhs, config, output)
    }
}

fn unsupported_provider_error(capability: &'static str, reason: CpuProviderUnsupported) -> Error {
    Error::unsupported(
        OP,
        format!("configured CPU {capability} provider reported unsupported: {reason:?}"),
    )
}

impl DotGeneralRuntime {
    #[allow(clippy::too_many_arguments)]
    fn execute_into(
        &self,
        bundle_identity: &Arc<CpuProviderBundleInner>,
        context: &CpuContext,
        buffers: &mut BufferPool,
        cache: &mut GemmAnalysisCache,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        mut output: TensorWrite<'_>,
    ) -> Result<()> {
        cache.bind_provider_bundle(bundle_identity);
        let validated = validate_dot_general(&lhs, &rhs, &output, config, accumulation)?;
        let provider_context = CpuProviderContext::new(context, CpuKernelParallelism::Inner);

        if let Some(general) = &self.general {
            let request = validated.request(&lhs, &rhs, &mut output, accumulation);
            match general.dot_general(&provider_context, request)? {
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
            return execute_gemm_plan(
                self.gemm.as_ref(),
                &provider_context,
                plan,
                &lhs,
                &rhs,
                accumulation,
                &mut output,
            );
        }

        self.execute_canonical_gemm(
            &provider_context,
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
        provider_context: &CpuProviderContext<'_>,
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
        )?;
        let rhs_canonical = match materialize_canonical_operand(
            self.layout.as_ref(),
            provider_context,
            buffers,
            rhs,
            &rhs_perm,
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
            match crate::gemm::prepare_provider_gemm_canonical(
                cache,
                cache_slot,
                &lhs,
                &rhs,
                output,
                &canonical_config,
            ) {
                Ok(Some(plan)) => execute_gemm_plan(
                    self.gemm.as_ref(),
                    provider_context,
                    plan,
                    &lhs,
                    &rhs,
                    accumulation,
                    output,
                ),
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

    fn execute_grouped(
        &self,
        context: &CpuContext,
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
        if self.grouped_scheduling == GroupedGemmScheduling::EngineOuter
            && context.num_threads() > 1
            && config.jobs().len() > 1
        {
            return context.install_if_needed(|| match &mut output {
                TensorWrite::Tensor(Tensor::F32(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    context,
                    &lhs,
                    &rhs,
                    config,
                    output.host_data_mut()?,
                    0,
                    |view| TensorViewMut::F32(view),
                ),
                TensorWrite::Tensor(Tensor::F64(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    context,
                    &lhs,
                    &rhs,
                    config,
                    output.host_data_mut()?,
                    0,
                    |view| TensorViewMut::F64(view),
                ),
                TensorWrite::Tensor(Tensor::C32(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    context,
                    &lhs,
                    &rhs,
                    config,
                    output.host_data_mut()?,
                    0,
                    |view| TensorViewMut::C32(view),
                ),
                TensorWrite::Tensor(Tensor::C64(output)) => execute_grouped_outer_typed(
                    self.gemm.as_ref(),
                    context,
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
                        context,
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
                        context,
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
                        context,
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
                        context,
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
            });
        }
        let kernel_parallelism = CpuKernelParallelism::Inner;
        let provider_context = CpuProviderContext::new(context, kernel_parallelism);
        let request = CpuGroupedGemmRequest::new(
            &lhs,
            &rhs,
            &mut output,
            config.jobs(),
            config.accumulation(),
        );
        match self.gemm.grouped_gemm(&provider_context, request)? {
            CpuProviderOutcome::Executed => Ok(()),
            CpuProviderOutcome::Unsupported(reason) => {
                Err(unsupported_provider_error("grouped-GEMM", reason))
            }
        }
    }
}

fn execute_gemm_plan(
    provider: &dyn CpuGemmProvider,
    context: &CpuProviderContext<'_>,
    plan: crate::gemm::ProviderGemmPlan,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    accumulation: DotGeneralAccumulation,
    output: &mut TensorWrite<'_>,
) -> Result<()> {
    let batch_count = plan.batch_count();
    let request = plan.request(lhs, rhs, output, accumulation);
    let outcome = if batch_count == 1 {
        provider.gemm(context, request)?
    } else {
        provider.strided_batched_gemm(context, request)?
    };
    match outcome {
        CpuProviderOutcome::Executed => Ok(()),
        CpuProviderOutcome::Unsupported(reason) => Err(unsupported_provider_error("GEMM", reason)),
    }
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
    TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Host(T::pool_acquire_zeroed(buffers, element_count)),
        crate::default_placement(),
    )
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
            "CPU contraction providers support floating and complex dtypes",
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
    context: &CpuProviderContext<'_>,
    buffers: &mut BufferPool,
    input: &TensorRead<'_>,
    permutation: &[usize],
) -> Result<Tensor> {
    let input_view = transposed_read_view(input, permutation)?;
    let mut output =
        allocate_canonical_operand(buffers, input_view.dtype(), input_view.shape().to_vec())?;
    let input = TensorRead::from_view(input_view);
    let outcome = {
        let mut output_write = TensorWrite::from_tensor(&mut output);
        let request = CpuLayoutTransformRequest::new(
            &input,
            &mut output_write,
            CpuLayoutTransformIntent::CanonicalColumnMajor,
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

#[allow(clippy::too_many_arguments)]
fn execute_grouped_outer_typed<T>(
    provider: &dyn CpuGemmProvider,
    context: &CpuContext,
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
    let output_base = usize::try_from(output_base).map_err(|_| {
        Error::invalid_argument(
            "grouped_gemm",
            "output",
            "grouped-GEMM output base offset is negative",
        )
    })?;
    for job in config.jobs() {
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
        if end > output_storage.len() {
            return Err(Error::invalid_argument(
                "grouped_gemm",
                "jobs",
                "grouped-GEMM output range exceeds host storage",
            ));
        }
    }

    let output_address = output_storage.as_mut_ptr() as usize;
    let provider_context = CpuProviderContext::new(context, CpuKernelParallelism::Sequential);
    config.jobs().par_iter().try_for_each(|job| {
        let len = job.rows() * job.cols();
        let start = output_base + job.out_offset();
        // SAFETY: the common grouped validator proves pairwise-disjoint output
        // ranges; the preflight above proves each range is inside this one host
        // allocation. Each Rayon task receives exactly its own range.
        let output_slice =
            unsafe { std::slice::from_raw_parts_mut((output_address as *mut T).add(start), len) };
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
        match provider.grouped_gemm(&provider_context, request)? {
            CpuProviderOutcome::Executed => Ok(()),
            CpuProviderOutcome::Unsupported(reason) => {
                Err(unsupported_provider_error("grouped-GEMM", reason))
            }
        }
    })
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
}

impl CpuProviderBundleBuilder {
    /// Replace the GEMM-family provider slot.
    pub fn gemm_provider(mut self, provider: Arc<dyn CpuGemmProvider>) -> Self {
        self.gemm = Some(provider);
        self.grouped_scheduling = GroupedGemmScheduling::ProviderOwned;
        self
    }

    /// Permit the engine to fan out grouped GEMM into concurrent single-job calls.
    ///
    /// The installed GEMM provider must be safe for concurrent calls and must
    /// honor [`CpuKernelParallelism::Sequential`] without creating inner
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
        Ok(CpuProviderBundle {
            inner: Arc::new(CpuProviderBundleInner {
                dot_general: DotGeneralRuntime {
                    general: self.general,
                    gemm,
                    layout,
                    general_policy: self.general_policy,
                    grouped_scheduling: self.grouped_scheduling,
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
        let storage_len = match tensor.buffer() {
            Buffer::Host(storage) => storage.len(),
            Buffer::Backend(_) => return Err(crate::cpu_backend_buffer_error(OP)),
        };
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
