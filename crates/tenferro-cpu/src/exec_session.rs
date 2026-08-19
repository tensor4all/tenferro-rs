use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::{Tensor, TensorRead, TensorValue, TensorWrite};
use std::any::TypeId;
use std::sync::Arc;
use tenferro_tensor::backend::{BackendSession, ElementwiseFusionPlan, GroupedGemmConfig};
use tenferro_tensor::{
    CompareDir, ContractionScalar, DType, DotGeneralConfig, ElementwiseReadOp, GatherConfig,
    PadConfig, ScatterConfig, SharedTensorAllocationDomain, SliceConfig, TensorView, TypedTensor,
};
use tenferro_tensor::{
    DotGeneralAccumulation, SessionCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};

use super::backend::{
    elementwise_read_into_fallback_with_pool, reclaim_typed, tag_fresh_output, FreshCpuOutput,
};
use super::indexed_plan_cache::IndexedPlanCache;
use super::provider::{CpuExecutionContext, CpuOperationEntry, CpuProviderOutcome};
use super::CpuProviderBundle;
use super::{
    analytic, copy_tensor_read_into, elementwise, gemm, indexing, materialize_tensor_read,
    reduction, structural,
};

/// Marker for the concrete erased CPU execution-session target.
#[doc(hidden)]
pub(super) struct CpuExecSessionMarker;

/// Borrowed CPU execution session used by scheduler-owned extension regions.
#[doc(hidden)]
pub struct CpuExecSession<'a> {
    pub(crate) entry: CpuOperationEntry<'a>,
    pub(crate) entered: Option<CpuExecutionContext<'a>>,
    pub(crate) buffers: &'a mut BufferPool,
    pub(crate) gemm_analysis_cache: &'a mut gemm::GemmAnalysisCache,
    pub(crate) indexed_plan_cache: &'a mut IndexedPlanCache,
    pub(crate) providers: &'a CpuProviderBundle,
    pub(crate) backend_kind: super::CpuBackendKind,
    pub(crate) allocation_domain: Option<&'a Arc<dyn SharedTensorAllocationDomain>>,
}

fn pooled_zero_tensor<T>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + Clone + 'static,
{
    let element_count =
        tenferro_tensor::validate::checked_shape_product("dot_general", "output", &shape)?;
    TypedTensor::from_vec_col_major(shape, T::pool_acquire_zeroed(buffers, element_count))
}

fn allocate_dot_output(
    buffers: &mut BufferPool,
    dtype: DType,
    shape: Vec<usize>,
) -> crate::Result<Tensor> {
    match dtype {
        DType::F32 => pooled_zero_tensor(buffers, shape).map(Tensor::F32),
        DType::F64 => pooled_zero_tensor(buffers, shape).map(Tensor::F64),
        DType::C32 => pooled_zero_tensor(buffers, shape).map(Tensor::C32),
        DType::C64 => pooled_zero_tensor(buffers, shape).map(Tensor::C64),
        dtype => Err(crate::Error::unsupported_dtype(
            "dot_general",
            dtype,
            crate::cpu_contraction_unsupported_dtype_message(dtype),
        )),
    }
}

fn flatten_compact_read(
    input: TensorRead<'_>,
    element_count: usize,
) -> crate::Result<TensorRead<'_>> {
    macro_rules! flatten {
        ($variant:ident, $view:expr) => {
            Ok(TensorRead::from_view(TensorView::$variant(
                $view.try_reshape(&[element_count])?,
            )))
        };
    }

    match input {
        TensorRead::Tensor(Tensor::F32(tensor)) => flatten!(F32, tensor.as_view()),
        TensorRead::Tensor(Tensor::F64(tensor)) => flatten!(F64, tensor.as_view()),
        TensorRead::Tensor(Tensor::I32(tensor)) => flatten!(I32, tensor.as_view()),
        TensorRead::Tensor(Tensor::I64(tensor)) => flatten!(I64, tensor.as_view()),
        TensorRead::Tensor(Tensor::Bool(tensor)) => flatten!(Bool, tensor.as_view()),
        TensorRead::Tensor(Tensor::C32(tensor)) => flatten!(C32, tensor.as_view()),
        TensorRead::Tensor(Tensor::C64(tensor)) => flatten!(C64, tensor.as_view()),
        TensorRead::View(TensorView::F32(view)) => flatten!(F32, view),
        TensorRead::View(TensorView::F64(view)) => flatten!(F64, view),
        TensorRead::View(TensorView::I32(view)) => flatten!(I32, view),
        TensorRead::View(TensorView::I64(view)) => flatten!(I64, view),
        TensorRead::View(TensorView::Bool(view)) => flatten!(Bool, view),
        TensorRead::View(TensorView::C32(view)) => flatten!(C32, view),
        TensorRead::View(TensorView::C64(view)) => flatten!(C64, view),
    }
}

impl CpuExecSession<'_> {
    /// Return the provider selected by the owning CPU backend.
    #[doc(hidden)]
    pub fn kind(&self) -> super::CpuBackendKind {
        self.backend_kind
    }

    /// Return the resource-domain identity selected for this session.
    #[doc(hidden)]
    pub fn domain_id(&self) -> super::CpuDomainId {
        self.entry.domain_id()
    }

    /// Return the shared allocator selected for this execution session.
    #[doc(hidden)]
    pub fn shared_allocation_domain(&self) -> Option<Arc<dyn SharedTensorAllocationDomain>> {
        self.allocation_domain.cloned()
    }

    /// Run a CPU-owned linalg kernel inside this already-entered session.
    #[doc(hidden)]
    pub fn with_linalg_pool<R: Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        self.run_native_with_context(op)
    }

    fn run_native<R: Send>(
        &mut self,
        op: impl FnOnce(&mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let buffers = &mut *self.buffers;
        if let Some(context) = self.entered {
            return context.with_native_parallelism(|| op(buffers));
        }
        let mode = self.entry.preferred_engine_mode();
        self.entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| op(buffers))
            })
            .map_err(|error| crate::Error::backend_source("CPU native execution", error))?
    }

    fn run_native_fresh<R: FreshCpuOutput + Send>(
        &mut self,
        op: impl FnOnce(&mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let buffers = &mut *self.buffers;
        if let Some(context) = self.entered {
            return context.with_native_parallelism(|| {
                let mut output = op(buffers)?;
                output.tag_fresh(context.domain_id());
                Ok(output)
            });
        }
        let mode = self.entry.preferred_engine_mode();
        self.entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| {
                    let mut output = op(buffers)?;
                    output.tag_fresh(context.domain_id());
                    Ok(output)
                })
            })
            .map_err(|error| crate::Error::backend_source("CPU native execution", error))?
    }

    #[cfg(feature = "cpu-faer")]
    pub(crate) fn with_faer_parallelism(
        &mut self,
        callback: impl FnOnce(faer::Par) -> crate::Result<()> + Send,
    ) -> crate::Result<()> {
        self.run_native_with_context(|context, _| callback(context.faer_parallelism()))
    }

    fn run_native_with_context<R: Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let buffers = &mut *self.buffers;
        if let Some(context) = self.entered {
            return context.with_native_parallelism(|| op(&context, buffers));
        }
        let mode = self.entry.preferred_engine_mode();
        self.entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| op(context, buffers))
            })
            .map_err(|error| crate::Error::backend_source("CPU native execution", error))?
    }

    fn run_native_fresh_with_context<R: FreshCpuOutput + Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        let buffers = &mut *self.buffers;
        if let Some(context) = self.entered {
            return context.with_native_parallelism(|| {
                let mut output = op(&context, buffers)?;
                output.tag_fresh(context.domain_id());
                Ok(output)
            });
        }
        let mode = self.entry.preferred_engine_mode();
        self.entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| {
                    let mut output = op(context, buffers)?;
                    output.tag_fresh(context.domain_id());
                    Ok(output)
                })
            })
            .map_err(|error| crate::Error::backend_source("CPU native execution", error))?
    }

    fn run_native_fresh_with_indexed_context<R: FreshCpuOutput + Send>(
        &mut self,
        op: impl FnOnce(
                &CpuExecutionContext<'_>,
                &mut BufferPool,
                &mut IndexedPlanCache,
            ) -> crate::Result<R>
            + Send,
    ) -> crate::Result<R> {
        let buffers = &mut *self.buffers;
        let indexed_plan_cache = &mut *self.indexed_plan_cache;
        if let Some(context) = self.entered {
            return context.with_native_parallelism(|| {
                let mut output = op(&context, buffers, indexed_plan_cache)?;
                output.tag_fresh(context.domain_id());
                Ok(output)
            });
        }
        let mode = self.entry.preferred_engine_mode();
        self.entry
            .enter(mode, |context| {
                context.with_native_parallelism(|| {
                    let mut output = op(context, buffers, indexed_plan_cache)?;
                    output.tag_fresh(context.domain_id());
                    Ok(output)
                })
            })
            .map_err(|error| crate::Error::backend_source("CPU native execution", error))?
    }
}

impl TensorDeviceTransfer for CpuExecSession<'_> {
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        if tensor.backend_family().is_some() {
            return Err(crate::Error::runtime_state(
                "CpuBackend::download_to_host",
                "CPU backend received a backend buffer; download the tensor to host with its owning backend before CPU execution",
            ));
        }
        tensor.tensor_view().duplicate()
    }

    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        if tensor.backend_family().is_some() {
            return Err(crate::Error::runtime_state(
                "CpuBackend::upload_host_tensor",
                "CPU backend upload_host_tensor expects a host tensor; download backend buffers to host before CPU execution",
            ));
        }
        tensor.tensor_view().duplicate()
    }
}

/// Simple delegation that reuses an entered managed session when available.
macro_rules! delegate {
    ($name:ident($($arg:ident : $ty:ty),*) => $body:expr) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> {
            self.run_native_fresh(|_| $body)
        }
    };
}

/// Delegation for operations whose outputs can be allocated from the session pool.
macro_rules! delegate_with_pool {
    ($name:ident($($arg:ident : $ty:ty),*) => $callee:path) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> {
            self.run_native_fresh(|buffers| $callee(buffers, $($arg),*))
        }
    };
}

impl TensorElementwise for CpuExecSession<'_> {
    // Elementwise — direct delegation within the current session scope.
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            tenferro_tensor::backend::elementwise_read_into_with_context(
                op,
                inputs,
                out,
                &exec_context,
                |inputs, out| elementwise_read_into_fallback_with_pool(buffers, op, inputs, out),
            )
        })
    }

    delegate_with_pool!(add(lhs: &Tensor, rhs: &Tensor) => elementwise::add_with_pool);

    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.run_native_fresh(|buffers| elementwise::add_read_with_pool(buffers, lhs, rhs))
    }

    delegate_with_pool!(sub(lhs: &Tensor, rhs: &Tensor) => elementwise::sub_with_pool);
    delegate_with_pool!(sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::sub_read_with_pool);
    delegate_with_pool!(mul(lhs: &Tensor, rhs: &Tensor) => elementwise::mul_with_pool);
    delegate_with_pool!(mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::mul_read_with_pool);
    delegate_with_pool!(neg(input: &Tensor) => elementwise::neg_with_pool);
    delegate_with_pool!(neg_read(input: TensorRead<'_>) => elementwise::neg_read_with_pool);
    delegate_with_pool!(conj(input: &Tensor) => elementwise::conj_with_pool);
    delegate_with_pool!(conj_read(input: TensorRead<'_>) => elementwise::conj_read_with_pool);
    delegate_with_pool!(div(lhs: &Tensor, rhs: &Tensor) => elementwise::div_with_pool);
    delegate_with_pool!(div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::div_read_with_pool);
    delegate_with_pool!(rem(lhs: &Tensor, rhs: &Tensor) => elementwise::rem_with_pool);
    delegate_with_pool!(rem_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::rem_read_with_pool);
    delegate_with_pool!(abs(input: &Tensor) => elementwise::abs_with_pool);
    delegate_with_pool!(abs_read(input: TensorRead<'_>) => elementwise::abs_read_with_pool);
    delegate_with_pool!(sign(input: &Tensor) => elementwise::sign_with_pool);
    delegate_with_pool!(sign_read(input: TensorRead<'_>) => elementwise::sign_read_with_pool);
    delegate_with_pool!(maximum(lhs: &Tensor, rhs: &Tensor) => elementwise::maximum_with_pool);
    delegate_with_pool!(maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::maximum_read_with_pool);
    delegate_with_pool!(minimum(lhs: &Tensor, rhs: &Tensor) => elementwise::minimum_with_pool);
    delegate_with_pool!(minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::minimum_read_with_pool);
    delegate_with_pool!(compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) => elementwise::compare_with_pool);
    delegate_with_pool!(compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) => elementwise::compare_read_with_pool);
    delegate_with_pool!(select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) => elementwise::select_with_pool);
    delegate_with_pool!(select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) => elementwise::select_read_with_pool);
    delegate_with_pool!(clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) => elementwise::clamp_with_pool);
    delegate_with_pool!(clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) => elementwise::clamp_read_with_pool);
}

impl TensorAnalytic for CpuExecSession<'_> {
    // Analytic
    delegate_with_pool!(exp(input: &Tensor) => analytic::exp_with_pool);
    delegate_with_pool!(exp_read(input: TensorRead<'_>) => analytic::exp_read_with_pool);
    delegate_with_pool!(log(input: &Tensor) => analytic::log_with_pool);
    delegate_with_pool!(log_read(input: TensorRead<'_>) => analytic::log_read_with_pool);
    delegate_with_pool!(sin(input: &Tensor) => analytic::sin_with_pool);
    delegate_with_pool!(sin_read(input: TensorRead<'_>) => analytic::sin_read_with_pool);
    delegate_with_pool!(cos(input: &Tensor) => analytic::cos_with_pool);
    delegate_with_pool!(cos_read(input: TensorRead<'_>) => analytic::cos_read_with_pool);
    delegate_with_pool!(tanh(input: &Tensor) => analytic::tanh_with_pool);
    delegate_with_pool!(tanh_read(input: TensorRead<'_>) => analytic::tanh_read_with_pool);
    delegate_with_pool!(sqrt(input: &Tensor) => analytic::sqrt_with_pool);
    delegate_with_pool!(sqrt_read(input: TensorRead<'_>) => analytic::sqrt_read_with_pool);
    delegate_with_pool!(rsqrt(input: &Tensor) => analytic::rsqrt_with_pool);
    delegate_with_pool!(rsqrt_read(input: TensorRead<'_>) => analytic::rsqrt_read_with_pool);
    delegate_with_pool!(pow(lhs: &Tensor, rhs: &Tensor) => analytic::pow_with_pool);
    delegate_with_pool!(pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => analytic::pow_read_with_pool);
    delegate_with_pool!(expm1(input: &Tensor) => analytic::expm1_with_pool);
    delegate_with_pool!(expm1_read(input: TensorRead<'_>) => analytic::expm1_read_with_pool);
    delegate_with_pool!(log1p(input: &Tensor) => analytic::log1p_with_pool);
    delegate_with_pool!(log1p_read(input: TensorRead<'_>) => analytic::log1p_read_with_pool);
}

impl TensorStructural for CpuExecSession<'_> {
    // Structural
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        // INVARIANT: compact tensors need no kernel and no engine entry; the
        // Arc clone is the only work, so it must not pay the multi-thread pool
        // entry cost (O(us) on a 4-thread pool). Views still go through the
        // entry path because they may materialize via strided kernels.
        if matches!(input, TensorRead::Tensor(_)) {
            return materialize_tensor_read(self.buffers, "CpuBackend::to_contiguous_read", input);
        }
        self.run_native_fresh(|buffers| {
            materialize_tensor_read(buffers, "CpuBackend::to_contiguous_read", input)
        })
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()> {
        self.run_native(|_| copy_tensor_read_into("CpuBackend::copy_read_into", src, dst))
    }

    delegate_with_pool!(transpose(input: &Tensor, perm: &[usize]) => structural::transpose_with_pool);
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh(|buffers| structural::transpose_read_with_pool(buffers, input, perm))
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        // INVARIANT: typed_reshape performs a serial host copy (to_vec); no
        // parallel kernel runs, so the engine entry is pure overhead on
        // multi-thread pools.
        structural::reshape(input, shape)
    }

    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        match &input {
            // INVARIANT: compact inputs take the serial host-copy path, so
            // they must not pay the engine entry; views may materialize via
            // strided kernels and keep the entry.
            TensorRead::Tensor(tensor) => structural::reshape(tensor, shape),
            TensorRead::View(_) => self.run_native_fresh(|buffers| {
                structural::reshape_read_with_pool(buffers, input, shape)
            }),
        }
    }

    delegate_with_pool!(broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) => structural::broadcast_in_dim_with_pool);
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native_fresh(|buffers| {
            structural::broadcast_in_dim_read_with_pool(buffers, input, shape, dims)
        })
    }

    delegate_with_pool!(cast(input: &Tensor, to: crate::DType) => structural::cast_with_pool);
    delegate_with_pool!(extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::extract_diagonal_with_pool);
    delegate_with_pool!(embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::embed_diagonal_with_pool);
    delegate_with_pool!(tril(input: &Tensor, k: i64) => structural::tril_with_pool);
    delegate_with_pool!(triu(input: &Tensor, k: i64) => structural::triu_with_pool);
}

impl TensorReduction for CpuExecSession<'_> {
    // Reduction
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, _| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum(input, axes, &exec_context)
        })
    }

    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_sum_squares_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum_squares_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, _| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_prod(input, axes, &exec_context)
        })
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_prod_read(buffers, input, axes, &exec_context)
        })
    }

    delegate!(reduce_max(input: &Tensor, axes: &[usize]) => reduction::reduce_max(input, axes));

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh(|buffers| reduction::reduce_max_read(buffers, input, axes))
    }

    delegate!(reduce_min(input: &Tensor, axes: &[usize]) => reduction::reduce_min(input, axes));

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh(|buffers| reduction::reduce_min_read(buffers, input, axes))
    }
}

impl CpuExecSession<'_> {
    fn execute_dot_allocated(
        &mut self,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        let dtype = lhs.dtype();
        if rhs.dtype() != dtype {
            return Err(crate::Error::dtype_mismatch(
                "dot_general",
                dtype,
                rhs.dtype(),
            ));
        }
        let output_shape = tenferro_tensor::backend::dot_general_output_shape(
            lhs.shape(),
            rhs.shape(),
            config,
            "dot_general",
        )?;
        self.providers.preflight_dot_general(&self.entry)?;
        let accumulation = DotGeneralAccumulation {
            lhs_conj,
            rhs_conj,
            alpha: ContractionScalar::one(dtype)?,
            beta: ContractionScalar::zero(dtype)?,
        };

        // Uninitialized fast path: beta == 0 here by construction, so the
        // dot output is fully overwritten. Take it only when the GEMM
        // provider is the guaranteed consumer (no general-contraction
        // provider) and exposes the full-overwrite witness. The uninit
        // checkout holds the scratch pool exclusively, so only the direct
        // GEMM plan is attempted; anything else falls back below.
        let providers = self.providers;
        let runtime = providers.dot_general();
        if runtime.general.is_none() && runtime.gemm.uninit_provider().is_some() {
            let mut output = crate::dot_runtime::UninitTensor::acquire(
                self.buffers,
                dtype,
                output_shape.clone(),
            )?;
            let outcome = runtime.execute_dot_into_uninit(
                providers.inner(),
                &self.entry,
                self.entered.as_ref(),
                self.gemm_analysis_cache,
                cache_slot,
                &lhs,
                &rhs,
                config,
                accumulation,
                &output_shape,
                output.as_uninit_bytes_mut(),
            );
            match outcome {
                Ok(CpuProviderOutcome::Executed) => {
                    // SAFETY: the GEMM provider's unsafe impl guarantees every
                    // destination element is initialized before `Executed`.
                    let mut output = unsafe { output.assume_init()? };
                    tag_fresh_output(&mut output, self.entry.domain_id());
                    return Ok(output);
                }
                Ok(CpuProviderOutcome::Unsupported(_)) => {
                    // Discard the uninit checkout; fall back to the zeroed path.
                }
                Err(error) => return Err(error),
            }
        }

        let mut output = allocate_dot_output(self.buffers, dtype, output_shape)?;
        self.providers.execute_dot_general_into_scoped(
            &self.entry,
            self.entered.as_ref(),
            self.buffers,
            self.gemm_analysis_cache,
            cache_slot,
            lhs,
            rhs,
            config,
            accumulation,
            TensorWrite::from_tensor(&mut output),
        )?;
        tag_fresh_output(&mut output, self.entry.domain_id());
        Ok(output)
    }
}

impl TensorDot for CpuExecSession<'_> {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.execute_dot_allocated(
            None,
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
            false,
            false,
        )
    }

    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.execute_dot_allocated(None, lhs, rhs, config, false, false)
    }

    fn dot_general_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let accumulation = DotGeneralAccumulation::overwrite(lhs.dtype())?;
        self.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
    }

    fn dot_general_read_into_accum(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.providers.execute_dot_general_into_scoped(
            &self.entry,
            self.entered.as_ref(),
            self.buffers,
            self.gemm_analysis_cache,
            None,
            lhs,
            rhs,
            config,
            accumulation,
            out,
        )
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.execute_dot_allocated(
            None,
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
            lhs_conj,
            rhs_conj,
        )
    }
}

impl SessionCachedDot for CpuExecSession<'_> {
    fn dot_general_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.execute_dot_allocated(
            cache_slot,
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
            false,
            false,
        )
    }

    fn dot_general_with_conj_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.execute_dot_allocated(
            cache_slot,
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
            lhs_conj,
            rhs_conj,
        )
    }

    fn dot_general_read_into_accum_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.providers.execute_dot_general_into_scoped(
            &self.entry,
            self.entered.as_ref(),
            self.buffers,
            self.gemm_analysis_cache,
            cache_slot,
            lhs,
            rhs,
            config,
            accumulation,
            out,
        )
    }

    fn grouped_gemm_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &GroupedGemmConfig<'_>,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.providers.execute_grouped_gemm_scoped(
            &self.entry,
            self.entered.as_ref(),
            lhs,
            rhs,
            config,
            out,
        )
    }
}

impl TensorIndexing for CpuExecSession<'_> {
    // Indexing
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.run_native_fresh_with_indexed_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::gather_with_pool(
                buffers,
                cache,
                &exec_context,
                operand,
                start_indices,
                config,
            )
        })
    }
    fn scatter(
        &mut self,
        operand: &Tensor,
        indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        self.run_native_fresh_with_indexed_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::scatter_with_pool(
                buffers,
                cache,
                &exec_context,
                operand,
                indices,
                updates,
                config,
            )
        })
    }
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_slice_with_pool(buffers, &exec_context, input, config)
        })
    }
    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native_fresh_with_indexed_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::dynamic_slice_with_pool(
                buffers,
                cache,
                &exec_context,
                input,
                starts,
                slice_sizes,
            )
        })
    }
    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor> {
        self.run_native_fresh_with_indexed_context(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::dynamic_update_slice_with_pool(
                buffers,
                cache,
                &exec_context,
                operand,
                update,
                starts,
            )
        })
    }
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_pad_with_pool(buffers, &exec_context, input, config)
        })
    }
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_concatenate_with_pool(buffers, &exec_context, inputs, axis)
        })
    }
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::reverse_with_pool(buffers, &exec_context, input, axes)
        })
    }
}

impl TensorBuffer for CpuExecSession<'_> {
    fn reclaim_buffer(&mut self, tensor: Tensor) {
        match tensor {
            Tensor::F32(t) => reclaim_typed(self.buffers, t),
            Tensor::F64(t) => reclaim_typed(self.buffers, t),
            Tensor::I32(t) => reclaim_typed(self.buffers, t),
            Tensor::I64(t) => reclaim_typed(self.buffers, t),
            Tensor::Bool(t) => reclaim_typed(self.buffers, t),
            Tensor::C32(t) => reclaim_typed(self.buffers, t),
            Tensor::C64(t) => reclaim_typed(self.buffers, t),
        }
    }
}

impl TensorFusion for CpuExecSession<'_> {
    fn execute_elementwise_fusion(
        &mut self,
        inputs: &[&Tensor],
        plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        self.run_native_fresh_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            elementwise::elementwise_fusion_with_pool(buffers, &exec_context, inputs, plan)
        })
    }

    fn execute_broadcast_multiply(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<Tensor>> {
        self.run_native_fresh(|buffers| {
            elementwise::broadcast_multiply_read_with_pool(
                buffers, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
        })
    }

    fn execute_broadcast_multiply_value(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<TensorValue>> {
        let domain = self.entry.domain_id();
        self.run_native(|buffers| {
            elementwise::broadcast_multiply_value_with_pool_and_tag(
                buffers,
                lhs,
                lhs_shape,
                lhs_dims,
                rhs,
                rhs_shape,
                rhs_dims,
                |tensor| tag_fresh_output(tensor, domain),
            )
        })
    }
}

impl BackendSession for CpuExecSession<'_> {
    fn vdot_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        tenferro_tensor::backend::validate_vdot_read(&lhs, &rhs)?;
        crate::blas1::validate_cpu_read("BackendSession::vdot_read", &lhs)?;
        crate::blas1::validate_cpu_read("BackendSession::vdot_read", &rhs)?;
        let element_count = tenferro_tensor::validate::checked_shape_product(
            "BackendSession::vdot_read",
            "shape",
            lhs.shape(),
        )?;
        let (lhs, rhs, rank) = if lhs.shape().len() > 2
            && lhs.is_col_major_contiguous()?
            && rhs.is_col_major_contiguous()?
        {
            (
                flatten_compact_read(lhs, element_count)?,
                flatten_compact_read(rhs, element_count)?,
                1,
            )
        } else {
            let rank = lhs.shape().len();
            (lhs, rhs, rank)
        };
        let axes = (0..rank).collect::<Vec<_>>();
        let config = DotGeneralConfig {
            lhs_contracting_dims: axes.clone(),
            rhs_contracting_dims: axes,
            lhs_batch_dims: Vec::new(),
            rhs_batch_dims: Vec::new(),
        };
        self.execute_dot_allocated(None, lhs, rhs, &config, true, false)
    }

    fn norm_squared_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        tenferro_tensor::backend::validate_norm_squared_read(&input)?;
        crate::blas1::validate_cpu_read("BackendSession::norm_squared_read", &input)?;
        self.run_native_fresh(|buffers| reduction::norm_squared_read(buffers, input))
    }

    fn axpby_read_into_accum(
        &mut self,
        alpha: ContractionScalar,
        x: TensorRead<'_>,
        beta: ContractionScalar,
        y: TensorWrite<'_>,
    ) -> crate::Result<()> {
        tenferro_tensor::backend::validate_axpby_read_into_accum(alpha, &x, beta, &y)?;
        self.run_native_with_context(|context, buffers| {
            crate::blas1::axpby_read_into_accum(context, buffers, alpha, x, beta, y)
        })
    }

    fn session_type_id(&self) -> TypeId {
        TypeId::of::<CpuExecSessionMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

#[cfg(test)]
mod tests;
