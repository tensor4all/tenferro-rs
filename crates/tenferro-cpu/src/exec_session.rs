use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::engine::guard_unwind;
use crate::{CpuEngine, CpuPlanCache, Tensor, TensorRead, TensorValue, TensorWrite};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::backend::{ElementwiseFusionPlan, Engine};
use tenferro_tensor::{
    ActivationOp, CompareDir, ContractionScalar, DType, DotGeneralConfig, ElementwiseReadOp,
    GatherConfig, PadConfig, ScatterConfig, SliceConfig, TensorView, TypedTensor,
};
use tenferro_tensor::{
    DotGeneralAccumulation, SessionCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};

use super::backend::elementwise_read_into_fallback_with_pool;
use super::provider::CpuExecutionContext;
use super::{
    activation, analytic, copy_tensor_read_into, elementwise, indexing,
    materialize_tensor_read_in_domain, reduction, structural,
};

/// Native-engine marker for [`CpuEngine`]; private to this crate so no other
/// crate can create a token that claims to be a CPU engine.
pub(super) struct CpuEngineMarker;

/// Route an eager `dot_general` request to a slot of the engine's plan cache.
///
/// A prepared program keys a contraction plan by its instruction index, because
/// a compiled program has exactly one such index per operation. An eager call
/// has none, so this derives one from the request's geometry: the dtype, the two
/// operand shapes, the four axis lists and the conjugation flags, all of which
/// the plan key already names.
///
/// The slot only *routes*. [`crate::contraction::BinaryCache`] still compares the
/// full key - including both operand layouts, which this fingerprint deliberately
/// leaves out because reading them here would cost the very allocation the cache
/// is meant to avoid - before reusing a plan, so two requests that collide
/// rebuild instead of sharing a plan that does not apply. A collision therefore
/// costs a miss, never a wrong answer.
fn eager_dot_cache_slot(
    capacity: usize,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> Option<usize> {
    use std::hash::{Hash, Hasher};

    if capacity == 0 {
        return None;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    lhs.dtype().hash(&mut hasher);
    lhs.shape().hash(&mut hasher);
    rhs.shape().hash(&mut hasher);
    config.lhs_contracting_dims.hash(&mut hasher);
    config.rhs_contracting_dims.hash(&mut hasher);
    config.lhs_batch_dims.hash(&mut hasher);
    config.rhs_batch_dims.hash(&mut hasher);
    lhs_conj.hash(&mut hasher);
    rhs_conj.hash(&mut hasher);
    Some(hasher.finish() as usize % capacity)
}

fn pooled_zero_tensor<T>(
    buffers: &mut BufferPool,
    shape: impl tenferro_tensor::IntoShapeVec,
) -> crate::Result<TypedTensor<T>>
where
    T: PoolScalar + Clone + 'static,
{
    let shape = shape.into_shape_vec();
    let element_count =
        tenferro_tensor::validate::checked_shape_product("dot_general", "output", &shape)?;
    TypedTensor::from_vec_col_major(shape, T::pool_acquire_zeroed(buffers, element_count))
}

fn allocate_dot_output(
    buffers: &mut BufferPool,
    dtype: DType,
    shape: impl tenferro_tensor::IntoShapeVec,
) -> crate::Result<Tensor> {
    let shape = shape.into_shape_vec();
    match dtype {
        DType::F32 => pooled_zero_tensor(buffers, &shape).map(Tensor::from_typed::<f32>),
        DType::F64 => pooled_zero_tensor(buffers, &shape).map(Tensor::from_typed::<f64>),
        DType::C32 => pooled_zero_tensor(buffers, &shape)
            .map(Tensor::from_typed::<tenferro_tensor::Complex32>),
        DType::C64 => pooled_zero_tensor(buffers, &shape)
            .map(Tensor::from_typed::<tenferro_tensor::Complex64>),
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
                $view.reshape_view(&[element_count])?,
            )))
        };
    }
    /// The typed tensor behind a read adapter's tensor, or the refusal this adapter reports.
    fn read_refusal(tensor: &Tensor) -> crate::Error {
        crate::Error::unsupported_dtype(
            "flatten_compact_read",
            tensor.dtype(),
            "an externally defined payload is not a compact runtime read",
        )
    }

    match input {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => flatten!(
                F32,
                tensor
                    .as_typed::<f32>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::F64 => flatten!(
                F64,
                tensor
                    .as_typed::<f64>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::I32 => flatten!(
                I32,
                tensor
                    .as_typed::<i32>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::I64 => flatten!(
                I64,
                tensor
                    .as_typed::<i64>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::Bool => flatten!(
                Bool,
                tensor
                    .as_typed::<bool>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::C32 => flatten!(
                C32,
                tensor
                    .as_typed::<Complex32>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            DType::C64 => flatten!(
                C64,
                tensor
                    .as_typed::<Complex64>()
                    .ok_or_else(|| read_refusal(tensor))?
                    .as_view()
            ),
            // A caller-owned payload has no compact runtime read.
            DType::External(_) => Err(read_refusal(tensor)),
        },
        TensorRead::View(TensorView::F32(view)) => flatten!(F32, view),
        TensorRead::View(TensorView::F64(view)) => flatten!(F64, view),
        TensorRead::View(TensorView::I32(view)) => flatten!(I32, view),
        TensorRead::View(TensorView::I64(view)) => flatten!(I64, view),
        TensorRead::View(TensorView::Bool(view)) => flatten!(Bool, view),
        TensorRead::View(TensorView::C32(view)) => flatten!(C32, view),
        TensorRead::View(TensorView::C64(view)) => flatten!(C64, view),
    }
}

impl CpuEngine {
    /// Replace this engine's effective batch policy, returning the previous one.
    pub(crate) fn replace_batch_policy(
        &mut self,
        policy: crate::CpuBatchPolicy,
    ) -> crate::CpuBatchPolicy {
        self.set_batch_policy(policy)
    }

    fn run_native<R: Send>(
        &mut self,
        op: impl FnOnce(&mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        self.check_entry("CpuEngine")?;
        let (context, buffers, _, broken) = self.parts();
        guard_unwind(buffers, broken, |buffers| {
            context.with_native_parallelism(|| op(buffers))
        })
    }

    #[cfg(feature = "native")]
    pub(crate) fn with_faer_parallelism(
        &mut self,
        callback: impl FnOnce(faer::Par) -> crate::Result<()> + Send,
    ) -> crate::Result<()> {
        self.check_entry("CpuEngine")?;
        let (context, buffers, _, broken) = self.parts();
        guard_unwind(buffers, broken, |_| context.with_faer_pool(callback))
    }

    fn run_native_with_context<R: Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> crate::Result<R> + Send,
    ) -> crate::Result<R> {
        self.check_entry("CpuEngine")?;
        let (context, buffers, _, broken) = self.parts();
        guard_unwind(buffers, broken, |buffers| {
            context.with_native_parallelism(|| op(&context, buffers))
        })
    }

    fn run_native_with_plans<R: Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool, &CpuPlanCache) -> crate::Result<R>
            + Send,
    ) -> crate::Result<R> {
        self.check_entry("CpuEngine")?;
        let (context, buffers, plans, broken) = self.parts();
        guard_unwind(buffers, broken, |buffers| {
            context.with_native_parallelism(|| op(&context, buffers, plans))
        })
    }

    /// Run a contraction-provider call with the engine's token, buffers and
    /// plans; the provider decides its own parallelism from the token.
    fn run_contraction<R>(
        &mut self,
        op: impl FnOnce(&cpueinsum::Exec<'_>, &mut BufferPool, &CpuPlanCache) -> crate::Result<R>,
    ) -> crate::Result<R> {
        self.check_entry("dot_general")?;
        let context = CpuExecutionContext::new(&self.resources, self.batch_policy);
        let _retention = self
            .resources
            .retention_guard(self.buffers.max_retained_capacity_bytes());
        let exec = context.contraction_exec()?;
        let plans = &self.plans;
        guard_unwind(&mut self.buffers, &mut self.broken, |buffers| {
            op(&exec, buffers, plans)
        })
    }
}

impl TensorDeviceTransfer for CpuEngine {
    fn download_to_host(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        self.check_entry("CpuEngine::download_to_host")?;
        if tensor.backend_family().is_some() {
            return Err(crate::Error::runtime_state(
                "CpuEngine::download_to_host",
                "CPU backend received a backend buffer; download the tensor to host with its owning backend before CPU execution",
            ));
        }
        tensor.tensor_view().duplicate()
    }

    fn upload_host_tensor(&mut self, tensor: TensorRead<'_>) -> crate::Result<Tensor> {
        self.check_entry("CpuEngine::upload_host_tensor")?;
        if tensor.backend_family().is_some() {
            return Err(crate::Error::runtime_state(
                "CpuEngine::upload_host_tensor",
                "CPU backend upload_host_tensor expects a host tensor; download backend buffers to host before CPU execution",
            ));
        }
        tensor.tensor_view().duplicate()
    }
}

/// Delegation for operations whose outputs can be allocated from the session pool.
macro_rules! delegate_with_pool {
    ($name:ident($($arg:ident : $ty:ty),*) => $callee:path) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> {
            self.run_native(|buffers| $callee(buffers, $($arg),*))
        }
    };
}

/// Delegation for pooled operations whose kernels take the session's strided
/// execution context.
macro_rules! delegate_with_pool_context {
    ($name:ident($($arg:ident : $ty:ty),*) => $callee:path) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> {
            self.run_native_with_context(|context, buffers| {
                $callee(buffers, &context.strided_exec_context(), $($arg),*)
            })
        }
    };
}

impl TensorElementwise for CpuEngine {
    // Elementwise — direct delegation within the current session scope.
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            tenferro_internal_cpu_kernels::elementwise_read_into_with_context(
                op,
                inputs,
                out,
                &exec_context,
                |inputs, out| {
                    elementwise_read_into_fallback_with_pool(
                        buffers,
                        &exec_context,
                        op,
                        inputs,
                        out,
                    )
                },
            )
        })
    }

    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            elementwise::add_read_with_pool(buffers, &context.strided_exec_context(), lhs, rhs)
        })
    }

    delegate_with_pool_context!(sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::sub_read_with_pool);
    delegate_with_pool_context!(mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::mul_read_with_pool);
    delegate_with_pool_context!(neg_read(input: TensorRead<'_>) => elementwise::neg_read_with_pool);
    delegate_with_pool_context!(conj_read(input: TensorRead<'_>) => elementwise::conj_read_with_pool);
    delegate_with_pool_context!(div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::div_read_with_pool);
    delegate_with_pool_context!(rem(lhs: &Tensor, rhs: &Tensor) => elementwise::rem_with_pool);
    delegate_with_pool_context!(rem_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::rem_read_with_pool);
    delegate_with_pool_context!(abs_read(input: TensorRead<'_>) => elementwise::abs_read_with_pool);
    delegate_with_pool_context!(sign_read(input: TensorRead<'_>) => elementwise::sign_read_with_pool);
    delegate_with_pool_context!(maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::maximum_read_with_pool);
    delegate_with_pool_context!(minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::minimum_read_with_pool);
    delegate_with_pool_context!(compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) => elementwise::compare_read_with_pool);
    delegate_with_pool_context!(select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) => elementwise::select_read_with_pool);
    delegate_with_pool_context!(clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) => elementwise::clamp_read_with_pool);
}

impl TensorAnalytic for CpuEngine {
    // Analytic
    delegate_with_pool!(exp_read(input: TensorRead<'_>) => analytic::exp_read_with_pool);
    delegate_with_pool!(log_read(input: TensorRead<'_>) => analytic::log_read_with_pool);
    delegate_with_pool!(sin_read(input: TensorRead<'_>) => analytic::sin_read_with_pool);
    delegate_with_pool!(cos_read(input: TensorRead<'_>) => analytic::cos_read_with_pool);
    delegate_with_pool!(tanh_read(input: TensorRead<'_>) => analytic::tanh_read_with_pool);
    delegate_with_pool!(sqrt_read(input: TensorRead<'_>) => analytic::sqrt_read_with_pool);
    delegate_with_pool!(rsqrt_read(input: TensorRead<'_>) => analytic::rsqrt_read_with_pool);
    delegate_with_pool!(pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => analytic::pow_read_with_pool);
    delegate_with_pool!(expm1_read(input: TensorRead<'_>) => analytic::expm1_read_with_pool);
    delegate_with_pool!(log1p_read(input: TensorRead<'_>) => analytic::log1p_read_with_pool);
    delegate_with_pool!(erf_read(input: TensorRead<'_>) => analytic::erf_read_with_pool);
}

impl TensorStructural for CpuEngine {
    // Structural
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        // Preserve the host-compact entry policy. Managed copies must enter
        // the configured execution scope for provider mapping and allocation,
        // just as views enter it for strided materialization.
        if input.as_tensor().is_none() {
            // Only a view is actually copied here; an owned read passes through.
            self.resources.count_read_staging();
        }
        let domain = self.shared_allocation_domain().cloned();
        if matches!(input, TensorRead::Tensor(_)) && input.backend_family().is_none() {
            self.check_entry("CpuEngine::to_contiguous_read")?;
            return materialize_tensor_read_in_domain(
                &mut self.buffers,
                "CpuEngine::to_contiguous_read",
                input,
                domain.as_deref(),
            );
        }
        self.run_native(|buffers| {
            materialize_tensor_read_in_domain(
                buffers,
                "CpuEngine::to_contiguous_read",
                input,
                domain.as_deref(),
            )
        })
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> crate::Result<()> {
        self.run_native(|_| copy_tensor_read_into("CpuEngine::copy_read_into", src, dst))
    }

    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|buffers| structural::transpose_read_with_pool(buffers, input, perm))
    }

    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        self.check_entry("CpuEngine::reshape_read")?;
        if let TensorRead::Tensor(tensor) = &input {
            // INVARIANT: a compact input keeps the serial host copy and its typed
            // refusal for an external payload, without the engine entry or a
            // pool buffer, unless a parallel-capable context can run the copy
            // above the strided scheduler's parallel threshold. Such a copy
            // materializes on the pool; `run_native` (not `run_native_fresh`)
            // keeps the input placement that the copy puts on the output, as the
            // reshape contract requires.
            let parallel = self.thread_budget().get() > 1;
            if !structural::reshape_uses_pool(tensor, parallel)? {
                return structural::reshape(tensor, shape);
            }
            return self
                .run_native(|buffers| structural::reshape_read_with_pool(buffers, input, shape));
        }
        self.run_native(|buffers| structural::reshape_read_with_pool(buffers, input, shape))
    }

    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native(|buffers| {
            structural::broadcast_in_dim_read_with_pool(buffers, input, shape, dims)
        })
    }

    fn tril_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        k: i64,
    ) -> crate::Result<Tensor> {
        self.run_native(|buffers| structural::tril_read_with_pool(buffers, input, k))
    }

    fn triu_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        k: i64,
    ) -> crate::Result<Tensor> {
        self.run_native(|buffers| structural::triu_read_with_pool(buffers, input, k))
    }

    fn extract_diagonal_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.run_native(|buffers| {
            structural::extract_diagonal_read_with_pool(buffers, input, axis_a, axis_b)
        })
    }

    fn embed_diagonal_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.run_native(|buffers| {
            structural::embed_diagonal_read_with_pool(buffers, input, axis_a, axis_b)
        })
    }

    delegate_with_pool!(cast(input: &Tensor, to: crate::DType) => structural::cast_with_pool);
    delegate_with_pool!(extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::extract_diagonal_with_pool);
    delegate_with_pool!(embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::embed_diagonal_with_pool);
    delegate_with_pool!(tril(input: &Tensor, k: i64) => structural::tril_with_pool);
    delegate_with_pool!(triu(input: &Tensor, k: i64) => structural::triu_with_pool);
}

impl TensorReduction for CpuEngine {
    // Reduction

    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_sum_squares_read(
        &mut self,
        input: TensorRead<'_>,
        axes: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_sum_squares_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_prod_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_max_read(buffers, input, axes, &exec_context)
        })
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            reduction::reduce_min_read(buffers, input, axes, &exec_context)
        })
    }
}

impl CpuEngine {
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
        let accumulation = DotGeneralAccumulation {
            lhs_conj,
            rhs_conj,
            alpha: ContractionScalar::one(dtype)?,
            beta: ContractionScalar::zero(dtype)?,
        };
        if matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64) {
            return self.run_contraction(|exec, buffers, plans| {
                crate::contraction::dot_fresh(
                    exec,
                    buffers,
                    plans,
                    cache_slot,
                    lhs,
                    rhs,
                    config,
                    accumulation,
                    output_shape,
                )
            });
        }
        let mut output = allocate_dot_output(&mut self.buffers, dtype, output_shape)?;
        self.dot_general_read_into_accum_cached(
            cache_slot,
            lhs,
            rhs,
            config,
            accumulation,
            TensorWrite::from_tensor(&mut output),
        )?;
        Ok(output)
    }
}

impl TensorDot for CpuEngine {
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        let slot = eager_dot_cache_slot(
            self.plans.contraction_capacity(),
            &lhs,
            &rhs,
            config,
            false,
            false,
        );
        self.execute_dot_allocated(slot, lhs, rhs, config, false, false)
    }

    fn dot_general_with_conj_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        // The contraction lane dispatches on the read's view and consumes its
        // strides, so a conjugated operand needs no staging copy here (#2214).
        let slot = eager_dot_cache_slot(
            self.plans.contraction_capacity(),
            &lhs,
            &rhs,
            config,
            lhs_conj,
            rhs_conj,
        );
        self.execute_dot_allocated(slot, lhs, rhs, config, lhs_conj, rhs_conj)
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
        let slot = eager_dot_cache_slot(
            self.plans.contraction_capacity(),
            &lhs,
            &rhs,
            config,
            accumulation.lhs_conj,
            accumulation.rhs_conj,
        );
        self.dot_general_read_into_accum_cached(slot, lhs, rhs, config, accumulation, out)
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        let lhs = TensorRead::from_tensor(lhs);
        let rhs = TensorRead::from_tensor(rhs);
        let slot = eager_dot_cache_slot(
            self.plans.contraction_capacity(),
            &lhs,
            &rhs,
            config,
            lhs_conj,
            rhs_conj,
        );
        self.execute_dot_allocated(slot, lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

impl SessionCachedDot for CpuEngine {
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

    fn dot_general_read_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        // The caller's slot reaches the same plan cache (keyed by layout, config
        // and conjugation), so a view operand is consumed instead of staged
        // (#2214). Do not copy CUDA's policy of ignoring the slot.
        self.execute_dot_allocated(cache_slot, lhs, rhs, config, false, false)
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

    fn dot_general_with_conj_read_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.execute_dot_allocated(cache_slot, lhs, rhs, config, lhs_conj, rhs_conj)
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
        if matches!(
            lhs.dtype(),
            DType::F32 | DType::F64 | DType::C32 | DType::C64
        ) {
            return self.run_contraction(|exec, buffers, plans| {
                crate::contraction::dot_into(
                    exec,
                    buffers,
                    plans,
                    cache_slot,
                    lhs,
                    rhs,
                    config,
                    accumulation,
                    out,
                )
            });
        }
        // Only floating and complex contractions have a CPU numerical route;
        // integer and Bool operands are rejected before the output is touched.
        Err(crate::Error::unsupported_dtype(
            "dot_general",
            lhs.dtype(),
            crate::cpu_contraction_unsupported_dtype_message(lhs.dtype()),
        ))
    }
}

impl TensorIndexing for CpuEngine {
    // Indexing
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.run_native_with_plans(|context, buffers, cache| {
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
        self.run_native_with_plans(|context, buffers, cache| {
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
    fn scatter_read(
        &mut self,
        operand: tenferro_tensor::TensorRead<'_>,
        indices: tenferro_tensor::TensorRead<'_>,
        updates: tenferro_tensor::TensorRead<'_>,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        self.run_native_with_plans(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::scatter_read_with_pool(
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
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_slice_with_pool(buffers, &exec_context, input, config)
        })
    }
    fn slice_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        config: &SliceConfig,
    ) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::slice_read_with_pool(buffers, &exec_context, input, config)
        })
    }
    fn dynamic_slice_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        starts: tenferro_tensor::TensorRead<'_>,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native_with_plans(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::dynamic_slice_read_with_pool(
                buffers,
                cache,
                &exec_context,
                input,
                starts,
                slice_sizes,
            )
        })
    }
    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native_with_plans(|context, buffers, cache| {
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
        self.run_native_with_plans(|context, buffers, cache| {
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
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_pad_with_pool(buffers, &exec_context, input, config)
        })
    }
    fn pad_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        config: &PadConfig,
    ) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::pad_read_with_pool(buffers, &exec_context, input, config)
        })
    }
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::try_concatenate_with_pool(buffers, &exec_context, inputs, axis)
        })
    }
    fn concatenate_read(
        &mut self,
        inputs: &[tenferro_tensor::TensorRead<'_>],
        axis: usize,
    ) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::concatenate_read_with_pool(buffers, &exec_context, inputs, axis)
        })
    }
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::reverse_with_pool(buffers, &exec_context, input, axes)
        })
    }
    fn reverse_read(
        &mut self,
        input: tenferro_tensor::TensorRead<'_>,
        axes: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            indexing::reverse_read_with_pool(buffers, &exec_context, input, axes)
        })
    }
}

/// Typed view canonicalization runs on the engine.
impl<T, R> tenferro_tensor::TensorViewCanonicalization<T, R> for CpuEngine
where
    T: tenferro_tensor::TensorScalar + PoolScalar,
    R: tenferro_tensor::TensorRank,
    R::Shape: Send + Sync,
    R::Strides: Send + Sync,
{
    fn to_contiguous(
        &mut self,
        view: &tenferro_tensor::TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>> {
        self.run_native(|buffers| {
            crate::structural::typed_materialize_view_with_pool(
                buffers,
                view,
                "CpuEngine::to_contiguous",
            )
        })
    }

    fn copy_into(
        &mut self,
        src: &tenferro_tensor::TypedTensorView<'_, T, R>,
        dst: &mut tenferro_tensor::TypedTensorViewMut<'_, T, R>,
    ) -> crate::Result<()> {
        self.run_native(|_| {
            crate::structural::typed_copy_view_into(src, dst, "CpuEngine::copy_into")
        })
    }
}

impl TensorBuffer for CpuEngine {
    fn reclaim_buffer(&mut self, tensor: Tensor) {
        crate::backend::reclaim_tensor(&mut self.buffers, tensor);
    }
}

impl TensorFusion for CpuEngine {
    fn execute_elementwise_fusion(
        &mut self,
        inputs: &[&Tensor],
        plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        self.run_native_with_context(|context, buffers| {
            let exec_context = context.strided_exec_context();
            tenferro_cpu_fused::elementwise_fusion_with_pool(buffers, &exec_context, inputs, plan)
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
        self.run_native_with_context(|context, buffers| {
            elementwise::broadcast_multiply_read_with_pool(
                buffers,
                &context.strided_exec_context(),
                lhs,
                lhs_shape,
                lhs_dims,
                rhs,
                rhs_shape,
                rhs_dims,
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
        self.run_native_with_context(|context, buffers| {
            elementwise::broadcast_multiply_value_with_pool_and_tag(
                buffers,
                &context.strided_exec_context(),
                lhs,
                lhs_shape,
                lhs_dims,
                rhs,
                rhs_shape,
                rhs_dims,
                |_| {},
            )
        })
    }
}

impl Engine for CpuEngine {
    fn gather_read(
        &mut self,
        operand: tenferro_tensor::TensorRead<'_>,
        start_indices: tenferro_tensor::TensorRead<'_>,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.run_native_with_plans(|context, buffers, cache| {
            let exec_context = context.strided_exec_context();
            indexing::gather_read_with_pool(
                buffers,
                cache,
                &exec_context,
                operand,
                start_indices,
                config,
            )
        })
    }
    fn vdot_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        tenferro_tensor::backend::validate_vdot_read(&lhs, &rhs)?;
        crate::blas1::validate_cpu_read("Engine::vdot_read", &lhs)?;
        crate::blas1::validate_cpu_read("Engine::vdot_read", &rhs)?;
        let element_count = tenferro_tensor::validate::checked_shape_product(
            "Engine::vdot_read",
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
        let axes = (0..rank).collect::<tinyvec::TinyVec<[usize; 4]>>();
        let config = DotGeneralConfig {
            lhs_contracting_dims: axes.clone(),
            rhs_contracting_dims: axes,
            lhs_batch_dims: Default::default(),
            rhs_batch_dims: Default::default(),
        };
        self.execute_dot_allocated(None, lhs, rhs, &config, true, false)
    }

    fn norm_squared_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        tenferro_tensor::backend::validate_norm_squared_read(&input)?;
        crate::blas1::validate_cpu_read("Engine::norm_squared_read", &input)?;
        self.run_native(|buffers| reduction::norm_squared_read(buffers, input))
    }

    fn fused_activation_read(
        &mut self,
        op: ActivationOp,
        input: TensorRead<'_>,
    ) -> crate::Result<Option<Tensor>> {
        crate::blas1::validate_cpu_read(op.label(), &input)?;
        self.run_native(|buffers| activation::fused_activation_read_with_pool(buffers, op, input))
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

    fn native_engine(&mut self) -> Option<tenferro_tensor::NativeSessionRef<'_>> {
        // SAFETY: `CpuEngineMarker` is private to this crate, and this is the
        // only place a token carrying it is created; it always points to a
        // `CpuEngine`, exclusively borrowed for the token lifetime.
        Some(unsafe { tenferro_tensor::NativeSessionRef::new::<CpuEngineMarker, _>(self) })
    }
}

#[cfg(test)]
mod tests;
