//! Engine-kind registration for the eager runtime and the session-free host
//! read used by leaf construction.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(test)]
use std::sync::Arc;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{EngineId, EngineRegistration, ExecutionEngine, RuntimeConfigError};
#[cfg(test)]
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, ElementwiseReadOp, Engine, GatherConfig, PadConfig,
    ScatterConfig, SliceConfig, TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
    TensorWrite,
};
use tenferro_tensor::{DType, MemoryKind, Result as TensorResult, Tensor, TensorRead};

/// Copy an owned host read into a fresh compact host tensor, or `None` when the
/// CPU engine's host clone would refuse it.
///
/// This mirrors `tenferro-cpu`'s `clone_host_tensor_read` acceptance: host
/// placement and a preset scalar. Everything else — a view read, a
/// backend-family buffer, a device or managed placement, a caller-owned external
/// scalar — needs an engine, where that engine reports its own typed error.
pub(crate) fn cpu_host_owned_read(input: &TensorRead<'_>) -> Option<TensorResult<Tensor>> {
    if !matches!(input, TensorRead::Tensor(_))
        || input.backend_family().is_some()
        || !matches!(
            input.placement().memory_kind,
            MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
        )
        || matches!(input.dtype(), DType::External(_))
    {
        return None;
    }
    Some(input.clone().tensor_view().duplicate())
}

/// The runtime registration for the kind of `engine`, or `None` for an engine
/// kind the eager runtime cannot register by itself.
///
/// The registration carries only immutable preparation metadata; the engine
/// itself is never stored.
pub(crate) fn engine_kind_registration(
    engine: &mut dyn ExecutionEngine,
) -> Option<Result<EngineRegistration, RuntimeConfigError>> {
    let engine_id = engine.target().engine_id().clone();
    if let Some(cpu) = engine.as_any_mut().downcast_mut::<CpuEngine>() {
        return Some(tenferro_cpu::runtime_engine_registration_with_id(
            cpu, engine_id,
        ));
    }
    #[cfg(feature = "cuda")]
    if let Some(cuda) = engine
        .as_any_mut()
        .downcast_mut::<tenferro_gpu::cuda::CudaEngine>()
    {
        return Some(tenferro_gpu::cuda::cuda_runtime_engine_registration(cuda));
    }
    None
}

/// The typed rejection for an engine kind the eager runtime cannot register.
///
/// The engine id is part of the message because the decision is made from the
/// concrete engine type, not from the id: a foreign engine can report any id it
/// likes, and the diagnostic must say which engine was refused.
pub(crate) fn unregisterable_engine_error(engine_id: &EngineId) -> crate::Error {
    crate::Error::unsupported(
        "EagerRuntime::ensure_engine",
        crate::ErrorPhase::Execution,
        format!("engine id {engine_id:?} is not a registerable eager engine kind"),
    )
}

#[cfg(test)]
#[derive(Debug)]
pub struct RecordingEngine {
    materializations: Arc<AtomicUsize>,
    inner: CpuEngine,
}

#[cfg(test)]
impl RecordingEngine {
    /// A CPU engine that counts `to_contiguous_read` materializations.
    pub(crate) fn new(materializations: Arc<AtomicUsize>) -> Self {
        Self {
            materializations,
            inner: CpuEngine::serial(),
        }
    }
}

#[cfg(test)]
impl ExecutionEngine for RecordingEngine {
    fn target(&self) -> tenferro_runtime::EngineTarget {
        self.inner.target()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self.inner.as_any_mut()
    }
}

#[cfg(test)]
macro_rules! delegate_recording_backend_methods {
    ($(fn $method:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        $(
            fn $method(&mut self, $($arg: $ty),*) -> $ret {
                self.inner.$method($($arg),*)
            }
        )*
    };
}

#[cfg(test)]
macro_rules! delegate_owned_read_methods {
    ($(fn $method:ident($label:literal; $($read:ident),+ $(; $($extra:ident: $extra_ty:ty),*)?);)*) => {
        $(
            fn $method(
                &mut self,
                $($read: TensorRead<'_>,)+
                $($($extra: $extra_ty),*)?
            ) -> TensorResult<Tensor> {
                $(let $read = tenferro_tensor::backend::read_owned_tensor($label, $read)?;)+
                self.inner.$method($(TensorRead::from_tensor($read)),+ $(, $($extra),*)?)
            }
        )*
    };
}

#[cfg(test)]
impl Engine for RecordingEngine {
    // The gather read entry belongs to `Engine`; it forwards the reads so the
    // `to_contiguous_read` count stays a caller-staging count (#2223).
    delegate_recording_backend_methods! {
        fn gather_read(operand: TensorRead<'_>, start_indices: TensorRead<'_>, config: &GatherConfig) -> TensorResult<Tensor>;
    }
}

#[cfg(test)]
impl TensorElementwise for RecordingEngine {
    delegate_recording_backend_methods! {
        fn elementwise_read_into(op: ElementwiseReadOp, inputs: &[TensorRead<'_>], out: TensorWrite<'_>) -> TensorResult<()>;
        fn mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
    }
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor> {
        let lhs = tenferro_tensor::backend::read_owned_tensor("add", lhs)?;
        let rhs = tenferro_tensor::backend::read_owned_tensor("add", rhs)?;
        {
            self.inner
                .add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
        }
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn sub_read("sub"; lhs, rhs);
        fn neg_read("neg"; input);
        fn conj_read("conj"; input);
        fn div_read("div"; lhs, rhs);
        fn abs_read("abs"; input);
        fn sign_read("sign"; input);
        fn maximum_read("maximum"; lhs, rhs);
        fn minimum_read("minimum"; lhs, rhs);
        fn compare_read("compare"; lhs, rhs; dir: &CompareDir);
        fn select_read("select"; pred, on_true, on_false);
        fn clamp_read("clamp"; input, lower, upper);
    }
}

#[cfg(test)]
impl TensorAnalytic for RecordingEngine {
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn exp_read("exp"; input);
        fn log_read("log"; input);
        fn sin_read("sin"; input);
        fn cos_read("cos"; input);
        fn tanh_read("tanh"; input);
        fn sqrt_read("sqrt"; input);
        fn rsqrt_read("rsqrt"; input);
        fn pow_read("pow"; lhs, rhs);
        fn expm1_read("expm1"; input);
        fn log1p_read("log1p"; input);
        fn erf_read("erf"; input);
    }
}

#[cfg(test)]
impl TensorStructural for RecordingEngine {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.materializations.fetch_add(1, Ordering::Relaxed);
        self.inner.to_contiguous_read(input)
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> TensorResult<()> {
        self.inner.copy_read_into(src, dst)
    }

    delegate_recording_backend_methods! {
        fn cast(input: &Tensor, to: DType) -> TensorResult<Tensor>;
        fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn triu(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn transpose_read(input: TensorRead<'_>, perm: &[usize]) -> TensorResult<Tensor>;
        // The read entries that consume a borrowed operand's own layout. Forwarding the
        // read as-is is what the CPU engine itself does, so a caller that stages the view
        // before the hook is the only thing that can raise `to_contiguous_read` here
        // (#2223).
        fn extract_diagonal_read(input: TensorRead<'_>, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal_read(input: TensorRead<'_>, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril_read(input: TensorRead<'_>, k: i64) -> TensorResult<Tensor>;
        fn triu_read(input: TensorRead<'_>, k: i64) -> TensorResult<Tensor>;
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn reshape_read("reshape"; input; shape: &[usize]);
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected borrowed views. Reproduce it explicitly rather than
    // forwarding a view, which would widen the accepted input surface.
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn broadcast_in_dim_read("broadcast_in_dim"; input; shape: &[usize], dims: &[usize]);
    }
}

#[cfg(test)]
impl TensorReduction for RecordingEngine {
    delegate_recording_backend_methods! {
        fn reduce_sum_squares_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected views, which is not the same as forwarding a view to
    // the inner backend. Reproduce the old default explicitly.
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    delegate_owned_read_methods! {
        fn reduce_sum_read("reduce_sum"; input; axes: &[usize]);
        fn reduce_prod_read("reduce_prod"; input; axes: &[usize]);
        fn reduce_max_read("reduce_max"; input; axes: &[usize]);
        fn reduce_min_read("reduce_min"; input; axes: &[usize]);
    }
}

#[cfg(test)]
impl TensorIndexing for RecordingEngine {
    delegate_recording_backend_methods! {
        fn gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> TensorResult<Tensor>;
        fn scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> TensorResult<Tensor>;
        fn slice(input: &Tensor, config: &SliceConfig) -> TensorResult<Tensor>;
        fn dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> TensorResult<Tensor>;
        fn dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> TensorResult<Tensor>;
        fn pad(input: &Tensor, config: &PadConfig) -> TensorResult<Tensor>;
        fn concatenate(inputs: &[&Tensor], axis: usize) -> TensorResult<Tensor>;
        fn reverse(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        // The read entries that consume a borrowed operand's own layout; see the
        // structural block above for why they forward the read unchanged (#2223).
        fn scatter_read(operand: TensorRead<'_>, scatter_indices: TensorRead<'_>, updates: TensorRead<'_>, config: &ScatterConfig) -> TensorResult<Tensor>;
        fn slice_read(input: TensorRead<'_>, config: &SliceConfig) -> TensorResult<Tensor>;
        fn dynamic_slice_read(input: TensorRead<'_>, starts: TensorRead<'_>, slice_sizes: &[usize]) -> TensorResult<Tensor>;
        fn dynamic_update_slice_read(operand: TensorRead<'_>, update: TensorRead<'_>, starts: TensorRead<'_>) -> TensorResult<Tensor>;
        fn pad_read(input: TensorRead<'_>, config: &PadConfig) -> TensorResult<Tensor>;
        fn concatenate_read(inputs: &[TensorRead<'_>], axis: usize) -> TensorResult<Tensor>;
        fn reverse_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

#[cfg(test)]
impl TensorDot for RecordingEngine {
    delegate_recording_backend_methods! {
        fn dot_general_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, config: &DotGeneralConfig) -> TensorResult<Tensor>;
    }
}

#[cfg(test)]
impl TensorFusion for RecordingEngine {}
#[cfg(test)]
impl tenferro_tensor::SessionCachedDot for RecordingEngine {}
#[cfg(test)]
impl TensorBuffer for RecordingEngine {}
#[cfg(test)]
impl TensorDeviceTransfer for RecordingEngine {
    delegate_recording_backend_methods! {
        fn download_to_host(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
        fn upload_host_tensor(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

#[cfg(test)]
mod tests;
