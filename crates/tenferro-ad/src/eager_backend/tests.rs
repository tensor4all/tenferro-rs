//! Module-local eager backend fixtures.
//!
//! Keep all counters here: production eager dispatch must remain uninstrumented.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::*;

#[derive(Debug)]
pub struct RecordingBackend {
    materializations: Arc<AtomicUsize>,
    session_entries: Arc<AtomicUsize>,
    inner: CpuBackend,
}

impl EagerBackend {
    pub(crate) fn recording_cpu(materializations: Arc<AtomicUsize>) -> Self {
        Self::recording_cpu_with_sessions(materializations, Arc::new(AtomicUsize::new(0)))
    }

    pub(crate) fn recording_cpu_with_sessions(
        materializations: Arc<AtomicUsize>,
        session_entries: Arc<AtomicUsize>,
    ) -> Self {
        Self::Recording(RecordingBackend {
            materializations,
            session_entries,
            inner: CpuBackend::new(),
        })
    }
}

macro_rules! delegate_recording_backend_methods {
    ($(fn $method:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        $(
            fn $method(&mut self, $($arg: $ty),*) -> $ret {
                self.inner.$method($($arg),*)
            }
        )*
    };
}

impl BackendRuntimeCache for RecordingBackend {
    type RuntimeCache = ();
}

impl TensorElementwise for RecordingBackend {
    delegate_recording_backend_methods! {
        fn add(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn sub(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn mul(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn neg(input: &Tensor) -> TensorResult<Tensor>;
        fn conj(input: &Tensor) -> TensorResult<Tensor>;
        fn div(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn abs(input: &Tensor) -> TensorResult<Tensor>;
        fn sign(input: &Tensor) -> TensorResult<Tensor>;
        fn maximum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn minimum(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult<Tensor>;
        fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult<Tensor>;
        fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult<Tensor>;
    }
}

impl TensorAnalytic for RecordingBackend {
    delegate_recording_backend_methods! {
        fn exp(input: &Tensor) -> TensorResult<Tensor>;
        fn log(input: &Tensor) -> TensorResult<Tensor>;
        fn sin(input: &Tensor) -> TensorResult<Tensor>;
        fn cos(input: &Tensor) -> TensorResult<Tensor>;
        fn tanh(input: &Tensor) -> TensorResult<Tensor>;
        fn sqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn rsqrt(input: &Tensor) -> TensorResult<Tensor>;
        fn pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult<Tensor>;
        fn expm1(input: &Tensor) -> TensorResult<Tensor>;
        fn log1p(input: &Tensor) -> TensorResult<Tensor>;
    }
}

impl TensorStructural for RecordingBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        self.materializations.fetch_add(1, Ordering::Relaxed);
        self.inner.to_contiguous_read(input)
    }

    fn copy_read_into(&mut self, src: TensorRead<'_>, dst: TensorWrite<'_>) -> TensorResult<()> {
        self.inner.copy_read_into(src, dst)
    }

    delegate_recording_backend_methods! {
        fn transpose(input: &Tensor, perm: &[usize]) -> TensorResult<Tensor>;
        fn reshape(input: &Tensor, shape: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
        fn cast(input: &Tensor, to: DType) -> TensorResult<Tensor>;
        fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn triu(input: &Tensor, k: i64) -> TensorResult<Tensor>;
    }
}

impl TensorReduction for RecordingBackend {
    delegate_recording_backend_methods! {
        fn reduce_sum(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

impl TensorIndexing for RecordingBackend {
    delegate_recording_backend_methods! {
        fn gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> TensorResult<Tensor>;
        fn scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> TensorResult<Tensor>;
        fn slice(input: &Tensor, config: &SliceConfig) -> TensorResult<Tensor>;
        fn dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> TensorResult<Tensor>;
        fn dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> TensorResult<Tensor>;
        fn pad(input: &Tensor, config: &PadConfig) -> TensorResult<Tensor>;
        fn concatenate(inputs: &[&Tensor], axis: usize) -> TensorResult<Tensor>;
        fn reverse(input: &Tensor, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

impl TensorDot for RecordingBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> TensorResult<Tensor> {
        self.inner.dot_general(lhs, rhs, config)
    }
}

impl TensorFusion for RecordingBackend {}
impl TensorBuffer for RecordingBackend {}
impl TensorDeviceTransfer for RecordingBackend {}
impl BackendCachedDot for RecordingBackend {}
impl BackendSessionHost for RecordingBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        self.session_entries.fetch_add(1, Ordering::Relaxed);
        f(self)
    }
}
impl TensorBackend for RecordingBackend {}

#[test]
fn production_module_contains_no_recording_fixture_implementation() {
    let source = include_str!("../eager_backend.rs");
    for forbidden in [
        "struct RecordingBackend",
        "delegate_recording_backend_methods",
        "impl TensorElementwise for RecordingBackend",
    ] {
        assert!(
            !source.contains(forbidden),
            "production fixture leak: {forbidden}"
        );
    }
}
