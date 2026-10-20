use std::any::Any;

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{
    EngineId, EngineTarget, ErrorPhase, ExecutionContextIdentity, ExecutionEngine,
};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, Engine, ErrorKind, GatherConfig,
    PadConfig, ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBuffer,
    TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead,
    TensorReduction, TensorStructural, TensorWrite,
};

use super::{engine_kind_registration, unregisterable_engine_error};

type TensorResult<T> = tenferro_tensor::Result<T>;

/// An execution engine of a kind the eager runtime does not know.
#[derive(Debug)]
struct ForeignEngine {
    target: EngineTarget,
}

impl ExecutionEngine for ForeignEngine {
    fn target(&self) -> EngineTarget {
        self.target.clone()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

macro_rules! unused_engine_methods {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $ty),*) -> $ret {
                $(let _ = &$arg;)*
                panic!(concat!(stringify!($name), " is not used by the foreign-engine test"))
            }
        )+
    };
}

impl TensorElementwise for ForeignEngine {
    unused_engine_methods! {
        fn elementwise_read_into(op: ElementwiseReadOp, inputs: &[TensorRead<'_>], out: TensorWrite<'_>) -> TensorResult<()>;
        fn add_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn neg_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn conj_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn abs_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sign_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) -> TensorResult<Tensor>;
        fn select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> TensorResult<Tensor>;
        fn clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

impl TensorAnalytic for ForeignEngine {
    unused_engine_methods! {
        fn exp_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn log_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sin_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn cos_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn tanh_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn sqrt_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn rsqrt_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult<Tensor>;
        fn expm1_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn log1p_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn erf_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

impl TensorStructural for ForeignEngine {
    unused_engine_methods! {
        fn to_contiguous_read(input: TensorRead<'_>) -> TensorResult<Tensor>;
        fn cast(input: &Tensor, to: DType) -> TensorResult<Tensor>;
        fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult<Tensor>;
        fn tril(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn triu(input: &Tensor, k: i64) -> TensorResult<Tensor>;
        fn transpose_read(input: TensorRead<'_>, perm: &[usize]) -> TensorResult<Tensor>;
        fn reshape_read(input: TensorRead<'_>, shape: &[usize]) -> TensorResult<Tensor>;
        fn broadcast_in_dim_read(input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> TensorResult<Tensor>;
    }
}

impl TensorReduction for ForeignEngine {
    unused_engine_methods! {
        fn reduce_sum_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_prod_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_max_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_min_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

impl TensorIndexing for ForeignEngine {
    unused_engine_methods! {
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

impl TensorDot for ForeignEngine {
    unused_engine_methods! {
        fn dot_general_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, config: &DotGeneralConfig) -> TensorResult<Tensor>;
    }
}

impl TensorFusion for ForeignEngine {}
impl TensorBuffer for ForeignEngine {}

impl TensorDeviceTransfer for ForeignEngine {
    unused_engine_methods! {
        fn download_to_host(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
        fn upload_host_tensor(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

impl tenferro_tensor::SessionCachedDot for ForeignEngine {}
impl Engine for ForeignEngine {}

#[test]
fn cpu_engine_registration_carries_the_engine_id_and_cpu_identity() {
    let engine_id = EngineId::new("tenferro.test.ad.cpu.v1").expect("engine id");
    let mut engine = CpuEngine::serial().with_runtime_engine_id(engine_id.clone());

    let registration = engine_kind_registration(&mut engine)
        .expect("the CPU engine kind is registerable")
        .expect("the CPU registration is valid");
    assert_eq!(registration.engine_id(), &engine_id);
    assert_eq!(
        registration.context_identity(),
        ExecutionContextIdentity::of::<CpuEngine>()
    );
}

#[test]
fn a_foreign_engine_kind_is_refused_with_a_typed_error() {
    let engine_id = EngineId::new("tenferro.test.ad.foreign.v1").expect("engine id");
    let mut engine = ForeignEngine {
        target: EngineTarget::new(
            engine_id.clone(),
            ExecutionContextIdentity::of::<ForeignEngine>(),
            None,
        ),
    };

    assert!(
        engine_kind_registration(&mut engine).is_none(),
        "the eager runtime has no registration for a foreign engine kind"
    );

    let error = unregisterable_engine_error(&engine_id);
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.phase(), Some(ErrorPhase::Execution));
    assert!(error.to_string().contains(engine_id.as_str()));
}
