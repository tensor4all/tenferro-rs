use std::any::Any;
use std::num::NonZeroU64;
use std::sync::Arc;

use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, Engine, GatherConfig, PadConfig,
    ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead, TensorReduction,
    TensorStructural, TensorWrite,
};

use super::super::{
    EngineId, EngineTarget, EventDomainId, EventDomainRun, ExecutionContextIdentity,
    ExecutionEngine, ImmediateEventDomainDriver, Output, RegistrationIdentity, RuntimeEpoch,
    RuntimeId,
};

type TensorResult<T> = tenferro_tensor::Result<T>;

/// A test engine that keeps every [`ExecutionEngine`] hook at its default.
///
/// Only the structural copy hook is real and can be made to fail; every other
/// [`Engine`] operation is unreachable because these tests exercise the
/// runtime-facing defaults, not tensor kernels.
#[derive(Debug)]
struct HookEngine {
    fail_copy: bool,
}

impl HookEngine {
    fn new() -> Self {
        Self { fail_copy: false }
    }

    fn with_failing_copy() -> Self {
        Self { fail_copy: true }
    }
}

impl ExecutionEngine for HookEngine {
    fn target(&self) -> EngineTarget {
        EngineTarget::new(
            EngineId::new("tenferro.test.engine.hooks.v1").expect("engine id"),
            ExecutionContextIdentity::of::<HookEngine>(),
            None,
        )
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
                panic!(concat!(stringify!($name), " is not used by the engine hook tests"))
            }
        )+
    };
}

impl TensorElementwise for HookEngine {
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> TensorResult<()> {
        let _ = (op, inputs, out);
        unreachable!("the elementwise into hook is not used by the engine hook tests")
    }

    unused_engine_methods! {
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

impl TensorAnalytic for HookEngine {
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

impl TensorStructural for HookEngine {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult<Tensor> {
        if self.fail_copy {
            return Err(tenferro_tensor::Error::unsupported(
                "HookEngine::to_contiguous_read",
                "injected copy failure",
            ));
        }
        input.clone().tensor_view().duplicate()
    }

    unused_engine_methods! {
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

impl TensorReduction for HookEngine {
    unused_engine_methods! {
        fn reduce_sum_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_prod_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_max_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
        fn reduce_min_read(input: TensorRead<'_>, axes: &[usize]) -> TensorResult<Tensor>;
    }
}

impl TensorIndexing for HookEngine {
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

impl TensorDot for HookEngine {
    unused_engine_methods! {
        fn dot_general_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, config: &DotGeneralConfig) -> TensorResult<Tensor>;
    }
}

impl TensorFusion for HookEngine {}
impl TensorBuffer for HookEngine {}

impl TensorDeviceTransfer for HookEngine {
    unused_engine_methods! {
        fn download_to_host(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
        fn upload_host_tensor(tensor: TensorRead<'_>) -> TensorResult<Tensor>;
    }
}

impl tenferro_tensor::SessionCachedDot for HookEngine {}
impl Engine for HookEngine {}

fn test_domain(value: u64) -> EventDomainId {
    EventDomainId::runtime_created_for_test(
        RuntimeId::from_nonzero(NonZeroU64::new(value).expect("runtime id")),
        RuntimeEpoch::from_nonzero(NonZeroU64::new(value + 1).expect("runtime epoch")),
        RegistrationIdentity::new(
            NonZeroU64::new(value + 2).expect("registration issuer"),
            NonZeroU64::new(value + 3).expect("registration ordinal"),
        ),
    )
}

#[test]
fn engine_target_accessors_expose_the_allocation_domain() {
    let domain = tenferro_tensor::AllocationDomainId::fresh();
    let engine_id = EngineId::new("tenferro.test.engine.hooks.v1").expect("engine id");
    let target = EngineTarget::new(
        engine_id.clone(),
        ExecutionContextIdentity::of::<HookEngine>(),
        Some(domain),
    );

    assert_eq!(target.engine_id(), &engine_id);
    assert_eq!(
        target.context_identity(),
        ExecutionContextIdentity::of::<HookEngine>()
    );
    assert_eq!(target.allocation_domain(), Some(domain));
}

#[test]
fn execution_engine_defaults_keep_no_caches_and_drain_immediately() {
    let mut engine = HookEngine::new();

    let caches = engine.take_extension_caches();
    assert!(caches.is_empty());
    engine.restore_extension_caches(caches);
    engine.mark_broken();

    let driver = engine.event_domain_driver();
    let domain = test_domain(11);
    let run = driver.begin_run(domain).expect("immediate run");
    assert_eq!(run.domain(), domain);
    engine.retire_run(run).expect("immediate run drains");
}

#[derive(Debug)]
struct FailingRun {
    domain: EventDomainId,
}

impl EventDomainRun for FailingRun {
    fn domain(&self) -> EventDomainId {
        self.domain
    }

    fn enqueue(
        &mut self,
        _dependencies: &[Arc<dyn super::super::EventToken>],
        _launch: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Arc<dyn super::super::EventToken>> {
        unreachable!("the failing run is only retired")
    }

    fn drain(&mut self) -> crate::Result<()> {
        Err(crate::Error::runtime_state(
            "engine-hook-test-failing-run",
            crate::ErrorPhase::Execution,
            "injected drain failure",
        ))
    }
}

#[test]
fn retire_run_propagates_the_run_drain_error() {
    let mut engine = HookEngine::new();
    let error = engine
        .retire_run(Box::new(FailingRun {
            domain: test_domain(21),
        }))
        .expect_err("the injected drain failure must propagate");
    assert!(matches!(
        error,
        crate::Error::RuntimeState {
            op: "engine-hook-test-failing-run",
            message,
            ..
        } if message == "injected drain failure"
    ));
}

#[test]
fn output_read_into_owned_and_debug_cover_both_variants() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("tensor");
    let borrowed = Output::Borrowed(TensorRead::from_tensor(&tensor));
    assert_eq!(borrowed.read().shape(), &[2]);
    assert!(borrowed.into_owned().is_none());

    let owned =
        Output::Owned(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).expect("owned tensor"));
    assert_eq!(owned.read().shape(), &[1]);
    assert_eq!(
        owned
            .into_owned()
            .expect("owned output")
            .as_slice::<f64>()
            .expect("f64"),
        &[3.0]
    );

    let owned_debug = format!(
        "{:?}",
        Output::Owned(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("tensor"))
    );
    assert!(owned_debug.contains("Output::Owned"));
    assert!(owned_debug.contains("shape"));
    let borrowed_debug = format!("{:?}", Output::Borrowed(TensorRead::from_tensor(&tensor)));
    assert!(borrowed_debug.contains("Output::Borrowed"));
    assert!(borrowed_debug.contains("shape"));
}

#[test]
fn output_into_tensor_copies_borrowed_outputs_and_propagates_copy_errors() {
    let mut engine = HookEngine::new();
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("tensor");

    let copied = Output::Borrowed(TensorRead::from_tensor(&tensor))
        .into_tensor(&mut engine)
        .expect("borrowed output copies through the engine");
    assert_eq!(copied.as_slice::<f64>().expect("f64"), &[1.0, 2.0]);

    let owned = Output::Owned(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("owned tensor"),
    )
    .into_tensor(&mut engine)
    .expect("owned output needs no copy");
    assert_eq!(owned.as_slice::<f64>().expect("f64"), &[1.0, 2.0]);

    let mut failing = HookEngine::with_failing_copy();
    let error = Output::Borrowed(TensorRead::from_tensor(&tensor))
        .into_tensor(&mut failing)
        .expect_err("the engine copy failure must propagate");
    assert!(matches!(
        error,
        tenferro_tensor::Error::Unsupported {
            op: "HookEngine::to_contiguous_read",
            ..
        }
    ));
}

#[test]
fn immediate_event_domain_driver_is_the_default_driver() {
    let engine = HookEngine::new();
    let driver = engine.event_domain_driver();
    assert!(driver.supports_scoped_read_only());
    let _ = ImmediateEventDomainDriver::new();
}
