use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tenferro_ops::{dim_expr::DimExpr, ShapeRelation};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir, DType,
    DotGeneralConfig, GatherConfig, MemoryKind, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorRead, TensorReduction, TensorStructural,
};

use super::super::GraphExecutor;
use crate::exec::{ExecInstruction, ExecOp, ExecProgram, ExecSlot};
use crate::extension::execute_lowered_program_with_backend_cache;
use crate::extension_cache::ExtensionCacheStore;
use crate::extension_runtime::ExtensionExecutionContext;
use crate::shape_constraint::{ConstraintSource, ShapeGuard};
use crate::{Error, GraphCompiler, TracedTensor};

#[derive(Debug)]
struct CountingBackend {
    uploads: Arc<AtomicUsize>,
    uploads_in_session: Arc<AtomicUsize>,
    dispatches: Arc<AtomicUsize>,
    session_entries: Arc<AtomicUsize>,
    session_depth: Arc<AtomicUsize>,
}

macro_rules! unreachable_backend_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                $(let _ = &$arg;)*
                panic!(concat!(stringify!($name), " should not be called by this test"))
            }
        )+
    };
}

impl TensorDeviceTransfer for CountingBackend {
    fn upload_host_tensor(&mut self, tensor: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        if self.session_depth.load(Ordering::Relaxed) != 0 {
            self.uploads_in_session.fetch_add(1, Ordering::Relaxed);
        }
        Ok(tensor.clone())
    }
}

impl BackendRuntimeCache for CountingBackend {
    type RuntimeCache = ();
}

impl TensorElementwise for CountingBackend {
    unreachable_backend_methods! {
        add(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sub(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        mul(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        conj(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        abs(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sign(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> tenferro_tensor::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> tenferro_tensor::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> tenferro_tensor::Result<Tensor>;
    }

    fn neg(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        self.dispatches.fetch_add(1, Ordering::Relaxed);
        Ok(input.clone())
    }
}

impl TensorAnalytic for CountingBackend {
    unreachable_backend_methods! {
        exp(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        log(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sin(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        cos(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        tanh(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        sqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        rsqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        pow(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
        expm1(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        log1p(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorStructural for CountingBackend {
    unreachable_backend_methods! {
        transpose(input: &Tensor, perm: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> tenferro_tensor::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor>;
        cast(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorReduction for CountingBackend {
    unreachable_backend_methods! {
        reduce_sum(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_prod(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_max(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_min(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorIndexing for CountingBackend {
    unreachable_backend_methods! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> tenferro_tensor::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> tenferro_tensor::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> tenferro_tensor::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> tenferro_tensor::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> tenferro_tensor::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> tenferro_tensor::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorDot for CountingBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        self.dispatches.fetch_add(1, Ordering::Relaxed);
        Ok(lhs.clone())
    }
}

impl TensorFusion for CountingBackend {}
impl TensorBuffer for CountingBackend {}
impl BackendCachedDot for CountingBackend {}
impl BackendSessionHost for CountingBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        self.session_entries.fetch_add(1, Ordering::Relaxed);
        self.session_depth.fetch_add(1, Ordering::Relaxed);
        let result = f(self);
        self.session_depth.fetch_sub(1, Ordering::Relaxed);
        result
    }
}
impl TensorBackend for CountingBackend {}

fn counting_backend() -> (
    CountingBackend,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let uploads = Arc::new(AtomicUsize::new(0));
    let uploads_in_session = Arc::new(AtomicUsize::new(0));
    let dispatches = Arc::new(AtomicUsize::new(0));
    let session_entries = Arc::new(AtomicUsize::new(0));
    let session_depth = Arc::new(AtomicUsize::new(0));
    (
        CountingBackend {
            uploads: uploads.clone(),
            uploads_in_session,
            dispatches: dispatches.clone(),
            session_entries: session_entries.clone(),
            session_depth,
        },
        uploads,
        dispatches,
        session_entries,
    )
}

#[test]
fn host_native_and_session_ffi_share_one_backend_session() {
    let uploads = Arc::new(AtomicUsize::new(0));
    let uploads_in_session = Arc::new(AtomicUsize::new(0));
    let dispatches = Arc::new(AtomicUsize::new(0));
    let session_entries = Arc::new(AtomicUsize::new(0));
    let session_depth = Arc::new(AtomicUsize::new(0));
    let mut backend = CountingBackend {
        uploads: uploads.clone(),
        uploads_in_session: uploads_in_session.clone(),
        dispatches: dispatches.clone(),
        session_entries: session_entries.clone(),
        session_depth,
    };
    let program = ExecProgram {
        instructions: vec![
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![0],
                output_slots: vec![1],
                dtype: DType::F64,
                output_shapes: vec![vec![]].into(),
                output_extents: vec![vec![]].into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::Negate,
                input_slots: vec![1],
                output_slots: vec![2],
                dtype: DType::F64,
                output_shapes: vec![vec![]].into(),
                output_extents: vec![vec![]].into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::DotGeneral(DotGeneralConfig {
                    lhs_contracting_dims: vec![],
                    rhs_contracting_dims: vec![],
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
                }),
                input_slots: vec![2, 1],
                output_slots: vec![3],
                dtype: DType::F64,
                output_shapes: vec![vec![]].into(),
                output_extents: vec![vec![]].into(),
                last_use: vec![true, true],
            },
        ],
        input_slots: vec![0],
        output_slots: vec![3],
        n_slots: 4,
        shape_guards: vec![],
    };
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut slots = Vec::new();
    let mut cache = ();

    let outputs = crate::segment::eval_exec_segmented_slots_with_cache_and_workspace(
        &mut backend,
        &program,
        vec![ExecSlot::Owned(input)],
        &mut slots,
        &mut cache,
        None,
    )
    .unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(uploads.load(Ordering::Relaxed), 1);
    assert_eq!(uploads_in_session.load(Ordering::Relaxed), 1);
    assert_eq!(dispatches.load(Ordering::Relaxed), 2);
    assert_eq!(session_entries.load(Ordering::Relaxed), 1);
}

fn fused_host_program() -> ExecProgram {
    let scalar_metadata = || (vec![vec![]].into(), vec![vec![]].into());
    let vector_metadata = || {
        (
            vec![vec![DimExpr::Const(2)]].into(),
            vec![vec![tenferro_ops::ShapeExtent::exact(DimExpr::Const(2))]].into(),
        )
    };
    let (output_shapes, output_extents) = vector_metadata();
    let first = ExecInstruction {
        op: ExecOp::Negate,
        input_slots: vec![0],
        output_slots: vec![1],
        dtype: DType::F64,
        output_shapes,
        output_extents,
        last_use: vec![false],
    };
    let (output_shapes, output_extents) = vector_metadata();
    let second = ExecInstruction {
        op: ExecOp::Negate,
        input_slots: vec![1],
        output_slots: vec![2],
        dtype: DType::F64,
        output_shapes,
        output_extents,
        last_use: vec![true],
    };
    let (output_shapes, output_extents) = vector_metadata();
    let ffi = ExecInstruction {
        op: ExecOp::DotGeneral(DotGeneralConfig {
            lhs_contracting_dims: vec![],
            rhs_contracting_dims: vec![],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        }),
        input_slots: vec![2, 0],
        output_slots: vec![3],
        dtype: DType::F64,
        output_shapes,
        output_extents,
        last_use: vec![false, false],
    };
    let (output_shapes, output_extents) = scalar_metadata();
    let host = ExecInstruction {
        op: ExecOp::ShapeOf { axis: 0 },
        input_slots: vec![0],
        output_slots: vec![4],
        dtype: DType::F64,
        output_shapes,
        output_extents,
        last_use: vec![false],
    };
    let (output_shapes, output_extents) = scalar_metadata();
    let third = ExecInstruction {
        op: ExecOp::Negate,
        input_slots: vec![4],
        output_slots: vec![5],
        dtype: DType::F64,
        output_shapes,
        output_extents,
        last_use: vec![false],
    };
    let (output_shapes, output_extents) = scalar_metadata();
    let fourth = ExecInstruction {
        op: ExecOp::Negate,
        input_slots: vec![5],
        output_slots: vec![6],
        dtype: DType::F64,
        output_shapes,
        output_extents,
        last_use: vec![true],
    };
    ExecProgram {
        instructions: vec![first, second, ffi, host, third, fourth],
        input_slots: vec![0],
        output_slots: vec![3, 6],
        n_slots: 7,
        shape_guards: vec![],
    }
}

#[test]
fn fused_and_host_segments_share_one_backend_session_for_owned_outputs() {
    let (mut backend, _, dispatches, sessions) = counting_backend();
    let mut slots = Vec::new();
    let mut cache = ();

    let outputs = crate::segment::eval_exec_segmented_slots_with_cache_and_workspace(
        &mut backend,
        &fused_host_program(),
        vec![ExecSlot::Owned(
            Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        )],
        &mut slots,
        &mut cache,
        None,
    )
    .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(dispatches.load(Ordering::Relaxed), 5);
    assert_eq!(sessions.load(Ordering::Relaxed), 1);
}

#[test]
fn fused_and_host_segments_share_one_backend_session_for_value_outputs() {
    let (mut backend, _, dispatches, sessions) = counting_backend();
    let mut slots = Vec::new();
    let mut cache = ();

    let outputs = crate::segment::eval_exec_segmented_slot_values_with_cache_and_workspace(
        &mut backend,
        &fused_host_program(),
        vec![ExecSlot::Owned(
            Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        )],
        &mut slots,
        &mut cache,
        None,
    )
    .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(dispatches.load(Ordering::Relaxed), 5);
    assert_eq!(sessions.load(Ordering::Relaxed), 1);
}

fn constant_guard(lhs: usize, rhs: usize, family_id: &'static str) -> ShapeGuard {
    ShapeGuard {
        source: ConstraintSource {
            family_id,
            instruction_index: Some(0),
        },
        relation: ShapeRelation::Equal,
        lhs: DimExpr::Const(lhs),
        rhs: DimExpr::Const(rhs),
    }
}

fn scalar_default_program(guard: ShapeGuard) -> crate::GraphProgram {
    let input = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let output = input.neg().unwrap();
    let mut program = GraphCompiler::new().compile(&output).unwrap();
    program.exec.shape_guards = vec![guard];
    program
}

fn explicit_input_program(guard: ShapeGuard) -> (TracedTensor, crate::GraphProgram) {
    let input = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let output = input.neg().unwrap();
    let mut program = GraphCompiler::new()
        .compile_with_input_specs(&output, &[(&input, DType::F64, &[2])])
        .unwrap();
    program.exec.shape_guards = vec![guard];
    (input, program)
}

fn unary_exec_program(guard: ShapeGuard) -> ExecProgram {
    ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Negate,
            input_slots: vec![0],
            output_slots: vec![1],
            dtype: DType::F64,
            output_shapes: vec![vec![]].into(),
            output_extents: vec![vec![]].into(),
            last_use: vec![false],
        }],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
        shape_guards: vec![guard],
    }
}

fn scalar_input() -> Tensor {
    Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap()
}

fn assert_graph_guard_error(error: Error) {
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "runtime.graph-preflight.v1",
            lhs_value: 1,
            rhs_value: 2,
            ..
        }
    ));
}

fn assert_counts(
    uploads: &AtomicUsize,
    dispatches: &AtomicUsize,
    session_entries: &AtomicUsize,
    expected_uploads: usize,
    expected_dispatches: usize,
    expected_session_entries: usize,
) {
    assert_eq!(uploads.load(Ordering::Relaxed), expected_uploads);
    assert_eq!(dispatches.load(Ordering::Relaxed), expected_dispatches);
    assert_eq!(
        session_entries.load(Ordering::Relaxed),
        expected_session_entries
    );
}

#[test]
fn graph_run_wrappers_validate_before_default_upload() {
    let bad = scalar_default_program(constant_guard(1, 2, "runtime.graph-preflight.v1"));
    let good = scalar_default_program(constant_guard(2, 2, "runtime.graph-preflight.v1"));

    let (backend, uploads, dispatches, sessions) = counting_backend();
    assert_graph_guard_error(GraphExecutor::new(backend).run_many(&bad).unwrap_err());
    assert_counts(&uploads, &dispatches, &sessions, 0, 0, 0);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    assert_graph_guard_error(
        GraphExecutor::new(backend)
            .run_many_values(&bad)
            .unwrap_err(),
    );
    assert_counts(&uploads, &dispatches, &sessions, 0, 0, 0);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    assert_graph_guard_error(
        GraphExecutor::new(backend)
            .run_many_with_input_reads(&bad, &[])
            .unwrap_err(),
    );
    assert_counts(&uploads, &dispatches, &sessions, 0, 0, 0);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    assert_graph_guard_error(
        GraphExecutor::new(backend)
            .run_many_values_with_input_reads(&bad, &[])
            .unwrap_err(),
    );
    assert_counts(&uploads, &dispatches, &sessions, 0, 0, 0);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    GraphExecutor::new(backend).run_many(&good).unwrap();
    assert_counts(&uploads, &dispatches, &sessions, 1, 1, 1);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    GraphExecutor::new(backend).run_many_values(&good).unwrap();
    assert_counts(&uploads, &dispatches, &sessions, 1, 1, 1);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    GraphExecutor::new(backend)
        .run_many_with_input_reads(&good, &[])
        .unwrap();
    assert_counts(&uploads, &dispatches, &sessions, 1, 1, 1);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    GraphExecutor::new(backend)
        .run_many_values_with_input_reads(&good, &[])
        .unwrap();
    assert_counts(&uploads, &dispatches, &sessions, 1, 1, 1);
}

#[test]
fn graph_run_preflight_uses_explicit_input_shape() {
    let axis = DimExpr::InputDim {
        input_idx: 0,
        axis: 0,
    };
    let (input, bad) = explicit_input_program(ShapeGuard {
        source: ConstraintSource {
            family_id: "runtime.graph-explicit-preflight.v1",
            instruction_index: Some(0),
        },
        relation: ShapeRelation::Equal,
        lhs: axis.clone(),
        rhs: DimExpr::Const(3),
    });
    let (good_input, good) = explicit_input_program(ShapeGuard {
        source: ConstraintSource {
            family_id: "runtime.graph-explicit-preflight.v1",
            instruction_index: Some(0),
        },
        relation: ShapeRelation::Equal,
        lhs: axis,
        rhs: DimExpr::Const(2),
    });
    let bound = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();

    let (backend, uploads, dispatches, sessions) = counting_backend();
    let error = GraphExecutor::new(backend)
        .run_many_with_inputs(&bad, &[(&input, &bound)])
        .unwrap_err();
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "runtime.graph-explicit-preflight.v1",
            lhs_value: 2,
            rhs_value: 3,
            ..
        }
    ));
    assert_counts(&uploads, &dispatches, &sessions, 0, 0, 0);

    let (backend, uploads, dispatches, sessions) = counting_backend();
    GraphExecutor::new(backend)
        .run_many_with_inputs(&good, &[(&good_input, &bound)])
        .unwrap();
    assert_counts(&uploads, &dispatches, &sessions, 0, 1, 1);
}

fn deferred_zero_guard_program(rhs: usize) -> ExecProgram {
    unary_exec_program(ShapeGuard {
        source: ConstraintSource {
            family_id: "runtime.deferred-zero-preflight.v1",
            instruction_index: Some(0),
        },
        relation: ShapeRelation::Equal,
        lhs: DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        rhs: DimExpr::Const(rhs),
    })
}

fn owned_deferred_zero_selection(
) -> Vec<super::super::SelectedInput<'static, 'static, &'static Tensor>> {
    vec![super::super::SelectedInput::DeferredZero {
        dtype: DType::F64,
        shape: vec![2],
    }]
}

fn borrowed_deferred_zero_selection(
) -> Vec<super::super::SelectedInput<'static, 'static, TensorRead<'static>>> {
    vec![super::super::SelectedInput::DeferredZero {
        dtype: DType::F64,
        shape: vec![2],
    }]
}

fn assert_deferred_zero_tensor(tensor: &Tensor) {
    assert_eq!(tensor.dtype(), DType::F64);
    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
    assert_eq!(tensor.placement().memory_kind, MemoryKind::UnpinnedHost);
    assert_eq!(tensor.placement().device, None);
}

#[test]
fn owned_deferred_zero_factory_runs_only_after_guard_validation() {
    let bad = deferred_zero_guard_program(3);
    let good = deferred_zero_guard_program(2);

    let (mut backend, _, _, _) = counting_backend();
    let mut factory_calls = 0;
    let error = super::super::materialize_inputs(
        &bad,
        owned_deferred_zero_selection(),
        &mut backend,
        |dtype, shape| {
            factory_calls += 1;
            super::super::zeros_tensor(dtype, shape)
        },
    )
    .unwrap_err();
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(factory_calls, 0);

    let outputs = super::super::materialize_inputs(
        &good,
        owned_deferred_zero_selection(),
        &mut backend,
        |dtype, shape| {
            factory_calls += 1;
            assert_eq!(dtype, DType::F64);
            assert_eq!(shape, vec![2]);
            super::super::zeros_tensor(dtype, shape)
        },
    )
    .unwrap();
    assert_eq!(factory_calls, 1);
    assert_deferred_zero_tensor(&outputs[0]);
}

#[test]
fn borrowed_deferred_zero_factory_runs_only_after_guard_validation() {
    let bad = deferred_zero_guard_program(3);
    let good = deferred_zero_guard_program(2);

    let (mut backend, _, _, _) = counting_backend();
    let mut factory_calls = 0;
    let result = super::super::materialize_input_reads(
        &bad,
        borrowed_deferred_zero_selection(),
        &mut backend,
        |dtype, shape| {
            factory_calls += 1;
            super::super::zeros_tensor(dtype, shape)
        },
    );
    let Err(error) = result else {
        panic!("bad guard must fail before borrowed deferred-zero materialization");
    };
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(factory_calls, 0);

    let outputs = super::super::materialize_input_reads(
        &good,
        borrowed_deferred_zero_selection(),
        &mut backend,
        |dtype, shape| {
            factory_calls += 1;
            assert_eq!(dtype, DType::F64);
            assert_eq!(shape, vec![2]);
            super::super::zeros_tensor(dtype, shape)
        },
    )
    .unwrap();
    assert_eq!(factory_calls, 1);
    let ExecSlot::Owned(output) = &outputs[0] else {
        panic!("deferred borrowed input must materialize into an owned slot");
    };
    assert_deferred_zero_tensor(output);
}

#[test]
fn deferred_zero_factory_errors_follow_metadata_and_guard_validation() {
    let (input, mut invalid_metadata) =
        explicit_input_program(constant_guard(1, 2, "runtime.deferred-zero-error-order.v1"));
    invalid_metadata.inputs[0].shape = vec![3];
    let bound = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]).unwrap();
    let (backend, uploads, dispatches, sessions) = counting_backend();
    let error = GraphExecutor::new(backend)
        .run_with_inputs(&invalid_metadata, &[(&input, &bound)])
        .unwrap_err();
    assert!(matches!(error, Error::PlaceholderShapeMismatch { .. }));
    assert_counts(&uploads, &dispatches, &sessions, 0, 0, 0);

    let injected_error = || Error::InvalidCompiledGraph {
        message: "injected deferred-zero factory failure".to_string(),
    };
    let (mut backend, _, _, _) = counting_backend();
    let mut factory_calls = 0;
    let error = super::super::materialize_inputs(
        &deferred_zero_guard_program(3),
        owned_deferred_zero_selection(),
        &mut backend,
        |_, _| {
            factory_calls += 1;
            Err(injected_error())
        },
    )
    .unwrap_err();
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(factory_calls, 0);

    let error = super::super::materialize_inputs(
        &deferred_zero_guard_program(2),
        owned_deferred_zero_selection(),
        &mut backend,
        |_, _| {
            factory_calls += 1;
            Err(injected_error())
        },
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidCompiledGraph { .. }));
    assert_eq!(factory_calls, 1);
}

#[test]
fn owner_scoped_execution_paths_validate_before_dispatch() {
    let bad = unary_exec_program(constant_guard(1, 2, "runtime.owner-preflight.v1"));
    let good = unary_exec_program(constant_guard(2, 2, "runtime.owner-preflight.v1"));

    let (mut backend, uploads, dispatches, _) = counting_backend();
    let mut caches = ExtensionCacheStore::new();
    let mut ctx = ExtensionExecutionContext::new(&mut backend, &mut caches);
    let error = ctx
        .execute_core_exec_program_unsegmented(&bad, vec![scalar_input()])
        .unwrap_err();
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(uploads.load(Ordering::Relaxed), 0);
    assert_eq!(dispatches.load(Ordering::Relaxed), 0);
    ctx.execute_core_exec_program_unsegmented(&good, vec![scalar_input()])
        .unwrap();
    assert_eq!(dispatches.load(Ordering::Relaxed), 1);

    let (mut backend, uploads, dispatches, _) = counting_backend();
    let mut backend_cache = ();
    let error = execute_lowered_program_with_backend_cache(
        &mut backend,
        &bad,
        vec![scalar_input()],
        &mut backend_cache,
    )
    .unwrap_err();
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(uploads.load(Ordering::Relaxed), 0);
    assert_eq!(dispatches.load(Ordering::Relaxed), 0);
    execute_lowered_program_with_backend_cache(
        &mut backend,
        &good,
        vec![scalar_input()],
        &mut backend_cache,
    )
    .unwrap();
    assert_eq!(dispatches.load(Ordering::Relaxed), 1);
}
