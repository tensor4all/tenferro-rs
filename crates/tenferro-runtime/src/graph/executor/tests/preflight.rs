use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tenferro_ops::{dim_expr::DimExpr, ShapeRelation};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, CompareDir, DType, DotGeneralConfig,
    GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBackend,
    TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion, TensorIndexing,
    TensorReduction, TensorStructural,
};

use super::super::GraphExecutor;
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use crate::extension::execute_lowered_program_with_backend_cache;
use crate::extension_cache::ExtensionCacheStore;
use crate::extension_runtime::ExtensionExecutionContext;
use crate::shape_constraint::{ConstraintSource, ShapeGuard};
use crate::{Error, GraphCompiler, TracedTensor};

#[derive(Debug)]
struct CountingBackend {
    uploads: Arc<AtomicUsize>,
    dispatches: Arc<AtomicUsize>,
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
    unreachable_backend_methods! {
        dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorFusion for CountingBackend {}
impl TensorBuffer for CountingBackend {}
impl BackendCachedDot for CountingBackend {}
impl BackendSessionHost for CountingBackend {}
impl TensorBackend for CountingBackend {}

fn counting_backend() -> (CountingBackend, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let uploads = Arc::new(AtomicUsize::new(0));
    let dispatches = Arc::new(AtomicUsize::new(0));
    (
        CountingBackend {
            uploads: uploads.clone(),
            dispatches: dispatches.clone(),
        },
        uploads,
        dispatches,
    )
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
    expected_uploads: usize,
    expected_dispatches: usize,
) {
    assert_eq!(uploads.load(Ordering::Relaxed), expected_uploads);
    assert_eq!(dispatches.load(Ordering::Relaxed), expected_dispatches);
}

#[test]
fn graph_run_wrappers_validate_before_default_upload() {
    let bad = scalar_default_program(constant_guard(1, 2, "runtime.graph-preflight.v1"));
    let good = scalar_default_program(constant_guard(2, 2, "runtime.graph-preflight.v1"));

    let (backend, uploads, dispatches) = counting_backend();
    assert_graph_guard_error(GraphExecutor::new(backend).run_many(&bad).unwrap_err());
    assert_counts(&uploads, &dispatches, 0, 0);

    let (backend, uploads, dispatches) = counting_backend();
    assert_graph_guard_error(
        GraphExecutor::new(backend)
            .run_many_values(&bad)
            .unwrap_err(),
    );
    assert_counts(&uploads, &dispatches, 0, 0);

    let (backend, uploads, dispatches) = counting_backend();
    assert_graph_guard_error(
        GraphExecutor::new(backend)
            .run_many_with_input_reads(&bad, &[])
            .unwrap_err(),
    );
    assert_counts(&uploads, &dispatches, 0, 0);

    let (backend, uploads, dispatches) = counting_backend();
    assert_graph_guard_error(
        GraphExecutor::new(backend)
            .run_many_values_with_input_reads(&bad, &[])
            .unwrap_err(),
    );
    assert_counts(&uploads, &dispatches, 0, 0);

    let (backend, uploads, dispatches) = counting_backend();
    GraphExecutor::new(backend).run_many(&good).unwrap();
    assert_counts(&uploads, &dispatches, 1, 1);

    let (backend, uploads, dispatches) = counting_backend();
    GraphExecutor::new(backend).run_many_values(&good).unwrap();
    assert_counts(&uploads, &dispatches, 1, 1);

    let (backend, uploads, dispatches) = counting_backend();
    GraphExecutor::new(backend)
        .run_many_with_input_reads(&good, &[])
        .unwrap();
    assert_counts(&uploads, &dispatches, 1, 1);

    let (backend, uploads, dispatches) = counting_backend();
    GraphExecutor::new(backend)
        .run_many_values_with_input_reads(&good, &[])
        .unwrap();
    assert_counts(&uploads, &dispatches, 1, 1);
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

    let (backend, uploads, dispatches) = counting_backend();
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
    assert_counts(&uploads, &dispatches, 0, 0);

    let (backend, uploads, dispatches) = counting_backend();
    GraphExecutor::new(backend)
        .run_many_with_inputs(&good, &[(&good_input, &bound)])
        .unwrap();
    assert_counts(&uploads, &dispatches, 0, 1);
}

#[test]
fn owner_scoped_execution_paths_validate_before_dispatch() {
    let bad = unary_exec_program(constant_guard(1, 2, "runtime.owner-preflight.v1"));
    let good = unary_exec_program(constant_guard(2, 2, "runtime.owner-preflight.v1"));

    let (mut backend, uploads, dispatches) = counting_backend();
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

    let (mut backend, uploads, dispatches) = counting_backend();
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
