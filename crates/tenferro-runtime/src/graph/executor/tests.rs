use std::any::Any;
use std::hash::Hasher;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use tenferro_cpu::CpuBackend;
use tenferro_ops::{dim_expr::DimExpr, ext_op::ExtensionOp, ShapeRelation, SymDim};
use tenferro_tensor::{DType, Tensor, TensorRead, TensorValue, TypedTensor};

use super::GraphExecutor;
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use crate::extension_runtime::{ExtensionExecutionContext, ExtensionRuntime};
use crate::shape_constraint::{ConstraintSource, ShapeGuard};
use crate::{Error, GraphCompiler, TracedTensor};

const COUNTED_FAMILY: &str = "runtime.counted-identity.v1";

#[derive(Clone, Debug)]
struct CountedIdentityOp;

impl ExtensionOp for CountedIdentityOp {
    fn family_id(&self) -> &'static str {
        COUNTED_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

#[derive(Debug)]
struct CountedIdentityRuntime {
    calls: Arc<AtomicUsize>,
}

impl ExtensionRuntime<CpuBackend> for CountedIdentityRuntime {
    fn family_id(&self) -> &'static str {
        COUNTED_FAMILY
    }

    fn execute(
        &self,
        _op: &dyn ExtensionOp,
        inputs: &[&Tensor],
        _ctx: &mut ExtensionExecutionContext<'_, CpuBackend>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![inputs[0].clone()])
    }

    fn execute_reads(
        &self,
        _op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        _ctx: &mut ExtensionExecutionContext<'_, CpuBackend>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![inputs[0].to_tensor()?])
    }
}

fn scaled_guard_program(guard: ShapeGuard) -> ExecProgram {
    ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Extension(Arc::new(CountedIdentityOp)),
            input_slots: vec![0, 1],
            output_slots: vec![2],
            dtype: DType::F64,
            output_shapes: vec![vec![DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            }]]
            .into(),
            output_extents: vec![vec![]].into(),
            last_use: vec![false, false],
        }],
        input_slots: vec![0, 1],
        output_slots: vec![2],
        n_slots: 3,
        shape_guards: vec![guard],
    }
}

fn scaled_guard(family: &'static str, instruction_index: Option<usize>) -> ShapeGuard {
    ShapeGuard {
        source: ConstraintSource {
            family_id: family,
            instruction_index,
        },
        relation: ShapeRelation::Equal,
        lhs: DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        rhs: DimExpr::Mul(
            Box::new(DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }),
            Box::new(DimExpr::Const(2)),
        ),
    }
}

fn f64_zeros(shape: Vec<usize>) -> Tensor {
    Tensor::F64(TypedTensor::zeros(shape).unwrap())
}

fn counted_executor(calls: Arc<AtomicUsize>) -> GraphExecutor<CpuBackend> {
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .extension_executor_mut()
        .registry_mut()
        .register(Arc::new(CountedIdentityRuntime { calls }))
        .unwrap();
    executor
}

#[test]
fn executor_rejects_shape_guard_before_dispatch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut executor = counted_executor(calls.clone());
    let program = scaled_guard_program(scaled_guard(COUNTED_FAMILY, Some(0)));

    let error = executor
        .eval_exec_ir(&program, vec![f64_zeros(vec![7]), f64_zeros(vec![3])])
        .unwrap_err();

    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: COUNTED_FAMILY,
            instruction_index: Some(0),
            lhs_value: 7,
            rhs_value: 6,
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let output = executor
        .eval_exec_ir(&program, vec![f64_zeros(vec![6]), f64_zeros(vec![3])])
        .unwrap();
    assert_eq!(output[0].shape(), &[6]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn executor_shape_guards_cover_value_and_borrowed_entry_variants() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut executor = counted_executor(calls.clone());
    let program = scaled_guard_program(scaled_guard(COUNTED_FAMILY, Some(0)));

    let error = executor
        .eval_exec_ir_values(&program, vec![f64_zeros(vec![7]), f64_zeros(vec![3])])
        .unwrap_err();
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    executor
        .eval_exec_ir_values(&program, vec![f64_zeros(vec![6]), f64_zeros(vec![3])])
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let bad_inputs = vec![f64_zeros(vec![7]), f64_zeros(vec![3])];
    let error = executor
        .eval_exec_ir_non_consuming(&program, &bad_inputs)
        .unwrap_err();
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(bad_inputs[0].shape(), &[7]);
    let good_inputs = vec![f64_zeros(vec![6]), f64_zeros(vec![3])];
    executor
        .eval_exec_ir_non_consuming(&program, &good_inputs)
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);

    let error = executor
        .eval_exec_ir_non_consuming_values(&program, &bad_inputs)
        .unwrap_err();
    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    executor
        .eval_exec_ir_non_consuming_values(&program, &good_inputs)
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[test]
fn executor_reports_shape_guard_evaluation_causes_before_dispatch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut executor = counted_executor(calls.clone());
    let axis_guard = ShapeGuard {
        source: ConstraintSource {
            family_id: "runtime.axis-oob.v1",
            instruction_index: Some(4),
        },
        relation: ShapeRelation::Equal,
        lhs: DimExpr::InputDim {
            input_idx: 0,
            axis: 1,
        },
        rhs: DimExpr::Const(0),
    };
    let error = executor
        .eval_exec_ir(
            &scaled_guard_program(axis_guard),
            vec![f64_zeros(vec![6]), f64_zeros(vec![3])],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::ShapeConstraintEvaluation {
            family: "runtime.axis-oob.v1",
            instruction_index: Some(4),
            cause: crate::ShapeConstraintEvalError::AxisOutOfBounds {
                input_idx: 0,
                axis: 1,
                rank: 1,
            },
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    let division_guard = ShapeGuard {
        source: ConstraintSource {
            family_id: "runtime.division-zero.v1",
            instruction_index: Some(5),
        },
        relation: ShapeRelation::Equal,
        lhs: DimExpr::FloorDiv(Box::new(DimExpr::Const(1)), Box::new(DimExpr::Const(0))),
        rhs: DimExpr::Const(0),
    };
    let error = executor
        .eval_exec_ir(
            &scaled_guard_program(division_guard),
            vec![f64_zeros(vec![6]), f64_zeros(vec![3])],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::ShapeConstraintEvaluation {
            family: "runtime.division-zero.v1",
            instruction_index: Some(5),
            cause: crate::ShapeConstraintEvalError::DivisionByZero,
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn executor_evaluates_shape_guards_in_stored_order() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut executor = counted_executor(calls.clone());
    let mut program = scaled_guard_program(scaled_guard("runtime.first-guard.v1", Some(2)));
    program.shape_guards.push(ShapeGuard {
        source: ConstraintSource {
            family_id: "runtime.second-guard.v1",
            instruction_index: Some(3),
        },
        relation: ShapeRelation::Equal,
        lhs: DimExpr::FloorDiv(Box::new(DimExpr::Const(1)), Box::new(DimExpr::Const(0))),
        rhs: DimExpr::Const(0),
    });

    let error = executor
        .eval_exec_ir(&program, vec![f64_zeros(vec![7]), f64_zeros(vec![3])])
        .unwrap_err();

    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "runtime.first-guard.v1",
            instruction_index: Some(2),
            lhs_value: 7,
            rhs_value: 6,
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn executor_validates_guards_before_segmenting_multiple_instructions() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut executor = counted_executor(calls.clone());
    let mut program = scaled_guard_program(scaled_guard(COUNTED_FAMILY, Some(0)));
    program.instructions.push(ExecInstruction {
        op: ExecOp::Extension(Arc::new(CountedIdentityOp)),
        input_slots: vec![2, 1],
        output_slots: vec![3],
        dtype: DType::F64,
        output_shapes: vec![vec![DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        }]]
        .into(),
        output_extents: vec![vec![]].into(),
        last_use: vec![false, false],
    });
    program.output_slots = vec![3];
    program.n_slots = 4;

    let error = executor
        .eval_exec_ir(&program, vec![f64_zeros(vec![7]), f64_zeros(vec![3])])
        .unwrap_err();

    assert!(matches!(error, Error::ShapeConstraintViolation { .. }));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn executor_validates_zero_instruction_guards_and_preserves_input_count_precedence() {
    let guard = ShapeGuard {
        source: ConstraintSource {
            family_id: "runtime.zero-instruction.v1",
            instruction_index: None,
        },
        relation: ShapeRelation::Equal,
        lhs: DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        rhs: DimExpr::InputDim {
            input_idx: 1,
            axis: 0,
        },
    };
    let program = ExecProgram {
        instructions: Vec::new(),
        input_slots: vec![0, 1],
        output_slots: vec![0],
        n_slots: 2,
        shape_guards: vec![guard],
    };
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let count_error = executor
        .eval_exec_ir(&program, vec![f64_zeros(vec![2])])
        .unwrap_err();
    assert!(
        matches!(count_error, Error::Internal(message) if message.contains("expected 2 inputs"))
    );

    let guard_error = executor
        .eval_exec_ir(&program, vec![f64_zeros(vec![2]), f64_zeros(vec![3])])
        .unwrap_err();
    assert!(matches!(
        guard_error,
        Error::ShapeConstraintViolation {
            family: "runtime.zero-instruction.v1",
            lhs_value: 2,
            rhs_value: 3,
            ..
        }
    ));
}

#[test]
fn executor_with_no_shape_guards_dispatches_unchanged() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut executor = counted_executor(calls.clone());
    let mut program = scaled_guard_program(scaled_guard(COUNTED_FAMILY, Some(0)));
    program.shape_guards.clear();

    let output = executor
        .eval_exec_ir(&program, vec![f64_zeros(vec![7]), f64_zeros(vec![3])])
        .unwrap();

    assert_eq!(output[0].shape(), &[7]);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn nonsegmented_extension_program_path_validates_shape_guards() {
    let program = ExecProgram {
        instructions: Vec::new(),
        input_slots: vec![0, 1],
        output_slots: vec![0],
        n_slots: 2,
        shape_guards: vec![scaled_guard("runtime.nonsegmented.v1", None)],
    };
    let mut backend = CpuBackend::new();

    let error = crate::exec::eval_exec_ir_unsegmented_with_cache(
        &mut backend,
        &program,
        vec![f64_zeros(vec![7]), f64_zeros(vec![3])],
    )
    .unwrap_err();

    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "runtime.nonsegmented.v1",
            instruction_index: None,
            lhs_value: 7,
            rhs_value: 6,
            ..
        }
    ));
}

#[test]
fn borrowed_input_execution_retains_executor_slot_workspace_capacity() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[4])])
        .unwrap();
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let mut executor = GraphExecutor::new(CpuBackend::new());

    assert_eq!(executor.borrowed_slot_workspace_capacity, 0);

    let outputs = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert!(
        executor.borrowed_slot_workspace_capacity >= program.exec.n_slots,
        "borrowed execution should retain reusable slot capacity; capacity={}, n_slots={}",
        executor.borrowed_slot_workspace_capacity,
        program.exec.n_slots
    );
}

#[test]
fn borrowed_input_value_execution_retains_workspace_and_lazy_output() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let y = x.transpose(&[1, 0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
        .unwrap();
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let outputs = executor
        .run_many_values_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert!(matches!(outputs[0], TensorValue::View(_)));
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert!(
        executor.borrowed_slot_workspace_capacity >= program.exec.n_slots,
        "borrowed value execution should retain reusable slot capacity; capacity={}, n_slots={}",
        executor.borrowed_slot_workspace_capacity,
        program.exec.n_slots
    );
}

#[test]
fn single_value_wrappers_preserve_lazy_outputs_and_debug_state() {
    let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let y = x.transpose(&[1, 0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let debug = format!("{executor:?}");
    assert!(debug.contains("GraphExecutor"));
    assert!(debug.contains("backend_type"));

    let value = executor.run_value(&program).unwrap();
    assert!(matches!(value, TensorValue::View(_)));
    assert_eq!(value.shape(), &[2, 2]);

    let placeholder = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let transposed = placeholder.transpose(&[1, 0]).unwrap();
    let program = compiler
        .compile_with_input_specs(&transposed, &[(&placeholder, DType::F64, &[2, 2])])
        .unwrap();
    let bound =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());

    let value = executor
        .run_value_with_inputs(&program, &[(&placeholder, &bound)])
        .unwrap();
    assert!(matches!(value, TensorValue::View(_)));
    assert_eq!(value.shape(), &[2, 2]);

    let value = executor
        .run_value_with_input_reads(&program, &[(&placeholder, TensorRead::from_tensor(&bound))])
        .unwrap();
    assert!(matches!(value, TensorValue::View(_)));
    assert_eq!(value.shape(), &[2, 2]);

    executor.reclaim_value_outputs(vec![TensorValue::from_tensor(
        Tensor::from_vec_col_major(vec![1], vec![9.0_f64]).unwrap(),
    )]);
}

#[test]
fn deferred_zero_tensor_covers_supported_dtypes_and_overflow() {
    for dtype in [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ] {
        let zero = super::zeros_tensor(dtype, vec![2]).unwrap();
        assert_eq!(zero.dtype(), dtype);
        assert_eq!(zero.shape(), &[2]);
    }

    let err = super::zeros_tensor(DType::Bool, vec![usize::MAX, 2]).unwrap_err();
    assert!(matches!(err, Error::InvalidCompiledGraph { .. }));
}

#[test]
fn borrowed_input_workspace_does_not_retype_static_slot_vec() {
    let source = include_str!("../executor.rs");

    assert!(
        !source.contains("Vec::from_raw_parts"),
        "borrowed input workspace must not retype Vec allocations across ExecSlot lifetimes"
    );
    assert!(
        !source.contains("BorrowedSlotWorkspace"),
        "borrowed input execution should use a lifetime-local workspace"
    );
}

#[test]
fn compile_with_input_specs_rejects_computed_placeholder_specs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();

    let err = compiler
        .compile_with_input_specs(&y, &[(&y, DType::F64, &[2])])
        .unwrap_err();

    assert!(matches!(err, Error::UnexpectedBinding { binding_index: 0 }));
}
