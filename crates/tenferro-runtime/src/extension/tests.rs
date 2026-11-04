use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, ValidationError};

use super::*;
use crate::program::{ProgramShapeRelation, SemanticOpRef};
use crate::{TraceContext, TracedTensor};

#[derive(Clone, Debug)]
struct TestExtension {
    input_count: usize,
    output_count: usize,
    inferred_outputs: usize,
}

impl ExtensionOp for TestExtension {
    fn family_id(&self) -> &'static str {
        "test.extension"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_count(&self) -> usize {
        self.input_count
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        let shape = ctx.input_shape(0)?.to_vec();
        Ok((0..self.inferred_outputs)
            .map(|_| (dtype, shape.clone()))
            .collect())
    }
}

fn symbolic_input(trace: &TraceContext) -> TracedTensor {
    trace.input_symbolic_shape(DType::F64, 1).unwrap()
}

/// Assert a program carries exactly one guard, allowing either operand order.
fn assert_one_axis_equality_guard(
    trace: &TraceContext,
    outputs: &[&TracedTensor],
    lhs_input: usize,
    rhs_input: usize,
) {
    let values: Vec<_> = outputs.iter().map(|output| output.trace_value()).collect();
    let graph = trace.finish(&values).unwrap();
    let guards = graph.program().shape_guards();

    assert_eq!(guards.len(), 1, "one guard per constrained operation");
    let guard = &guards[0];
    assert_eq!(guard.relation(), ProgramShapeRelation::Equal);
    let lhs = DimExpr::InputDim {
        input_idx: lhs_input,
        axis: 0,
    };
    let rhs = DimExpr::InputDim {
        input_idx: rhs_input,
        axis: 0,
    };
    assert!(
        (guard.lhs() == &lhs && guard.rhs() == &rhs)
            || (guard.lhs() == &rhs && guard.rhs() == &lhs),
        "unexpected guard: {guard:?}"
    );
}

#[test]
fn apply_returns_error_for_input_count_mismatch() {
    let trace = TraceContext::new();
    let op = Arc::new(TestExtension {
        input_count: 2,
        output_count: 1,
        inferred_outputs: 1,
    });
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();

    let Err(err) = apply(op, &[&x]) else {
        panic!("input count mismatch should be an error");
    };

    let message = err.to_string();
    assert!(message.contains("test.extension"), "{message}");
    assert!(message.contains("expects 2 inputs, got 1"), "{message}");
}

#[test]
fn apply_returns_error_for_output_metadata_count_mismatch() {
    let trace = TraceContext::new();
    let op = Arc::new(TestExtension {
        input_count: 1,
        output_count: 2,
        inferred_outputs: 1,
    });
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();

    let Err(err) = apply(op, &[&x]) else {
        panic!("output metadata mismatch should be an error");
    };

    match err {
        Error::TensorRuntime(tenferro_tensor::Error::Validation { op, source }) => {
            assert_eq!(op, "extension");
            assert!(matches!(
                source,
                ValidationError::InvalidArgument {
                    argument: "output metadata",
                    message,
                } if message.contains("family_id=\"test.extension\"")
                    && message.contains("declared 2 outputs")
            ));
        }
        other => panic!("expected structured extension validation error, got {other:?}"),
    }
}

#[test]
fn apply_standard_op_builds_core_op_without_extension() {
    // An extension whose traced fast path lowers to core operations leaves a core
    // graph with no extension carrier and no shape obligation of its own.
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let y = trace
        .from_vec_col_major(vec![2], vec![3.0_f64, 4.0])
        .unwrap();

    let outputs = apply_standard_op(StdTensorOp::Add, &[&x, &y]).expect("the lowering builds");
    let graph = trace
        .finish(
            &outputs
                .iter()
                .map(|output| output.trace_value())
                .collect::<Vec<_>>(),
        )
        .unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].rank, 1);
    assert_eq!(outputs[0].dtype, DType::F64);
    assert!(graph.program().shape_guards().is_empty());
    let operations: Vec<_> = graph.program().operations().map(|op| op.op()).collect();
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0],
        SemanticOpRef::Core(crate::program::CoreSemanticOp::Add)
    ));
}

#[derive(Clone, Debug)]
struct CountedConstraintExtension {
    calls: Arc<AtomicUsize>,
}

impl ExtensionOp for CountedConstraintExtension {
    fn family_id(&self) -> &'static str {
        "test.counted-constraint"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| Arc::ptr_eq(&self.calls, &other.calls))
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        2
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        ctx.require_axes_equal((0, 0), (1, 0))?;
        let meta = (ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec());
        Ok(vec![meta.clone(), meta])
    }
}

fn counted_constraint_outputs(calls: &Arc<AtomicUsize>) -> (TraceContext, Vec<TracedTensor>) {
    let trace = TraceContext::new();
    let lhs = symbolic_input(&trace);
    let rhs = symbolic_input(&trace);
    let outputs = apply(
        Arc::new(CountedConstraintExtension {
            calls: Arc::clone(calls),
        }),
        &[&lhs, &rhs],
    )
    .expect("the constrained extension builds");
    (trace, outputs)
}

/// A one-in/one-out extension that counts how often its inference runs.
#[derive(Clone, Debug)]
struct CountedExtension {
    calls: Arc<AtomicUsize>,
}

impl ExtensionOp for CountedExtension {
    fn family_id(&self) -> &'static str {
        "test.counted-extension"
    }

    fn payload_hash(&self, _hasher: &mut dyn std::hash::Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| Arc::ptr_eq(&self.calls, &other.calls))
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

#[test]
fn constraint_scope_analysis_invokes_extension_inference_once_and_attaches_one_scope() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (trace, outputs) = counted_constraint_outputs(&calls);

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_one_axis_equality_guard(&trace, &[&outputs[0], &outputs[1]], 0, 1);
}

#[test]
fn graph_local_constraint_scope_extension_child_does_not_reinfer_or_duplicate_ancestor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (trace, outputs) = counted_constraint_outputs(&calls);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let child = apply(
        Arc::new(TestExtension {
            input_count: 1,
            output_count: 1,
            inferred_outputs: 1,
        }),
        &[&outputs[0]],
    )
    .expect("the child extension builds")
    .remove(0);

    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "a child operation must not re-run its ancestor's inference"
    );
    assert_one_axis_equality_guard(&trace, &[&child], 0, 1);
}

#[test]
fn graph_local_constraint_scope_expanded_child_does_not_reinfer_or_duplicate_ancestor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (trace, outputs) = counted_constraint_outputs(&calls);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    // A core-op child is the surviving lowerer for an expanded extension path.
    let child = apply_standard_op(StdTensorOp::Neg, &[&outputs[0]])
        .expect("the core-op child builds")
        .remove(0);

    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "a lowered child must not re-run its ancestor's inference"
    );
    assert_one_axis_equality_guard(&trace, &[&child], 0, 1);
}

/// The deleted test counted analysis *visits*; P2 has no analysis traversal to
/// count (`metadata::test_support::with_visit_count` is gone), so the test is
/// named for the half that survives: one inference per appended operation, none
/// for the chain already behind it.
#[test]
fn extension_chain_runs_one_inference_per_appended_operation() {
    // A chain of N extension operations runs N inferences.
    const DEPTH: usize = 16;
    let calls = Arc::new(AtomicUsize::new(0));
    let trace = TraceContext::new();
    let mut value = symbolic_input(&trace);

    for _ in 0..DEPTH {
        value = apply(
            Arc::new(CountedExtension {
                calls: Arc::clone(&calls),
            }),
            &[&value],
        )
        .expect("each chain link builds")
        .remove(0);
    }

    assert_eq!(
        calls.load(Ordering::Relaxed),
        DEPTH,
        "one inference per appended operation, none for the chain already built"
    );
}

#[test]
fn graph_analysis_resolves_unregistered_multi_output_parent_on_demand() {
    // A multi-output extension exposes usable metadata for every declared output,
    // so a downstream operation can consume either of them.
    let trace = TraceContext::new();
    let input = symbolic_input(&trace);

    let outputs = apply(
        Arc::new(TestExtension {
            input_count: 1,
            output_count: 2,
            inferred_outputs: 2,
        }),
        &[&input],
    )
    .expect("the multi-output extension builds");

    assert_eq!(outputs.len(), 2);
    for output in &outputs {
        assert_eq!(output.dtype, DType::F64);
        assert_eq!(output.rank, 1);
    }
    let sum = outputs[0]
        .add(&outputs[1])
        .expect("both outputs are usable");
    let graph = trace.finish(&[sum.trace_value()]).unwrap();
    assert_eq!(graph.program().outputs().len(), 1);
    assert_eq!(sum.rank, 1);
}

#[test]
fn standard_op_analysis_uses_const_exact_parent_metadata_for_shape_validation() {
    let trace = TraceContext::new();
    let build = |values: [f64; 2]| {
        trace
            .from_vec_col_major(vec![2], values.to_vec())
            .expect("the concrete input builds")
    };
    let reshape = StdTensorOp::Reshape {
        to_shape: vec![DimExpr::Const(2), DimExpr::Const(1)],
    };
    let lhs = apply_standard_op(reshape.clone(), &[&build([1.0, 2.0])])
        .expect("reshape builds")
        .remove(0);
    let rhs = apply_standard_op(reshape, &[&build([3.0, 4.0])])
        .expect("reshape builds")
        .remove(0);

    let output = apply_standard_op(
        StdTensorOp::Concatenate {
            axis: 1,
            input_count: 2,
        },
        &[&lhs, &rhs],
    )
    .expect("concatenate validates against exact parent metadata")
    .remove(0);

    assert_eq!(
        output.sym_shape(),
        Some([SymDim::from(2), SymDim::from(2)].as_slice())
    );
}

#[test]
fn append_then_analyze_registers_output_metadata() {
    // Appending a core op registers the output's metadata in the context, so a
    // downstream reader does not have to re-derive rank or dtype.
    let trace = TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let y = trace
        .from_vec_col_major(vec![2], vec![3.0_f64, 4.0])
        .unwrap();

    let output = apply_standard_op(StdTensorOp::Add, &[&x, &y])
        .expect("the core op builds")
        .remove(0);

    let metadata = trace.value_metadata(output.trace_value()).unwrap();
    assert_eq!(metadata.dtype(), DType::F64);
    assert_eq!(metadata.shape().len(), 1);
}
