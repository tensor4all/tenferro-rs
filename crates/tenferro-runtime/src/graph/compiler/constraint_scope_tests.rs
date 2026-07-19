use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use computegraph::types::ValueKey;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{ShapeRelation, SymDim};
use tenferro_tensor::{DType, Tensor};

use super::{GraphCompiler, TracedTensor};
use crate::shape_constraint::{
    ConstraintScopeChain, ConstraintSource, LocalShapeConstraint, ScopedShapeConstraint,
    ShapeConstraintScope,
};
use crate::{Error, ShapeConstraintEvalError};

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScopedConstraintExtension {
    outputs: usize,
}

impl ExtensionOp for ScopedConstraintExtension {
    fn family_id(&self) -> &'static str {
        "test.graph-scope"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.outputs);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
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
        self.outputs
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let lhs = ctx.input_axis(0, 0)?;
        let rhs = ctx.input_axis(1, 0)?;
        ctx.require_equal(lhs, 2 * rhs)?;
        let dtype = ctx.input_dtype(0)?;
        let shape = ctx.input_shape(0)?.to_vec();
        Ok((0..self.outputs).map(|_| (dtype, shape.clone())).collect())
    }
}

fn constrained(outputs: usize) -> Vec<TracedTensor> {
    constrained_shapes(outputs, 6, 3)
}

fn constrained_shapes(outputs: usize, lhs_dim: usize, rhs_dim: usize) -> Vec<TracedTensor> {
    let lhs = TracedTensor::from_vec_col_major(vec![lhs_dim], vec![1.0_f64; lhs_dim]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![rhs_dim], vec![2.0_f64; rhs_dim]).unwrap();
    crate::extension::apply(
        Arc::new(ScopedConstraintExtension { outputs }),
        &[&lhs, &rhs],
    )
    .unwrap()
}

#[test]
fn graph_compiler_rejects_concrete_scoped_constraint_contradiction() {
    let output = constrained_shapes(1, 7, 3).remove(0);

    let error = GraphCompiler::new().compile(&output).unwrap_err();

    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "test.graph-scope",
            relation: ShapeRelation::Equal,
            lhs_value: 7,
            rhs_value: 6,
            ..
        }
    ));
}

#[test]
fn graph_compiler_discharges_equal_concrete_scope_without_guard() {
    let output = constrained_shapes(1, 6, 3).remove(0);

    let program = GraphCompiler::new().compile(&output).unwrap();

    assert!(program.exec.shape_guards.is_empty());
}

#[test]
fn constraint_scope_survives_unary_and_reshape_and_discharges_concretely() {
    let extension = constrained(1).remove(0);
    let output = extension.neg().unwrap().reshape(&[6]).unwrap();

    assert_eq!(output.constraint_scopes.materialize().len(), 1);
    let program = GraphCompiler::new().compile(&output).unwrap();
    assert!(program.exec.shape_guards.is_empty());
}

#[test]
fn constraint_scope_shared_unary_branches_merge_and_discharge_without_duplication() {
    let extension = constrained(1).remove(0);
    let left = extension.neg().unwrap();
    let right = extension.neg().unwrap();
    let output = left.add(&right).unwrap();

    assert_eq!(output.constraint_scopes.materialize().len(), 1);
    let program = GraphCompiler::new().compile(&output).unwrap();
    assert!(program.exec.shape_guards.is_empty());
}

#[test]
fn compile_many_clones_shared_constraint_scope_once() {
    let outputs = constrained(2);
    let mut compiler = GraphCompiler::new();

    let (program, scope_clones) = super::test_support::with_constraint_scope_clone_count(|| {
        compiler.compile_many(&[&outputs[0], &outputs[1]])
    });

    assert_eq!(program.unwrap().output_count(), 2);
    assert_eq!(scope_clones, 1);
}

#[test]
fn constraint_scope_multi_output_keeps_other_live_and_prunes_all_dead() {
    let outputs = constrained_shapes(2, 7, 3);
    assert!(matches!(
        GraphCompiler::new().compile(&outputs[1]),
        Err(Error::ShapeConstraintViolation {
            family: "test.graph-scope",
            lhs_value: 7,
            rhs_value: 6,
            ..
        })
    ));

    let mut unrelated = TracedTensor::from_vec_col_major(vec![1], vec![9.0_f64]).unwrap();
    unrelated.constraint_scopes =
        ConstraintScopeChain::merge([&unrelated.constraint_scopes, &outputs[0].constraint_scopes]);
    let dead_program = GraphCompiler::new().compile(&unrelated).unwrap();
    assert!(dead_program.exec.shape_guards.is_empty());
}

#[test]
fn constraint_scope_checkpoint_preserves_chain() {
    let mut output = constrained(1).remove(0);
    let before = output.constraint_scopes.materialize();

    crate::ad_support::checkpoint_tensor(
        &mut output,
        Arc::new(Tensor::from_vec_col_major(vec![6], vec![3.0_f64; 6]).unwrap()),
    )
    .unwrap();

    let after = output.constraint_scopes.materialize();
    assert_eq!(after.len(), 1);
    assert!(Arc::ptr_eq(&before[0], &after[0]));
}

#[test]
fn constraint_scope_live_missing_input_returns_typed_evaluation_error() {
    let mut output = constrained(1).remove(0);
    let origin = output.graph.values()[output.val].key.clone();
    let missing = ValueKey::Input(crate::traced::next_input_key());
    let scope = Arc::new(ShapeConstraintScope::new(vec![ScopedShapeConstraint {
        origins: vec![origin],
        inputs: vec![missing],
        local: LocalShapeConstraint {
            source: ConstraintSource {
                family_id: "test.missing-scoped-input",
                instruction_index: None,
            },
            relation: ShapeRelation::Equal,
            lhs: DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
            rhs: DimExpr::Const(1),
        },
    }]));
    output.constraint_scopes = ConstraintScopeChain::with_scope(scope, []);

    let error = GraphCompiler::new().compile(&output).unwrap_err();

    assert!(matches!(
        error,
        Error::ShapeConstraintEvaluation {
            family: "test.missing-scoped-input",
            instruction_index: Some(_),
            cause: ShapeConstraintEvalError::MissingInput {
                input_idx: 0,
                input_count: 1,
            },
            ..
        }
    ));
}

#[test]
fn constraint_scope_live_missing_axis_returns_typed_evaluation_error() {
    let mut output = constrained(1).remove(0);
    let origin = output.graph.values()[output.val].key.clone();
    let input = output.graph.operations()[0].inputs[0].clone();
    let input = match input {
        computegraph::types::ValueRef::External(key) => key,
        computegraph::types::ValueRef::Local(local) => output.graph.values()[local].key.clone(),
    };
    let scope = Arc::new(ShapeConstraintScope::new(vec![ScopedShapeConstraint {
        origins: vec![origin],
        inputs: vec![input],
        local: LocalShapeConstraint {
            source: ConstraintSource {
                family_id: "test.missing-scoped-axis",
                instruction_index: None,
            },
            relation: ShapeRelation::Equal,
            lhs: DimExpr::InputDim {
                input_idx: 0,
                axis: 99,
            },
            rhs: DimExpr::Const(1),
        },
    }]));
    output.constraint_scopes = ConstraintScopeChain::with_scope(scope, []);

    let error = GraphCompiler::new().compile(&output).unwrap_err();

    assert!(matches!(
        error,
        Error::ShapeConstraintEvaluation {
            family: "test.missing-scoped-axis",
            instruction_index: Some(_),
            cause: ShapeConstraintEvalError::AxisOutOfBounds {
                input_idx: 0,
                axis: 99,
                rank: 1,
            },
            ..
        }
    ));
}

fn attach_violated_scope_to_layout_output(
    mut output: TracedTensor,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
) -> TracedTensor {
    let origin = output.graph.values()[output.val].key.clone();
    let lhs_input = lhs.graph.values()[lhs.val].key.clone();
    let rhs_input = rhs.graph.values()[rhs.val].key.clone();
    let scope = Arc::new(ShapeConstraintScope::new(vec![ScopedShapeConstraint {
        origins: vec![origin],
        inputs: vec![lhs_input, rhs_input],
        local: LocalShapeConstraint {
            source: ConstraintSource {
                family_id: "test.eliminated-layout-scope",
                instruction_index: None,
            },
            relation: ShapeRelation::Equal,
            lhs: DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
            rhs: DimExpr::Mul(
                Box::new(DimExpr::Const(2)),
                Box::new(DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                }),
            ),
        },
    }]));
    output.constraint_scopes = ConstraintScopeChain::with_scope(scope, []);
    output
}

fn assert_eliminated_layout_keeps_live_constraint(output: TracedTensor) {
    let error = GraphCompiler::new()
        .compile(&output)
        .expect_err("optimizer elimination must not discard a live graph-scoped constraint");
    assert!(
        matches!(
            error,
            Error::ShapeConstraintViolation {
                family: "test.eliminated-layout-scope",
                lhs_value: 7,
                rhs_value: 6,
                ..
            }
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn graph_scoped_constraint_survives_identity_reshape_elimination() {
    let lhs = TracedTensor::from_vec_col_major(vec![7], vec![1.0_f64; 7]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let base = lhs.add(&rhs.reduce_sum(Some(&[0])).unwrap()).unwrap();
    let output = base.reshape(&[7]).unwrap();

    assert_eliminated_layout_keeps_live_constraint(attach_violated_scope_to_layout_output(
        output, &lhs, &rhs,
    ));
}

#[test]
fn graph_scoped_constraint_survives_identity_transpose_elimination() {
    let lhs = TracedTensor::from_vec_col_major(vec![7], vec![1.0_f64; 7]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let base = lhs.add(&rhs.reduce_sum(Some(&[0])).unwrap()).unwrap();
    let output = base.transpose(&[0]).unwrap();

    assert_eliminated_layout_keeps_live_constraint(attach_violated_scope_to_layout_output(
        output, &lhs, &rhs,
    ));
}
