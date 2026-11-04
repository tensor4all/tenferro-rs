//! Contracts for the shape obligations an extension contributes.
//!
//! An extension records symbolic equalities through
//! [`tenferro_ops::ExtensionShapeContext`]. The semantic program keeps them as
//! shape guards and discharges the ones it can already decide; the ones whose
//! operands are concrete at compilation are evaluated there, so a contradiction
//! is rejected at compile time and only an undecided equality reaches execution.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_ops::{ShapeRelation, SymDim};
use tenferro_tensor::{DType, Tensor};

use super::GraphCompiler;
use crate::{CompiledGraph, Error, TracedTensor};

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
        let lhs = ctx.input_axis(0, 0)?;
        let rhs = ctx.input_axis(1, 0)?;
        ctx.require_equal(lhs, 2 * rhs)?;
        let dtype = ctx.input_dtype(0)?;
        let shape = ctx.input_shape(0)?.to_vec();
        Ok((0..self.outputs).map(|_| (dtype, shape.clone())).collect())
    }
}

/// Build a trace with two symbolic inputs and `outputs` constrained outputs.
fn symbolic_constrained(outputs: usize) -> (TracedTensor, TracedTensor, Vec<TracedTensor>) {
    let trace = crate::TraceContext::new();
    let lhs = trace.input_symbolic_shape(DType::F64, 1).unwrap();
    let rhs = trace.input_symbolic_shape(DType::F64, 1).unwrap();
    let outputs = crate::extension::apply(
        Arc::new(ScopedConstraintExtension { outputs }),
        &[&lhs, &rhs],
    )
    .unwrap();
    (lhs, rhs, outputs)
}

/// Build a trace whose two inputs carry data but advertise symbolic extents.
fn symbolic_default_bound(outputs: usize, lhs_dim: usize, rhs_dim: usize) -> Vec<TracedTensor> {
    let trace = crate::TraceContext::new();
    let lhs = trace
        .from_tensor_symbolic_shape(
            Tensor::from_vec_col_major(vec![lhs_dim], vec![1.0_f64; lhs_dim]).unwrap(),
        )
        .unwrap();
    let rhs = trace
        .from_tensor_symbolic_shape(
            Tensor::from_vec_col_major(vec![rhs_dim], vec![2.0_f64; rhs_dim]).unwrap(),
        )
        .unwrap();
    crate::extension::apply(
        Arc::new(ScopedConstraintExtension { outputs }),
        &[&lhs, &rhs],
    )
    .unwrap()
}

/// Build a trace whose two inputs carry concrete default bindings.
fn constrained_shapes(outputs: usize, lhs_dim: usize, rhs_dim: usize) -> Vec<TracedTensor> {
    let trace = crate::TraceContext::new();
    let lhs = trace
        .from_vec_col_major(vec![lhs_dim], vec![1.0_f64; lhs_dim])
        .unwrap();
    let rhs = trace
        .from_vec_col_major(vec![rhs_dim], vec![2.0_f64; rhs_dim])
        .unwrap();
    crate::extension::apply(
        Arc::new(ScopedConstraintExtension { outputs }),
        &[&lhs, &rhs],
    )
    .unwrap()
}

fn compile_with_shapes(
    output: &TracedTensor,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    lhs_dim: usize,
    rhs_dim: usize,
) -> crate::Result<CompiledGraph> {
    let lhs_shape = [lhs_dim];
    let rhs_shape = [rhs_dim];
    GraphCompiler::new().compile_with_input_specs(
        output,
        &[
            (lhs, DType::F64, lhs_shape.as_slice()),
            (rhs, DType::F64, rhs_shape.as_slice()),
        ],
    )
}

fn assert_violation(program: crate::Result<CompiledGraph>, lhs_value: usize, rhs_value: usize) {
    let error = program.expect_err("a contradicting obligation must be rejected");
    assert!(
        matches!(
            error,
            Error::ShapeConstraintViolation {
                family: "test.graph-scope",
                relation: ShapeRelation::Equal,
                lhs_value: actual_lhs,
                rhs_value: actual_rhs,
                ..
            } if actual_lhs == lhs_value && actual_rhs == rhs_value
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn graph_compiler_rejects_concrete_scoped_constraint_contradiction() {
    let (lhs, rhs, outputs) = symbolic_constrained(1);

    assert_violation(compile_with_shapes(&outputs[0], &lhs, &rhs, 7, 3), 7, 6);
}

#[test]
fn graph_compiler_discharges_equal_concrete_scope_without_guard() {
    let (lhs, rhs, outputs) = symbolic_constrained(1);

    let program = compile_with_shapes(&outputs[0], &lhs, &rhs, 6, 3).unwrap();

    assert!(program.program().shape_guards().is_empty());
}

#[test]
fn graph_compiler_keeps_default_bound_scope_as_runtime_guard() {
    // A default binding fixes the data but advertises symbolic extents, so the
    // equality stays undecided and the compiled program keeps it as a guard.
    let output = symbolic_default_bound(1, 6, 3).remove(0);

    let program = GraphCompiler::new().compile(&output).unwrap();

    let guards = program.program().shape_guards();
    assert_eq!(guards.len(), 1);
    assert_eq!(
        guards[0].relation(),
        crate::program::ProgramShapeRelation::Equal
    );
}

#[test]
fn constraint_scope_survives_unary_and_reshape_and_discharges_concretely() {
    let (lhs, rhs, mut outputs) = symbolic_constrained(1);
    let extension = outputs.remove(0);
    let output = extension.neg().unwrap().reshape([6]).unwrap();

    let undecided = GraphCompiler::new().compile(&output).unwrap();
    assert_eq!(undecided.program().shape_guards().len(), 1);

    let program = compile_with_shapes(&output, &lhs, &rhs, 6, 3).unwrap();
    assert!(program.program().shape_guards().is_empty());
}

#[test]
fn constraint_scope_shared_unary_branches_merge_and_discharge_without_duplication() {
    let (lhs, rhs, mut outputs) = symbolic_constrained(1);
    let extension = outputs.remove(0);
    let left = extension.neg().unwrap();
    let right = extension.neg().unwrap();
    let output = left.add(&right).unwrap();

    let undecided = GraphCompiler::new().compile(&output).unwrap();
    assert_eq!(
        undecided.program().shape_guards().len(),
        1,
        "a constraint carried through two branches stays one obligation"
    );

    let program = compile_with_shapes(&output, &lhs, &rhs, 6, 3).unwrap();
    assert!(program.program().shape_guards().is_empty());
}

#[test]
fn compile_many_clones_shared_constraint_scope_once() {
    let (_, _, outputs) = symbolic_constrained(2);
    let mut compiler = GraphCompiler::new();

    let program = compiler.compile_many(&[&outputs[0], &outputs[1]]).unwrap();

    assert_eq!(program.output_count(), 2);
    // Two compiled outputs of one constrained operation share a single obligation.
    assert_eq!(program.program().shape_guards().len(), 1);
}

#[test]
fn constraint_scope_multi_output_keeps_other_live_and_prunes_all_dead() {
    let outputs = constrained_shapes(2, 7, 3);
    let error = GraphCompiler::new().compile(&outputs[1]).unwrap_err();
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "test.graph-scope",
            lhs_value: 7,
            rhs_value: 6,
            ..
        }
    ));

    // A value that does not depend on the constrained operation carries no
    // obligation of its own, even though it shares the context.
    let unrelated = outputs[0]
        .context()
        .from_vec_col_major(vec![1], vec![9.0_f64])
        .unwrap();
    let dead_program = GraphCompiler::new().compile(&unrelated).unwrap();
    assert!(dead_program.program().shape_guards().is_empty());
}

#[test]
fn constraint_scope_checkpoint_preserves_chain() {
    // A checkpoint replaces the constrained value by a bound alias, and the AD
    // snapshot still sees the pre-checkpoint definition. The obligation therefore
    // survives the checkpoint instead of being dropped with the replaced value.
    let (_, _, mut outputs) = symbolic_constrained(1);
    let constrained = outputs.remove(0);
    let concrete = Arc::new(crate::checkpoint::RetainedValue::from_tensor(
        Tensor::from_vec_col_major(vec![6], vec![3.0_f64; 6]).unwrap(),
    ));
    let value = constrained
        .context()
        .checkpoint_value(constrained.trace_value(), concrete)
        .unwrap();

    let snapshot = constrained.context().ad_snapshot(&[value]).unwrap();

    let guards = snapshot.program.shape_guards();
    assert_eq!(guards.len(), 1);
    assert_eq!(
        guards[0].relation(),
        crate::program::ProgramShapeRelation::Equal
    );
}

#[test]
fn graph_scoped_constraint_survives_identity_reshape_elimination() {
    // The reshape is a layout no-op, and the obligation of the operation beneath
    // it is still checked against the bound shapes.
    let base = constrained_shapes(1, 7, 3).remove(0);
    let output = base.reshape([7]).unwrap();

    let error = GraphCompiler::new()
        .compile(&output)
        .expect_err("an eliminated identity reshape must not hide the obligation");
    assert!(
        matches!(
            error,
            Error::ShapeConstraintViolation {
                family: "test.graph-scope",
                lhs_value: 7,
                rhs_value: 6,
                ..
            }
        ),
        "unexpected error: {error:?}"
    );
}

#[test]
fn graph_scoped_constraint_survives_identity_transpose_elimination() {
    // The transpose permutes nothing, so lowering may drop it; the symbolic
    // obligation must still be in the staged program afterwards.
    let (_, _, mut outputs) = symbolic_constrained(1);
    let output = outputs.remove(0).transpose(&[0]).unwrap();

    let program = GraphCompiler::new().compile(&output).unwrap();
    assert_eq!(program.program().shape_guards().len(), 1);
    let staged = crate::compiler::semantic_staging::stage_semantic_program(
        program.program(),
        program.compiler_options(),
    )
    .unwrap();
    assert_eq!(
        staged.shape_guards.len(),
        1,
        "lowering must not drop a live shape obligation"
    );
}

/// Explicit input specs define the compiled input order, independent of the
/// order in which the context created the placeholders.
#[test]
fn compile_with_input_specs_reorders_inputs_by_explicit_order() {
    let trace = crate::TraceContext::new();
    let a = trace.input_symbolic_shape(crate::DType::F64, 1).unwrap();
    let b = trace.input_symbolic_shape(crate::DType::F64, 1).unwrap();
    let a_key = a.input_key().expect("symbolic traced tensor has input key");
    let b_key = b.input_key().expect("symbolic traced tensor has input key");
    let c = a.mul(&b).unwrap();

    // Request b first, then a.
    let program = GraphCompiler::new()
        .compile_with_input_specs(
            &c,
            &[(&b, crate::DType::F64, &[2]), (&a, crate::DType::F64, &[2])],
        )
        .unwrap();

    assert_eq!(program.input_count(), 2);
    assert_eq!(program.input_keys().len(), 2);
    assert_eq!(
        program.input_key_index(&b_key),
        Some(0),
        "b must be first in explicit order"
    );
    assert_eq!(
        program.input_key_index(&a_key),
        Some(1),
        "a must be second in explicit order"
    );
}
