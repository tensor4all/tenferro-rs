use super::*;
use crate::program::{CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder};
use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;
use tenferro_ops::{
    ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp},
    SymDim,
};

#[test]
fn compile_publishes_semantic_program_and_separate_default_bindings() {
    let trace = crate::TraceContext::new();
    let input = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let output = input.neg().unwrap();

    let program = GraphCompiler::new().compile(&output).unwrap();

    assert_eq!(program.program().inputs().len(), 1);
    assert_eq!(program.program().outputs().len(), 1);
    assert_eq!(program.program().operations().count(), 1);
    assert_eq!(program.bindings().len(), 1);
    assert_eq!(
        program
            .program()
            .value_metadata(program.program().inputs()[0])
            .unwrap()
            .shape(),
        &[ShapeExtent::Exact(DimExpr::Const(2))]
    );
    assert_eq!(
        program
            .program()
            .value_metadata(program.program().outputs()[0])
            .unwrap()
            .dtype(),
        DType::F64
    );
}

#[test]
fn compiler_options_constructor_keeps_the_given_options() {
    let options = CompilerOptions {
        optimizer: crate::OptimizerConfig {
            dot_decomposer: true,
            ..crate::OptimizerConfig::default()
        },
    };
    let compiler = GraphCompiler::with_compiler_options(options);
    assert!(compiler.compiler_options().optimizer.dot_decomposer);
    assert!(compiler.extension_caches().is_empty());
}

#[test]
fn compile_many_refuses_outputs_from_two_contexts() {
    let first = crate::TraceContext::new();
    let second = crate::TraceContext::new();
    let left = first
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap()
        .neg()
        .unwrap();
    let right = second
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap()
        .neg()
        .unwrap();

    let error = GraphCompiler::new()
        .compile_many(&[&left, &right])
        .unwrap_err();
    assert!(error.to_string().contains("different contexts"), "{error}");
}

#[test]
fn compile_preserves_symbolic_default_input_extent_identity() {
    let trace = crate::TraceContext::new();
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let input = trace.from_tensor_symbolic_shape(tensor).unwrap();
    let output = input.neg().unwrap();

    let program = GraphCompiler::new().compile(&output).unwrap();

    assert!(matches!(
        program
            .program()
            .value_metadata(program.program().inputs()[0])
            .unwrap()
            .shape(),
        [ShapeExtent::Exact(DimExpr::InputDim {
            input_idx: 0,
            axis: 0
        })]
    ));
}

#[test]
fn compile_frozen_program_rejects_static_unbound_shape_guard_mismatch() {
    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Neg, &[lhs]).unwrap()[0];
    builder
        .add_shape_guards_to_output(
            output,
            [ProgramShapeGuard::new(
                ProgramShapeRelation::Equal,
                DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                },
            )],
        )
        .unwrap();
    let frozen = builder.finish(&[output]).unwrap();

    let err = GraphCompiler::new()
        .compile_frozen_program(&frozen)
        .unwrap_err();

    assert!(matches!(
        err,
        Error::ShapeConstraintViolation {
            lhs_value: 2,
            rhs_value: 3,
            ..
        }
    ));
    let _ = rhs;
}

#[test]
fn compile_frozen_program_defers_dynamic_unbound_shape_guard() {
    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::from_metadata(
            ProgramValueMetadata::from_extents(DType::F64, [ShapeExtent::Unknown]),
        ))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Neg, &[lhs]).unwrap()[0];
    builder
        .add_shape_guards_to_output(
            output,
            [ProgramShapeGuard::new(
                ProgramShapeRelation::Equal,
                DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                },
            )],
        )
        .unwrap();
    let frozen = builder.finish(&[output]).unwrap();

    GraphCompiler::new()
        .compile_frozen_program(&frozen)
        .unwrap();
    let _ = rhs;
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PrunableTestOp {
    pruned: bool,
}

impl ExtensionOp for PrunableTestOp {
    fn family_id(&self) -> &'static str {
        "tenferro-runtime.test-prunable.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(u8::from(self.pruned));
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|that| self == that)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        if self.pruned {
            1
        } else {
            3
        }
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
        Ok((0..self.output_count())
            .map(|_| (dtype, shape.clone()))
            .collect())
    }

    fn prune_outputs(&self, live_outputs: &[bool]) -> Option<Arc<dyn ExtensionOp>> {
        (!self.pruned && live_outputs == [false, true, false])
            .then(|| Arc::new(Self { pruned: true }) as Arc<dyn ExtensionOp>)
    }
}

#[test]
fn compile_prunes_extension_outputs_with_replacement_op() {
    let trace = crate::TraceContext::new();
    let input = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let outputs =
        crate::extension::apply(Arc::new(PrunableTestOp { pruned: false }), &[&input]).unwrap();

    let program = GraphCompiler::new().compile(&outputs[1]).unwrap();
    let staging = stage_semantic_program(program.program(), program.compiler_options()).unwrap();
    let pruned_instruction = staging
        .instructions
        .iter()
        .find_map(|inst| match &inst.op {
            crate::exec::ExecOp::Extension(op)
                if op.family_id() == "tenferro-runtime.test-prunable.v1" =>
            {
                Some((
                    inst.output_slots.clone(),
                    format!("{op:?}"),
                    op.output_count(),
                ))
            }
            _ => None,
        })
        .expect("compiled program should contain the test extension");

    assert_eq!(pruned_instruction.0.len(), 1);
    assert!(pruned_instruction.1.contains("pruned: true"));
    assert_eq!(pruned_instruction.2, 1);
}

/// `finish` + `compile_traced_graph` keeps the same input keys, order and
/// pruning as `compile_many` (#2552 S4); before, the finish route compiled
/// with no input keys.
#[test]
fn finish_route_and_compile_many_agree_on_input_keys_and_pruning() {
    let trace = crate::TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let unused = trace
        .from_vec_col_major(vec![2], vec![9.0_f64, 9.0])
        .unwrap();
    let b = trace
        .from_vec_col_major(vec![2], vec![3.0_f64, 4.0])
        .unwrap();
    let c = b.mul(&a).unwrap();
    let _ = unused.neg().unwrap();

    let many = GraphCompiler::new().compile_many(&[&c]).unwrap();
    let graph = trace.finish(&[c.trace_value()]).unwrap();
    let finished = GraphCompiler::new().compile_traced_graph(&graph).unwrap();

    assert_eq!(finished.input_keys(), many.input_keys());
    assert_eq!(finished.input_count(), many.input_count());
    assert_eq!(finished.input_count(), 2, "the unused input is pruned");
    let unused_key = unused.input_key().expect("bound input has a key");
    assert!(!finished.input_keys().contains(&unused_key));
}

#[test]
fn compiled_graph_input_keys_preserve_order_for_binary_graph() {
    let trace = crate::TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let b = trace
        .from_vec_col_major(vec![2], vec![3.0_f64, 4.0])
        .unwrap();
    let a_key = a.input_key().expect("concrete traced tensor has input key");
    let b_key = b.input_key().expect("concrete traced tensor has input key");
    let c = a.mul(&b).unwrap();

    let program = GraphCompiler::new().compile_many(&[&c]).unwrap();

    assert_eq!(program.input_count(), 2);
    assert_eq!(program.input_keys().len(), 2);
    let keys: Vec<_> = program.input_keys().to_vec();
    assert!(keys.contains(&a_key), "input_keys must contain a's key");
    assert!(keys.contains(&b_key), "input_keys must contain b's key");
    // input_key_index maps each key to its position
    assert_eq!(
        program.input_key_index(&a_key),
        keys.iter().position(|k| k == &a_key)
    );
    assert_eq!(
        program.input_key_index(&b_key),
        keys.iter().position(|k| k == &b_key)
    );
}
