use std::collections::HashMap;

use computegraph::compile::{CompiledProgram, Instruction};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeExtent;

use crate::compiler::{compile_std_to_exec_with_options_and_constraints, CompilerOptions};
use crate::exec::ExecProgram;
use crate::program::{SemanticOpRef, SemanticProgram};
use crate::{Error, ErrorPhase, Result};

/// Lower one frozen semantic artifact into the temporary execution staging IR.
///
/// This is the sole forward semantic-to-staging adapter. It remains
/// crate-private and is deleted with execution staging in Phase 5.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "wired into GraphCompiler in Phase 3 A1 task 2")
)]
pub(crate) fn lower_semantic_to_exec_staging(
    program: &SemanticProgram,
    options: CompilerOptions,
) -> Result<ExecProgram> {
    let mut slots = HashMap::with_capacity(program.inputs().len());
    let mut input_slots = Vec::with_capacity(program.inputs().len());
    let mut input_dtypes = Vec::with_capacity(program.inputs().len());
    let mut input_shapes = Vec::with_capacity(program.inputs().len());

    for &input in program.inputs() {
        let slot = slots.len();
        if slots.insert(input, slot).is_some() {
            return Err(invalid_semantic_program(
                "semantic input appears more than once",
            ));
        }
        let metadata = program.value_metadata(input).map_err(|source| {
            Error::runtime_state_source("semantic_staging", ErrorPhase::Compile, source)
        })?;
        input_slots.push(slot);
        input_dtypes.push(metadata.dtype());
        input_shapes.push(exact_shape(metadata.shape())?);
    }

    let mut instructions = Vec::with_capacity(program.operations().len());
    for operation in program.operations() {
        let inputs = operation
            .inputs()
            .iter()
            .map(|input| {
                slots.get(input).copied().ok_or_else(|| {
                    invalid_semantic_program("operation references an unavailable input")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = Vec::with_capacity(operation.outputs().len());
        for &output in operation.outputs() {
            let slot = slots.len();
            if slots.insert(output, slot).is_some() {
                return Err(invalid_semantic_program(
                    "semantic value has more than one producer",
                ));
            }
            outputs.push(slot);
        }
        let operation = match operation.op() {
            SemanticOpRef::Core(core) => StdTensorOp::from(core),
            SemanticOpRef::Extension(extension) => StdTensorOp::Extension(extension.clone_arc()),
        };
        instructions.push(Instruction {
            operation,
            inputs,
            outputs,
        });
    }

    let output_slots = program
        .outputs()
        .iter()
        .map(|output| {
            slots
                .get(output)
                .copied()
                .ok_or_else(|| invalid_semantic_program("program output is unavailable"))
        })
        .collect::<Result<Vec<_>>>()?;
    let staged = CompiledProgram {
        instructions,
        input_slots,
        output_slots,
        n_slots: slots.len(),
    };
    compile_std_to_exec_with_options_and_constraints(
        &staged,
        &input_dtypes,
        &input_shapes,
        options,
        &[],
        &input_shapes,
    )
}

fn exact_shape(extents: &[ShapeExtent<DimExpr>]) -> Result<Vec<DimExpr>> {
    extents
        .iter()
        .map(|extent| match extent {
            ShapeExtent::Exact(expression) => Ok(expression.clone()),
            ShapeExtent::UpperBound(_) | ShapeExtent::Unknown => Err(Error::unsupported(
                "semantic_staging",
                ErrorPhase::Compile,
                "execution staging requires exact symbolic extents",
            )),
        })
        .collect()
}

fn invalid_semantic_program(message: &'static str) -> Error {
    Error::runtime_state("semantic_staging", ErrorPhase::Compile, message)
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::hash::Hasher;
    use std::sync::Arc;

    use tenferro_ops::dim_expr::DimExpr;
    use tenferro_ops::ext_op::{
        ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp,
    };
    use tenferro_ops::{ShapeExtent, SymDim};
    use tenferro_tensor::DType;

    use crate::compiler::CompilerOptions;
    use crate::exec::ExecOp;
    use crate::program::{
        CoreSemanticOp, ProgramInputSpec, ProgramValueMetadata, SemanticProgramBuilder,
    };

    use super::lower_semantic_to_exec_staging;

    #[derive(Clone, Debug)]
    struct PairWithGuard;

    impl ExtensionOp for PairWithGuard {
        fn family_id(&self) -> &'static str {
            "tenferro-runtime.test-pair-with-guard.v1"
        }

        fn payload_hash(&self, hasher: &mut dyn Hasher) {
            hasher.write_u8(1);
        }

        fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
            other.as_any().is::<Self>()
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
            context: &mut tenferro_ops::ExtensionShapeContext<'_>,
        ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
            let extent = context.input_axis(0, 0)?;
            context.require_equal(extent, SymDim::from(2))?;
            let metadata = (context.input_dtype(0)?, context.input_shape(0)?.to_vec());
            Ok(vec![metadata.clone(), metadata])
        }
    }

    #[test]
    fn lowers_ordered_core_program_to_execution_staging() {
        let mut builder = SemanticProgramBuilder::new();
        let lhs = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let rhs = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let sum = builder.add_op(CoreSemanticOp::Add, &[lhs, rhs]).unwrap()[0];
        let neg = builder.add_op(CoreSemanticOp::Neg, &[sum]).unwrap()[0];
        let frozen = builder.finish(&[neg, sum]).unwrap();

        let staging =
            lower_semantic_to_exec_staging(&frozen.program, CompilerOptions::default()).unwrap();

        assert_eq!(staging.input_slots, vec![0, 1]);
        assert_eq!(staging.output_slots, vec![3, 2]);
        assert_eq!(staging.n_slots, 4);
        assert_eq!(staging.instructions.len(), 2);
        assert!(matches!(staging.instructions[0].op, ExecOp::Add));
        assert!(matches!(staging.instructions[1].op, ExecOp::Negate));
    }

    #[test]
    fn preserves_extension_outputs_and_guards_but_not_bindings() {
        let mut builder = SemanticProgramBuilder::new();
        let input = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                }],
            ))
            .unwrap();
        builder
            .bind_input(
                input,
                Arc::new(
                    tenferro_tensor::Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
                        .unwrap(),
                ),
            )
            .unwrap();
        let pair = builder
            .add_extension(Arc::new(PairWithGuard), &[input])
            .unwrap();
        let frozen = builder.finish(&[pair[1], pair[0]]).unwrap();

        let staging =
            lower_semantic_to_exec_staging(&frozen.program, CompilerOptions::default()).unwrap();

        assert_eq!(frozen.bindings.len(), 1);
        assert_eq!(staging.input_slots, vec![0]);
        assert_eq!(staging.output_slots, vec![2, 1]);
        assert_eq!(staging.instructions.len(), 1);
        assert_eq!(staging.instructions[0].output_slots, vec![1, 2]);
        assert!(matches!(
            &staging.instructions[0].op,
            ExecOp::Extension(op)
                if op.family_id() == "tenferro-runtime.test-pair-with-guard.v1"
        ));
        assert_eq!(staging.shape_guards.len(), 1);
    }

    #[test]
    fn rejects_non_exact_semantic_extents_before_staging() {
        let mut builder = SemanticProgramBuilder::new();
        let input = builder
            .input(ProgramInputSpec::from_metadata(
                ProgramValueMetadata::from_extents(
                    DType::F64,
                    [ShapeExtent::UpperBound(DimExpr::Const(8))],
                ),
            ))
            .unwrap();
        let frozen = builder.finish(&[input]).unwrap();

        let error = lower_semantic_to_exec_staging(&frozen.program, CompilerOptions::default())
            .unwrap_err();

        assert!(matches!(error, crate::Error::Unsupported { .. }));
        assert_eq!(error.phase(), Some(crate::ErrorPhase::Compile));
    }
}
