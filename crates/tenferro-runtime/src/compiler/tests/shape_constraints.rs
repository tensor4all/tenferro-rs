use std::{any::Any, sync::Arc};

use computegraph::compile::{CompiledProgram, Instruction};
use tenferro_ops::{
    dim_expr::DimExpr, ext_op::ExtensionOp, std_tensor_op::StdTensorOp, ShapeRelation, SymDim,
};
use tenferro_tensor::DType;

use super::{compile_std_to_exec, dim_shape, make_std_instr};
use crate::{Error, ShapeGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstraintFixture {
    ScaledAxisEquality,
    ScaledAxisEqualityMultiOutput,
    WithoutOutput,
    InvalidAxis,
}

impl ExtensionOp for ConstraintFixture {
    fn family_id(&self) -> &'static str {
        match self {
            Self::ScaledAxisEquality => "test.compiler-scaled-axis-equality.v1",
            Self::ScaledAxisEqualityMultiOutput => {
                "test.compiler-scaled-axis-equality-multi-output.v1"
            }
            Self::WithoutOutput => "test.compiler-constraint-without-output.v1",
            Self::InvalidAxis => "test.compiler-invalid-constraint-axis.v1",
        }
    }

    fn payload_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        hasher.write_u8(match self {
            Self::ScaledAxisEquality => 0,
            Self::ScaledAxisEqualityMultiOutput => 1,
            Self::WithoutOutput => 2,
            Self::InvalidAxis => 3,
        });
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        match self {
            Self::ScaledAxisEquality
            | Self::ScaledAxisEqualityMultiOutput
            | Self::WithoutOutput => 2,
            Self::InvalidAxis => 1,
        }
    }

    fn output_count(&self) -> usize {
        match self {
            Self::ScaledAxisEqualityMultiOutput => 2,
            Self::WithoutOutput => 0,
            Self::ScaledAxisEquality | Self::InvalidAxis => 1,
        }
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        match self {
            Self::ScaledAxisEquality | Self::ScaledAxisEqualityMultiOutput => {
                let lhs = ctx.input_axis(0, 0)?;
                let rhs = ctx.input_axis(1, 0)?;
                ctx.require_equal(lhs, rhs * 2)?;
                let meta = (ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec());
                Ok(match self {
                    Self::ScaledAxisEqualityMultiOutput => vec![meta.clone(), meta],
                    _ => vec![meta],
                })
            }
            Self::WithoutOutput => {
                ctx.require_axes_equal((0, 0), (1, 0))?;
                Ok(Vec::new())
            }
            Self::InvalidAxis => {
                ctx.require_axes_equal((0, 1), (0, 0))?;
                Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
            }
        }
    }
}

fn scaled_axis_program(
    instructions: Vec<Instruction<StdTensorOp>>,
) -> CompiledProgram<StdTensorOp> {
    CompiledProgram {
        instructions,
        input_slots: vec![0, 1],
        output_slots: vec![2],
        n_slots: 3,
    }
}

fn assert_guard_equation(guard: &ShapeGuard, expected_lhs: DimExpr, expected_rhs: DimExpr) {
    assert!(
        (guard.lhs == expected_lhs && guard.rhs == expected_rhs)
            || (guard.lhs == expected_rhs && guard.rhs == expected_lhs),
        "unexpected normalized guard: {guard:?}"
    );
}

#[test]
fn compiler_retains_shape_guards_and_rejects_concrete_contradictions() {
    let extension = StdTensorOp::Extension(Arc::new(ConstraintFixture::ScaledAxisEquality));
    let program = scaled_axis_program(vec![make_std_instr(extension, vec![0, 1], vec![2])]);
    let dtypes = [DType::F64, DType::F64];

    let concrete =
        compile_std_to_exec(&program, &dtypes, &[dim_shape(&[6]), dim_shape(&[3])]).unwrap();
    assert!(concrete.shape_guards.is_empty());

    assert!(matches!(
        compile_std_to_exec(&program, &dtypes, &[dim_shape(&[7]), dim_shape(&[3])],),
        Err(Error::ShapeConstraintViolation {
            family: "test.compiler-scaled-axis-equality.v1",
            instruction_index: Some(0),
            relation: ShapeRelation::Equal,
            lhs_value: 7,
            rhs_value: 6,
            ..
        })
    ));

    let symbolic = compile_std_to_exec(
        &program,
        &dtypes,
        &[
            vec![DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            }],
            vec![DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }],
        ],
    )
    .unwrap();
    assert_eq!(symbolic.shape_guards.len(), 1);
}

#[test]
fn compiler_retains_shape_guards_with_nested_reordered_input_expressions() {
    let extension = StdTensorOp::Extension(Arc::new(ConstraintFixture::ScaledAxisEquality));
    let program = scaled_axis_program(vec![make_std_instr(extension, vec![1, 0], vec![2])]);
    let first = DimExpr::add(
        DimExpr::InputDim {
            input_idx: 3,
            axis: 1,
        },
        DimExpr::Const(1),
    );
    let second = DimExpr::floor_div(
        DimExpr::mul(
            DimExpr::InputDim {
                input_idx: 2,
                axis: 0,
            },
            DimExpr::Const(4),
        ),
        DimExpr::Const(2),
    );

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[vec![first], vec![second]],
    )
    .unwrap();

    assert_eq!(exec.shape_guards.len(), 1);
    let guard = &exec.shape_guards[0];
    assert_eq!(guard.source.instruction_index, Some(0));
    let normalized_first = DimExpr::add(
        DimExpr::Const(1),
        DimExpr::InputDim {
            input_idx: 3,
            axis: 1,
        },
    );
    let normalized_second = DimExpr::floor_div(
        DimExpr::mul(
            DimExpr::Const(4),
            DimExpr::InputDim {
                input_idx: 2,
                axis: 0,
            },
        ),
        DimExpr::Const(2),
    );
    assert_guard_equation(
        guard,
        normalized_second,
        DimExpr::mul(DimExpr::Const(2), normalized_first),
    );
}

#[test]
fn compiler_shape_guard_provenance_uses_final_instruction_indices() {
    let extension = || StdTensorOp::Extension(Arc::new(ConstraintFixture::ScaledAxisEquality));
    let program = CompiledProgram {
        instructions: vec![
            make_std_instr(StdTensorOp::Neg, vec![0], vec![2]),
            make_std_instr(extension(), vec![1, 0], vec![3]),
            make_std_instr(extension(), vec![0, 1], vec![4]),
        ],
        input_slots: vec![0, 1],
        output_slots: vec![3, 4],
        n_slots: 5,
    };
    let first = DimExpr::InputDim {
        input_idx: 0,
        axis: 0,
    };
    let second = DimExpr::InputDim {
        input_idx: 1,
        axis: 0,
    };

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[vec![first.clone()], vec![second.clone()]],
    )
    .unwrap();

    assert_eq!(exec.instructions.len(), 2);
    assert_eq!(exec.shape_guards.len(), 2);
    let first_guard = exec
        .shape_guards
        .iter()
        .find(|guard| guard.source.instruction_index == Some(0))
        .unwrap();
    assert_guard_equation(
        first_guard,
        second.clone(),
        DimExpr::mul(DimExpr::Const(2), first.clone()),
    );
    let second_guard = exec
        .shape_guards
        .iter()
        .find(|guard| guard.source.instruction_index == Some(1))
        .unwrap();
    assert_guard_equation(second_guard, first, DimExpr::mul(DimExpr::Const(2), second));
}

#[test]
fn compiler_preserves_multi_output_constraint_when_first_output_is_unused() {
    let program = CompiledProgram {
        instructions: vec![make_std_instr(
            StdTensorOp::Extension(Arc::new(ConstraintFixture::ScaledAxisEqualityMultiOutput)),
            vec![0, 1],
            vec![2, 3],
        )],
        input_slots: vec![0, 1],
        output_slots: vec![3],
        n_slots: 4,
    };

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[
            vec![DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            }],
            vec![DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }],
        ],
    )
    .unwrap();

    assert_eq!(exec.instructions.len(), 1);
    assert_eq!(exec.instructions[0].output_slots, vec![2, 3]);
    assert_eq!(exec.shape_guards.len(), 1);
    assert_eq!(exec.shape_guards[0].source.instruction_index, Some(0));
}

#[test]
fn compiler_rejects_duplicate_output_slot_producers_before_optimization() {
    let program = CompiledProgram {
        instructions: vec![
            make_std_instr(StdTensorOp::Neg, vec![0], vec![1]),
            make_std_instr(StdTensorOp::Neg, vec![0], vec![1]),
        ],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    };

    assert!(matches!(
        compile_std_to_exec(&program, &[DType::F64], &[dim_shape(&[2])]),
        Err(Error::InvalidCompiledGraph { ref message })
            if message.contains("output slot 1")
                && message.contains("instructions 0 and 1")
    ));
}

#[test]
fn compiler_rejects_out_of_range_output_slot_before_optimization() {
    let program = CompiledProgram {
        instructions: vec![make_std_instr(StdTensorOp::Neg, vec![0], vec![2])],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    };

    assert!(matches!(
        compile_std_to_exec(&program, &[DType::F64], &[dim_shape(&[2])]),
        Err(Error::InvalidCompiledGraph { message })
            if message == "output slot 2 is outside slot table of length 2"
    ));
}

#[test]
fn compiler_rejects_constraint_origin_eliminated_from_final_stream() {
    let program = CompiledProgram {
        instructions: vec![make_std_instr(
            StdTensorOp::Extension(Arc::new(ConstraintFixture::ScaledAxisEquality)),
            vec![0, 1],
            vec![2],
        )],
        input_slots: vec![0, 1],
        output_slots: vec![0],
        n_slots: 3,
    };

    assert!(matches!(
        compile_std_to_exec(
            &program,
            &[DType::F64, DType::F64],
            &[
                vec![DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                }],
                vec![DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                }],
            ],
        ),
        Err(Error::InvalidCompiledGraph { ref message })
            if message.contains("absent from the final instruction stream")
                && message.contains("output slots [2]")
    ));
}

#[test]
fn constraint_origin_without_outputs_hits_instruction_invariant_first() {
    let program = CompiledProgram {
        instructions: vec![make_std_instr(
            StdTensorOp::Extension(Arc::new(ConstraintFixture::WithoutOutput)),
            vec![0, 1],
            vec![],
        )],
        input_slots: vec![0, 1],
        output_slots: vec![],
        n_slots: 2,
    };

    assert!(matches!(
        compile_std_to_exec(
            &program,
            &[DType::F64, DType::F64],
            &[dim_shape(&[2]), dim_shape(&[2])],
        ),
        Err(Error::InvalidCompiledGraph { ref message })
            if message == "instruction has no outputs"
    ));
}

#[test]
fn compiler_preserves_constraint_inference_errors() {
    let program = CompiledProgram {
        instructions: vec![make_std_instr(
            StdTensorOp::Extension(Arc::new(ConstraintFixture::InvalidAxis)),
            vec![0],
            vec![1],
        )],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    };

    let error = compile_std_to_exec(&program, &[DType::F64], &[dim_shape(&[2])]).unwrap_err();
    assert!(
        matches!(
            &error,
            Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
                op: "test.compiler-invalid-constraint-axis.v1",
                message,
            }) if message
                == "extension family \"test.compiler-invalid-constraint-axis.v1\" axis 1 out of bounds for input 0 rank 1"
        ),
        "unexpected compiler error: {error:?}"
    );
}
