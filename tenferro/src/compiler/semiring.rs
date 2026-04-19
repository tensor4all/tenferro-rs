//! Semiring compile path (non-mainline, legacy).
//!
//! This submodule holds the `SemiringOp<Alg>` → `ExecProgram` lowering. Per
//! `docs/design/design_v3/30-algebra-and-tropical.md` ("Recommended Fate Of
//! `SemiringOp`") this path is **not** the traced AD mainline and is scheduled
//! for removal in Stage 6 of the `design_v3` migration plan. It remains
//! functional so the legacy tropical integration tests
//! (`tenferro/tests/tropical.rs`) continue to cover the semiring pipeline
//! until Stage 4 ships the external tropical crate.
//!
//! New code must not depend on this module. Mainline traced code goes through
//! [`super::compile_std_to_exec`].

use computegraph::compile::CompiledProgram;
use tenferro_algebra::Algebra;
use tenferro_ops::dim_expr::DimExpr;
#[allow(deprecated)]
use tenferro_ops::semiring_op::SemiringOp;
use tenferro_ops::semiring_op_kind::SemiringOpKind;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::TensorScalar;

use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use crate::shape_infer::infer_output_shapes;

use super::{dot_dimension_sorter, populate_last_use, transpose_folding};

/// Compile a `SemiringOp<Alg>` graph into an `ExecProgram`.
///
/// **Deprecated**: non-mainline per
/// `docs/design/design_v3/30-algebra-and-tropical.md`; scheduled for removal
/// in Stage 6 of the `design_v3` migration plan.
#[deprecated(
    since = "design_v3-stage-2",
    note = "non-mainline per docs/design/design_v3/30-algebra-and-tropical.md; scheduled for removal in Stage 6"
)]
#[allow(deprecated)]
pub fn compile_semiring_to_exec<Alg>(
    prog: &CompiledProgram<SemiringOp<Alg>>,
    input_shapes: &[Vec<DimExpr>],
) -> ExecProgram
where
    Alg: Algebra + Send + Sync + 'static,
    Alg::Scalar: TensorScalar,
{
    assert_eq!(
        prog.input_slots.len(),
        input_shapes.len(),
        "compile_semiring_to_exec: input shape count must match input slot count"
    );

    let dtype = <Alg::Scalar as TensorScalar>::dtype();
    let mut slot_shapes: Vec<Option<Vec<DimExpr>>> = vec![None; prog.n_slots];
    for (index, &slot) in prog.input_slots.iter().enumerate() {
        slot_shapes[slot] = Some(input_shapes[index].clone());
    }

    let instructions = prog
        .instructions
        .iter()
        .map(|instr| {
            let input_shapes_owned: Vec<Vec<DimExpr>> = instr
                .inputs
                .iter()
                .map(|&slot| {
                    slot_shapes[slot].clone().unwrap_or_else(|| {
                        panic!("compile_semiring_to_exec: missing shape for slot {slot}")
                    })
                })
                .collect();
            let input_shapes_refs: Vec<&[DimExpr]> =
                input_shapes_owned.iter().map(Vec::as_slice).collect();
            let output_shapes = infer_semiring_output_shapes(&instr.op.kind, &input_shapes_refs);
            assert_eq!(
                output_shapes.len(),
                instr.outputs.len(),
                "compile_semiring_to_exec: {:?} inferred {} output shapes for {} output slots",
                instr.op.kind,
                output_shapes.len(),
                instr.outputs.len()
            );

            for (slot, shape) in instr.outputs.iter().zip(output_shapes.iter()) {
                slot_shapes[*slot] = Some(shape.clone());
            }

            ExecInstruction {
                op: semiring_to_exec_op(&instr.op.kind),
                input_slots: instr.inputs.clone(),
                output_slots: instr.outputs.clone(),
                dtype,
                output_shapes,
                last_use: Vec::new(),
            }
        })
        .collect();

    let mut program = ExecProgram {
        instructions,
        input_slots: prog.input_slots.clone(),
        output_slots: prog.output_slots.clone(),
        n_slots: prog.n_slots,
    };
    dot_dimension_sorter(&mut program);
    transpose_folding(&mut program);
    populate_last_use(&mut program);
    program
}

fn semiring_to_exec_op(kind: &SemiringOpKind) -> ExecOp {
    match kind {
        SemiringOpKind::Add => ExecOp::Add,
        SemiringOpKind::Mul => ExecOp::Multiply,
        SemiringOpKind::DotGeneral(config) => ExecOp::DotGeneral(config.clone()),
        SemiringOpKind::ReduceSum { axes } => ExecOp::ReduceSum { axes: axes.clone() },
        SemiringOpKind::Transpose { perm } => ExecOp::Transpose { perm: perm.clone() },
        SemiringOpKind::Reshape { shape } => ExecOp::Reshape {
            shape: DimExpr::from_concrete(shape),
        },
        SemiringOpKind::BroadcastInDim { shape, dims } => ExecOp::BroadcastInDim {
            shape: DimExpr::from_concrete(shape),
            dims: dims.clone(),
        },
        SemiringOpKind::ExtractDiag { axis_a, axis_b } => ExecOp::ExtractDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
        SemiringOpKind::EmbedDiag { axis_a, axis_b } => ExecOp::EmbedDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
    }
}

fn infer_semiring_output_shapes(
    kind: &SemiringOpKind,
    input_shapes: &[&[DimExpr]],
) -> Vec<Vec<DimExpr>> {
    let op = match kind {
        SemiringOpKind::Add => StdTensorOp::Add,
        SemiringOpKind::Mul => StdTensorOp::Mul,
        SemiringOpKind::DotGeneral(config) => StdTensorOp::DotGeneral {
            config: config.clone(),
        },
        SemiringOpKind::ReduceSum { axes } => StdTensorOp::ReduceSum { axes: axes.clone() },
        SemiringOpKind::Transpose { perm } => StdTensorOp::Transpose { perm: perm.clone() },
        SemiringOpKind::Reshape { shape } => StdTensorOp::Reshape {
            to_shape: DimExpr::from_concrete(shape),
        },
        SemiringOpKind::BroadcastInDim { shape, dims } => StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(shape),
            dims: dims.clone(),
        },
        SemiringOpKind::ExtractDiag { axis_a, axis_b } => StdTensorOp::ExtractDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
        SemiringOpKind::EmbedDiag { axis_a, axis_b } => StdTensorOp::EmbedDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
    };
    infer_output_shapes(&op, input_shapes)
}
