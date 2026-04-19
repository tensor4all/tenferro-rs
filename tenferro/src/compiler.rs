use computegraph::compile::CompiledProgram;
use tenferro_algebra::Algebra;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::semiring_op::SemiringOp;
use tenferro_ops::semiring_op_kind::SemiringOpKind;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig, TensorScalar};

use crate::shape_infer::{infer_output_dtype, infer_output_shapes};

use super::exec::{ExecInstruction, ExecOp, ExecProgram};

pub fn compile_std_to_exec(
    prog: &CompiledProgram<StdTensorOp>,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
) -> ExecProgram {
    assert_eq!(
        prog.input_slots.len(),
        input_dtypes.len(),
        "compile_std_to_exec: input dtype count must match input slot count"
    );
    assert_eq!(
        prog.input_slots.len(),
        input_shapes.len(),
        "compile_std_to_exec: input shape count must match input slot count"
    );

    let mut slot_dtypes: Vec<Option<DType>> = vec![None; prog.n_slots];
    let mut slot_shapes: Vec<Option<Vec<DimExpr>>> = vec![None; prog.n_slots];

    for (index, &slot) in prog.input_slots.iter().enumerate() {
        slot_dtypes[slot] = Some(input_dtypes[index]);
        slot_shapes[slot] = Some(input_shapes[index].clone());
    }

    let instructions = prog
        .instructions
        .iter()
        .map(|instr| {
            let input_dtypes: Vec<DType> = instr
                .inputs
                .iter()
                .map(|&slot| {
                    slot_dtypes[slot].unwrap_or_else(|| {
                        panic!("compile_std_to_exec: missing dtype for slot {slot}")
                    })
                })
                .collect();
            let input_shapes_owned: Vec<Vec<DimExpr>> = instr
                .inputs
                .iter()
                .map(|&slot| {
                    slot_shapes[slot].clone().unwrap_or_else(|| {
                        panic!("compile_std_to_exec: missing shape for slot {slot}")
                    })
                })
                .collect();
            let input_shapes_refs: Vec<&[DimExpr]> =
                input_shapes_owned.iter().map(Vec::as_slice).collect();

            let output_dtype = infer_output_dtype(&instr.op, &input_dtypes);
            let output_shapes = infer_output_shapes(&instr.op, &input_shapes_refs);
            assert_eq!(
                output_shapes.len(),
                instr.outputs.len(),
                "compile_std_to_exec: {:?} inferred {} output shapes for {} output slots",
                instr.op,
                output_shapes.len(),
                instr.outputs.len()
            );

            for (slot, shape) in instr.outputs.iter().zip(output_shapes.iter()) {
                slot_dtypes[*slot] = Some(output_dtype);
                slot_shapes[*slot] = Some(shape.clone());
            }

            ExecInstruction {
                op: std_to_exec_op(&instr.op),
                input_slots: instr.inputs.clone(),
                output_slots: instr.outputs.clone(),
                dtype: output_dtype,
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

fn std_to_exec_op(op: &StdTensorOp) -> ExecOp {
    match op {
        StdTensorOp::Add => ExecOp::Add,
        StdTensorOp::Mul => ExecOp::Multiply,
        StdTensorOp::Neg => ExecOp::Negate,
        StdTensorOp::Conj => ExecOp::Conj,
        StdTensorOp::Div => ExecOp::Divide,
        StdTensorOp::Abs => ExecOp::Abs,
        StdTensorOp::Sign => ExecOp::Sign,
        StdTensorOp::Maximum => ExecOp::Maximum,
        StdTensorOp::Minimum => ExecOp::Minimum,
        StdTensorOp::Compare(dir) => ExecOp::Compare(dir.clone()),
        StdTensorOp::Select => ExecOp::Select,
        StdTensorOp::Clamp => ExecOp::Clamp,
        StdTensorOp::Exp => ExecOp::Exp,
        StdTensorOp::Log => ExecOp::Log,
        StdTensorOp::Sin => ExecOp::Sin,
        StdTensorOp::Cos => ExecOp::Cos,
        StdTensorOp::Tanh => ExecOp::Tanh,
        StdTensorOp::Sqrt => ExecOp::Sqrt,
        StdTensorOp::Rsqrt => ExecOp::Rsqrt,
        StdTensorOp::Pow => ExecOp::Pow,
        StdTensorOp::Expm1 => ExecOp::Expm1,
        StdTensorOp::Log1p => ExecOp::Log1p,
        StdTensorOp::Transpose { perm } => ExecOp::Transpose { perm: perm.clone() },
        StdTensorOp::Reshape { to_shape, .. } => ExecOp::Reshape {
            shape: to_shape.clone(),
        },
        StdTensorOp::BroadcastInDim { shape, dims } => ExecOp::BroadcastInDim {
            shape: shape.clone(),
            dims: dims.clone(),
        },
        StdTensorOp::Convert { to, .. } => ExecOp::Convert { to: *to },
        StdTensorOp::Constant { dtype, bytes } => ExecOp::Constant {
            dtype: *dtype,
            bytes: bytes.clone(),
        },
        StdTensorOp::DotGeneral { config, .. } => ExecOp::DotGeneral(config.clone()),
        StdTensorOp::NaryEinsum { subscripts, .. } => ExecOp::NaryEinsum {
            subscripts: subscripts.clone(),
        },
        StdTensorOp::ReduceSum { axes, .. } => ExecOp::ReduceSum { axes: axes.clone() },
        StdTensorOp::ReduceProd { axes, .. } => ExecOp::ReduceProd { axes: axes.clone() },
        StdTensorOp::ReduceMax { axes, .. } => ExecOp::ReduceMax { axes: axes.clone() },
        StdTensorOp::ReduceMin { axes, .. } => ExecOp::ReduceMin { axes: axes.clone() },
        StdTensorOp::ExtractDiag { axis_a, axis_b } => ExecOp::ExtractDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
        StdTensorOp::EmbedDiag { axis_a, axis_b } => ExecOp::EmbedDiag {
            axis_a: *axis_a,
            axis_b: *axis_b,
        },
        StdTensorOp::Tril { k } => ExecOp::Tril { k: *k },
        StdTensorOp::Triu { k } => ExecOp::Triu { k: *k },
        StdTensorOp::Gather(config) => ExecOp::Gather(config.clone()),
        StdTensorOp::Scatter(config) => ExecOp::Scatter(config.clone()),
        StdTensorOp::Slice(config) => ExecOp::Slice(config.clone()),
        StdTensorOp::DynamicSlice { slice_sizes } => ExecOp::DynamicSlice {
            slice_sizes: slice_sizes.clone(),
        },
        StdTensorOp::Pad(config) => ExecOp::Pad(config.clone()),
        StdTensorOp::Concatenate { axis } => ExecOp::Concatenate { axis: *axis },
        StdTensorOp::Reverse { axes } => ExecOp::Reverse { axes: axes.clone() },
        StdTensorOp::ShapeOf { axis } => ExecOp::ShapeOf { axis: *axis },
        StdTensorOp::DynamicTruncate { axis } => ExecOp::DynamicTruncate { axis: *axis },
        StdTensorOp::PadToMatch { axis } => ExecOp::PadToMatch { axis: *axis },
        StdTensorOp::Cholesky { .. } => ExecOp::Cholesky,
        StdTensorOp::Lu { .. } => ExecOp::Lu,
        StdTensorOp::Svd { eps, .. } => ExecOp::Svd { eps: *eps },
        StdTensorOp::Qr { .. } => ExecOp::Qr,
        StdTensorOp::Eigh { eps, .. } => ExecOp::Eigh { eps: *eps },
        StdTensorOp::Eig { .. } => ExecOp::Eig,
        StdTensorOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
            ..
        } => ExecOp::TriangularSolve {
            left_side: *left_side,
            lower: *lower,
            transpose_a: *transpose_a,
            unit_diagonal: *unit_diagonal,
        },
        StdTensorOp::ValidateNonsingular { .. } => ExecOp::ValidateNonsingular,
    }
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
        SemiringOpKind::ReduceSum { axes } => StdTensorOp::ReduceSum {
            axes: axes.clone(),
            input_shape: require_input_shape(kind, input_shapes, 0).to_vec(),
        },
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

fn require_input_shape<'a>(
    kind: &SemiringOpKind,
    input_shapes: &'a [&[DimExpr]],
    index: usize,
) -> &'a [DimExpr] {
    input_shapes.get(index).copied().unwrap_or_else(|| {
        panic!(
            "semiring shape inference for {kind:?} requires input index {index}, got {}",
            input_shapes.len()
        )
    })
}

fn populate_last_use(program: &mut ExecProgram) {
    let all_input_slots: Vec<Vec<usize>> = program
        .instructions
        .iter()
        .map(|instr| instr.input_slots.clone())
        .collect();
    for (idx, instr) in program.instructions.iter_mut().enumerate() {
        instr.last_use = compute_last_use(
            &instr.input_slots,
            idx,
            &all_input_slots,
            &program.output_slots,
        );
    }
}

fn compute_last_use(
    input_slots: &[usize],
    current_idx: usize,
    all_input_slots: &[Vec<usize>],
    output_slots: &[usize],
) -> Vec<bool> {
    input_slots
        .iter()
        .map(|&slot| {
            if output_slots.contains(&slot) {
                return false;
            }
            for later_inputs in &all_input_slots[current_idx + 1..] {
                if later_inputs.contains(&slot) {
                    return false;
                }
            }
            true
        })
        .collect()
}

// ============================================================================
// Pass 1: DotDimensionSorter
// ============================================================================
//
// Sort contracting dimensions of DotGeneral so that downstream execution sees
// a stable canonical ordering.

/// Sort contracting dimensions of all DotGeneral instructions in place.
pub fn dot_dimension_sorter(program: &mut ExecProgram) {
    for instr in &mut program.instructions {
        if let ExecOp::DotGeneral(config) = &mut instr.op {
            sort_contracting_dims(config);
        }
    }
}

fn sort_contracting_dims(config: &mut DotGeneralConfig) {
    let lhs = &config.lhs_contracting_dims;
    let rhs = &config.rhs_contracting_dims;

    if lhs.is_empty() {
        return;
    }

    if consecutive_if_sorted(lhs) && !is_sorted(lhs) {
        let perm = argsort(lhs);
        config.lhs_contracting_dims = apply_perm(lhs, &perm);
        config.rhs_contracting_dims = apply_perm(rhs, &perm);
    } else if consecutive_if_sorted(rhs) && !is_sorted(rhs) {
        let perm = argsort(rhs);
        config.lhs_contracting_dims = apply_perm(lhs, &perm);
        config.rhs_contracting_dims = apply_perm(rhs, &perm);
    }
}

fn consecutive_if_sorted(dims: &[usize]) -> bool {
    if dims.is_empty() {
        return true;
    }
    let min_val = *dims.iter().min().expect("non-empty");
    let max_val = *dims.iter().max().expect("non-empty");
    max_val - min_val == dims.len() - 1
}

fn is_sorted(dims: &[usize]) -> bool {
    dims.windows(2).all(|w| w[0] <= w[1])
}

fn argsort(dims: &[usize]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..dims.len()).collect();
    indices.sort_by_key(|&i| dims[i]);
    indices
}

fn apply_perm(source: &[usize], perm: &[usize]) -> Vec<usize> {
    perm.iter().map(|&p| source[p]).collect()
}

// ============================================================================
// Pass 2: TransposeFolding
// ============================================================================
//
// Absorb Transpose instructions that feed directly into DotGeneral by
// adjusting the DotGeneral dimension numbers and bypassing the Transpose.

/// Fold Transpose instructions into DotGeneral dimension numbers.
pub fn transpose_folding(program: &mut ExecProgram) {
    loop {
        let changed = transpose_fold_one_pass(program);
        if !changed {
            break;
        }
    }
}

fn transpose_fold_one_pass(program: &mut ExecProgram) -> bool {
    let mut changed = false;

    for index in 0..program.instructions.len() {
        if !matches!(program.instructions[index].op, ExecOp::DotGeneral(_)) {
            continue;
        }

        if try_fold_operand(program, index, 0) {
            changed = true;
        }
        if program.instructions[index].input_slots.len() > 1 && try_fold_operand(program, index, 1)
        {
            changed = true;
        }
    }

    changed
}

fn try_fold_operand(program: &mut ExecProgram, dot_idx: usize, operand_idx: usize) -> bool {
    let input_slot = program.instructions[dot_idx].input_slots[operand_idx];
    let Some(producer_idx) = find_producer(program, input_slot) else {
        return false;
    };

    let perm = match &program.instructions[producer_idx].op {
        ExecOp::Transpose { perm } => perm.clone(),
        _ => return false,
    };

    let config = match &program.instructions[dot_idx].op {
        ExecOp::DotGeneral(config) => config.clone(),
        _ => return false,
    };

    if !is_transpose_foldable(&config, operand_idx, &perm) {
        return false;
    }

    let new_config = fold_transpose_into_dot(&config, operand_idx, &perm);
    let original_input = program.instructions[producer_idx].input_slots[0];
    program.instructions[dot_idx].op = ExecOp::DotGeneral(new_config);
    program.instructions[dot_idx].input_slots[operand_idx] = original_input;
    true
}

fn find_producer(program: &ExecProgram, slot: usize) -> Option<usize> {
    program
        .instructions
        .iter()
        .position(|inst| inst.output_slots.contains(&slot))
}

fn is_transpose_foldable(config: &DotGeneralConfig, operand_idx: usize, perm: &[usize]) -> bool {
    let (contracting_dims, batch_dims) = if operand_idx == 0 {
        (
            config.lhs_contracting_dims.as_slice(),
            config.lhs_batch_dims.as_slice(),
        )
    } else {
        (
            config.rhs_contracting_dims.as_slice(),
            config.rhs_batch_dims.as_slice(),
        )
    };
    let rank = perm.len();

    if !is_valid_permutation(perm, rank) {
        return false;
    }

    let Some(free_dims) = free_axes(rank, contracting_dims, batch_dims) else {
        return false;
    };

    is_role_group_order_preserved(&free_dims, perm)
        && is_role_group_order_preserved(contracting_dims, perm)
        && is_role_group_order_preserved(batch_dims, perm)
}

fn free_axes(rank: usize, contracting_dims: &[usize], batch_dims: &[usize]) -> Option<Vec<usize>> {
    let mut used = vec![false; rank];
    for &axis in contracting_dims.iter().chain(batch_dims.iter()) {
        if axis >= rank || used[axis] {
            return None;
        }
        used[axis] = true;
    }

    Some((0..rank).filter(|&axis| !used[axis]).collect())
}

fn is_valid_permutation(perm: &[usize], rank: usize) -> bool {
    let mut seen = vec![false; rank];
    for &axis in perm {
        if axis >= rank || seen[axis] {
            return false;
        }
        seen[axis] = true;
    }
    true
}

fn map_axes(axes: &[usize], perm: &[usize]) -> Option<Vec<usize>> {
    axes.iter().map(|&axis| perm.get(axis).copied()).collect()
}

fn is_role_group_order_preserved(axes: &[usize], perm: &[usize]) -> bool {
    let Some(mapped_axes) = map_axes(axes, perm) else {
        return false;
    };
    is_strictly_increasing(&mapped_axes)
}

fn is_strictly_increasing(values: &[usize]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn fold_transpose_into_dot(
    config: &DotGeneralConfig,
    operand_idx: usize,
    perm: &[usize],
) -> DotGeneralConfig {
    let mut new_config = config.clone();
    if operand_idx == 0 {
        new_config.lhs_contracting_dims = config
            .lhs_contracting_dims
            .iter()
            .map(|&dim| perm[dim])
            .collect();
        new_config.lhs_batch_dims = config.lhs_batch_dims.iter().map(|&dim| perm[dim]).collect();
    } else {
        new_config.rhs_contracting_dims = config
            .rhs_contracting_dims
            .iter()
            .map(|&dim| perm[dim])
            .collect();
        new_config.rhs_batch_dims = config.rhs_batch_dims.iter().map(|&dim| perm[dim]).collect();
    }
    new_config
}
