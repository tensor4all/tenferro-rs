use computegraph::compile::CompiledProgram;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeExtent;
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::shape_infer::{
    infer_extension_output_meta, infer_output_dtype, infer_output_extents, infer_output_shapes,
};

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
    let mut slot_extents: Vec<Option<Vec<ShapeExtent<DimExpr>>>> = vec![None; prog.n_slots];

    for (index, &slot) in prog.input_slots.iter().enumerate() {
        slot_dtypes[slot] = Some(input_dtypes[index]);
        slot_shapes[slot] = Some(input_shapes[index].clone());
        slot_extents[slot] = Some(exact_extents_from_shape(&input_shapes[index]));
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
            let input_extents_owned: Vec<Vec<ShapeExtent<DimExpr>>> = instr
                .inputs
                .iter()
                .map(|&slot| {
                    slot_extents[slot].clone().unwrap_or_else(|| {
                        panic!("compile_std_to_exec: missing extents for slot {slot}")
                    })
                })
                .collect();

            let (output_dtype, output_shapes, output_extents): (
                DType,
                Vec<Vec<DimExpr>>,
                Vec<Vec<ShapeExtent<DimExpr>>>,
            ) = if let StdTensorOp::Extension(ext) = &instr.op {
                let metas =
                    infer_extension_output_meta(ext.as_ref(), &input_dtypes, &input_shapes_refs);
                assert_eq!(
                    metas.len(),
                    instr.outputs.len(),
                    "compile_std_to_exec: extension family_id={:?} \
                         inferred {} output metas for {} output slots",
                    ext.family_id(),
                    metas.len(),
                    instr.outputs.len()
                );
                // Current compiler supports a single dtype per instruction;
                // per spec Section 7, extensions must keep a uniform dtype
                // across all outputs. Surface a clean panic if violated.
                let dtypes_consistent = metas.iter().all(|(dtype, _)| *dtype == metas[0].0);
                assert!(
                    dtypes_consistent,
                    "compile_std_to_exec: extension family_id={:?} returned \
                         multiple output dtypes {:?}; multi-dtype extensions are \
                         not yet supported in the compiled path",
                    ext.family_id(),
                    metas.iter().map(|(dtype, _)| *dtype).collect::<Vec<_>>()
                );
                let dtype = metas[0].0;
                let shapes: Vec<Vec<DimExpr>> =
                    metas.into_iter().map(|(_dtype, shape)| shape).collect();
                let extents = exact_extents_from_shapes(&shapes);
                (dtype, shapes, extents)
            } else {
                let dtype = infer_output_dtype(&instr.op, &input_dtypes);
                let shapes = infer_output_shapes(&instr.op, &input_shapes_refs);
                let extents = infer_output_extents(&instr.op, &input_shapes_refs);
                assert_eq!(
                    shapes.len(),
                    instr.outputs.len(),
                    "compile_std_to_exec: {:?} inferred {} output shapes for {} output slots",
                    instr.op,
                    shapes.len(),
                    instr.outputs.len()
                );
                assert_eq!(
                    extents.len(),
                    instr.outputs.len(),
                    "compile_std_to_exec: {:?} inferred {} output extents for {} output slots",
                    instr.op,
                    extents.len(),
                    instr.outputs.len()
                );
                let resolved_extents =
                    resolve_output_extents(extents, &input_shapes_owned, &input_extents_owned);
                (dtype, shapes, resolved_extents)
            };

            for ((slot, shape), extents) in instr
                .outputs
                .iter()
                .zip(output_shapes.iter())
                .zip(output_extents.iter())
            {
                slot_dtypes[*slot] = Some(output_dtype);
                slot_shapes[*slot] = Some(shape.clone());
                slot_extents[*slot] = Some(extents.clone());
            }

            ExecInstruction {
                op: std_to_exec_op(&instr.op),
                input_slots: instr.inputs.clone(),
                output_slots: instr.outputs.clone(),
                dtype: output_dtype,
                output_shapes,
                output_extents,
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
    dot_decomposer(&mut program, input_shapes);
    eliminate_dead_code(&mut program);
    populate_last_use(&mut program);
    program
}

fn exact_extents_from_shape(shape: &[DimExpr]) -> Vec<ShapeExtent<DimExpr>> {
    shape.iter().cloned().map(ShapeExtent::exact).collect()
}

fn exact_extents_from_shapes(shapes: &[Vec<DimExpr>]) -> Vec<Vec<ShapeExtent<DimExpr>>> {
    shapes
        .iter()
        .map(|shape| exact_extents_from_shape(shape))
        .collect()
}

fn resolve_output_extents(
    extents: Vec<Vec<ShapeExtent<DimExpr>>>,
    input_shapes: &[Vec<DimExpr>],
    input_extents: &[Vec<ShapeExtent<DimExpr>>],
) -> Vec<Vec<ShapeExtent<DimExpr>>> {
    extents
        .into_iter()
        .map(|shape_extents| {
            shape_extents
                .into_iter()
                .map(|extent| resolve_extent(extent, input_shapes, input_extents))
                .collect()
        })
        .collect()
}

fn resolve_extent(
    extent: ShapeExtent<DimExpr>,
    input_shapes: &[Vec<DimExpr>],
    input_extents: &[Vec<ShapeExtent<DimExpr>>],
) -> ShapeExtent<DimExpr> {
    match extent {
        ShapeExtent::Exact(dim) => match dim_expr_extent_kind(&dim, input_extents) {
            ExtentKind::Exact => ShapeExtent::exact(resolve_dim_expr(&dim, input_shapes)),
            ExtentKind::UpperBound => {
                ShapeExtent::upper_bound(resolve_dim_expr(&dim, input_shapes))
            }
            ExtentKind::Unknown => ShapeExtent::unknown(),
        },
        ShapeExtent::UpperBound(dim) => match dim_expr_extent_kind(&dim, input_extents) {
            ExtentKind::Unknown => ShapeExtent::unknown(),
            ExtentKind::Exact | ExtentKind::UpperBound => {
                ShapeExtent::upper_bound(resolve_dim_expr(&dim, input_shapes))
            }
        },
        ShapeExtent::Unknown => ShapeExtent::unknown(),
    }
}

fn resolve_dim_expr(expr: &DimExpr, input_shapes: &[Vec<DimExpr>]) -> DimExpr {
    match expr {
        DimExpr::Const(value) => DimExpr::Const(*value),
        DimExpr::InputDim { input_idx, axis } => input_shapes
            .get(*input_idx)
            .and_then(|shape| shape.get(*axis))
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "compile_std_to_exec: InputDim({}, {}) cannot be resolved from {} input shapes",
                    input_idx,
                    axis,
                    input_shapes.len()
                )
            }),
        DimExpr::Add(a, b) => DimExpr::add(
            resolve_dim_expr(a, input_shapes),
            resolve_dim_expr(b, input_shapes),
        ),
        DimExpr::Sub(a, b) => DimExpr::sub(
            resolve_dim_expr(a, input_shapes),
            resolve_dim_expr(b, input_shapes),
        ),
        DimExpr::Mul(a, b) => DimExpr::mul(
            resolve_dim_expr(a, input_shapes),
            resolve_dim_expr(b, input_shapes),
        ),
        DimExpr::FloorDiv(a, b) => DimExpr::floor_div(
            resolve_dim_expr(a, input_shapes),
            resolve_dim_expr(b, input_shapes),
        ),
        DimExpr::Min(a, b) => DimExpr::min(
            resolve_dim_expr(a, input_shapes),
            resolve_dim_expr(b, input_shapes),
        ),
        DimExpr::Max(a, b) => DimExpr::max(
            resolve_dim_expr(a, input_shapes),
            resolve_dim_expr(b, input_shapes),
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtentKind {
    Exact,
    UpperBound,
    Unknown,
}

fn dim_expr_extent_kind(expr: &DimExpr, input_extents: &[Vec<ShapeExtent<DimExpr>>]) -> ExtentKind {
    match expr {
        DimExpr::Const(_) => ExtentKind::Exact,
        DimExpr::InputDim { input_idx, axis } => input_extents
            .get(*input_idx)
            .and_then(|shape| shape.get(*axis))
            .map(extent_kind)
            .unwrap_or(ExtentKind::Unknown),
        DimExpr::Add(a, b) | DimExpr::Mul(a, b) | DimExpr::Min(a, b) | DimExpr::Max(a, b) => {
            combine_monotonic_kinds(
                dim_expr_extent_kind(a, input_extents),
                dim_expr_extent_kind(b, input_extents),
            )
        }
        DimExpr::FloorDiv(a, b) => match (
            dim_expr_extent_kind(a, input_extents),
            dim_expr_extent_kind(b, input_extents),
        ) {
            (ExtentKind::Exact, ExtentKind::Exact) => ExtentKind::Exact,
            (ExtentKind::UpperBound, ExtentKind::Exact) => ExtentKind::UpperBound,
            _ => ExtentKind::Unknown,
        },
        DimExpr::Sub(a, b) => match (
            dim_expr_extent_kind(a, input_extents),
            dim_expr_extent_kind(b, input_extents),
        ) {
            (ExtentKind::Exact, ExtentKind::Exact) => ExtentKind::Exact,
            _ => ExtentKind::Unknown,
        },
    }
}

fn extent_kind(extent: &ShapeExtent<DimExpr>) -> ExtentKind {
    match extent {
        ShapeExtent::Exact(_) => ExtentKind::Exact,
        ShapeExtent::UpperBound(_) => ExtentKind::UpperBound,
        ShapeExtent::Unknown => ExtentKind::Unknown,
    }
}

fn combine_monotonic_kinds(lhs: ExtentKind, rhs: ExtentKind) -> ExtentKind {
    match (lhs, rhs) {
        (ExtentKind::Unknown, _) | (_, ExtentKind::Unknown) => ExtentKind::Unknown,
        (ExtentKind::Exact, ExtentKind::Exact) => ExtentKind::Exact,
        _ => ExtentKind::UpperBound,
    }
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
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        } => ExecOp::GatherDynamicSliceSizes {
            offset_dims: offset_dims.clone(),
            collapsed_slice_dims: collapsed_slice_dims.clone(),
            start_index_map: start_index_map.clone(),
            index_vector_dim: *index_vector_dim,
            slice_sizes: slice_sizes.clone(),
        },
        StdTensorOp::Scatter(config) => ExecOp::Scatter(config.clone()),
        StdTensorOp::Slice(config) => ExecOp::Slice(config.clone()),
        StdTensorOp::DynamicSlice { slice_sizes } => ExecOp::DynamicSlice {
            slice_sizes: slice_sizes.clone(),
        },
        StdTensorOp::Pad(config) => ExecOp::Pad(config.clone()),
        StdTensorOp::Concatenate { axis, .. } => ExecOp::Concatenate { axis: *axis },
        StdTensorOp::Reverse { axes } => ExecOp::Reverse { axes: axes.clone() },
        StdTensorOp::ShapeOf { axis } => ExecOp::ShapeOf { axis: *axis },
        StdTensorOp::DynamicTruncate { axis } => ExecOp::DynamicTruncate { axis: *axis },
        StdTensorOp::PadToMatch { axis } => ExecOp::PadToMatch { axis: *axis },
        StdTensorOp::Cholesky { .. } => ExecOp::Cholesky,
        StdTensorOp::Lu { .. } => ExecOp::Lu,
        StdTensorOp::FullPivLu { .. } => ExecOp::FullPivLu,
        StdTensorOp::FullPivLuSolve { transpose_a } => ExecOp::FullPivLuSolve {
            transpose_a: *transpose_a,
        },
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
        StdTensorOp::Extension(ext) => ExecOp::Extension(ext.clone()),
    }
}

pub(crate) fn populate_last_use(program: &mut ExecProgram) {
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

// ============================================================================
// Pass 3: DotDecomposer
// ============================================================================
//
// Canonicalize `ExecOp::DotGeneral` into a shape consumable by canonical
// batched GEMM kernels. Runs after `transpose_folding` so that pre-existing
// foldable transposes are already absorbed. Per-instruction `output_shapes`
// (populated by `shape_infer`) gives the shapes needed to emit `Reshape`
// instructions that merge free/contracting groups and restore the original
// output shape.
//
// Canonical form (tenferro column-major with batch trailing):
//
//   LHS: [M?, K, B0, B1, ...]
//     - lhs_contracting_dims = [|free_L|']
//     - lhs_batch_dims       = [|free_L|' + 1, ..., |free_L|' + nb]
//     where |free_L|' = 1 if the original LHS had any free dim, else 0.
//
//   RHS: [K, N?, B0, B1, ...]
//     - rhs_contracting_dims = [0]
//     - rhs_batch_dims       = [|free_R|' + 1, ..., |free_R|' + nb]
//     where |free_R|' = 1 if the original RHS had any free dim, else 0.
//
// The XLA-style algorithm per non-canonical DotGeneral:
//
//   1. Transpose LHS to target order (free ++ contracting ++ batch).
//   2. Reshape LHS to merge multiple free/contracting dims.
//   3. Transpose RHS to target order (contracting ++ free ++ batch).
//   4. Reshape RHS to merge multiple free/contracting dims.
//   5. Emit canonical `DotGeneral`.
//   6. Reshape output back to the original shape (required when either side
//      had multiple free dims). Without this, downstream consumers would
//      observe a rank-collapsed tensor.
//
// Only steps that are non-trivial for the specific operand are emitted.

/// Canonicalize all non-canonical `DotGeneral` instructions in `program`.
///
/// `input_shapes` are the program-input shapes (matching `program.input_slots`).
/// They are needed because `ExecProgram` does not otherwise carry input-slot
/// shape metadata.
pub fn dot_decomposer(program: &mut ExecProgram, input_shapes: &[Vec<DimExpr>]) {
    assert_eq!(
        program.input_slots.len(),
        input_shapes.len(),
        "dot_decomposer: input shape count must match input slot count"
    );

    let mut slot_shapes: Vec<Option<Vec<DimExpr>>> = vec![None; program.n_slots];
    let mut slot_extents: Vec<Option<Vec<ShapeExtent<DimExpr>>>> = vec![None; program.n_slots];
    for (index, &slot) in program.input_slots.iter().enumerate() {
        slot_shapes[slot] = Some(input_shapes[index].clone());
        slot_extents[slot] = Some(exact_extents_from_shape(&input_shapes[index]));
    }
    for instr in &program.instructions {
        for ((slot, shape), extents) in instr
            .output_slots
            .iter()
            .zip(instr.output_shapes.iter())
            .zip(instr.output_extents.iter())
        {
            slot_shapes[*slot] = Some(shape.clone());
            slot_extents[*slot] = Some(extents.clone());
        }
    }

    let mut new_instructions: Vec<ExecInstruction> = Vec::with_capacity(program.instructions.len());
    let mut n_slots = program.n_slots;

    for instr in &program.instructions {
        if let ExecOp::DotGeneral(config) = &instr.op {
            if instr.input_slots.len() == 2 && !config.lhs_contracting_dims.is_empty() {
                let lhs_slot = instr.input_slots[0];
                let rhs_slot = instr.input_slots[1];
                let lhs_shape = require_slot_shape(&slot_shapes, lhs_slot);
                let rhs_shape = require_slot_shape(&slot_shapes, rhs_slot);
                let lhs_extents = require_slot_extents(&slot_extents, lhs_slot);
                let rhs_extents = require_slot_extents(&slot_extents, rhs_slot);
                if !is_dot_canonical(config, lhs_shape.len(), rhs_shape.len()) {
                    decompose_dot(
                        instr,
                        config,
                        lhs_slot,
                        rhs_slot,
                        lhs_shape,
                        rhs_shape,
                        lhs_extents,
                        rhs_extents,
                        &mut n_slots,
                        &mut new_instructions,
                    );
                    continue;
                }
            }
        }
        new_instructions.push(instr.clone());
    }

    program.instructions = new_instructions;
    program.n_slots = n_slots;
}

fn require_slot_shape(slot_shapes: &[Option<Vec<DimExpr>>], slot: usize) -> &[DimExpr] {
    slot_shapes[slot]
        .as_ref()
        .unwrap_or_else(|| panic!("dot_decomposer: missing shape for slot {slot}"))
        .as_slice()
}

fn require_slot_extents(
    slot_extents: &[Option<Vec<ShapeExtent<DimExpr>>>],
    slot: usize,
) -> &[ShapeExtent<DimExpr>] {
    slot_extents[slot]
        .as_ref()
        .unwrap_or_else(|| panic!("dot_decomposer: missing extents for slot {slot}"))
        .as_slice()
}

/// Canonical form predicate for a `DotGeneral` with the given operand ranks.
fn is_dot_canonical(config: &DotGeneralConfig, lhs_rank: usize, rhs_rank: usize) -> bool {
    if config.lhs_contracting_dims.len() != 1 || config.rhs_contracting_dims.len() != 1 {
        return false;
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return false;
    }
    let nb = config.lhs_batch_dims.len();

    // LHS: [M?, K, B...]. free_L is 0 or 1 elems.
    let lhs_has_free = match lhs_rank.checked_sub(nb + 1) {
        Some(0) => false,
        Some(1) => true,
        _ => return false,
    };
    let lhs_expected_contracting = if lhs_has_free { 1 } else { 0 };
    let lhs_expected_batch: Vec<usize> =
        ((lhs_expected_contracting + 1)..(lhs_expected_contracting + 1 + nb)).collect();
    if config.lhs_contracting_dims != vec![lhs_expected_contracting]
        || config.lhs_batch_dims != lhs_expected_batch
    {
        return false;
    }

    // RHS: [K, N?, B...]. free_R is 0 or 1 elems, contracting is always at 0.
    let rhs_has_free = match rhs_rank.checked_sub(nb + 1) {
        Some(0) => false,
        Some(1) => true,
        _ => return false,
    };
    let rhs_free_count = usize::from(rhs_has_free);
    let rhs_expected_batch: Vec<usize> =
        ((rhs_free_count + 1)..(rhs_free_count + 1 + nb)).collect();
    if config.rhs_contracting_dims != vec![0] || config.rhs_batch_dims != rhs_expected_batch {
        return false;
    }

    true
}

fn free_axes_of(rank: usize, contracting: &[usize], batch: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn product_of_input_dims(input_idx: usize, start: usize, end: usize) -> DimExpr {
    assert!(end > start, "product_of_input_dims: empty range");
    let mut result = DimExpr::InputDim {
        input_idx,
        axis: start,
    };
    for axis in (start + 1)..end {
        result = DimExpr::mul(result, DimExpr::InputDim { input_idx, axis });
    }
    result
}

/// Emit decomposed instructions for a single non-canonical DotGeneral.
#[allow(clippy::too_many_arguments)]
fn decompose_dot(
    instr: &ExecInstruction,
    config: &DotGeneralConfig,
    lhs_slot: usize,
    rhs_slot: usize,
    lhs_shape: &[DimExpr],
    rhs_shape: &[DimExpr],
    lhs_extents: &[ShapeExtent<DimExpr>],
    rhs_extents: &[ShapeExtent<DimExpr>],
    n_slots: &mut usize,
    new_instructions: &mut Vec<ExecInstruction>,
) {
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    let nb = config.lhs_batch_dims.len();
    assert_eq!(
        nb,
        config.rhs_batch_dims.len(),
        "dot_decomposer: mismatched batch dim count"
    );

    let lhs_free = free_axes_of(
        lhs_rank,
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_axes_of(
        rhs_rank,
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );
    let fi_l = lhs_free.len();
    let ci_l = config.lhs_contracting_dims.len();
    let fi_r = rhs_free.len();
    let ci_r = config.rhs_contracting_dims.len();

    let lhs_target_perm: Vec<usize> = lhs_free
        .iter()
        .chain(config.lhs_contracting_dims.iter())
        .chain(config.lhs_batch_dims.iter())
        .copied()
        .collect();
    let (lhs_after_transpose_slot, lhs_after_transpose_shape, lhs_after_transpose_extents) =
        emit_transpose_if_needed(
            lhs_slot,
            lhs_shape,
            lhs_extents,
            &lhs_target_perm,
            instr.dtype,
            n_slots,
            new_instructions,
        );

    let (lhs_canon_slot, lhs_canon_shape, lhs_canon_extents) = if fi_l > 1 || ci_l > 1 {
        emit_merge_reshape(
            lhs_after_transpose_slot,
            &lhs_after_transpose_shape,
            &lhs_after_transpose_extents,
            fi_l,
            ci_l,
            nb,
            instr.dtype,
            n_slots,
            new_instructions,
        )
    } else {
        (
            lhs_after_transpose_slot,
            lhs_after_transpose_shape,
            lhs_after_transpose_extents,
        )
    };

    let rhs_target_perm: Vec<usize> = config
        .rhs_contracting_dims
        .iter()
        .chain(rhs_free.iter())
        .chain(config.rhs_batch_dims.iter())
        .copied()
        .collect();
    let (rhs_after_transpose_slot, rhs_after_transpose_shape, rhs_after_transpose_extents) =
        emit_transpose_if_needed(
            rhs_slot,
            rhs_shape,
            rhs_extents,
            &rhs_target_perm,
            instr.dtype,
            n_slots,
            new_instructions,
        );

    let (rhs_canon_slot, rhs_canon_shape, rhs_canon_extents) = if fi_r > 1 || ci_r > 1 {
        // For RHS the transpose target puts contracting first, then free.
        emit_merge_reshape_rhs(
            rhs_after_transpose_slot,
            &rhs_after_transpose_shape,
            &rhs_after_transpose_extents,
            fi_r,
            ci_r,
            nb,
            instr.dtype,
            n_slots,
            new_instructions,
        )
    } else {
        (
            rhs_after_transpose_slot,
            rhs_after_transpose_shape,
            rhs_after_transpose_extents,
        )
    };

    let lhs_free_count_canon = usize::from(fi_l > 0);
    let rhs_free_count_canon = usize::from(fi_r > 0);
    let canonical_config = DotGeneralConfig {
        lhs_contracting_dims: vec![lhs_free_count_canon],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: ((lhs_free_count_canon + 1)..(lhs_free_count_canon + 1 + nb)).collect(),
        rhs_batch_dims: ((rhs_free_count_canon + 1)..(rhs_free_count_canon + 1 + nb)).collect(),
    };

    let needs_output_reshape = fi_l > 1 || fi_r > 1;

    let canonical_output_slot = if needs_output_reshape {
        let s = *n_slots;
        *n_slots += 1;
        s
    } else {
        instr.output_slots[0]
    };

    // Compute canonical output shape directly: [M?, N?, batch...] referencing
    // the canonical operand shapes we just built.
    let mut canonical_output_shape: Vec<DimExpr> = Vec::new();
    let mut canonical_output_extents: Vec<ShapeExtent<DimExpr>> = Vec::new();
    if fi_l > 0 {
        canonical_output_shape.push(lhs_canon_shape[0].clone());
        canonical_output_extents.push(lhs_canon_extents[0].clone());
    }
    if fi_r > 0 {
        canonical_output_shape.push(rhs_canon_shape[rhs_free_count_canon].clone());
        canonical_output_extents.push(rhs_canon_extents[rhs_free_count_canon].clone());
    }
    for axis_offset in 0..nb {
        canonical_output_shape
            .push(lhs_canon_shape[lhs_free_count_canon + 1 + axis_offset].clone());
        canonical_output_extents
            .push(lhs_canon_extents[lhs_free_count_canon + 1 + axis_offset].clone());
    }

    new_instructions.push(ExecInstruction {
        op: ExecOp::DotGeneral(canonical_config),
        input_slots: vec![lhs_canon_slot, rhs_canon_slot],
        output_slots: vec![canonical_output_slot],
        dtype: instr.dtype,
        output_shapes: vec![canonical_output_shape.clone()],
        output_extents: vec![canonical_output_extents],
        last_use: Vec::new(),
    });

    if needs_output_reshape {
        // Target output shape: original DotGeneral output =
        //   [lhs_free_sizes..., rhs_free_sizes..., batch_sizes...]
        // Expressed via `InputDim{1, axis}` (original LHS) and
        // `InputDim{2, axis}` (original RHS) so the Reshape can evaluate
        // dynamic sizes at runtime.
        let mut to_shape: Vec<DimExpr> = Vec::new();
        for &axis in &lhs_free {
            to_shape.push(DimExpr::InputDim { input_idx: 1, axis });
        }
        for &axis in &rhs_free {
            to_shape.push(DimExpr::InputDim { input_idx: 2, axis });
        }
        for &axis in &config.lhs_batch_dims {
            to_shape.push(DimExpr::InputDim { input_idx: 1, axis });
        }

        // Metadata `output_shapes` uses the original DotGeneral output shape
        // directly, which we cloned from `instr` above.
        let metadata_shape = instr.output_shapes[0].clone();

        new_instructions.push(ExecInstruction {
            op: ExecOp::Reshape {
                shape: to_shape.clone(),
            },
            input_slots: vec![canonical_output_slot, lhs_slot, rhs_slot],
            output_slots: vec![instr.output_slots[0]],
            dtype: instr.dtype,
            output_shapes: vec![metadata_shape],
            output_extents: instr.output_extents.clone(),
            last_use: Vec::new(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_transpose_if_needed(
    input_slot: usize,
    input_shape: &[DimExpr],
    input_extents: &[ShapeExtent<DimExpr>],
    target_perm: &[usize],
    dtype: DType,
    n_slots: &mut usize,
    new_instructions: &mut Vec<ExecInstruction>,
) -> (usize, Vec<DimExpr>, Vec<ShapeExtent<DimExpr>>) {
    let is_identity = target_perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return (input_slot, input_shape.to_vec(), input_extents.to_vec());
    }
    let transposed_shape: Vec<DimExpr> = target_perm
        .iter()
        .map(|&axis| input_shape[axis].clone())
        .collect();
    let transposed_extents: Vec<ShapeExtent<DimExpr>> = target_perm
        .iter()
        .map(|&axis| input_extents[axis].clone())
        .collect();
    let out_slot = *n_slots;
    *n_slots += 1;
    new_instructions.push(ExecInstruction {
        op: ExecOp::Transpose {
            perm: target_perm.to_vec(),
        },
        input_slots: vec![input_slot],
        output_slots: vec![out_slot],
        dtype,
        output_shapes: vec![transposed_shape.clone()],
        output_extents: vec![transposed_extents.clone()],
        last_use: Vec::new(),
    });
    (out_slot, transposed_shape, transposed_extents)
}

/// LHS merge: [free..., contracting..., batch...] -> [M?, K, batch...].
#[allow(clippy::too_many_arguments)]
fn emit_merge_reshape(
    input_slot: usize,
    input_shape: &[DimExpr],
    input_extents: &[ShapeExtent<DimExpr>],
    fi: usize,
    ci: usize,
    nb: usize,
    dtype: DType,
    n_slots: &mut usize,
    new_instructions: &mut Vec<ExecInstruction>,
) -> (usize, Vec<DimExpr>, Vec<ShapeExtent<DimExpr>>) {
    // Runtime `shape` (op parameter) uses `InputDim{0, axis}` referring to
    // this Reshape's own input slot.
    let mut to_shape: Vec<DimExpr> = Vec::new();
    if fi > 0 {
        if fi == 1 {
            to_shape.push(DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            });
        } else {
            to_shape.push(product_of_input_dims(0, 0, fi));
        }
    }
    if ci == 1 {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: fi,
        });
    } else {
        to_shape.push(product_of_input_dims(0, fi, fi + ci));
    }
    for k in 0..nb {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: fi + ci + k,
        });
    }

    // Metadata `output_shapes` uses original-input DimExprs so downstream
    // passes can reason about the resulting dims.
    let mut output_shape_meta: Vec<DimExpr> = Vec::new();
    let mut output_extents_meta: Vec<ShapeExtent<DimExpr>> = Vec::new();
    if fi > 0 {
        output_shape_meta.push(merge_span(input_shape, 0, fi));
        output_extents_meta.push(merge_extent_span(input_shape, input_extents, 0, fi));
    }
    output_shape_meta.push(merge_span(input_shape, fi, fi + ci));
    output_extents_meta.push(merge_extent_span(input_shape, input_extents, fi, fi + ci));
    for k in 0..nb {
        output_shape_meta.push(input_shape[fi + ci + k].clone());
        output_extents_meta.push(input_extents[fi + ci + k].clone());
    }

    let out_slot = *n_slots;
    *n_slots += 1;
    new_instructions.push(ExecInstruction {
        op: ExecOp::Reshape { shape: to_shape },
        input_slots: vec![input_slot],
        output_slots: vec![out_slot],
        dtype,
        output_shapes: vec![output_shape_meta.clone()],
        output_extents: vec![output_extents_meta.clone()],
        last_use: Vec::new(),
    });
    (out_slot, output_shape_meta, output_extents_meta)
}

/// RHS merge: [contracting..., free..., batch...] -> [K, N?, batch...].
#[allow(clippy::too_many_arguments)]
fn emit_merge_reshape_rhs(
    input_slot: usize,
    input_shape: &[DimExpr],
    input_extents: &[ShapeExtent<DimExpr>],
    fi: usize,
    ci: usize,
    nb: usize,
    dtype: DType,
    n_slots: &mut usize,
    new_instructions: &mut Vec<ExecInstruction>,
) -> (usize, Vec<DimExpr>, Vec<ShapeExtent<DimExpr>>) {
    let mut to_shape: Vec<DimExpr> = Vec::new();
    if ci == 1 {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        });
    } else {
        to_shape.push(product_of_input_dims(0, 0, ci));
    }
    if fi > 0 {
        if fi == 1 {
            to_shape.push(DimExpr::InputDim {
                input_idx: 0,
                axis: ci,
            });
        } else {
            to_shape.push(product_of_input_dims(0, ci, ci + fi));
        }
    }
    for k in 0..nb {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: ci + fi + k,
        });
    }

    let mut output_shape_meta: Vec<DimExpr> = Vec::new();
    let mut output_extents_meta: Vec<ShapeExtent<DimExpr>> = Vec::new();
    output_shape_meta.push(merge_span(input_shape, 0, ci));
    output_extents_meta.push(merge_extent_span(input_shape, input_extents, 0, ci));
    if fi > 0 {
        output_shape_meta.push(merge_span(input_shape, ci, ci + fi));
        output_extents_meta.push(merge_extent_span(input_shape, input_extents, ci, ci + fi));
    }
    for k in 0..nb {
        output_shape_meta.push(input_shape[ci + fi + k].clone());
        output_extents_meta.push(input_extents[ci + fi + k].clone());
    }

    let out_slot = *n_slots;
    *n_slots += 1;
    new_instructions.push(ExecInstruction {
        op: ExecOp::Reshape { shape: to_shape },
        input_slots: vec![input_slot],
        output_slots: vec![out_slot],
        dtype,
        output_shapes: vec![output_shape_meta.clone()],
        output_extents: vec![output_extents_meta.clone()],
        last_use: Vec::new(),
    });
    (out_slot, output_shape_meta, output_extents_meta)
}

fn merge_span(shape: &[DimExpr], start: usize, end: usize) -> DimExpr {
    assert!(end > start, "merge_span: empty range");
    let mut result = shape[start].clone();
    for axis in (start + 1)..end {
        result = DimExpr::mul(result, shape[axis].clone());
    }
    result
}

fn merge_extent_span(
    shape: &[DimExpr],
    extents: &[ShapeExtent<DimExpr>],
    start: usize,
    end: usize,
) -> ShapeExtent<DimExpr> {
    assert!(end > start, "merge_extent_span: empty range");
    let merged_dim = merge_span(shape, start, end);
    let mut has_upper_bound = false;

    for extent in &extents[start..end] {
        match extent {
            ShapeExtent::Exact(_) => {}
            ShapeExtent::UpperBound(_) => has_upper_bound = true,
            ShapeExtent::Unknown => return ShapeExtent::unknown(),
        }
    }

    if has_upper_bound {
        ShapeExtent::upper_bound(merged_dim)
    } else {
        ShapeExtent::exact(merged_dim)
    }
}

// ============================================================================
// Pass 4: DeadCodeElimination
// ============================================================================
//
// Remove instructions whose outputs are never consumed by a program output
// or by a later instruction. `transpose_folding` + `dot_decomposer` can leave
// dead `Transpose` instructions (the folded-out producer is bypassed but
// remains in the program), and DCE reclaims that wasted runtime work.

/// Drop instructions with no downstream consumer. Preserves instructions with
/// observable side effects (`ValidateNonsingular`).
pub fn eliminate_dead_code(program: &mut ExecProgram) {
    let mut live_slots: std::collections::HashSet<usize> =
        program.output_slots.iter().copied().collect();
    let mut keep = vec![false; program.instructions.len()];
    for idx in (0..program.instructions.len()).rev() {
        let instr = &program.instructions[idx];
        let has_live_output = instr.output_slots.iter().any(|s| live_slots.contains(s));
        let is_side_effecting = matches!(&instr.op, ExecOp::ValidateNonsingular);
        if has_live_output || is_side_effecting {
            keep[idx] = true;
            for &slot in &instr.input_slots {
                live_slots.insert(slot);
            }
        }
    }
    let new_instructions: Vec<ExecInstruction> = program
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(i, instr)| keep[i].then(|| instr.clone()))
        .collect();
    program.instructions = new_instructions;
}
