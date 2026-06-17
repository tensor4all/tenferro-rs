use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ShapeExtent;
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use crate::Result;

use super::{exact_extents_from_shape, invalid_compiled_graph, missing_slot_meta};

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
pub fn dot_decomposer(program: &mut ExecProgram, input_shapes: &[Vec<DimExpr>]) -> Result<()> {
    if program.input_slots.len() != input_shapes.len() {
        return Err(invalid_compiled_graph(format!(
            "dot_decomposer input shape count {} must match input slot count {}",
            input_shapes.len(),
            program.input_slots.len()
        )));
    }

    let mut slot_shapes: Vec<Option<Vec<DimExpr>>> = vec![None; program.n_slots];
    let mut slot_extents: Vec<Option<Vec<ShapeExtent<DimExpr>>>> = vec![None; program.n_slots];
    for (index, &slot) in program.input_slots.iter().enumerate() {
        if slot >= program.n_slots {
            return Err(invalid_compiled_graph(format!(
                "dot_decomposer input slot {slot} is outside slot table of length {}",
                program.n_slots
            )));
        }
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
            if *slot >= program.n_slots {
                return Err(invalid_compiled_graph(format!(
                    "dot_decomposer output slot {slot} is outside slot table of length {}",
                    program.n_slots
                )));
            }
            slot_shapes[*slot] = Some(shape.clone());
            slot_extents[*slot] = Some(extents.clone());
        }
    }

    let mut new_instructions: Vec<ExecInstruction> = Vec::with_capacity(program.instructions.len());
    let mut n_slots = program.n_slots;

    for instr in &program.instructions {
        let dot_config_and_conj = match &instr.op {
            ExecOp::DotGeneral(config) => Some((config, false, false)),
            ExecOp::DotGeneralWithConj {
                config,
                lhs_conj,
                rhs_conj,
            } => Some((config, *lhs_conj, *rhs_conj)),
            _ => None,
        };

        if let Some((config, lhs_conj, rhs_conj)) = dot_config_and_conj {
            if instr.input_slots.len() == 2 && !config.lhs_contracting_dims.is_empty() {
                let lhs_slot = instr.input_slots[0];
                let rhs_slot = instr.input_slots[1];
                let lhs_shape = require_slot_shape(&slot_shapes, lhs_slot)?;
                let rhs_shape = require_slot_shape(&slot_shapes, rhs_slot)?;
                let lhs_extents = require_slot_extents(&slot_extents, lhs_slot)?;
                let rhs_extents = require_slot_extents(&slot_extents, rhs_slot)?;
                if !is_dot_canonical(config, lhs_shape.len(), rhs_shape.len()) {
                    let mut builder = InstructionBuilder {
                        n_slots: &mut n_slots,
                        instructions: &mut new_instructions,
                    };
                    decompose_dot(
                        DotDecomposeInput {
                            instr,
                            config,
                            lhs: OperandMeta {
                                slot: lhs_slot,
                                shape: lhs_shape,
                                extents: lhs_extents,
                            },
                            rhs: OperandMeta {
                                slot: rhs_slot,
                                shape: rhs_shape,
                                extents: rhs_extents,
                            },
                            lhs_conj,
                            rhs_conj,
                        },
                        &mut builder,
                    )?;
                    continue;
                }
            }
        }
        new_instructions.push(instr.clone());
    }

    program.instructions = new_instructions;
    program.n_slots = n_slots;
    Ok(())
}

#[derive(Clone, Copy)]
struct OperandMeta<'a> {
    slot: usize,
    shape: &'a [DimExpr],
    extents: &'a [ShapeExtent<DimExpr>],
}

struct EmittedOperand {
    slot: usize,
    shape: Vec<DimExpr>,
    extents: Vec<ShapeExtent<DimExpr>>,
}

struct DotDecomposeInput<'a> {
    instr: &'a ExecInstruction,
    config: &'a DotGeneralConfig,
    lhs: OperandMeta<'a>,
    rhs: OperandMeta<'a>,
    lhs_conj: bool,
    rhs_conj: bool,
}

struct InstructionBuilder<'a> {
    n_slots: &'a mut usize,
    instructions: &'a mut Vec<ExecInstruction>,
}

#[derive(Clone, Copy)]
struct MergeReshapeSpec {
    free_count: usize,
    contracting_count: usize,
    batch_count: usize,
    side: MergeSide,
}

#[derive(Clone, Copy)]
enum MergeSide {
    Lhs,
    Rhs,
}

impl InstructionBuilder<'_> {
    fn next_slot(&mut self) -> usize {
        let slot = *self.n_slots;
        *self.n_slots += 1;
        slot
    }

    fn push(&mut self, instr: ExecInstruction) {
        self.instructions.push(instr);
    }
}

fn require_slot_shape(slot_shapes: &[Option<Vec<DimExpr>>], slot: usize) -> Result<&[DimExpr]> {
    Ok(slot_shapes
        .get(slot)
        .and_then(Option::as_ref)
        .ok_or_else(|| missing_slot_meta("shape", slot))?
        .as_slice())
}

fn require_slot_extents(
    slot_extents: &[Option<Vec<ShapeExtent<DimExpr>>>],
    slot: usize,
) -> Result<&[ShapeExtent<DimExpr>]> {
    Ok(slot_extents
        .get(slot)
        .and_then(Option::as_ref)
        .ok_or_else(|| missing_slot_meta("extents", slot))?
        .as_slice())
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
fn decompose_dot(input: DotDecomposeInput<'_>, builder: &mut InstructionBuilder<'_>) -> Result<()> {
    let instr = input.instr;
    let config = input.config;
    let lhs = input.lhs;
    let rhs = input.rhs;

    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();
    let nb = config.lhs_batch_dims.len();
    if nb != config.rhs_batch_dims.len() {
        return Err(invalid_compiled_graph(format!(
            "dot_decomposer: lhs batch dim count {} must match rhs batch dim count {}",
            nb,
            config.rhs_batch_dims.len()
        )));
    }

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
    let lhs_after_transpose = emit_transpose_if_needed(lhs, &lhs_target_perm, instr.dtype, builder);

    let lhs_canon = if fi_l > 1 || ci_l > 1 {
        emit_merge_reshape(
            &lhs_after_transpose,
            MergeReshapeSpec {
                free_count: fi_l,
                contracting_count: ci_l,
                batch_count: nb,
                side: MergeSide::Lhs,
            },
            instr.dtype,
            builder,
        )
    } else {
        lhs_after_transpose
    };

    let rhs_target_perm: Vec<usize> = config
        .rhs_contracting_dims
        .iter()
        .chain(rhs_free.iter())
        .chain(config.rhs_batch_dims.iter())
        .copied()
        .collect();
    let rhs_after_transpose = emit_transpose_if_needed(rhs, &rhs_target_perm, instr.dtype, builder);

    let rhs_canon = if fi_r > 1 || ci_r > 1 {
        emit_merge_reshape(
            &rhs_after_transpose,
            MergeReshapeSpec {
                free_count: fi_r,
                contracting_count: ci_r,
                batch_count: nb,
                side: MergeSide::Rhs,
            },
            instr.dtype,
            builder,
        )
    } else {
        rhs_after_transpose
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
        builder.next_slot()
    } else {
        instr.output_slots[0]
    };

    // Compute canonical output shape directly: [M?, N?, batch...] referencing
    // the canonical operand shapes we just built.
    let mut canonical_output_shape: Vec<DimExpr> = Vec::new();
    let mut canonical_output_extents: Vec<ShapeExtent<DimExpr>> = Vec::new();
    if fi_l > 0 {
        canonical_output_shape.push(lhs_canon.shape[0].clone());
        canonical_output_extents.push(lhs_canon.extents[0].clone());
    }
    if fi_r > 0 {
        canonical_output_shape.push(rhs_canon.shape[rhs_free_count_canon].clone());
        canonical_output_extents.push(rhs_canon.extents[rhs_free_count_canon].clone());
    }
    for axis_offset in 0..nb {
        canonical_output_shape
            .push(lhs_canon.shape[lhs_free_count_canon + 1 + axis_offset].clone());
        canonical_output_extents
            .push(lhs_canon.extents[lhs_free_count_canon + 1 + axis_offset].clone());
    }

    builder.push(ExecInstruction {
        op: if input.lhs_conj || input.rhs_conj {
            ExecOp::DotGeneralWithConj {
                config: canonical_config,
                lhs_conj: input.lhs_conj,
                rhs_conj: input.rhs_conj,
            }
        } else {
            ExecOp::DotGeneral(canonical_config)
        },
        input_slots: vec![lhs_canon.slot, rhs_canon.slot],
        output_slots: vec![canonical_output_slot],
        dtype: instr.dtype,
        output_shapes: vec![canonical_output_shape.clone()].into(),
        output_extents: vec![canonical_output_extents].into(),
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

        builder.push(ExecInstruction {
            op: ExecOp::Reshape {
                shape: to_shape.clone(),
            },
            input_slots: vec![canonical_output_slot, lhs.slot, rhs.slot],
            output_slots: vec![instr.output_slots[0]],
            dtype: instr.dtype,
            output_shapes: vec![metadata_shape].into(),
            output_extents: instr.output_extents.clone(),
            last_use: Vec::new(),
        });
    }
    Ok(())
}

fn emit_transpose_if_needed(
    operand: OperandMeta<'_>,
    target_perm: &[usize],
    dtype: DType,
    builder: &mut InstructionBuilder<'_>,
) -> EmittedOperand {
    let is_identity = target_perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return EmittedOperand {
            slot: operand.slot,
            shape: operand.shape.to_vec(),
            extents: operand.extents.to_vec(),
        };
    }
    let transposed_shape: Vec<DimExpr> = target_perm
        .iter()
        .map(|&axis| operand.shape[axis].clone())
        .collect();
    let transposed_extents: Vec<ShapeExtent<DimExpr>> = target_perm
        .iter()
        .map(|&axis| operand.extents[axis].clone())
        .collect();
    let out_slot = builder.next_slot();
    builder.push(ExecInstruction {
        op: ExecOp::Transpose {
            perm: target_perm.to_vec(),
        },
        input_slots: vec![operand.slot],
        output_slots: vec![out_slot],
        dtype,
        output_shapes: vec![transposed_shape.clone()].into(),
        output_extents: vec![transposed_extents.clone()].into(),
        last_use: Vec::new(),
    });
    EmittedOperand {
        slot: out_slot,
        shape: transposed_shape,
        extents: transposed_extents,
    }
}

fn emit_merge_reshape(
    operand: &EmittedOperand,
    spec: MergeReshapeSpec,
    dtype: DType,
    builder: &mut InstructionBuilder<'_>,
) -> EmittedOperand {
    let to_shape = match spec.side {
        MergeSide::Lhs => lhs_merge_shape(spec),
        MergeSide::Rhs => rhs_merge_shape(spec),
    };
    let (output_shape_meta, output_extents_meta) = match spec.side {
        MergeSide::Lhs => lhs_merge_meta(operand, spec),
        MergeSide::Rhs => rhs_merge_meta(operand, spec),
    };

    let out_slot = builder.next_slot();
    builder.push(ExecInstruction {
        op: ExecOp::Reshape { shape: to_shape },
        input_slots: vec![operand.slot],
        output_slots: vec![out_slot],
        dtype,
        output_shapes: vec![output_shape_meta.clone()].into(),
        output_extents: vec![output_extents_meta.clone()].into(),
        last_use: Vec::new(),
    });
    EmittedOperand {
        slot: out_slot,
        shape: output_shape_meta,
        extents: output_extents_meta,
    }
}

fn lhs_merge_shape(spec: MergeReshapeSpec) -> Vec<DimExpr> {
    // Runtime `shape` (op parameter) uses `InputDim{0, axis}` referring to
    // this Reshape's own input slot.
    let mut to_shape: Vec<DimExpr> = Vec::new();
    if spec.free_count > 0 {
        if spec.free_count == 1 {
            to_shape.push(DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            });
        } else {
            to_shape.push(product_of_input_dims(0, 0, spec.free_count));
        }
    }
    if spec.contracting_count == 1 {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: spec.free_count,
        });
    } else {
        to_shape.push(product_of_input_dims(
            0,
            spec.free_count,
            spec.free_count + spec.contracting_count,
        ));
    }
    for k in 0..spec.batch_count {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: spec.free_count + spec.contracting_count + k,
        });
    }
    to_shape
}

fn rhs_merge_shape(spec: MergeReshapeSpec) -> Vec<DimExpr> {
    let mut to_shape: Vec<DimExpr> = Vec::new();
    if spec.contracting_count == 1 {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        });
    } else {
        to_shape.push(product_of_input_dims(0, 0, spec.contracting_count));
    }
    if spec.free_count > 0 {
        if spec.free_count == 1 {
            to_shape.push(DimExpr::InputDim {
                input_idx: 0,
                axis: spec.contracting_count,
            });
        } else {
            to_shape.push(product_of_input_dims(
                0,
                spec.contracting_count,
                spec.contracting_count + spec.free_count,
            ));
        }
    }
    for k in 0..spec.batch_count {
        to_shape.push(DimExpr::InputDim {
            input_idx: 0,
            axis: spec.contracting_count + spec.free_count + k,
        });
    }
    to_shape
}

fn lhs_merge_meta(
    operand: &EmittedOperand,
    spec: MergeReshapeSpec,
) -> (Vec<DimExpr>, Vec<ShapeExtent<DimExpr>>) {
    let mut shape_meta: Vec<DimExpr> = Vec::new();
    let mut extents_meta: Vec<ShapeExtent<DimExpr>> = Vec::new();
    if spec.free_count > 0 {
        shape_meta.push(merge_span(&operand.shape, 0, spec.free_count));
        extents_meta.push(merge_extent_span(
            &operand.shape,
            &operand.extents,
            0,
            spec.free_count,
        ));
    }
    shape_meta.push(merge_span(
        &operand.shape,
        spec.free_count,
        spec.free_count + spec.contracting_count,
    ));
    extents_meta.push(merge_extent_span(
        &operand.shape,
        &operand.extents,
        spec.free_count,
        spec.free_count + spec.contracting_count,
    ));
    for k in 0..spec.batch_count {
        shape_meta.push(operand.shape[spec.free_count + spec.contracting_count + k].clone());
        extents_meta.push(operand.extents[spec.free_count + spec.contracting_count + k].clone());
    }
    (shape_meta, extents_meta)
}

fn rhs_merge_meta(
    operand: &EmittedOperand,
    spec: MergeReshapeSpec,
) -> (Vec<DimExpr>, Vec<ShapeExtent<DimExpr>>) {
    let mut shape_meta: Vec<DimExpr> = Vec::new();
    let mut extents_meta: Vec<ShapeExtent<DimExpr>> = Vec::new();
    shape_meta.push(merge_span(&operand.shape, 0, spec.contracting_count));
    extents_meta.push(merge_extent_span(
        &operand.shape,
        &operand.extents,
        0,
        spec.contracting_count,
    ));
    if spec.free_count > 0 {
        shape_meta.push(merge_span(
            &operand.shape,
            spec.contracting_count,
            spec.contracting_count + spec.free_count,
        ));
        extents_meta.push(merge_extent_span(
            &operand.shape,
            &operand.extents,
            spec.contracting_count,
            spec.contracting_count + spec.free_count,
        ));
    }
    for k in 0..spec.batch_count {
        shape_meta.push(operand.shape[spec.contracting_count + spec.free_count + k].clone());
        extents_meta.push(operand.extents[spec.contracting_count + spec.free_count + k].clone());
    }
    (shape_meta, extents_meta)
}

fn merge_span(shape: &[DimExpr], start: usize, end: usize) -> DimExpr {
    assert!(end > start, "merge_span: empty range");
    let mut result = shape[start].clone();
    for dim in shape.iter().take(end).skip(start + 1) {
        result = DimExpr::mul(result, dim.clone());
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
