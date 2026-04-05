use computegraph::compile::CompiledProgram;
use tenferro_algebra::Algebra;
use tenferro_ops::semiring_op::SemiringOp;
use tenferro_ops::semiring_op_kind::SemiringOpKind;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig};

use super::exec::{ExecInstruction, ExecOp, ExecProgram};
use super::stablehlo::{StableHloInstruction, StableHloOp, StableHloProgram};

pub fn lower_to_stablehlo(prog: &CompiledProgram<StdTensorOp>) -> StableHloProgram {
    let instructions = prog
        .instructions
        .iter()
        .map(|instr| {
            let op = match &instr.op {
                StdTensorOp::Add => StableHloOp::Add,
                StdTensorOp::Mul => StableHloOp::Multiply,
                StdTensorOp::Neg => StableHloOp::Negate,
                StdTensorOp::Conj => StableHloOp::Conj,
                StdTensorOp::DotGeneral(c) => StableHloOp::DotGeneral(c.clone()),
                StdTensorOp::Transpose { perm } => StableHloOp::Transpose { perm: perm.clone() },
                StdTensorOp::Reshape { to_shape, .. } => StableHloOp::Reshape {
                    shape: to_shape.clone(),
                },
                StdTensorOp::BroadcastInDim { shape, dims } => StableHloOp::BroadcastInDim {
                    shape: shape.clone(),
                    dims: dims.clone(),
                },
                StdTensorOp::ReduceSum { axes, .. } => {
                    StableHloOp::ReduceSum { axes: axes.clone() }
                }
                StdTensorOp::ExtractDiag { axis_a, axis_b } => StableHloOp::ExtractDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                StdTensorOp::EmbedDiag { axis_a, axis_b } => StableHloOp::EmbedDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                StdTensorOp::Div => StableHloOp::Divide,
                StdTensorOp::Abs => StableHloOp::Abs,
                StdTensorOp::Sign => StableHloOp::Sign,
                StdTensorOp::Maximum => StableHloOp::Maximum,
                StdTensorOp::Minimum => StableHloOp::Minimum,
                StdTensorOp::Compare(dir) => StableHloOp::Compare(dir.clone()),
                StdTensorOp::Select => StableHloOp::Select,
                StdTensorOp::Clamp => StableHloOp::Clamp,
                StdTensorOp::Exp => StableHloOp::Exp,
                StdTensorOp::Log => StableHloOp::Log,
                StdTensorOp::Sin => StableHloOp::Sin,
                StdTensorOp::Cos => StableHloOp::Cos,
                StdTensorOp::Tanh => StableHloOp::Tanh,
                StdTensorOp::Sqrt => StableHloOp::Sqrt,
                StdTensorOp::Rsqrt => StableHloOp::Rsqrt,
                StdTensorOp::Pow => StableHloOp::Pow,
                StdTensorOp::Expm1 => StableHloOp::Expm1,
                StdTensorOp::Log1p => StableHloOp::Log1p,
                StdTensorOp::Cholesky => StableHloOp::Cholesky,
                StdTensorOp::Svd => StableHloOp::CustomCall {
                    target: "svd".to_string(),
                },
                StdTensorOp::Qr => StableHloOp::CustomCall {
                    target: "qr".to_string(),
                },
                StdTensorOp::Eigh => StableHloOp::CustomCall {
                    target: "eigh".to_string(),
                },
                StdTensorOp::Solve => StableHloOp::CustomCall {
                    target: "solve".to_string(),
                },
                _ => todo!("lower_to_stablehlo: unsupported op {:?}", instr.op),
            };
            StableHloInstruction {
                op,
                input_slots: instr.inputs.clone(),
                output_slots: instr.outputs.clone(),
                dtype: DType::F64,
            }
        })
        .collect();
    StableHloProgram {
        instructions,
        input_slots: prog.input_slots.clone(),
        output_slots: prog.output_slots.clone(),
        n_slots: prog.n_slots,
    }
}

pub fn lower_semiring_to_stablehlo<Alg>(prog: &CompiledProgram<SemiringOp<Alg>>) -> StableHloProgram
where
    Alg: Algebra + Send + Sync + 'static,
{
    let instructions = prog
        .instructions
        .iter()
        .map(|instr| {
            let op = match &instr.op.kind {
                SemiringOpKind::Add => StableHloOp::Add,
                SemiringOpKind::Mul => StableHloOp::Multiply,
                SemiringOpKind::DotGeneral(c) => StableHloOp::DotGeneral(c.clone()),
                SemiringOpKind::ReduceSum { axes } => StableHloOp::ReduceSum { axes: axes.clone() },
                SemiringOpKind::Transpose { perm } => StableHloOp::Transpose { perm: perm.clone() },
                SemiringOpKind::Reshape { shape } => StableHloOp::Reshape {
                    shape: shape.clone(),
                },
                SemiringOpKind::BroadcastInDim { shape, dims } => StableHloOp::BroadcastInDim {
                    shape: shape.clone(),
                    dims: dims.clone(),
                },
                SemiringOpKind::ExtractDiag { axis_a, axis_b } => StableHloOp::ExtractDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                SemiringOpKind::EmbedDiag { axis_a, axis_b } => StableHloOp::EmbedDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
            };
            StableHloInstruction {
                op,
                input_slots: instr.inputs.clone(),
                output_slots: instr.outputs.clone(),
                dtype: DType::F64,
            }
        })
        .collect();

    StableHloProgram {
        instructions,
        input_slots: prog.input_slots.clone(),
        output_slots: prog.output_slots.clone(),
        n_slots: prog.n_slots,
    }
}

pub fn compile_to_exec(stablehlo: &StableHloProgram) -> ExecProgram {
    // Run optimization passes on a mutable copy of the StableHLO program.
    let mut program = stablehlo.clone();
    dot_dimension_sorter(&mut program);
    reduction_simplification(&mut program);
    dot_decomposer(&mut program);
    transpose_folding(&mut program);

    let instructions: Vec<ExecInstruction> = program
        .instructions
        .iter()
        .enumerate()
        .map(|(idx, instr)| {
            let op = match &instr.op {
                StableHloOp::Add => ExecOp::Add,
                StableHloOp::Multiply => ExecOp::Multiply,
                StableHloOp::Negate => ExecOp::Negate,
                StableHloOp::Conj => ExecOp::Conj,
                StableHloOp::DotGeneral(c) => ExecOp::BatchedGemm(c.clone()),
                StableHloOp::Transpose { perm } => ExecOp::Permute { perm: perm.clone() },
                StableHloOp::Reshape { shape } => ExecOp::Reshape {
                    shape: shape.clone(),
                },
                StableHloOp::BroadcastInDim { shape, dims } => ExecOp::BroadcastInDim {
                    shape: shape.clone(),
                    dims: dims.clone(),
                },
                StableHloOp::ReduceSum { axes } => ExecOp::ReduceSum { axes: axes.clone() },
                StableHloOp::ExtractDiag { axis_a, axis_b } => ExecOp::ExtractDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                StableHloOp::EmbedDiag { axis_a, axis_b } => ExecOp::EmbedDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                StableHloOp::Divide => ExecOp::Divide,
                StableHloOp::Abs => ExecOp::Abs,
                StableHloOp::Sign => ExecOp::Sign,
                StableHloOp::Maximum => ExecOp::Maximum,
                StableHloOp::Minimum => ExecOp::Minimum,
                StableHloOp::Compare(dir) => ExecOp::Compare(dir.clone()),
                StableHloOp::Select => ExecOp::Select,
                StableHloOp::Clamp => ExecOp::Clamp,
                StableHloOp::Exp => ExecOp::Exp,
                StableHloOp::Log => ExecOp::Log,
                StableHloOp::Sin => ExecOp::Sin,
                StableHloOp::Cos => ExecOp::Cos,
                StableHloOp::Tanh => ExecOp::Tanh,
                StableHloOp::Sqrt => ExecOp::Sqrt,
                StableHloOp::Rsqrt => ExecOp::Rsqrt,
                StableHloOp::Pow => ExecOp::Pow,
                StableHloOp::Expm1 => ExecOp::Expm1,
                StableHloOp::Log1p => ExecOp::Log1p,
                StableHloOp::Cholesky => ExecOp::Cholesky,
                StableHloOp::CustomCall { target } => ExecOp::CustomCall {
                    target: target.clone(),
                },
                _ => todo!("compile_to_exec: unsupported op {:?}", instr.op),
            };

            let last_use = compute_last_use(
                &instr.input_slots,
                idx,
                &program.instructions,
                &program.output_slots,
            );

            ExecInstruction {
                op,
                input_slots: instr.input_slots.clone(),
                output_slots: instr.output_slots.clone(),
                dtype: instr.dtype,
                last_use,
            }
        })
        .collect();

    ExecProgram {
        instructions,
        input_slots: program.input_slots.clone(),
        output_slots: program.output_slots.clone(),
        n_slots: program.n_slots,
    }
}

fn compute_last_use(
    input_slots: &[usize],
    current_idx: usize,
    all_instructions: &[StableHloInstruction],
    output_slots: &[usize],
) -> Vec<bool> {
    input_slots
        .iter()
        .map(|&slot| {
            if output_slots.contains(&slot) {
                return false;
            }
            for later in &all_instructions[current_idx + 1..] {
                if later.input_slots.contains(&slot) {
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
// Sort contracting dimensions of DotGeneral so that DotDecomposer can
// canonicalize without inserting unnecessary transposes.
// Fires when contracting dims are consecutive-if-sorted but not yet sorted.

/// Sort contracting dimensions of all DotGeneral instructions in place.
pub fn dot_dimension_sorter(program: &mut StableHloProgram) {
    for instr in &mut program.instructions {
        if let StableHloOp::DotGeneral(config) = &mut instr.op {
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

/// Check if `dims`, once sorted, would be consecutive (no gaps).
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

/// Return the permutation that sorts `dims` (argsort).
fn argsort(dims: &[usize]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..dims.len()).collect();
    indices.sort_by_key(|&i| dims[i]);
    indices
}

/// Apply a permutation: `result[i] = source[perm[i]]`.
fn apply_perm(source: &[usize], perm: &[usize]) -> Vec<usize> {
    perm.iter().map(|&p| source[p]).collect()
}

// ============================================================================
// Pass 2: ReductionSimplification
// ============================================================================
//
// Verify that any ReduceSum feeding into a DotGeneral appears before that
// DotGeneral in program order. In SSA form this is guaranteed by construction,
// so this pass is effectively a validation no-op.

/// Verify (and enforce) that ReduceSum instructions that produce inputs for
/// DotGeneral appear before the DotGeneral in program order.
pub fn reduction_simplification(program: &mut StableHloProgram) {
    // Build a map: output_slot -> instruction index.
    let mut producer: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (idx, instr) in program.instructions.iter().enumerate() {
        for &slot in &instr.output_slots {
            producer.insert(slot, idx);
        }
    }

    // For each DotGeneral, check that any ReduceSum producer comes earlier.
    for (dot_idx, instr) in program.instructions.iter().enumerate() {
        if !matches!(instr.op, StableHloOp::DotGeneral(_)) {
            continue;
        }
        for &input_slot in &instr.input_slots {
            if let Some(&prod_idx) = producer.get(&input_slot) {
                if matches!(
                    program.instructions[prod_idx].op,
                    StableHloOp::ReduceSum { .. }
                ) {
                    debug_assert!(
                        prod_idx < dot_idx,
                        "ReductionSimplification: ReduceSum at index {} should \
                         precede DotGeneral at index {}",
                        prod_idx,
                        dot_idx
                    );
                }
            }
        }
    }
}

// ============================================================================
// Pass 3: DotDecomposer
// ============================================================================
//
// Canonicalize DotGeneral into [batch..., M, K] x [batch..., K, N] form.
// This pass inserts Transpose and Reshape instructions as needed.
//
// NOTE: Full implementation requires shape tracking through the program.
// Currently, this pass only applies when tensor shapes can be inferred from
// the program metadata. For the POC, the pass is a structural skeleton that
// handles the canonicality check and defers non-canonical cases.

/// Canonicalize DotGeneral instructions that are not already in canonical form.
///
/// Canonical form:
///   - Exactly 1 contracting dim
///   - Batch dims are leading `[0, 1, ..., nb-1]`
///   - Operand rank <= batch_dims.len() + 2
///
/// TODO: Full implementation requires shape tracking (adding `shape` info to
/// `StableHloInstruction` or maintaining a side table). For now, this pass
/// only applies to DotGeneral instructions already in canonical form (no-op)
/// or that can be canonicalized without shape info (single-contracting-dim
/// with non-leading batch dims, where we insert Transpose only).
pub fn dot_decomposer(program: &mut StableHloProgram) {
    // Collect indices of non-canonical DotGeneral instructions.
    let non_canonical_indices: Vec<usize> = program
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(idx, instr)| {
            if let StableHloOp::DotGeneral(config) = &instr.op {
                if !is_dot_canonical(config) {
                    return Some(idx);
                }
            }
            None
        })
        .collect();

    if non_canonical_indices.is_empty() {
        return;
    }

    // TODO: Shape tracking is needed to fully decompose non-canonical dots.
    // For now, handle the case where we have a single contracting dim but
    // batch dims are not leading — we can insert Transpose to fix that.
    // Multi-contracting-dim cases require shape info for Reshape and are
    // deferred.
    let mut new_instructions: Vec<StableHloInstruction> = Vec::new();
    let mut n_slots = program.n_slots;

    for instr in &program.instructions {
        if let StableHloOp::DotGeneral(config) = &instr.op {
            if !is_dot_canonical(config) && config.lhs_contracting_dims.len() == 1 {
                // Single contracting dim but batch dims not leading.
                // We can fix this by inserting transposes and rewriting
                // the DotGeneral dimension numbers.
                let (lhs_instrs, lhs_new_config, lhs_slot, new_n) = decompose_operand_transpose(
                    instr.input_slots[0],
                    &config.lhs_batch_dims,
                    &config.lhs_contracting_dims,
                    config.lhs_rank,
                    true,
                    instr.dtype,
                    n_slots,
                );
                n_slots = new_n;

                let (rhs_instrs, rhs_new_config, rhs_slot, new_n) = decompose_operand_transpose(
                    instr.input_slots[1],
                    &config.rhs_batch_dims,
                    &config.rhs_contracting_dims,
                    config.rhs_rank,
                    false,
                    instr.dtype,
                    n_slots,
                );
                n_slots = new_n;

                new_instructions.extend(lhs_instrs);
                new_instructions.extend(rhs_instrs);

                let new_config = DotGeneralConfig {
                    lhs_contracting_dims: lhs_new_config.0,
                    rhs_contracting_dims: rhs_new_config.0,
                    lhs_batch_dims: lhs_new_config.1,
                    rhs_batch_dims: rhs_new_config.1,
                    lhs_rank: config.lhs_rank,
                    rhs_rank: config.rhs_rank,
                };

                new_instructions.push(StableHloInstruction {
                    op: StableHloOp::DotGeneral(new_config),
                    input_slots: vec![lhs_slot, rhs_slot],
                    output_slots: instr.output_slots.clone(),
                    dtype: instr.dtype,
                });
                continue;
            }
        }
        new_instructions.push(instr.clone());
    }

    program.instructions = new_instructions;
    program.n_slots = n_slots;
}

/// For a single operand of a DotGeneral, compute the transpose needed to put
/// batch dims first, then either (non_contracting, contracting) for LHS
/// or (contracting, non_contracting) for RHS.
///
/// Returns: (new instructions to insert, (new_contracting, new_batch), output slot, new n_slots)
fn decompose_operand_transpose(
    input_slot: usize,
    batch_dims: &[usize],
    contracting_dims: &[usize],
    rank: usize,
    is_lhs: bool,
    dtype: DType,
    mut n_slots: usize,
) -> (
    Vec<StableHloInstruction>,
    (Vec<usize>, Vec<usize>),
    usize,
    usize,
) {
    let nb = batch_dims.len();

    // Check if already in canonical position (batch leading).
    let batch_leading =
        batch_dims.len() == (0..nb).len() && batch_dims.iter().enumerate().all(|(i, &d)| d == i);

    if batch_leading {
        // Already good — return identity.
        let new_batch: Vec<usize> = (0..nb).collect();
        let new_contracting = contracting_dims.to_vec();
        return (
            Vec::new(),
            (new_contracting, new_batch),
            input_slot,
            n_slots,
        );
    }

    // Compute non-contracting dims.
    let mut is_batch_or_contracting = vec![false; rank];
    for &d in batch_dims {
        if d < rank {
            is_batch_or_contracting[d] = true;
        }
    }
    for &d in contracting_dims {
        if d < rank {
            is_batch_or_contracting[d] = true;
        }
    }
    let non_contracting: Vec<usize> = (0..rank).filter(|&d| !is_batch_or_contracting[d]).collect();

    // Build target order: batch + non_contracting + contracting (LHS)
    //                  or batch + contracting + non_contracting (RHS)
    let target_order: Vec<usize> = if is_lhs {
        batch_dims
            .iter()
            .chain(non_contracting.iter())
            .chain(contracting_dims.iter())
            .copied()
            .collect()
    } else {
        batch_dims
            .iter()
            .chain(contracting_dims.iter())
            .chain(non_contracting.iter())
            .copied()
            .collect()
    };

    let is_identity = target_order.iter().enumerate().all(|(i, &d)| d == i);
    if is_identity {
        let new_batch: Vec<usize> = (0..nb).collect();
        let new_contracting = if is_lhs {
            vec![nb + non_contracting.len()]
        } else {
            vec![nb]
        };
        return (
            Vec::new(),
            (new_contracting, new_batch),
            input_slot,
            n_slots,
        );
    }

    // Insert a Transpose instruction.
    let transpose_output = n_slots;
    n_slots += 1;

    let transpose_instr = StableHloInstruction {
        op: StableHloOp::Transpose { perm: target_order },
        input_slots: vec![input_slot],
        output_slots: vec![transpose_output],
        dtype,
    };

    // After transpose, batch dims are leading.
    let new_batch: Vec<usize> = (0..nb).collect();
    let new_contracting = if is_lhs {
        // contracting is at position nb + non_contracting.len()
        vec![nb + non_contracting.len()]
    } else {
        // contracting is at position nb
        vec![nb]
    };

    (
        vec![transpose_instr],
        (new_contracting, new_batch),
        transpose_output,
        n_slots,
    )
}

/// Check if a DotGeneral is already in canonical form:
///   - exactly 1 contracting dim
///   - batch dims are leading [0, 1, ..., nb-1] for both lhs and rhs
fn is_dot_canonical(config: &DotGeneralConfig) -> bool {
    let nb = config.lhs_batch_dims.len();
    let leading_batch: Vec<usize> = (0..nb).collect();

    config.lhs_contracting_dims.len() == 1
        && config.lhs_batch_dims == leading_batch
        && config.rhs_batch_dims == leading_batch
}

// ============================================================================
// Pass 4: TransposeFolding
// ============================================================================
//
// Absorb Transpose instructions that feed directly into DotGeneral by
// adjusting the DotGeneral dimension numbers and bypassing the Transpose.
// Runs until fixed point (no more folds possible).

/// Fold Transpose instructions into DotGeneral dimension numbers.
pub fn transpose_folding(program: &mut StableHloProgram) {
    loop {
        let changed = transpose_fold_one_pass(program);
        if !changed {
            break;
        }
    }
}

fn transpose_fold_one_pass(program: &mut StableHloProgram) -> bool {
    let mut changed = false;

    for i in 0..program.instructions.len() {
        let is_dot = matches!(program.instructions[i].op, StableHloOp::DotGeneral(_));
        if !is_dot {
            continue;
        }

        // Try folding LHS (operand index 0).
        if try_fold_operand(program, i, 0) {
            changed = true;
        }

        // Try folding RHS (operand index 1).
        if program.instructions[i].input_slots.len() > 1 && try_fold_operand(program, i, 1) {
            changed = true;
        }
    }

    changed
}

/// Try to fold a Transpose feeding into the given operand of a DotGeneral.
/// Returns true if a fold was performed.
fn try_fold_operand(program: &mut StableHloProgram, dot_idx: usize, operand_idx: usize) -> bool {
    let input_slot = program.instructions[dot_idx].input_slots[operand_idx];

    // Find the producer of this slot.
    let producer_idx = match find_producer(program, input_slot) {
        Some(idx) => idx,
        None => return false,
    };

    // Check if the producer is a Transpose.
    let perm = match &program.instructions[producer_idx].op {
        StableHloOp::Transpose { perm } => perm.clone(),
        _ => return false,
    };

    // Extract the DotGeneral config.
    let config = match &program.instructions[dot_idx].op {
        StableHloOp::DotGeneral(c) => c.clone(),
        _ => return false,
    };

    // Check foldability.
    if !is_transpose_foldable(&config, operand_idx, &perm) {
        return false;
    }

    // Fold: apply the permutation to dimension numbers.
    let new_config = fold_transpose_into_dot(&config, operand_idx, &perm);

    // Get the original input (before Transpose).
    let original_input = program.instructions[producer_idx].input_slots[0];

    // Update the DotGeneral.
    program.instructions[dot_idx].op = StableHloOp::DotGeneral(new_config);
    program.instructions[dot_idx].input_slots[operand_idx] = original_input;

    true
}

/// Find which instruction produces a given slot.
fn find_producer(program: &StableHloProgram, slot: usize) -> Option<usize> {
    program
        .instructions
        .iter()
        .position(|inst| inst.output_slots.contains(&slot))
}

/// Check if a Transpose can be folded into DotGeneral for the given operand.
///
/// Foldability conditions:
/// - Batch dims must be unchanged by the permutation (perm[d] == d for each
///   batch dim).
/// - Exactly 1 contracting dimension.
fn is_transpose_foldable(config: &DotGeneralConfig, operand_idx: usize, perm: &[usize]) -> bool {
    let (batch_dims, contracting_dims) = if operand_idx == 0 {
        (&config.lhs_batch_dims, &config.lhs_contracting_dims)
    } else {
        (&config.rhs_batch_dims, &config.rhs_contracting_dims)
    };

    // Batch dims must be fixed points of the permutation.
    for &d in batch_dims {
        if d >= perm.len() || perm[d] != d {
            return false;
        }
    }

    // Must have exactly 1 contracting dimension.
    if contracting_dims.len() != 1 {
        return false;
    }

    true
}

/// Apply the transpose permutation to DotGeneral dimension numbers.
///
/// The key insight: if the DotGeneral previously read dim `d` from the
/// transposed tensor, it now needs to read dim `perm[d]` from the original
/// tensor (because Transpose maps original dim `perm[d]` to output dim `d`).
///
/// Wait — Transpose { perm } means output[i0, i1, ...] = input[i_{perm[0]}, i_{perm[1]}, ...].
/// So output dimension `d` corresponds to input dimension `perm[d]`.
/// The DotGeneral's dim numbers refer to the output of the Transpose.
/// To fold, we map each DotGeneral dim `d` to the input dim `perm[d]`.
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
            .map(|&d| perm[d])
            .collect();
        new_config.lhs_batch_dims = config.lhs_batch_dims.iter().map(|&d| perm[d]).collect();
    } else {
        new_config.rhs_contracting_dims = config
            .rhs_contracting_dims
            .iter()
            .map(|&d| perm[d])
            .collect();
        new_config.rhs_batch_dims = config.rhs_batch_dims.iter().map(|&d| perm[d]).collect();
    }
    new_config
}
