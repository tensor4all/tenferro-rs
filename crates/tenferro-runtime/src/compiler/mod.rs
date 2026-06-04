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
            let input_shapes_refs: Vec<&[DimExpr]> = instr
                .inputs
                .iter()
                .map(|&slot| {
                    slot_shapes[slot].as_deref().unwrap_or_else(|| {
                        panic!("compile_std_to_exec: missing shape for slot {slot}")
                    })
                })
                .collect();
            let input_extents_refs: Vec<&[ShapeExtent<DimExpr>]> = instr
                .inputs
                .iter()
                .map(|&slot| {
                    slot_extents[slot].as_deref().unwrap_or_else(|| {
                        panic!("compile_std_to_exec: missing extents for slot {slot}")
                    })
                })
                .collect();

            let (output_dtypes, output_shapes, output_extents) =
                if let StdTensorOp::Extension(ext) = &instr.operation {
                    let metas = infer_extension_output_meta(
                        ext.as_ref(),
                        &input_dtypes,
                        &input_shapes_refs,
                    );
                    assert_eq!(
                        metas.len(),
                        instr.outputs.len(),
                        "compile_std_to_exec: extension family_id={:?} \
                         inferred {} output metas for {} output slots",
                        ext.family_id(),
                        metas.len(),
                        instr.outputs.len()
                    );
                    let dtypes: Vec<DType> = metas.iter().map(|(dtype, _)| *dtype).collect();
                    let shapes: Vec<Vec<DimExpr>> =
                        metas.into_iter().map(|(_dtype, shape)| shape).collect();
                    let extents = exact_extents_from_shapes(&shapes);
                    (dtypes, shapes, extents)
                } else {
                    let dtype = infer_output_dtype(&instr.operation, &input_dtypes);
                    let shapes = infer_output_shapes(&instr.operation, &input_shapes_refs);
                    let extents = infer_output_extents(&instr.operation, &input_shapes_refs);
                    assert_eq!(
                        shapes.len(),
                        instr.outputs.len(),
                        "compile_std_to_exec: {:?} inferred {} output shapes for {} output slots",
                        instr.operation,
                        shapes.len(),
                        instr.outputs.len()
                    );
                    assert_eq!(
                        extents.len(),
                        instr.outputs.len(),
                        "compile_std_to_exec: {:?} inferred {} output extents for {} output slots",
                        instr.operation,
                        extents.len(),
                        instr.outputs.len()
                    );
                    let resolved_extents =
                        resolve_output_extents(extents, &input_shapes_refs, &input_extents_refs);
                    (vec![dtype; instr.outputs.len()], shapes, resolved_extents)
                };

            let instruction_dtype = output_dtypes
                .first()
                .copied()
                .unwrap_or_else(|| panic!("compile_std_to_exec: instruction has no outputs"));
            for (((slot, dtype), shape), extents) in instr
                .outputs
                .iter()
                .zip(output_dtypes.iter())
                .zip(output_shapes.iter())
                .zip(output_extents.iter())
            {
                slot_dtypes[*slot] = Some(*dtype);
                slot_shapes[*slot] = Some(shape.clone());
                slot_extents[*slot] = Some(extents.clone());
            }

            ExecInstruction {
                op: ExecOp::from_std_tensor_op(&instr.operation),
                input_slots: instr.inputs.clone(),
                output_slots: instr.outputs.clone(),
                dtype: instruction_dtype,
                output_shapes: output_shapes.into(),
                output_extents: output_extents.into(),
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
    conj_sinking(&mut program, input_dtypes, input_shapes);
    dot_dimension_sorter(&mut program);
    transpose_folding(&mut program);
    dot_conj_folding(&mut program);
    eliminate_dead_code(&mut program);
    populate_last_use(&mut program);
    program
}

pub(super) fn exact_extents_from_shape(shape: &[DimExpr]) -> Vec<ShapeExtent<DimExpr>> {
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
    input_shapes: &[&[DimExpr]],
    input_extents: &[&[ShapeExtent<DimExpr>]],
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
    input_shapes: &[&[DimExpr]],
    input_extents: &[&[ShapeExtent<DimExpr>]],
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

fn resolve_dim_expr(expr: &DimExpr, input_shapes: &[&[DimExpr]]) -> DimExpr {
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

fn dim_expr_extent_kind(expr: &DimExpr, input_extents: &[&[ShapeExtent<DimExpr>]]) -> ExtentKind {
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

#[derive(Clone)]
struct SlotMeta {
    dtype: DType,
    shape: Vec<DimExpr>,
    extents: Vec<ShapeExtent<DimExpr>>,
}

#[derive(Clone)]
struct ProducerInfo {
    op: ExecOp,
    input_slots: Vec<usize>,
    dtype: DType,
}

type ProducerMap = Vec<Option<ProducerInfo>>;

// ============================================================================
// Pass 0: ConjSinking
// ============================================================================
//
// Move `Conj` through shape-preserving standard ops so layout passes can still
// see through Transpose/Reshape and `dot_conj_folding` can finally fold input
// conjugation into the GEMM backend call.

/// Sink `Conj` through commuting standard ops.
///
/// `input_dtypes` and `input_shapes` are the program-input metadata matching
/// `program.input_slots`. They are required because `ExecProgram` stores
/// metadata on instruction outputs, but not on program inputs.
pub fn conj_sinking(
    program: &mut ExecProgram,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
) {
    assert_eq!(
        program.input_slots.len(),
        input_dtypes.len(),
        "conj_sinking: input dtype count must match input slot count"
    );
    assert_eq!(
        program.input_slots.len(),
        input_shapes.len(),
        "conj_sinking: input shape count must match input slot count"
    );

    loop {
        if !conj_sinking_one_pass(program, input_dtypes, input_shapes) {
            break;
        }
    }
}

fn conj_sinking_one_pass(
    program: &mut ExecProgram,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
) -> bool {
    let mut slot_meta = collect_slot_meta(program, input_dtypes, input_shapes);
    let mut redirect: Vec<usize> = (0..program.n_slots).collect();
    let mut producer_by_slot: ProducerMap = vec![None; program.n_slots];
    let mut conj_cache: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut new_instructions = Vec::with_capacity(program.instructions.len());
    let mut changed = false;

    let instructions = std::mem::take(&mut program.instructions);
    for mut instr in instructions {
        for slot in &mut instr.input_slots {
            *slot = resolve_slot_redirect(*slot, &redirect);
        }

        if matches!(instr.op, ExecOp::Conj) && instr.input_slots.len() == 1 {
            let input_slot = instr.input_slots[0];
            if let Some(producer) = producer_for_slot(&producer_by_slot, input_slot).cloned() {
                if matches!(producer.op, ExecOp::Conj) && producer.input_slots.len() == 1 {
                    let replacement = resolve_slot_redirect(producer.input_slots[0], &redirect);
                    for &slot in &instr.output_slots {
                        redirect[slot] = replacement;
                    }
                    changed = true;
                    continue;
                }

                if let Some(commuting_inputs) =
                    conj_commuting_input_indices(&producer, producer.input_slots.len(), &slot_meta)
                {
                    let mut input_slots = producer.input_slots;
                    for input_idx in commuting_inputs {
                        let input_slot = input_slots[input_idx];
                        input_slots[input_idx] = ConjSinkingState {
                            slot_meta: &mut slot_meta,
                            producer_by_slot: &mut producer_by_slot,
                            conj_cache: &mut conj_cache,
                            new_instructions: &mut new_instructions,
                            n_slots: &mut program.n_slots,
                            redirect: &mut redirect,
                        }
                        .ensure_conj_slot(input_slot);
                    }
                    instr.op = producer.op;
                    instr.input_slots = input_slots;
                    changed = true;
                }
            }
        }

        record_producer(&mut producer_by_slot, &instr);
        new_instructions.push(instr);
    }

    for slot in &mut program.output_slots {
        *slot = resolve_slot_redirect(*slot, &redirect);
    }
    program.instructions = new_instructions;
    changed
}

fn collect_slot_meta(
    program: &ExecProgram,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
) -> Vec<Option<SlotMeta>> {
    let mut slot_meta = vec![None; program.n_slots];
    for (index, &slot) in program.input_slots.iter().enumerate() {
        slot_meta[slot] = Some(SlotMeta {
            dtype: input_dtypes[index],
            shape: input_shapes[index].clone(),
            extents: exact_extents_from_shape(&input_shapes[index]),
        });
    }
    for instr in &program.instructions {
        let output_dtypes = instruction_output_dtypes(instr, &slot_meta);
        for (((&slot, dtype), shape), extents) in instr
            .output_slots
            .iter()
            .zip(output_dtypes.iter())
            .zip(instr.output_shapes.iter())
            .zip(instr.output_extents.iter())
        {
            slot_meta[slot] = Some(SlotMeta {
                dtype: *dtype,
                shape: shape.clone(),
                extents: extents.clone(),
            });
        }
    }
    slot_meta
}

fn instruction_output_dtypes(
    instr: &ExecInstruction,
    slot_meta: &[Option<SlotMeta>],
) -> Vec<DType> {
    let ExecOp::Extension(ext) = &instr.op else {
        return vec![instr.dtype; instr.output_slots.len()];
    };
    let input_dtypes: Vec<DType> = instr
        .input_slots
        .iter()
        .map(|&slot| {
            slot_meta[slot]
                .as_ref()
                .unwrap_or_else(|| panic!("collect_slot_meta: missing dtype for slot {slot}"))
                .dtype
        })
        .collect();
    let input_shapes: Vec<Vec<DimExpr>> = instr
        .input_slots
        .iter()
        .map(|&slot| {
            slot_meta[slot]
                .as_ref()
                .unwrap_or_else(|| panic!("collect_slot_meta: missing shape for slot {slot}"))
                .shape
                .clone()
        })
        .collect();
    let input_shape_refs: Vec<&[DimExpr]> = input_shapes.iter().map(Vec::as_slice).collect();
    let metas = infer_extension_output_meta(ext.as_ref(), &input_dtypes, &input_shape_refs);
    assert_eq!(
        metas.len(),
        instr.output_slots.len(),
        "collect_slot_meta: extension family_id={:?} inferred {} output metas for {} output slots",
        ext.family_id(),
        metas.len(),
        instr.output_slots.len()
    );
    metas.into_iter().map(|(dtype, _shape)| dtype).collect()
}

struct ConjSinkingState<'a> {
    slot_meta: &'a mut Vec<Option<SlotMeta>>,
    producer_by_slot: &'a mut ProducerMap,
    conj_cache: &'a mut std::collections::HashMap<usize, usize>,
    new_instructions: &'a mut Vec<ExecInstruction>,
    n_slots: &'a mut usize,
    redirect: &'a mut Vec<usize>,
}

impl ConjSinkingState<'_> {
    fn ensure_conj_slot(&mut self, slot: usize) -> usize {
        let slot = resolve_slot_redirect(slot, self.redirect);
        if let Some(producer) = producer_for_slot(self.producer_by_slot, slot) {
            if matches!(producer.op, ExecOp::Conj) && producer.input_slots.len() == 1 {
                return resolve_slot_redirect(producer.input_slots[0], self.redirect);
            }
        }
        if let Some(&conj_slot) = self.conj_cache.get(&slot) {
            return conj_slot;
        }

        let meta = self.slot_meta[slot]
            .clone()
            .unwrap_or_else(|| panic!("conj_sinking: missing metadata for slot {slot}"));
        let output_slot = *self.n_slots;
        *self.n_slots += 1;
        self.redirect.push(output_slot);
        self.slot_meta.push(Some(meta.clone()));

        let instr = ExecInstruction {
            op: ExecOp::Conj,
            input_slots: vec![slot],
            output_slots: vec![output_slot],
            dtype: meta.dtype,
            output_shapes: vec![meta.shape].into(),
            output_extents: vec![meta.extents].into(),
            last_use: Vec::new(),
        };
        record_producer(self.producer_by_slot, &instr);
        self.new_instructions.push(instr);
        self.conj_cache.insert(slot, output_slot);
        output_slot
    }
}

fn record_producer(producer_by_slot: &mut ProducerMap, instr: &ExecInstruction) {
    let producer = ProducerInfo {
        op: instr.op.clone(),
        input_slots: instr.input_slots.clone(),
        dtype: instr.dtype,
    };
    for &slot in &instr.output_slots {
        if slot >= producer_by_slot.len() {
            producer_by_slot.resize_with(slot + 1, || None);
        }
        producer_by_slot[slot] = Some(producer.clone());
    }
}

fn producer_for_slot(producer_by_slot: &ProducerMap, slot: usize) -> Option<&ProducerInfo> {
    producer_by_slot.get(slot).and_then(Option::as_ref)
}

fn resolve_slot_redirect(mut slot: usize, redirect: &[usize]) -> usize {
    while slot < redirect.len() && redirect[slot] != slot {
        slot = redirect[slot];
    }
    slot
}

fn conj_commuting_input_indices(
    producer: &ProducerInfo,
    input_len: usize,
    slot_meta: &[Option<SlotMeta>],
) -> Option<Vec<usize>> {
    match &producer.op {
        ExecOp::Add | ExecOp::Multiply | ExecOp::Divide | ExecOp::DotGeneral(_) => {
            Some((0..input_len).collect())
        }
        ExecOp::DotGeneralWithConj { .. } => Some((0..input_len).collect()),
        ExecOp::Convert { .. } => {
            let input_dtype = producer
                .input_slots
                .first()
                .and_then(|&slot| slot_meta.get(slot))
                .and_then(Option::as_ref)
                .map(|meta| meta.dtype)?;
            (dtype_supports_conj(input_dtype) && dtype_supports_conj(producer.dtype))
                .then_some(vec![0])
        }
        ExecOp::Negate
        | ExecOp::ReduceSum { .. }
        | ExecOp::ReduceProd { .. }
        | ExecOp::Transpose { .. }
        | ExecOp::Reshape { .. }
        | ExecOp::BroadcastInDim { .. }
        | ExecOp::Slice(_)
        | ExecOp::DynamicSlice { .. }
        | ExecOp::Pad(_)
        | ExecOp::Reverse { .. }
        | ExecOp::ExtractDiag { .. }
        | ExecOp::EmbedDiag { .. }
        | ExecOp::Tril { .. }
        | ExecOp::Triu { .. }
        | ExecOp::Gather(_) => Some(vec![0]),
        ExecOp::Concatenate { .. } => Some((0..input_len).collect()),
        _ => None,
    }
}

fn dtype_supports_conj(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64)
}

pub(crate) fn populate_last_use(program: &mut ExecProgram) {
    let mut output_slot = vec![false; program.n_slots];
    for &slot in &program.output_slots {
        output_slot[slot] = true;
    }

    let mut last_user: Vec<Option<usize>> = vec![None; program.n_slots];
    for (idx, instr) in program.instructions.iter().enumerate() {
        for &slot in &instr.input_slots {
            last_user[slot] = Some(idx);
        }
    }

    for (idx, instr) in program.instructions.iter_mut().enumerate() {
        instr.last_use = instr
            .input_slots
            .iter()
            .map(|&slot| !output_slot[slot] && last_user[slot] == Some(idx))
            .collect();
    }
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
        match &mut instr.op {
            ExecOp::DotGeneral(config) | ExecOp::DotGeneralWithConj { config, .. } => {
                sort_contracting_dims(config);
            }
            _ => {}
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
        let producer_by_slot = producer_indices_by_slot(program);
        let changed = transpose_fold_one_pass(program, &producer_by_slot);
        if !changed {
            break;
        }
    }
}

fn transpose_fold_one_pass(program: &mut ExecProgram, producer_by_slot: &[Option<usize>]) -> bool {
    let mut changed = false;

    for index in 0..program.instructions.len() {
        if !matches!(
            program.instructions[index].op,
            ExecOp::DotGeneral(_) | ExecOp::DotGeneralWithConj { .. }
        ) {
            continue;
        }

        if try_fold_operand(program, producer_by_slot, index, 0) {
            changed = true;
        }
        if program.instructions[index].input_slots.len() > 1
            && try_fold_operand(program, producer_by_slot, index, 1)
        {
            changed = true;
        }
    }

    changed
}

fn try_fold_operand(
    program: &mut ExecProgram,
    producer_by_slot: &[Option<usize>],
    dot_idx: usize,
    operand_idx: usize,
) -> bool {
    let input_slot = program.instructions[dot_idx].input_slots[operand_idx];
    let Some(producer_idx) = producer_by_slot.get(input_slot).copied().flatten() else {
        return false;
    };

    let perm = match &program.instructions[producer_idx].op {
        ExecOp::Transpose { perm } => perm.clone(),
        _ => return false,
    };

    let (config, existing_conj) = match &program.instructions[dot_idx].op {
        ExecOp::DotGeneral(config) => (config.clone(), None),
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => (config.clone(), Some((*lhs_conj, *rhs_conj))),
        _ => return false,
    };

    if !is_transpose_foldable(&config, operand_idx, &perm) {
        return false;
    }

    let new_config = fold_transpose_into_dot(&config, operand_idx, &perm);
    let original_input = program.instructions[producer_idx].input_slots[0];
    program.instructions[dot_idx].op = match existing_conj {
        Some((lhs_conj, rhs_conj)) => ExecOp::DotGeneralWithConj {
            config: new_config,
            lhs_conj,
            rhs_conj,
        },
        None => ExecOp::DotGeneral(new_config),
    };
    program.instructions[dot_idx].input_slots[operand_idx] = original_input;
    true
}

fn producer_indices_by_slot(program: &ExecProgram) -> Vec<Option<usize>> {
    let mut producer_by_slot = vec![None; program.n_slots];
    for (idx, instr) in program.instructions.iter().enumerate() {
        for &slot in &instr.output_slots {
            if slot >= producer_by_slot.len() {
                producer_by_slot.resize(slot + 1, None);
            }
            producer_by_slot[slot] = Some(idx);
        }
    }
    producer_by_slot
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

mod dot_decomposer;

pub use dot_decomposer::dot_decomposer;

// ============================================================================
// Pass 4: DotConjFolding
// ============================================================================
//
// After layout canonicalization, fold `Conj` producers on DotGeneral operands
// into backend-visible conjugation flags. This also handles layout-only chains
// such as `Reshape(Transpose(Conj(x)))` by rewiring the layout chain to consume
// `x` and marking the DotGeneral operand as conjugated.

/// Fold `Conj` inputs into `DotGeneralWithConj`.
pub fn dot_conj_folding(program: &mut ExecProgram) {
    let producer_by_slot = producer_index_by_slot(program);
    let use_counts = slot_use_counts(program);
    let mut layout_rewrites: Vec<(usize, usize)> = Vec::new();
    let mut shape_input_rewrites: Vec<(usize, usize)> = Vec::new();
    let mut dot_updates: Vec<DotConjUpdate> = Vec::new();

    for (dot_idx, instr) in program.instructions.iter().enumerate() {
        if instr.input_slots.len() < 2 {
            continue;
        }

        let (config, mut lhs_conj, mut rhs_conj) = match instr.op.clone() {
            ExecOp::DotGeneral(config) => (config, false, false),
            ExecOp::DotGeneralWithConj {
                config,
                lhs_conj,
                rhs_conj,
            } => (config, lhs_conj, rhs_conj),
            _ => continue,
        };

        let mut input_replacements = [None, None];
        if let Some(fold) = find_dot_operand_conj_fold(
            program,
            instr.input_slots[0],
            &producer_by_slot,
            &use_counts,
        ) {
            lhs_conj = !lhs_conj;
            input_replacements[0] = fold.dot_input_replacement;
            if let Some(rewrite) = fold.layout_input_rewrite {
                layout_rewrites.push(rewrite);
            }
            shape_input_rewrites.push(fold.shape_input_rewrite);
        }
        if let Some(fold) = find_dot_operand_conj_fold(
            program,
            instr.input_slots[1],
            &producer_by_slot,
            &use_counts,
        ) {
            rhs_conj = !rhs_conj;
            input_replacements[1] = fold.dot_input_replacement;
            if let Some(rewrite) = fold.layout_input_rewrite {
                layout_rewrites.push(rewrite);
            }
            shape_input_rewrites.push(fold.shape_input_rewrite);
        }

        dot_updates.push(DotConjUpdate {
            dot_idx,
            config,
            lhs_conj,
            rhs_conj,
            input_replacements,
        });
    }

    for (layout_idx, new_input_slot) in layout_rewrites {
        program.instructions[layout_idx].input_slots[0] = new_input_slot;
    }
    for instr in &mut program.instructions {
        if !matches!(instr.op, ExecOp::Reshape { .. }) || instr.input_slots.len() <= 1 {
            continue;
        }
        for slot in &mut instr.input_slots[1..] {
            for &(from, to) in &shape_input_rewrites {
                if *slot == from {
                    *slot = to;
                }
            }
        }
    }

    for update in dot_updates {
        let instr = &mut program.instructions[update.dot_idx];
        for (operand_idx, replacement) in update.input_replacements.into_iter().enumerate() {
            if let Some(slot) = replacement {
                instr.input_slots[operand_idx] = slot;
            }
        }
        instr.op = if update.lhs_conj || update.rhs_conj {
            ExecOp::DotGeneralWithConj {
                config: update.config,
                lhs_conj: update.lhs_conj,
                rhs_conj: update.rhs_conj,
            }
        } else {
            ExecOp::DotGeneral(update.config)
        };
    }
}

struct DotConjUpdate {
    dot_idx: usize,
    config: DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
    input_replacements: [Option<usize>; 2],
}

struct DotOperandConjFold {
    dot_input_replacement: Option<usize>,
    layout_input_rewrite: Option<(usize, usize)>,
    shape_input_rewrite: (usize, usize),
}

fn producer_index_by_slot(program: &ExecProgram) -> Vec<Option<usize>> {
    let mut producer_by_slot = vec![None; program.n_slots];
    for (idx, instr) in program.instructions.iter().enumerate() {
        for &slot in &instr.output_slots {
            if slot >= producer_by_slot.len() {
                producer_by_slot.resize(slot + 1, None);
            }
            producer_by_slot[slot] = Some(idx);
        }
    }
    producer_by_slot
}

fn slot_use_counts(program: &ExecProgram) -> Vec<usize> {
    let mut counts = vec![0usize; program.n_slots];
    for instr in &program.instructions {
        for &slot in &instr.input_slots {
            if slot >= counts.len() {
                counts.resize(slot + 1, 0);
            }
            counts[slot] += 1;
        }
    }
    for &slot in &program.output_slots {
        if slot >= counts.len() {
            counts.resize(slot + 1, 0);
        }
        counts[slot] += 1;
    }
    counts
}

fn find_dot_operand_conj_fold(
    program: &ExecProgram,
    mut slot: usize,
    producer_by_slot: &[Option<usize>],
    use_counts: &[usize],
) -> Option<DotOperandConjFold> {
    let mut layout_chain = Vec::new();
    let mut seen = std::collections::HashSet::new();
    while seen.insert(slot) {
        let producer_idx = producer_by_slot.get(slot).copied().flatten()?;
        let producer = &program.instructions[producer_idx];

        if matches!(producer.op, ExecOp::Conj) && producer.input_slots.len() == 1 {
            let source_slot = producer.input_slots[0];
            return Some(if let Some(&layout_idx) = layout_chain.last() {
                DotOperandConjFold {
                    dot_input_replacement: None,
                    layout_input_rewrite: Some((layout_idx, source_slot)),
                    shape_input_rewrite: (slot, source_slot),
                }
            } else {
                DotOperandConjFold {
                    dot_input_replacement: Some(source_slot),
                    layout_input_rewrite: None,
                    shape_input_rewrite: (slot, source_slot),
                }
            });
        }

        if !is_conj_transparent_layout_op(&producer.op)
            || producer.input_slots.len() != 1
            || producer.output_slots.len() != 1
            || use_counts.get(slot).copied().unwrap_or(0) != 1
        {
            return None;
        }
        layout_chain.push(producer_idx);
        slot = producer.input_slots[0];
    }
    None
}

fn is_conj_transparent_layout_op(op: &ExecOp) -> bool {
    matches!(
        op,
        ExecOp::Transpose { .. }
            | ExecOp::Reshape { .. }
            | ExecOp::BroadcastInDim { .. }
            | ExecOp::Slice(_)
            | ExecOp::DynamicSlice { .. }
            | ExecOp::Pad(_)
            | ExecOp::Reverse { .. }
            | ExecOp::ExtractDiag { .. }
            | ExecOp::EmbedDiag { .. }
            | ExecOp::Tril { .. }
            | ExecOp::Triu { .. }
            | ExecOp::Gather(_)
    )
}

// ============================================================================
// Pass 5: DeadCodeElimination
// ============================================================================
//
// Remove instructions whose outputs are never consumed by a program output
// or by a later instruction. `transpose_folding` + `dot_decomposer` can leave
// dead `Transpose` instructions (the folded-out producer is bypassed but
// remains in the program), and DCE reclaims that wasted runtime work.

/// Drop instructions with no downstream consumer.
pub fn eliminate_dead_code(program: &mut ExecProgram) {
    let mut live_slots = vec![false; program.n_slots];
    for &slot in &program.output_slots {
        if slot >= live_slots.len() {
            live_slots.resize(slot + 1, false);
        }
        live_slots[slot] = true;
    }
    let mut keep = vec![false; program.instructions.len()];
    for idx in (0..program.instructions.len()).rev() {
        let instr = &program.instructions[idx];
        let has_live_output = instr
            .output_slots
            .iter()
            .any(|&slot| live_slots.get(slot).copied().unwrap_or(false));
        if has_live_output {
            keep[idx] = true;
            for &slot in &instr.input_slots {
                if slot >= live_slots.len() {
                    live_slots.resize(slot + 1, false);
                }
                live_slots[slot] = true;
            }
        }
    }
    let instructions = std::mem::take(&mut program.instructions);
    program.instructions = instructions
        .into_iter()
        .enumerate()
        .filter_map(|(i, instr)| keep[i].then_some(instr))
        .collect();
}
