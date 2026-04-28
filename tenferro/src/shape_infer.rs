//! Output-shape and output-dtype inference for `StdTensorOp`.
//!
//! Called during `StdTensorOp -> ExecProgram` lowering to populate
//! `ExecInstruction::output_shapes` and `ExecInstruction::dtype`.

use std::collections::HashMap;

use tenferro_einsum::Subscripts;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_tensor::{DType, DotGeneralConfig, GatherConfig, PadConfig, SliceConfig};

/// Infer output dtype for a single instruction given its op and input dtypes.
///
/// Panics if the input dtypes are inconsistent for the op (shouldn't happen
/// in well-formed SSA programs). For `StdTensorOp::Extension`, prefer the
/// combined [`infer_extension_output_meta`] helper — this function only
/// returns the first output's dtype, which is sufficient for single-output
/// extensions but loses information for multi-output ones.
pub fn infer_output_dtype(op: &StdTensorOp, input_dtypes: &[DType]) -> DType {
    match op {
        StdTensorOp::Constant { dtype, .. } => *dtype,
        StdTensorOp::Convert { to, .. } => *to,
        StdTensorOp::Eig { input_dtype, .. } => match input_dtype {
            DType::F32 | DType::C32 => DType::C32,
            DType::F64 | DType::C64 => DType::C64,
            DType::I64 => DType::I64,
        },
        StdTensorOp::Extension(ext) => extension_first_output_dtype(ext.as_ref(), input_dtypes),
        StdTensorOp::Add
        | StdTensorOp::Mul
        | StdTensorOp::Neg
        | StdTensorOp::Conj
        | StdTensorOp::Div
        | StdTensorOp::Abs
        | StdTensorOp::Sign
        | StdTensorOp::Maximum
        | StdTensorOp::Minimum
        | StdTensorOp::Select
        | StdTensorOp::Clamp
        | StdTensorOp::Exp
        | StdTensorOp::Log
        | StdTensorOp::Sin
        | StdTensorOp::Cos
        | StdTensorOp::Tanh
        | StdTensorOp::Sqrt
        | StdTensorOp::Rsqrt
        | StdTensorOp::Pow
        | StdTensorOp::Expm1
        | StdTensorOp::Log1p
        | StdTensorOp::Transpose { .. }
        | StdTensorOp::Reshape { .. }
        | StdTensorOp::BroadcastInDim { .. }
        | StdTensorOp::ReduceSum { .. }
        | StdTensorOp::ReduceProd { .. }
        | StdTensorOp::ReduceMax { .. }
        | StdTensorOp::ReduceMin { .. }
        | StdTensorOp::ExtractDiag { .. }
        | StdTensorOp::EmbedDiag { .. }
        | StdTensorOp::Tril { .. }
        | StdTensorOp::Triu { .. }
        | StdTensorOp::Gather(_)
        | StdTensorOp::Scatter(_)
        | StdTensorOp::Slice(_)
        | StdTensorOp::DynamicSlice { .. }
        | StdTensorOp::Pad(_)
        | StdTensorOp::Concatenate { .. }
        | StdTensorOp::Reverse { .. }
        | StdTensorOp::DotGeneral { .. }
        | StdTensorOp::NaryEinsum { .. }
        | StdTensorOp::Cholesky { .. }
        | StdTensorOp::Lu { .. }
        | StdTensorOp::Svd { .. }
        | StdTensorOp::Qr { .. }
        | StdTensorOp::Eigh { .. }
        | StdTensorOp::TriangularSolve { .. }
        | StdTensorOp::ValidateNonsingular { .. }
        | StdTensorOp::DynamicTruncate { .. }
        | StdTensorOp::PadToMatch { .. } => input_dtypes[0],
        StdTensorOp::Compare(_) => input_dtypes[0],
        StdTensorOp::ShapeOf { .. } => DType::F64,
    }
}

/// Infer output shapes for a single instruction.
///
/// Returns a vector of shapes (one per output slot). For single-output ops,
/// the vector has length 1. For multi-output linalg ops the vector has one
/// entry per output.
pub fn infer_output_shapes(op: &StdTensorOp, input_shapes: &[&[DimExpr]]) -> Vec<Vec<DimExpr>> {
    match op {
        StdTensorOp::Add => vec![same_or_scalar_broadcast_shape(
            require_input(op, input_shapes, 0),
            require_input(op, input_shapes, 1),
        )],
        StdTensorOp::Mul => vec![same_or_scalar_broadcast_shape(
            require_input(op, input_shapes, 0),
            require_input(op, input_shapes, 1),
        )],
        StdTensorOp::Neg
        | StdTensorOp::Conj
        | StdTensorOp::Div
        | StdTensorOp::Abs
        | StdTensorOp::Sign
        | StdTensorOp::Maximum
        | StdTensorOp::Minimum
        | StdTensorOp::Compare(_)
        | StdTensorOp::Select
        | StdTensorOp::Clamp
        | StdTensorOp::Exp
        | StdTensorOp::Log
        | StdTensorOp::Sin
        | StdTensorOp::Cos
        | StdTensorOp::Tanh
        | StdTensorOp::Sqrt
        | StdTensorOp::Rsqrt
        | StdTensorOp::Pow
        | StdTensorOp::Expm1
        | StdTensorOp::Log1p
        | StdTensorOp::Convert { .. }
        | StdTensorOp::Tril { .. }
        | StdTensorOp::Triu { .. }
        | StdTensorOp::Reverse { .. }
        | StdTensorOp::Scatter(_)
        | StdTensorOp::Cholesky { .. }
        | StdTensorOp::ValidateNonsingular { .. } => {
            vec![require_input(op, input_shapes, 0).to_vec()]
        }
        StdTensorOp::Transpose { perm } => {
            vec![permute_shape(require_input(op, input_shapes, 0), perm)]
        }
        StdTensorOp::Reshape { to_shape, .. } => vec![to_shape.clone()],
        StdTensorOp::BroadcastInDim { shape, .. } => vec![shape.clone()],
        StdTensorOp::Constant { .. } => vec![Vec::new()],
        StdTensorOp::ReduceSum { axes, .. }
        | StdTensorOp::ReduceProd { axes, .. }
        | StdTensorOp::ReduceMax { axes, .. }
        | StdTensorOp::ReduceMin { axes, .. } => {
            vec![reduced_shape(require_input(op, input_shapes, 0), axes)]
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            vec![extract_diag_shape(
                require_input(op, input_shapes, 0),
                *axis_a,
                *axis_b,
            )]
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            vec![embed_diag_shape(
                require_input(op, input_shapes, 0),
                *axis_a,
                *axis_b,
            )]
        }
        StdTensorOp::Gather(config) => vec![gather_shape(
            require_input(op, input_shapes, 0),
            require_input(op, input_shapes, 1),
            config,
        )],
        StdTensorOp::Slice(config) => vec![slice_shape(require_input(op, input_shapes, 0), config)],
        StdTensorOp::DynamicSlice { slice_sizes } => {
            vec![slice_sizes.iter().copied().map(DimExpr::Const).collect()]
        }
        StdTensorOp::Pad(config) => vec![pad_shape(require_input(op, input_shapes, 0), config)],
        StdTensorOp::DotGeneral { config, .. } => vec![dot_general_shape(
            require_input(op, input_shapes, 0),
            require_input(op, input_shapes, 1),
            config,
        )],
        StdTensorOp::NaryEinsum { subscripts } => {
            vec![einsum_output_shape(subscripts, input_shapes)]
        }
        StdTensorOp::Concatenate { axis, .. } => vec![concatenate_shape(input_shapes, *axis)],
        StdTensorOp::ShapeOf { .. } => vec![Vec::new()],
        StdTensorOp::DynamicTruncate { axis } => {
            let shape = require_input(op, input_shapes, 0).to_vec();
            assert!(
                *axis < shape.len(),
                "DynamicTruncate axis {axis} out of bounds for rank {}",
                shape.len()
            );
            vec![shape]
        }
        StdTensorOp::PadToMatch { axis } => vec![pad_to_match_shape(
            require_input(op, input_shapes, 0),
            require_input(op, input_shapes, 1),
            *axis,
        )],
        StdTensorOp::Lu { .. } => lu_shapes(require_input(op, input_shapes, 0)),
        StdTensorOp::Svd { .. } => svd_shapes(require_input(op, input_shapes, 0)),
        StdTensorOp::Qr { .. } => qr_shapes(require_input(op, input_shapes, 0)),
        StdTensorOp::Eigh { .. } | StdTensorOp::Eig { .. } => {
            eig_like_shapes(require_input(op, input_shapes, 0))
        }
        StdTensorOp::TriangularSolve { .. } => vec![require_input(op, input_shapes, 1).to_vec()],
        StdTensorOp::Extension(ext) => {
            let metas = infer_extension_output_meta(ext.as_ref(), &[], input_shapes);
            metas.into_iter().map(|(_dtype, shape)| shape).collect()
        }
    }
}

/// Compute the full `(dtype, shape)` meta for every output of a
/// `StdTensorOp::Extension(op)` instruction.
///
/// This is the single source of truth for multi-output extensions —
/// [`infer_output_dtype`] only returns the first slot's dtype. Prefer this
/// when populating `ExecInstruction` fields.
///
/// `input_dtypes` must have length `op.n_inputs()` and be consistent with
/// `input_shapes`. `input_shapes` uses [`DimExpr`] expressions so shape
/// inference can flow through composed symbolic graphs.
pub fn infer_extension_output_meta(
    op: &dyn ExtensionOp,
    input_dtypes: &[DType],
    input_shapes: &[&[DimExpr]],
) -> Vec<(DType, Vec<DimExpr>)> {
    // Build per-input SymDim representations using the input index as the
    // synthetic tensor id. This preserves DimExpr::InputDim ↔ SymDim::TensorAxis
    // round-trips; the tensor_id namespace is local to this call.
    let symdim_storage: Vec<Vec<SymDim>> = input_shapes
        .iter()
        .enumerate()
        .map(|(input_idx, shape)| {
            shape
                .iter()
                .enumerate()
                .map(|(axis, dim)| dim_expr_to_sym_dim(dim, input_idx, axis))
                .collect()
        })
        .collect();
    let symdim_refs: Vec<&[SymDim]> = symdim_storage.iter().map(Vec::as_slice).collect();

    let metas = op.infer_output_meta(input_dtypes, &symdim_refs);

    let tensor_map: Vec<(u64, usize)> = (0..input_shapes.len())
        .map(|input_idx| (input_idx as u64, input_idx))
        .collect();

    metas
        .into_iter()
        .map(|(dtype, shape)| {
            let dim_exprs: Vec<DimExpr> = shape
                .iter()
                .map(|dim| {
                    dim.to_dim_expr(&tensor_map).unwrap_or_else(|err| {
                        panic!(
                            "ExtensionOp::infer_output_meta for family {:?} \
                             returned a SymDim that cannot be converted to DimExpr: {err}",
                            op.family_id(),
                        )
                    })
                })
                .collect();
            (dtype, dim_exprs)
        })
        .collect()
}

fn dim_expr_to_sym_dim(expr: &DimExpr, input_idx: usize, axis: usize) -> SymDim {
    // Prefer explicit const for concrete extents; otherwise expose as an
    // opaque TensorAxis handle. Inner expressions (Add/Mul/…) are not
    // expected in per-input shape vectors — callers pass them as raw
    // `InputDim` references — so a direct mapping is sufficient.
    match expr {
        DimExpr::Const(value) => SymDim::from(*value),
        DimExpr::InputDim {
            input_idx: idx,
            axis: ax,
        } => SymDim::tensor_axis(*idx as u64, *ax),
        _ => SymDim::tensor_axis(input_idx as u64, axis),
    }
}

fn extension_first_output_dtype(op: &dyn ExtensionOp, input_dtypes: &[DType]) -> DType {
    // For dtype-only queries we synthesise a rank-0 shape list per input so
    // the extension's `infer_output_meta` stays total even when shapes are
    // unknown to the caller. Extensions whose dtype depends on shape must
    // be handled through [`infer_extension_output_meta`], which
    // `compile_std_to_exec` prefers for the `Extension` arm.
    let empty_rows: Vec<&[SymDim]> = (0..op.n_inputs()).map(|_| [].as_slice()).collect();
    let metas = op.infer_output_meta(input_dtypes, &empty_rows);
    if metas.is_empty() {
        panic!(
            "ExtensionOp::infer_output_meta for family {:?} returned an \
             empty meta list; expected at least one output",
            op.family_id()
        );
    }
    metas[0].0
}

fn require_input<'a>(
    op: &StdTensorOp,
    input_shapes: &'a [&[DimExpr]],
    idx: usize,
) -> &'a [DimExpr] {
    input_shapes.get(idx).copied().unwrap_or_else(|| {
        panic!(
            "{op:?} expects input index {idx}, got {} input shapes",
            input_shapes.len()
        )
    })
}

fn permute_shape(input_shape: &[DimExpr], perm: &[usize]) -> Vec<DimExpr> {
    perm.iter().map(|&axis| input_shape[axis].clone()).collect()
}

fn reduced_shape(input_shape: &[DimExpr], axes: &[usize]) -> Vec<DimExpr> {
    input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| (!axes.contains(&axis)).then_some(dim.clone()))
        .collect()
}

fn same_or_scalar_broadcast_shape(lhs_shape: &[DimExpr], rhs_shape: &[DimExpr]) -> Vec<DimExpr> {
    if lhs_shape.is_empty() {
        rhs_shape.to_vec()
    } else if rhs_shape.is_empty() {
        lhs_shape.to_vec()
    } else {
        // Dynamic shape ops such as DynamicTruncate can only be inferred
        // approximately here, so for non-scalar inputs we preserve the
        // historical "follow lhs" behavior.
        lhs_shape.to_vec()
    }
}

fn extract_diag_shape(input_shape: &[DimExpr], axis_a: usize, axis_b: usize) -> Vec<DimExpr> {
    assert!(
        axis_a < input_shape.len() && axis_b < input_shape.len(),
        "ExtractDiag axes ({axis_a}, {axis_b}) out of bounds for rank {}",
        input_shape.len()
    );
    assert_ne!(axis_a, axis_b, "ExtractDiag requires distinct axes");
    let diag_output_axis = if axis_a < axis_b { axis_a } else { axis_a - 1 };
    let diag_dim = dim_min(input_shape[axis_a].clone(), input_shape[axis_b].clone());
    let mut output_shape = input_shape.to_vec();
    output_shape.remove(axis_b);
    output_shape[diag_output_axis] = diag_dim;
    output_shape
}

fn embed_diag_shape(input_shape: &[DimExpr], axis_a: usize, axis_b: usize) -> Vec<DimExpr> {
    assert!(
        axis_a < input_shape.len(),
        "EmbedDiag axis_a {axis_a} out of bounds for rank {}",
        input_shape.len()
    );
    assert!(
        axis_b <= input_shape.len(),
        "EmbedDiag axis_b {axis_b} out of bounds for rank {}",
        input_shape.len()
    );
    let mut output_shape = input_shape.to_vec();
    output_shape.insert(axis_b, input_shape[axis_a].clone());
    output_shape
}

fn dot_general_shape(
    lhs_shape: &[DimExpr],
    rhs_shape: &[DimExpr],
    config: &DotGeneralConfig,
) -> Vec<DimExpr> {
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();

    let lhs_free = (0..lhs_rank).filter(|axis| {
        !config.lhs_contracting_dims.contains(axis) && !config.lhs_batch_dims.contains(axis)
    });
    let rhs_free = (0..rhs_rank).filter(|axis| {
        !config.rhs_contracting_dims.contains(axis) && !config.rhs_batch_dims.contains(axis)
    });

    let mut output_shape = Vec::new();
    output_shape.extend(lhs_free.map(|axis| lhs_shape[axis].clone()));
    output_shape.extend(rhs_free.map(|axis| rhs_shape[axis].clone()));
    output_shape.extend(
        config
            .lhs_batch_dims
            .iter()
            .map(|&axis| lhs_shape[axis].clone()),
    );
    output_shape
}

fn gather_shape(
    operand_shape: &[DimExpr],
    index_shape: &[DimExpr],
    config: &GatherConfig,
) -> Vec<DimExpr> {
    assert_eq!(
        config.slice_sizes.len(),
        operand_shape.len(),
        "gather: slice_sizes rank mismatch"
    );

    let batch_shape = if config.index_vector_dim == index_shape.len() {
        index_shape.to_vec()
    } else {
        index_shape
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| (axis != config.index_vector_dim).then_some(dim.clone()))
            .collect()
    };

    let window_dims: Vec<usize> = (0..operand_shape.len())
        .filter(|dim| !config.collapsed_slice_dims.contains(dim))
        .collect();
    assert_eq!(
        config.offset_dims.len(),
        window_dims.len(),
        "gather: offset_dims length mismatch"
    );

    let out_rank = batch_shape.len() + config.offset_dims.len();
    let mut out_shape = vec![DimExpr::Const(0); out_rank];
    let mut out_axis_to_operand_dim = vec![None; out_rank];
    for (offset_axis, &out_axis) in config.offset_dims.iter().enumerate() {
        out_axis_to_operand_dim[out_axis] = Some(window_dims[offset_axis]);
    }

    let mut batch_axis = 0usize;
    for out_axis in 0..out_rank {
        if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
            out_shape[out_axis] = DimExpr::Const(config.slice_sizes[operand_dim]);
        } else {
            out_shape[out_axis] = batch_shape[batch_axis].clone();
            batch_axis += 1;
        }
    }

    out_shape
}

fn slice_shape(input_shape: &[DimExpr], config: &SliceConfig) -> Vec<DimExpr> {
    let rank = input_shape.len();
    assert_eq!(config.starts.len(), rank, "slice: starts rank mismatch");
    assert_eq!(config.limits.len(), rank, "slice: limits rank mismatch");
    assert_eq!(config.strides.len(), rank, "slice: strides rank mismatch");
    (0..rank)
        .map(|axis| {
            let span = config.limits[axis] - config.starts[axis];
            DimExpr::Const((span + config.strides[axis] - 1) / config.strides[axis])
        })
        .collect()
}

fn pad_shape(input_shape: &[DimExpr], config: &PadConfig) -> Vec<DimExpr> {
    let rank = input_shape.len();
    assert_eq!(
        config.edge_padding_low.len(),
        rank,
        "pad: edge_padding_low rank mismatch"
    );
    assert_eq!(
        config.edge_padding_high.len(),
        rank,
        "pad: edge_padding_high rank mismatch"
    );
    assert_eq!(
        config.interior_padding.len(),
        rank,
        "pad: interior_padding rank mismatch"
    );

    input_shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| {
            assert!(
                config.interior_padding[axis] >= 0,
                "pad: interior padding must be non-negative on axis {axis}"
            );
            if let DimExpr::Const(extent) = dim {
                let base = if *extent == 0 {
                    0
                } else {
                    (*extent as i64 - 1) * (config.interior_padding[axis] + 1) + 1
                };
                let padded = config.edge_padding_low[axis] + config.edge_padding_high[axis] + base;
                DimExpr::Const(
                    usize::try_from(padded)
                        .expect("pad: output extent must be representable as usize"),
                )
            } else if config.interior_padding[axis] == 0 {
                add_signed(
                    dim.clone(),
                    config.edge_padding_low[axis] + config.edge_padding_high[axis],
                )
            } else {
                let stride = DimExpr::Const((config.interior_padding[axis] + 1) as usize);
                let stretched = dim_add(
                    dim_mul(dim_sub(dim.clone(), DimExpr::Const(1)), stride),
                    DimExpr::Const(1),
                );
                add_signed(
                    stretched,
                    config.edge_padding_low[axis] + config.edge_padding_high[axis],
                )
            }
        })
        .collect()
}

fn add_signed(expr: DimExpr, amount: i64) -> DimExpr {
    if amount >= 0 {
        dim_add(expr, DimExpr::Const(amount as usize))
    } else {
        dim_sub(expr, DimExpr::Const((-amount) as usize))
    }
}

fn concatenate_shape(input_shapes: &[&[DimExpr]], axis: usize) -> Vec<DimExpr> {
    let first = input_shapes
        .first()
        .copied()
        .expect("concatenate expects at least one input shape");
    assert!(axis < first.len(), "concatenate axis {axis} out of bounds");
    let mut output_shape = first.to_vec();
    let axis_dim = input_shapes
        .iter()
        .skip(1)
        .fold(first[axis].clone(), |acc, shape| {
            dim_add(acc, shape[axis].clone())
        });
    output_shape[axis] = axis_dim;
    output_shape
}

fn einsum_output_shape(subscripts: &str, input_shapes: &[&[DimExpr]]) -> Vec<DimExpr> {
    let parsed = Subscripts::parse(subscripts)
        .unwrap_or_else(|err| panic!("invalid einsum subscripts {subscripts:?}: {err}"));
    assert_eq!(
        parsed.inputs.len(),
        input_shapes.len(),
        "einsum subscripts expect {} inputs, got {}",
        parsed.inputs.len(),
        input_shapes.len()
    );

    let mut label_dims: HashMap<u32, DimExpr> = HashMap::new();
    for (labels, shape) in parsed.inputs.iter().zip(input_shapes.iter()) {
        assert_eq!(
            labels.len(),
            shape.len(),
            "einsum input rank mismatch: labels={}, shape={}",
            labels.len(),
            shape.len()
        );
        for (&label, dim) in labels.iter().zip(shape.iter()) {
            if let Some(existing) = label_dims.get(&label) {
                if let (DimExpr::Const(lhs), DimExpr::Const(rhs)) = (existing, dim) {
                    assert_eq!(
                        lhs, rhs,
                        "einsum label {label} has inconsistent concrete sizes {lhs} vs {rhs}"
                    );
                }
            } else {
                label_dims.insert(label, dim.clone());
            }
        }
    }

    parsed
        .output
        .iter()
        .map(|label| {
            label_dims
                .get(label)
                .cloned()
                .unwrap_or_else(|| panic!("einsum output label {label} missing from inputs"))
        })
        .collect()
}

fn pad_to_match_shape(
    input_shape: &[DimExpr],
    reference_shape: &[DimExpr],
    axis: usize,
) -> Vec<DimExpr> {
    assert!(
        axis < input_shape.len(),
        "PadToMatch input axis {axis} out of bounds"
    );
    assert!(
        axis < reference_shape.len(),
        "PadToMatch reference axis {axis} out of bounds"
    );
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = dim_max(input_shape[axis].clone(), reference_shape[axis].clone());
    output_shape
}

fn matrix_parts(input_shape: &[DimExpr]) -> (&DimExpr, &DimExpr, &[DimExpr]) {
    assert!(
        input_shape.len() >= 2,
        "linalg op expects rank >= 2, got {}",
        input_shape.len()
    );
    (&input_shape[0], &input_shape[1], &input_shape[2..])
}

fn svd_shapes(input_shape: &[DimExpr]) -> Vec<Vec<DimExpr>> {
    let (m, n, batch) = matrix_parts(input_shape);
    let k = dim_min(m.clone(), n.clone());
    let mut u_shape = vec![m.clone(), k.clone()];
    u_shape.extend_from_slice(batch);
    let mut s_shape = vec![k.clone()];
    s_shape.extend_from_slice(batch);
    let mut vt_shape = vec![k, n.clone()];
    vt_shape.extend_from_slice(batch);
    vec![u_shape, s_shape, vt_shape]
}

fn qr_shapes(input_shape: &[DimExpr]) -> Vec<Vec<DimExpr>> {
    let (m, n, batch) = matrix_parts(input_shape);
    let k = dim_min(m.clone(), n.clone());
    let mut q_shape = vec![m.clone(), k.clone()];
    q_shape.extend_from_slice(batch);
    let mut r_shape = vec![k, n.clone()];
    r_shape.extend_from_slice(batch);
    vec![q_shape, r_shape]
}

fn lu_shapes(input_shape: &[DimExpr]) -> Vec<Vec<DimExpr>> {
    let (m, n, batch) = matrix_parts(input_shape);
    let k = dim_min(m.clone(), n.clone());
    let mut p_shape = vec![m.clone(), m.clone()];
    p_shape.extend_from_slice(batch);
    let mut l_shape = vec![m.clone(), k.clone()];
    l_shape.extend_from_slice(batch);
    let mut u_shape = vec![k, n.clone()];
    u_shape.extend_from_slice(batch);
    vec![p_shape, l_shape, u_shape, batch.to_vec()]
}

fn eig_like_shapes(input_shape: &[DimExpr]) -> Vec<Vec<DimExpr>> {
    let (n, _, batch) = matrix_parts(input_shape);
    let mut values_shape = vec![n.clone()];
    values_shape.extend_from_slice(batch);
    let mut vectors_shape = vec![n.clone(), n.clone()];
    vectors_shape.extend_from_slice(batch);
    vec![values_shape, vectors_shape]
}

fn dim_add(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => DimExpr::Const(lhs + rhs),
        (lhs, rhs) => DimExpr::add(lhs, rhs),
    }
}

fn dim_sub(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => DimExpr::Const(lhs - rhs),
        (lhs, rhs) => DimExpr::sub(lhs, rhs),
    }
}

fn dim_mul(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => DimExpr::Const(lhs * rhs),
        (lhs, rhs) => DimExpr::mul(lhs, rhs),
    }
}

fn dim_min(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => DimExpr::Const(lhs.min(rhs)),
        (lhs, rhs) => DimExpr::min(lhs, rhs),
    }
}

fn dim_max(lhs: DimExpr, rhs: DimExpr) -> DimExpr {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => DimExpr::Const(lhs.max(rhs)),
        (lhs, rhs) => DimExpr::max(lhs, rhs),
    }
}
