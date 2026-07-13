//! Output-shape and output-dtype inference for `StdTensorOp`.
//!
//! Called during `StdTensorOp -> ExecProgram` lowering to populate
//! `ExecInstruction::output_shapes` and `ExecInstruction::dtype`.

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{invoke_extension_shape_inference, ExtensionOp};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_ops::ShapeExtent;
use tenferro_tensor::{DType, DotGeneralConfig, GatherConfig, PadConfig, SliceConfig};

use crate::shape_constraint::{ConstraintSource, LocalShapeConstraint};
use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InferredExtensionMeta {
    pub(crate) output_metas: Vec<(DType, Vec<DimExpr>)>,
    pub(crate) constraints: Vec<LocalShapeConstraint>,
}

/// Promote two dtypes to the narrowest common dtype that avoids silent
/// precision loss, following the policy defined in [#811].
///
/// Rules:
/// - Bool + T -> T
/// - I32 + I32 -> I32
/// - I32 + I64 -> I64
/// - I64 + I64 → I64 (integer-preserving; Div/Pow use [`promote_dtype_div_like`])
/// - I32/I64 + F32/F64 → F64 (F32 does not have enough mantissa bits for arbitrary integers)
/// - I32/I64 + C32/C64 → C64
/// - F32 + F64 → F64
/// - F32 + C32 → C32
/// - F32/C32 + C64 → C64
/// - F64 + C32 → C64
///
/// [#811]: https://github.com/tensor4all/tenferro-rs/issues/811
pub fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    tenferro_tensor::validate::promote_dtype(lhs, rhs)
}

/// Promote an arbitrary number of dtypes by folding [`promote_dtype`].
///
/// Returns `DType::F64` for an empty iterator (the safest default).
pub fn promote_dtypes(dtypes: impl IntoIterator<Item = DType>) -> DType {
    dtypes
        .into_iter()
        .reduce(promote_dtype)
        .unwrap_or(DType::F64)
}

/// Convenience wrapper that picks the right promotion rule for a binary op.
pub fn promote_dtype_for_binary_op(op: &StdTensorOp, lhs: DType, rhs: DType) -> DType {
    match op {
        StdTensorOp::Pow => promote_dtype_pow_like(lhs, rhs),
        _ => promote_dtype(lhs, rhs),
    }
}

/// Like [`promote_dtype`], but for division-like ops where I64 / I64
/// should produce F64 to avoid integer truncation.
pub fn promote_dtype_div_like(lhs: DType, rhs: DType) -> DType {
    if matches!(lhs, DType::I32 | DType::I64) && matches!(rhs, DType::I32 | DType::I64) {
        return DType::F64;
    }
    promote_dtype(lhs, rhs)
}

/// Like [`promote_dtype`], but integer powers preserve integer dtype because
/// integer `pow` is defined as wrapping arithmetic with non-negative exponents.
pub fn promote_dtype_pow_like(lhs: DType, rhs: DType) -> DType {
    promote_dtype(lhs, rhs)
}

fn is_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::C32 | DType::C64)
}

fn ordered_dtype_op_name(op: &StdTensorOp) -> Option<&'static str> {
    // Complex values have no total order; keep magnitude ordering explicit by
    // rejecting ordered primitives before dtype promotion.
    match op {
        StdTensorOp::Compare(_) => Some("Compare"),
        StdTensorOp::Maximum => Some("Maximum"),
        StdTensorOp::Minimum => Some("Minimum"),
        StdTensorOp::Rem => Some("Rem"),
        StdTensorOp::Clamp => Some("Clamp"),
        StdTensorOp::ReduceMax { axes } if !axes.is_empty() => Some("ReduceMax"),
        StdTensorOp::ReduceMin { axes } if !axes.is_empty() => Some("ReduceMin"),
        _ => None,
    }
}

fn reject_complex_ordered_dtypes(op: &StdTensorOp, input_dtypes: &[DType]) -> Result<()> {
    let Some(op_name) = ordered_dtype_op_name(op) else {
        return Ok(());
    };
    if input_dtypes.iter().copied().any(is_complex_dtype) {
        return Err(shape_infer_error(format!(
            "{op_name} does not support complex dtypes because complex numbers have no total order"
        )));
    }
    Ok(())
}

/// Infer output dtype for a single instruction given its op and input dtypes.
///
/// For `StdTensorOp::Extension`, prefer the combined
/// `infer_extension_output_meta` helper when shape metadata is also needed.
/// This function returns only the first output's dtype.
pub fn infer_output_dtype(op: &StdTensorOp, input_dtypes: &[DType]) -> Result<DType> {
    reject_complex_ordered_dtypes(op, input_dtypes)?;

    let dtype = match op {
        StdTensorOp::Constant { dtype, .. } => *dtype,
        StdTensorOp::Convert { to, .. } => *to,
        StdTensorOp::Extension(ext) => {
            return extension_first_output_dtype(ext.as_ref(), input_dtypes)
        }
        StdTensorOp::Compare(_) => DType::Bool,
        StdTensorOp::Abs => real_dtype_for_abs(dtype_input(op, input_dtypes, 0)?),
        StdTensorOp::Select => promote_dtype(
            dtype_input(op, input_dtypes, 1)?,
            dtype_input(op, input_dtypes, 2)?,
        ),
        StdTensorOp::Clamp => promote_dtypes(input_dtypes.iter().copied()),
        // Binary / ternary / N-ary ops — promote input dtypes.
        StdTensorOp::Add
        | StdTensorOp::Sub
        | StdTensorOp::Mul
        | StdTensorOp::Div
        | StdTensorOp::Rem
        | StdTensorOp::Maximum
        | StdTensorOp::Minimum
        | StdTensorOp::DotGeneral { .. }
        | StdTensorOp::Concatenate { .. } => promote_dtype(
            dtype_input(op, input_dtypes, 0)?,
            dtype_input(op, input_dtypes, 1)?,
        ),
        StdTensorOp::Pow => promote_dtype_pow_like(
            dtype_input(op, input_dtypes, 0)?,
            dtype_input(op, input_dtypes, 1)?,
        ),
        StdTensorOp::Scatter(_) => promote_dtype(
            dtype_input(op, input_dtypes, 0)?,
            dtype_input(op, input_dtypes, 2)?,
        ),
        StdTensorOp::DynamicUpdateSlice => promote_dtype(
            dtype_input(op, input_dtypes, 0)?,
            dtype_input(op, input_dtypes, 1)?,
        ),
        // Unary / structural — output dtype equals input dtype.
        StdTensorOp::Neg
        | StdTensorOp::Conj
        | StdTensorOp::Sign
        | StdTensorOp::Exp
        | StdTensorOp::Log
        | StdTensorOp::Sin
        | StdTensorOp::Cos
        | StdTensorOp::Tanh
        | StdTensorOp::Sqrt
        | StdTensorOp::Rsqrt
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
        | StdTensorOp::GatherDynamicSliceSizes { .. }
        | StdTensorOp::DynamicSlice { .. }
        | StdTensorOp::DynamicTruncate { .. }
        | StdTensorOp::PadToMatch { .. }
        | StdTensorOp::Slice(_)
        | StdTensorOp::Pad(_)
        | StdTensorOp::Reverse { .. } => dtype_input(op, input_dtypes, 0)?,
        StdTensorOp::ShapeOf { .. } => DType::F64,
    };
    Ok(dtype)
}

fn dtype_input(op: &StdTensorOp, input_dtypes: &[DType], index: usize) -> Result<DType> {
    input_dtypes.get(index).copied().ok_or_else(|| {
        shape_infer_error(format!(
            "{op:?} missing input dtype at index {index}; got {} input dtypes",
            input_dtypes.len()
        ))
    })
}

fn real_dtype_for_abs(dtype: DType) -> DType {
    match dtype {
        DType::C32 => DType::F32,
        DType::C64 => DType::F64,
        other => other,
    }
}

/// Infer output shapes for a single instruction.
///
/// Returns a vector of shapes (one per output slot). For single-output ops,
/// the vector has length 1. Multi-output extension ops return one entry per
/// output.
pub fn infer_output_shapes(
    op: &StdTensorOp,
    input_shapes: &[&[DimExpr]],
) -> Result<Vec<Vec<DimExpr>>> {
    let shapes = match op {
        StdTensorOp::Add => vec![same_or_scalar_broadcast_shape(
            require_input(op, input_shapes, 0)?,
            require_input(op, input_shapes, 1)?,
        )?],
        StdTensorOp::Sub => vec![same_or_scalar_broadcast_shape(
            require_input(op, input_shapes, 0)?,
            require_input(op, input_shapes, 1)?,
        )?],
        StdTensorOp::Mul => vec![same_or_scalar_broadcast_shape(
            require_input(op, input_shapes, 0)?,
            require_input(op, input_shapes, 1)?,
        )?],
        StdTensorOp::Neg
        | StdTensorOp::Conj
        | StdTensorOp::Div
        | StdTensorOp::Rem
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
        | StdTensorOp::Scatter(_) => {
            vec![require_input(op, input_shapes, 0)?.to_vec()]
        }
        StdTensorOp::Transpose { perm } => {
            vec![permute_shape(require_input(op, input_shapes, 0)?, perm)?]
        }
        StdTensorOp::Reshape { to_shape, .. } => vec![to_shape.clone()],
        StdTensorOp::BroadcastInDim { shape, .. } => vec![shape.clone()],
        StdTensorOp::Constant { .. } => vec![Vec::new()],
        StdTensorOp::ReduceSum { axes, .. }
        | StdTensorOp::ReduceProd { axes, .. }
        | StdTensorOp::ReduceMax { axes, .. }
        | StdTensorOp::ReduceMin { axes, .. } => {
            vec![reduced_shape(require_input(op, input_shapes, 0)?, axes)?]
        }
        StdTensorOp::ExtractDiag { axis_a, axis_b } => {
            vec![extract_diag_shape(
                require_input(op, input_shapes, 0)?,
                *axis_a,
                *axis_b,
            )?]
        }
        StdTensorOp::EmbedDiag { axis_a, axis_b } => {
            vec![embed_diag_shape(
                require_input(op, input_shapes, 0)?,
                *axis_a,
                *axis_b,
            )?]
        }
        StdTensorOp::Gather(config) => vec![gather_shape(
            require_input(op, input_shapes, 0)?,
            require_input(op, input_shapes, 1)?,
            config,
        )?],
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            index_vector_dim,
            slice_sizes,
            ..
        } => {
            let resolved_slice_sizes: Vec<_> = slice_sizes
                .iter()
                .map(|dim| resolve_dim_expr_from_shapes(dim, input_shapes))
                .collect::<Result<_>>()?;
            vec![gather_shape_from_slice_sizes(
                require_input(op, input_shapes, 0)?,
                require_input(op, input_shapes, 1)?,
                offset_dims,
                collapsed_slice_dims,
                *index_vector_dim,
                &resolved_slice_sizes,
            )?]
        }
        StdTensorOp::Slice(config) => {
            vec![slice_shape(require_input(op, input_shapes, 0)?, config)?]
        }
        StdTensorOp::DynamicSlice { slice_sizes } => {
            vec![slice_sizes.iter().copied().map(DimExpr::Const).collect()]
        }
        StdTensorOp::DynamicUpdateSlice => vec![require_input(op, input_shapes, 0)?.to_vec()],
        StdTensorOp::Pad(config) => vec![pad_shape(require_input(op, input_shapes, 0)?, config)?],
        StdTensorOp::DotGeneral { config, .. } => vec![dot_general_shape(
            require_input(op, input_shapes, 0)?,
            require_input(op, input_shapes, 1)?,
            config,
        )?],
        StdTensorOp::Concatenate { axis, .. } => vec![concatenate_shape(input_shapes, *axis)?],
        StdTensorOp::ShapeOf { .. } => vec![Vec::new()],
        StdTensorOp::DynamicTruncate { axis } => {
            let shape = require_input(op, input_shapes, 0)?.to_vec();
            if *axis >= shape.len() {
                return Err(shape_infer_error(format!(
                    "DynamicTruncate axis {axis} out of bounds for rank {}",
                    shape.len()
                )));
            }
            vec![shape]
        }
        StdTensorOp::PadToMatch { axis } => vec![pad_to_match_shape(
            require_input(op, input_shapes, 0)?,
            require_input(op, input_shapes, 1)?,
            *axis,
        )?],
        StdTensorOp::Extension(ext) => {
            let metas = infer_extension_output_meta(ext.as_ref(), &[], input_shapes)?;
            metas.into_iter().map(|(_dtype, shape)| shape).collect()
        }
    };
    Ok(shapes)
}

/// Infer output shape extents for a single instruction.
///
/// Most existing shape rules produce exact dimensions. Runtime-sized operators
/// can instead report a known upper bound so metadata consumers do not treat a
/// bound expression as proof of exactness.
///
pub fn infer_output_extents(
    op: &StdTensorOp,
    input_shapes: &[&[DimExpr]],
) -> Result<Vec<Vec<ShapeExtent<DimExpr>>>> {
    match op {
        StdTensorOp::DynamicTruncate { axis } => {
            let shape = require_input(op, input_shapes, 0)?;
            if *axis >= shape.len() {
                return Err(shape_infer_error(format!(
                    "DynamicTruncate axis {axis} out of bounds for rank {}",
                    shape.len()
                )));
            }
            let mut extents: Vec<_> = shape.iter().cloned().map(ShapeExtent::exact).collect();
            extents[*axis] = ShapeExtent::upper_bound(shape[*axis].clone());
            Ok(vec![extents])
        }
        _ => Ok(infer_output_shapes(op, input_shapes)?
            .into_iter()
            .map(|shape| shape.into_iter().map(ShapeExtent::exact).collect())
            .collect()),
    }
}

/// Compute the full `(dtype, shape)` meta for every output of a
/// `StdTensorOp::Extension(op)` instruction.
///
/// This is the single source of truth for multi-output extensions —
/// [`infer_output_dtype`] only returns the first slot's dtype. Prefer this
/// when populating `ExecInstruction` fields.
///
/// `input_dtypes` must have length `op.input_count()` and be consistent with
/// `input_shapes`. `input_shapes` uses [`DimExpr`] expressions so shape
/// inference can flow through composed symbolic graphs.
pub fn infer_extension_output_meta(
    op: &dyn ExtensionOp,
    input_dtypes: &[DType],
    input_shapes: &[&[DimExpr]],
) -> Result<Vec<(DType, Vec<DimExpr>)>> {
    Ok(infer_extension_output_meta_with_constraints(op, input_dtypes, input_shapes)?.output_metas)
}

pub(crate) fn infer_extension_output_meta_with_constraints(
    op: &dyn ExtensionOp,
    input_dtypes: &[DType],
    input_shapes: &[&[DimExpr]],
) -> Result<InferredExtensionMeta> {
    // Give each extension input a call-local tensor id. These ids deliberately
    // do not reuse DimExpr program input indices: extension input order and
    // program input order are different namespaces.
    let symdim_storage: Vec<Vec<SymDim>> = input_shapes
        .iter()
        .enumerate()
        .map(|(input_idx, shape)| -> Result<Vec<SymDim>> {
            let tensor_id = extension_local_tensor_id(input_idx)?;
            Ok(shape
                .iter()
                .enumerate()
                .map(|(axis, dim)| match dim {
                    DimExpr::Const(value) => SymDim::from(*value),
                    _ => SymDim::tensor_axis(tensor_id, axis),
                })
                .collect())
        })
        .collect::<Result<_>>()?;
    let symdim_refs: Vec<&[SymDim]> = symdim_storage.iter().map(Vec::as_slice).collect();

    let inferred = invoke_extension_shape_inference(op, input_dtypes, &symdim_refs)?;

    let tensor_map: Vec<(u64, usize)> = (0..input_shapes.len())
        .map(|input_idx| Ok((extension_local_tensor_id(input_idx)?, input_idx)))
        .collect::<Result<_>>()?;

    let convert = |dim: &SymDim| {
        let local_expr = dim.to_dim_expr(&tensor_map).map_err(|err| {
            shape_infer_error(format!(
                "ExtensionOp::infer_output_meta for family {:?} returned a SymDim \
                 that cannot be converted to a local DimExpr: {err}",
                op.family_id()
            ))
        })?;
        resolve_dim_expr_from_shapes(&local_expr, input_shapes)
    };

    let output_metas = inferred
        .output_metas
        .into_iter()
        .map(|(dtype, shape)| {
            let dim_exprs = shape.iter().map(&convert).collect::<Result<_>>()?;
            Ok((dtype, dim_exprs))
        })
        .collect::<Result<_>>()?;
    let source = ConstraintSource {
        family_id: op.family_id(),
        instruction_index: None,
    };
    let constraints = inferred
        .constraints
        .into_iter()
        .map(|constraint| {
            Ok(LocalShapeConstraint {
                source: source.clone(),
                relation: constraint.relation(),
                lhs: convert(constraint.lhs())?,
                rhs: convert(constraint.rhs())?,
            })
        })
        .collect::<Result<_>>()?;

    Ok(InferredExtensionMeta {
        output_metas,
        constraints,
    })
}

fn extension_local_tensor_id(input_idx: usize) -> Result<u64> {
    let offset = u64::try_from(input_idx).map_err(|_| {
        shape_infer_error(format!(
            "extension input index {input_idx} does not fit the local symbolic namespace"
        ))
    })?;
    u64::MAX.checked_sub(offset).ok_or_else(|| {
        shape_infer_error(format!(
            "extension input index {input_idx} exhausts the local symbolic namespace"
        ))
    })
}

fn extension_first_output_dtype(op: &dyn ExtensionOp, input_dtypes: &[DType]) -> Result<DType> {
    // For dtype-only queries we synthesise a rank-0 shape list per input so
    // the extension's `infer_output_meta` stays total even when shapes are
    // unknown to the caller. Extensions whose dtype depends on shape must
    // be handled through [`infer_extension_output_meta`], which
    // `compile_std_to_exec` prefers for the `Extension` arm.
    let empty_rows: Vec<&[SymDim]> = (0..op.input_count()).map(|_| [].as_slice()).collect();
    let inferred = invoke_extension_shape_inference(op, input_dtypes, &empty_rows)?;
    inferred
        .output_metas
        .first()
        .map(|meta| meta.0)
        .ok_or_else(|| {
            shape_infer_error(format!(
                "ExtensionOp::infer_output_meta for family {:?} returned an empty meta list",
                op.family_id()
            ))
        })
}

fn require_input<'a>(
    op: &StdTensorOp,
    input_shapes: &'a [&[DimExpr]],
    idx: usize,
) -> Result<&'a [DimExpr]> {
    input_shapes.get(idx).copied().ok_or_else(|| {
        shape_infer_error(format!(
            "{op:?} expects input index {idx}, got {} input shapes",
            input_shapes.len()
        ))
    })
}

fn resolve_dim_expr_from_shapes(expr: &DimExpr, input_shapes: &[&[DimExpr]]) -> Result<DimExpr> {
    match expr {
        DimExpr::Const(value) => Ok(DimExpr::Const(*value)),
        DimExpr::InputDim { input_idx, axis } => {
            require_input_expr(input_shapes, *input_idx, *axis)
        }
        DimExpr::Add(a, b) => dim_add(
            resolve_dim_expr_from_shapes(a, input_shapes)?,
            resolve_dim_expr_from_shapes(b, input_shapes)?,
        ),
        DimExpr::Sub(a, b) => dim_sub(
            resolve_dim_expr_from_shapes(a, input_shapes)?,
            resolve_dim_expr_from_shapes(b, input_shapes)?,
        ),
        DimExpr::Mul(a, b) => dim_mul(
            resolve_dim_expr_from_shapes(a, input_shapes)?,
            resolve_dim_expr_from_shapes(b, input_shapes)?,
        ),
        DimExpr::FloorDiv(a, b) => Ok(DimExpr::floor_div(
            resolve_dim_expr_from_shapes(a, input_shapes)?,
            resolve_dim_expr_from_shapes(b, input_shapes)?,
        )),
        DimExpr::Min(a, b) => Ok(DimExpr::min(
            resolve_dim_expr_from_shapes(a, input_shapes)?,
            resolve_dim_expr_from_shapes(b, input_shapes)?,
        )),
        DimExpr::Max(a, b) => Ok(DimExpr::max(
            resolve_dim_expr_from_shapes(a, input_shapes)?,
            resolve_dim_expr_from_shapes(b, input_shapes)?,
        )),
    }
}

fn require_input_expr(
    input_shapes: &[&[DimExpr]],
    input_idx: usize,
    axis: usize,
) -> Result<DimExpr> {
    input_shapes
        .get(input_idx)
        .and_then(|shape| shape.get(axis))
        .cloned()
        .ok_or_else(|| {
            shape_infer_error(format!(
                "InputDim({}, {}) cannot be resolved from {} input shapes",
                input_idx,
                axis,
                input_shapes.len()
            ))
        })
}

fn permute_shape(input_shape: &[DimExpr], perm: &[usize]) -> Result<Vec<DimExpr>> {
    tenferro_tensor::validate::validate_permutation_axes("transpose", input_shape.len(), perm)
        .map_err(shape_infer_from_tensor_error)?;
    Ok(perm.iter().map(|&axis| input_shape[axis].clone()).collect())
}

fn validate_reduction_axes(input_shape: &[DimExpr], axes: &[usize]) -> Result<()> {
    tenferro_tensor::validate::validate_unique_axes(
        "shape_infer_reduction",
        "reduction",
        input_shape.len(),
        axes,
    )
    .map_err(shape_infer_from_tensor_error)
}

fn reduced_shape(input_shape: &[DimExpr], axes: &[usize]) -> Result<Vec<DimExpr>> {
    validate_reduction_axes(input_shape, axes)?;
    Ok(input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, dim)| (!axes.contains(&axis)).then_some(dim.clone()))
        .collect())
}

fn same_or_scalar_broadcast_shape(
    lhs_shape: &[DimExpr],
    rhs_shape: &[DimExpr],
) -> Result<Vec<DimExpr>> {
    let rank = lhs_shape.len().max(rhs_shape.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let lhs_dim = broadcast_aligned_dim(lhs_shape, rank, axis);
        let rhs_dim = broadcast_aligned_dim(rhs_shape, rank, axis);
        out.push(broadcast_dim(lhs_dim, rhs_dim)?);
    }
    Ok(out)
}

fn broadcast_aligned_dim(shape: &[DimExpr], output_rank: usize, output_axis: usize) -> DimExpr {
    if output_axis < output_rank - shape.len() {
        DimExpr::Const(1)
    } else {
        shape[output_axis - (output_rank - shape.len())].clone()
    }
}

fn broadcast_dim(lhs: DimExpr, rhs: DimExpr) -> Result<DimExpr> {
    if lhs == rhs {
        return Ok(lhs);
    }
    match (&lhs, &rhs) {
        (DimExpr::Const(1), _) => Ok(rhs),
        (_, DimExpr::Const(1)) => Ok(lhs),
        (DimExpr::Const(lhs_value), DimExpr::Const(rhs_value)) => Err(shape_infer_error(format!(
            "incompatible Add/Mul broadcast dimensions: {lhs_value} and {rhs_value}"
        ))),
        _ => Ok(dim_max(lhs, rhs)),
    }
}

fn shape_infer_error(message: impl Into<String>) -> Error {
    Error::InvalidCompiledGraph {
        message: message.into(),
    }
}

fn shape_infer_from_tensor_error(err: tenferro_tensor::Error) -> Error {
    shape_infer_error(err.to_string())
}

fn extract_diag_shape(
    input_shape: &[DimExpr],
    axis_a: usize,
    axis_b: usize,
) -> Result<Vec<DimExpr>> {
    if axis_a >= input_shape.len() || axis_b >= input_shape.len() {
        return Err(shape_infer_error(format!(
            "ExtractDiag axes ({axis_a}, {axis_b}) out of bounds for rank {}",
            input_shape.len()
        )));
    }
    if axis_a == axis_b {
        return Err(shape_infer_error("ExtractDiag requires distinct axes"));
    }
    let diag_output_axis = if axis_a < axis_b { axis_a } else { axis_a - 1 };
    let diag_dim = dim_min(input_shape[axis_a].clone(), input_shape[axis_b].clone());
    let mut output_shape = input_shape.to_vec();
    output_shape.remove(axis_b);
    output_shape[diag_output_axis] = diag_dim;
    Ok(output_shape)
}

fn embed_diag_shape(input_shape: &[DimExpr], axis_a: usize, axis_b: usize) -> Result<Vec<DimExpr>> {
    if axis_a >= input_shape.len() {
        return Err(shape_infer_error(format!(
            "EmbedDiag axis_a {axis_a} out of bounds for rank {}",
            input_shape.len()
        )));
    }
    if axis_b > input_shape.len() {
        return Err(shape_infer_error(format!(
            "EmbedDiag axis_b {axis_b} out of bounds for rank {}",
            input_shape.len()
        )));
    }
    let mut output_shape = input_shape.to_vec();
    output_shape.insert(axis_b, input_shape[axis_a].clone());
    Ok(output_shape)
}

fn dot_general_shape(
    lhs_shape: &[DimExpr],
    rhs_shape: &[DimExpr],
    config: &DotGeneralConfig,
) -> Result<Vec<DimExpr>> {
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    config
        .validate_dims_with_ranks(lhs_rank, rhs_rank)
        .map_err(|err| shape_infer_error(err.to_string()))?;

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
    Ok(output_shape)
}

fn gather_shape(
    operand_shape: &[DimExpr],
    index_shape: &[DimExpr],
    config: &GatherConfig,
) -> Result<Vec<DimExpr>> {
    let slice_sizes: Vec<_> = config
        .slice_sizes
        .iter()
        .copied()
        .map(DimExpr::Const)
        .collect();
    gather_shape_from_slice_sizes(
        operand_shape,
        index_shape,
        &config.offset_dims,
        &config.collapsed_slice_dims,
        config.index_vector_dim,
        &slice_sizes,
    )
}

fn gather_shape_from_slice_sizes(
    operand_shape: &[DimExpr],
    index_shape: &[DimExpr],
    offset_dims: &[usize],
    collapsed_slice_dims: &[usize],
    index_vector_dim: usize,
    slice_sizes: &[DimExpr],
) -> Result<Vec<DimExpr>> {
    if slice_sizes.len() != operand_shape.len() {
        return Err(shape_infer_error(format!(
            "gather: slice_sizes rank mismatch: got {}, expected {}",
            slice_sizes.len(),
            operand_shape.len()
        )));
    }
    validate_gather_slice_sizes_within_operand(operand_shape, slice_sizes)?;
    if index_vector_dim > index_shape.len() {
        return Err(shape_infer_error(format!(
            "gather: index_vector_dim {index_vector_dim} out of bounds for index rank {}",
            index_shape.len()
        )));
    }
    ensure_unique_axes("gather", "offset_dims", offset_dims)?;
    ensure_unique_axes("gather", "collapsed_slice_dims", collapsed_slice_dims)?;
    if collapsed_slice_dims
        .iter()
        .any(|&axis| axis >= operand_shape.len())
    {
        return Err(shape_infer_error(format!(
            "gather: collapsed_slice_dims {collapsed_slice_dims:?} out of bounds for operand rank {}",
            operand_shape.len()
        )));
    }

    let batch_shape = if index_vector_dim == index_shape.len() {
        index_shape.to_vec()
    } else {
        index_shape
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| (axis != index_vector_dim).then_some(dim.clone()))
            .collect()
    };

    let window_dims: Vec<usize> = (0..operand_shape.len())
        .filter(|dim| !collapsed_slice_dims.contains(dim))
        .collect();
    if offset_dims.len() != window_dims.len() {
        return Err(shape_infer_error(format!(
            "gather: offset_dims length mismatch: got {}, expected {}",
            offset_dims.len(),
            window_dims.len()
        )));
    }

    let out_rank = batch_shape.len() + offset_dims.len();
    let mut out_shape = vec![DimExpr::Const(0); out_rank];
    let mut out_axis_to_operand_dim = vec![None; out_rank];
    for (offset_axis, &out_axis) in offset_dims.iter().enumerate() {
        let Some(target) = out_axis_to_operand_dim.get_mut(out_axis) else {
            return Err(shape_infer_error(format!(
                "gather: offset_dim {out_axis} out of bounds for output rank {out_rank}"
            )));
        };
        *target = Some(window_dims[offset_axis]);
    }

    let mut batch_axis = 0usize;
    for out_axis in 0..out_rank {
        if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
            out_shape[out_axis] = slice_sizes[operand_dim].clone();
        } else {
            out_shape[out_axis] = batch_shape[batch_axis].clone();
            batch_axis += 1;
        }
    }

    Ok(out_shape)
}

fn validate_gather_slice_sizes_within_operand(
    operand_shape: &[DimExpr],
    slice_sizes: &[DimExpr],
) -> Result<()> {
    for (axis, (slice_size, dim_size)) in slice_sizes.iter().zip(operand_shape).enumerate() {
        if let (DimExpr::Const(slice_size), DimExpr::Const(dim_size)) = (slice_size, dim_size) {
            if slice_size > dim_size {
                return Err(shape_infer_error(format!(
                    "gather: slice_sizes[{axis}]={slice_size} exceeds operand dimension {dim_size}"
                )));
            }
        }
    }
    Ok(())
}

fn ensure_unique_axes(op: &'static str, role: &'static str, axes: &[usize]) -> Result<()> {
    for (idx, &axis) in axes.iter().enumerate() {
        if axes[..idx].contains(&axis) {
            return Err(shape_infer_error(format!(
                "{op}: duplicate {role} axis {axis}"
            )));
        }
    }
    Ok(())
}

fn slice_shape(input_shape: &[DimExpr], config: &SliceConfig) -> Result<Vec<DimExpr>> {
    let rank = input_shape.len();
    if config.starts.len() != rank || config.limits.len() != rank || config.strides.len() != rank {
        return Err(shape_infer_error(format!(
            "slice: config rank mismatch for rank {rank}: starts={}, limits={}, strides={}",
            config.starts.len(),
            config.limits.len(),
            config.strides.len()
        )));
    }
    (0..rank)
        .map(|axis| {
            let span = config.limits[axis]
                .checked_sub(config.starts[axis])
                .ok_or_else(|| {
                    shape_infer_error(format!(
                        "slice: limit {} is smaller than start {} on axis {axis}",
                        config.limits[axis], config.starts[axis]
                    ))
                })?;
            if config.strides[axis] == 0 {
                return Err(shape_infer_error(format!(
                    "slice: stride must be non-zero on axis {axis}"
                )));
            }
            Ok(DimExpr::Const(span.div_ceil(config.strides[axis])))
        })
        .collect()
}

fn pad_shape(input_shape: &[DimExpr], config: &PadConfig) -> Result<Vec<DimExpr>> {
    let rank = input_shape.len();
    if config.edge_padding_low.len() != rank
        || config.edge_padding_high.len() != rank
        || config.interior_padding.len() != rank
    {
        return Err(shape_infer_error(format!(
            "pad: config rank mismatch for rank {rank}: low={}, high={}, interior={}",
            config.edge_padding_low.len(),
            config.edge_padding_high.len(),
            config.interior_padding.len()
        )));
    }

    input_shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| {
            if config.interior_padding[axis] < 0 {
                return Err(shape_infer_error(format!(
                    "pad: interior padding must be non-negative on axis {axis}"
                )));
            }
            if let DimExpr::Const(extent) = dim {
                let base = if *extent == 0 {
                    0
                } else {
                    let extent = i64::try_from(*extent).map_err(|_| {
                        shape_infer_error(format!(
                            "pad: input extent {extent} exceeds i64 on axis {axis}"
                        ))
                    })?;
                    let stride = config.interior_padding[axis]
                        .checked_add(1)
                        .ok_or_else(|| {
                            shape_infer_error(format!(
                                "pad: interior padding overflow on axis {axis}"
                            ))
                        })?;
                    extent
                        .checked_sub(1)
                        .and_then(|value| value.checked_mul(stride))
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| {
                            shape_infer_error(format!(
                                "pad: stretched extent overflow on axis {axis}"
                            ))
                        })?
                };
                let padded = config.edge_padding_low[axis]
                    .checked_add(config.edge_padding_high[axis])
                    .and_then(|value| value.checked_add(base))
                    .ok_or_else(|| {
                        shape_infer_error(format!("pad: output extent overflow on axis {axis}"))
                    })?;
                Ok(DimExpr::Const(usize::try_from(padded).map_err(|_| {
                    shape_infer_error(format!(
                        "pad: output extent {padded} must be representable as usize on axis {axis}"
                    ))
                })?))
            } else if config.interior_padding[axis] == 0 {
                add_signed(
                    dim.clone(),
                    signed_padding_sum(
                        config.edge_padding_low[axis],
                        config.edge_padding_high[axis],
                        axis,
                    )?,
                )
            } else {
                let stride = DimExpr::Const(usize_from_nonnegative_i64(
                    config.interior_padding[axis]
                        .checked_add(1)
                        .ok_or_else(|| {
                            shape_infer_error(format!(
                                "pad: interior padding overflow on axis {axis}"
                            ))
                        })?,
                    "pad",
                )?);
                let stretched = dim_add(
                    dim_mul(dim_sub(dim.clone(), DimExpr::Const(1))?, stride)?,
                    DimExpr::Const(1),
                )?;
                add_signed(
                    stretched,
                    signed_padding_sum(
                        config.edge_padding_low[axis],
                        config.edge_padding_high[axis],
                        axis,
                    )?,
                )
            }
        })
        .collect()
}

fn add_signed(expr: DimExpr, amount: i64) -> Result<DimExpr> {
    if amount >= 0 {
        dim_add(
            expr,
            DimExpr::Const(usize_from_nonnegative_i64(amount, "add_signed")?),
        )
    } else {
        let magnitude = amount
            .checked_neg()
            .ok_or_else(|| shape_infer_error("add_signed: negative amount magnitude overflow"))?;
        dim_sub(
            expr,
            DimExpr::Const(usize_from_nonnegative_i64(magnitude, "add_signed")?),
        )
    }
}

fn signed_padding_sum(low: i64, high: i64, axis: usize) -> Result<i64> {
    low.checked_add(high)
        .ok_or_else(|| shape_infer_error(format!("pad: edge padding overflow on axis {axis}")))
}

fn concatenate_shape(input_shapes: &[&[DimExpr]], axis: usize) -> Result<Vec<DimExpr>> {
    let first = input_shapes
        .first()
        .copied()
        .ok_or_else(|| shape_infer_error("concatenate expects at least one input shape"))?;
    if axis >= first.len() {
        return Err(shape_infer_error(format!(
            "concatenate axis {axis} out of bounds for rank {}",
            first.len()
        )));
    }
    let mut output_shape = first.to_vec();
    let mut axis_dim = first[axis].clone();
    for (input_idx, shape) in input_shapes.iter().enumerate().skip(1) {
        if shape.len() != first.len() {
            return Err(shape_infer_error(format!(
                "concatenate input {input_idx} rank mismatch: got {}, expected {}",
                shape.len(),
                first.len()
            )));
        }
        for (dim_idx, (expected, actual)) in first.iter().zip(*shape).enumerate() {
            if dim_idx != axis && actual != expected {
                return Err(shape_infer_error(format!(
                    "concatenate dimension mismatch on non-axis dim {dim_idx}: input {input_idx} has {actual:?}, expected {expected:?}"
                )));
            }
        }
        axis_dim = dim_add(axis_dim, shape[axis].clone())?;
    }
    output_shape[axis] = axis_dim;
    Ok(output_shape)
}

fn pad_to_match_shape(
    input_shape: &[DimExpr],
    reference_shape: &[DimExpr],
    axis: usize,
) -> Result<Vec<DimExpr>> {
    if axis >= input_shape.len() {
        return Err(shape_infer_error(format!(
            "PadToMatch input axis {axis} out of bounds for rank {}",
            input_shape.len()
        )));
    }
    if axis >= reference_shape.len() {
        return Err(shape_infer_error(format!(
            "PadToMatch reference axis {axis} out of bounds for rank {}",
            reference_shape.len()
        )));
    }
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = dim_max(input_shape[axis].clone(), reference_shape[axis].clone());
    Ok(output_shape)
}

fn dim_add(lhs: DimExpr, rhs: DimExpr) -> Result<DimExpr> {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => {
            lhs.checked_add(rhs).map(DimExpr::Const).ok_or_else(|| {
                shape_infer_error(format!("dimension addition overflow: {lhs} + {rhs}"))
            })
        }
        (lhs, rhs) => Ok(DimExpr::add(lhs, rhs)),
    }
}

fn dim_sub(lhs: DimExpr, rhs: DimExpr) -> Result<DimExpr> {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => {
            lhs.checked_sub(rhs).map(DimExpr::Const).ok_or_else(|| {
                shape_infer_error(format!("dimension subtraction underflow: {lhs} - {rhs}"))
            })
        }
        (lhs, rhs) => Ok(DimExpr::sub(lhs, rhs)),
    }
}

fn dim_mul(lhs: DimExpr, rhs: DimExpr) -> Result<DimExpr> {
    match (lhs, rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => {
            lhs.checked_mul(rhs).map(DimExpr::Const).ok_or_else(|| {
                shape_infer_error(format!("dimension multiplication overflow: {lhs} * {rhs}"))
            })
        }
        (lhs, rhs) => Ok(DimExpr::mul(lhs, rhs)),
    }
}

fn usize_from_nonnegative_i64(value: i64, op: &'static str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| shape_infer_error(format!("{op}: value {value} must fit in usize")))
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

#[cfg(test)]
mod tests;
