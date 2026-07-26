use crate::ad::ADRuleResult;
use computegraph::types::{LocalValueId, OperationRole, ValueRef};
use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::DType;

use crate::ad::context::ShapeGuardContext;
use crate::ad::PrimitiveRuleBuilder;
use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;

/// AD rule support status for a core primitive operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_ops::ad::support::{primitive_ad_support, AdRuleSupport};
///
/// let add = primitive_ad_support(PrimitiveOpKind::Add);
/// assert_eq!(add.linearize, AdRuleSupport::Supported);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdRuleSupport {
    Supported,
    SupportedViaLinearize,
    NonDifferentiable,
    Unsupported,
}

/// AD support manifest entry for one core primitive operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_ops::ad::support::{primitive_ad_support, AdRuleSupport};
///
/// let compare = primitive_ad_support(PrimitiveOpKind::Compare);
/// assert_eq!(compare.kind, PrimitiveOpKind::Compare);
/// assert_eq!(compare.linearize, AdRuleSupport::NonDifferentiable);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimitiveAdSupport {
    /// Core primitive operation kind described by this manifest entry.
    pub kind: PrimitiveOpKind,
    /// Forward-mode graph emission support.
    pub linearize: AdRuleSupport,
    /// Transposed-linear graph emission support.
    pub transpose: AdRuleSupport,
}

macro_rules! direct_support {
    ($kind:ident) => {
        PrimitiveAdSupport {
            kind: PrimitiveOpKind::$kind,
            linearize: AdRuleSupport::Supported,
            transpose: AdRuleSupport::Supported,
        }
    };
}

macro_rules! nondiff_support {
    ($kind:ident) => {
        PrimitiveAdSupport {
            kind: PrimitiveOpKind::$kind,
            linearize: AdRuleSupport::NonDifferentiable,
            transpose: AdRuleSupport::NonDifferentiable,
        }
    };
}

/// Complete core primitive AD support manifest.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_ops::ad::support::PRIMITIVE_AD_SUPPORT;
///
/// assert_eq!(PRIMITIVE_AD_SUPPORT.len(), PrimitiveOpKind::COUNT);
/// ```
pub static PRIMITIVE_AD_SUPPORT: [PrimitiveAdSupport; PrimitiveOpKind::COUNT] = [
    direct_support!(Add),
    direct_support!(Sub),
    direct_support!(Mul),
    direct_support!(Neg),
    direct_support!(Conj),
    direct_support!(Div),
    nondiff_support!(Rem),
    direct_support!(Abs),
    direct_support!(Sign),
    direct_support!(Maximum),
    direct_support!(Minimum),
    nondiff_support!(Compare),
    direct_support!(Select),
    direct_support!(Clamp),
    direct_support!(Exp),
    direct_support!(Log),
    direct_support!(Sin),
    direct_support!(Cos),
    direct_support!(Tanh),
    direct_support!(Sqrt),
    direct_support!(Rsqrt),
    direct_support!(Pow),
    direct_support!(Expm1),
    direct_support!(Log1p),
    direct_support!(DotGeneral),
    direct_support!(ReduceSum),
    direct_support!(ReduceProd),
    direct_support!(ReduceMax),
    direct_support!(ReduceMin),
    direct_support!(Transpose),
    direct_support!(Reshape),
    direct_support!(BroadcastInDim),
    direct_support!(Convert),
    direct_support!(ExtractDiag),
    direct_support!(EmbedDiag),
    direct_support!(Tril),
    direct_support!(Triu),
    direct_support!(Gather),
    direct_support!(GatherDynamicSliceSizes),
    direct_support!(Scatter),
    direct_support!(Slice),
    direct_support!(DynamicSlice),
    direct_support!(DynamicUpdateSlice),
    direct_support!(Pad),
    direct_support!(Concatenate),
    direct_support!(Reverse),
    nondiff_support!(ShapeOf),
    direct_support!(DynamicTruncate),
    direct_support!(PadToMatch),
    nondiff_support!(Constant),
];

/// Return the complete core primitive AD support manifest.
///
/// # Examples
///
/// ```rust
/// let manifest = tenferro_ops::ad::support::all_primitive_ad_support();
/// assert_eq!(manifest.len(), tenferro_core_ops::PrimitiveOpKind::COUNT);
/// ```
pub fn all_primitive_ad_support() -> &'static [PrimitiveAdSupport; PrimitiveOpKind::COUNT] {
    &PRIMITIVE_AD_SUPPORT
}

/// Return the support manifest entry for one core primitive operation kind.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_ops::ad::support::primitive_ad_support;
///
/// let entry = primitive_ad_support(PrimitiveOpKind::ReduceSum);
/// assert_eq!(entry.kind, PrimitiveOpKind::ReduceSum);
/// ```
pub fn primitive_ad_support(kind: PrimitiveOpKind) -> &'static PrimitiveAdSupport {
    &PRIMITIVE_AD_SUPPORT[kind.as_index()]
}

/// Return whether a dtype is a real floating-point dtype.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DType;
/// use tenferro_ops::ad::support::is_real_dtype;
///
/// assert!(is_real_dtype(DType::F64));
/// assert!(!is_real_dtype(DType::C64));
/// ```
pub fn is_real_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
}

fn dim_expr_uses_only_input(expr: &DimExpr, expected_input_idx: usize) -> bool {
    match expr {
        DimExpr::Const(_) => true,
        DimExpr::InputDim { input_idx, .. } => *input_idx == expected_input_idx,
        DimExpr::Add(a, b)
        | DimExpr::Sub(a, b)
        | DimExpr::Mul(a, b)
        | DimExpr::FloorDiv(a, b)
        | DimExpr::Min(a, b)
        | DimExpr::Max(a, b) => {
            dim_expr_uses_only_input(a, expected_input_idx)
                && dim_expr_uses_only_input(b, expected_input_idx)
        }
    }
}

#[doc(hidden)]
pub fn constant_scalar(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    bytes: Vec<u8>,
) -> LocalValueId {
    builder.add_operation(
        StdTensorOp::Constant { dtype, bytes },
        vec![],
        OperationRole::Primary,
    )[0]
}

#[doc(hidden)]
pub fn zero_like(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    anchor: ValueRef<StdTensorOp>,
    anchor_rank: usize,
) -> LocalValueId {
    super::zeros::build_zero_like(builder, dtype, anchor, anchor_rank)
}

#[doc(hidden)]
pub fn one_like(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    anchor: ValueRef<StdTensorOp>,
    anchor_rank: usize,
) -> LocalValueId {
    super::zeros::build_one_like(builder, dtype, anchor, anchor_rank)
}

#[doc(hidden)]
pub fn identity_matrix(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    size: usize,
    batch_shape: &[DimExpr],
    shape_source: ValueRef<StdTensorOp>,
    shape_source_idx: usize,
) -> LocalValueId {
    let one = one_like(builder, dtype, shape_source.clone(), 0);
    let mut shape = Vec::with_capacity(1 + batch_shape.len());
    shape.push(DimExpr::Const(size));
    shape.extend_from_slice(batch_shape);

    let mut inputs = vec![ValueRef::Local(one)];
    if DimExpr::max_input_idx_all(&shape).is_some() {
        assert!(
            shape
                .iter()
                .all(|expr| dim_expr_uses_only_input(expr, shape_source_idx)),
            "identity_matrix shape expressions must reference only the provided shape_source_idx"
        );
        shape = DimExpr::remap_all(&shape, shape_source_idx, 1);
        inputs.push(shape_source);
    }

    let ones = builder.add_operation(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![],
        },
        inputs,
        OperationRole::Primary,
    )[0];
    builder.add_operation(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValueRef::Local(ones)],
        OperationRole::Primary,
    )[0]
}

pub(crate) fn is_differentiable_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::C32 | DType::C64)
}

pub(crate) fn linear_transpose_input_active(mode: &OperationRole, input_index: usize) -> bool {
    match mode {
        // Direct transpose-rule tests use `Primary` for globally linear
        // primitives. Only an explicit Linearized active mask suppresses an
        // inactive cotangent path.
        OperationRole::Primary => true,
        OperationRole::Linearized { active_mask } => {
            active_mask.get(input_index).copied().unwrap_or(false)
        }
    }
}

fn is_complex_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::C32 | DType::C64)
}

pub(crate) fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    tenferro_tensor::validate::promote_dtype(lhs, rhs)
}

pub(crate) fn promote_dtype_div_like(lhs: DType, rhs: DType) -> DType {
    if matches!(lhs, DType::I32 | DType::I64) && matches!(rhs, DType::I32 | DType::I64) {
        return DType::F64;
    }
    promote_dtype(lhs, rhs)
}

pub(crate) fn dtype_of_or_real(ctx: &mut ShapeGuardContext, val: &ValueRef<StdTensorOp>) -> DType {
    ctx.metadata_if_available(val)
        .map(|metadata| metadata.dtype)
        .unwrap_or(DType::F64)
}

#[doc(hidden)]
pub fn conjugate_primal_if_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<ValueRef<StdTensorOp>> {
    let dtype = ctx.dtype_of(&input)?;
    Ok(conjugate_primal_if_dtype_complex(builder, input, dtype))
}

#[doc(hidden)]
pub fn conjugate_primal_if_dtype_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    dtype: DType,
) -> ValueRef<StdTensorOp> {
    if is_complex_dtype(dtype) {
        ValueRef::Local(
            builder.add_operation(StdTensorOp::Conj, vec![input], OperationRole::Primary)[0],
        )
    } else {
        input
    }
}

pub(crate) fn conjugate_primal_if_any_dtype_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    dtypes: &[DType],
) -> ValueRef<StdTensorOp> {
    if dtypes.iter().copied().any(is_complex_dtype) {
        ValueRef::Local(
            builder.add_operation(StdTensorOp::Conj, vec![input], OperationRole::Primary)[0],
        )
    } else {
        input
    }
}

#[doc(hidden)]
pub fn conjugate_linear_if_dtype_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    dtype: DType,
) -> LocalValueId {
    if is_complex_dtype(dtype) {
        builder.add_operation(
            StdTensorOp::Conj,
            vec![ValueRef::Local(input)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0]
    } else {
        input
    }
}

pub(crate) fn project_linear_to_dtype(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    from: DType,
    to: DType,
) -> LocalValueId {
    if !is_real_dtype(to) || from == to {
        return input;
    }
    convert_linear_to_dtype(builder, input, from, to)
}

pub(crate) fn convert_linear_to_dtype(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    from: DType,
    to: DType,
) -> LocalValueId {
    if from == to {
        return input;
    }
    builder.add_operation(
        StdTensorOp::Convert { from, to },
        vec![ValueRef::Local(input)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    )[0]
}

pub(crate) fn convert_fixed_ref_to_dtype(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    from: DType,
    to: DType,
) -> ValueRef<StdTensorOp> {
    if from == to {
        return input;
    }
    ValueRef::Local(
        builder.add_operation(
            StdTensorOp::Convert { from, to },
            vec![input],
            OperationRole::Linearized {
                active_mask: vec![false],
            },
        )[0],
    )
}

#[cfg(test)]
mod tests;
