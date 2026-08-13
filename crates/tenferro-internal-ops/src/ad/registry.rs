use crate::ad::PrimitiveRuleBuilder;
use crate::ad::{ADRuleError, ADRuleKind, ADRuleResult, ResidualSpec};
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_core_ops::PrimitiveOpKind;

use super::context::ShapeGuardContext;
use super::transpose_input::{fixed_value_refs, metadata_value_refs, TransposeInputRef};
use super::{
    analytic, contraction, diagonal, dynamic, elementwise, indexing, semiring, structural,
};
use crate::std_tensor_op::StdTensorOp;

pub(crate) type LinearizeFn = fn(
    &StdTensorOp,
    &mut dyn PrimitiveRuleBuilder,
    &[ValueKey<StdTensorOp>],
    &[ValueKey<StdTensorOp>],
    &[Option<LocalValueId>],
    &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>>;

pub(crate) type TransposeFn = fn(
    &StdTensorOp,
    &mut dyn PrimitiveRuleBuilder,
    &[Option<LocalValueId>],
    &[TransposeInputRef<'_>],
    &OperationRole,
    &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>>;

pub(crate) trait PrimitiveAdRule: Send + Sync {
    fn kind(&self) -> PrimitiveOpKind;

    /// Declare which primal inputs this rule's transpose reads as tensor
    /// residuals. Inputs not declared may only be accessed through metadata.
    fn residual_mask(&self) -> ResidualSpec;

    fn linearize(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;

    fn transpose_rule(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[TransposeInputRef<'_>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}

struct FunctionPrimitiveAdRule {
    kind: PrimitiveOpKind,
    linearize: LinearizeFn,
    transpose_rule: TransposeFn,
    residual_mask: ResidualSpec,
}

impl PrimitiveAdRule for FunctionPrimitiveAdRule {
    fn kind(&self) -> PrimitiveOpKind {
        self.kind
    }

    fn residual_mask(&self) -> ResidualSpec {
        self.residual_mask
    }

    fn linearize(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        (self.linearize)(op, builder, primal_in, primal_out, tangent_in, ctx)
    }

    fn transpose_rule(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[TransposeInputRef<'_>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        (self.transpose_rule)(op, builder, cotangent_out, inputs, mode, ctx)
    }
}

pub(crate) fn primitive_ad_rule(kind: PrimitiveOpKind) -> Option<&'static dyn PrimitiveAdRule> {
    let rule = PRIMITIVE_AD_RULES[kind.as_index()];
    debug_assert_eq!(rule.kind(), kind);
    Some(rule)
}

pub(crate) fn missing_rule(kind: PrimitiveOpKind, rule: ADRuleKind) -> ADRuleError {
    ADRuleError::unsupported(format!("missing primitive AD rule for {kind:?}"), rule)
}

fn catalog_kind_mismatch(op: &StdTensorOp, rule: ADRuleKind) -> ADRuleError {
    ADRuleError::invalid_input(
        "tenferro-internal-ops primitive AD registry",
        rule,
        format!("AD registry rule was invoked with a mismatched operation: {op:?}"),
    )
}

macro_rules! catalog_payload {
    ($op:expr, $rule:expr, $pattern:pat => $payload:expr) => {
        match $op {
            $pattern => $payload,
            _ => return Err(catalog_kind_mismatch($op, $rule)),
        }
    };
}

static PRIMITIVE_AD_RULES: [&'static dyn PrimitiveAdRule; PrimitiveOpKind::COUNT] = [
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Add,
        linearize: linearize_add,
        transpose_rule: transpose_add,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Sub,
        linearize: linearize_sub,
        transpose_rule: transpose_sub,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Mul,
        linearize: linearize_mul,
        transpose_rule: transpose_mul,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Neg,
        linearize: linearize_neg,
        transpose_rule: transpose_neg,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Conj,
        linearize: linearize_conj,
        transpose_rule: transpose_conj,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Div,
        linearize: linearize_div,
        transpose_rule: transpose_div,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Rem,
        linearize: linearize_compare,
        transpose_rule: transpose_compare,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Abs,
        linearize: linearize_abs,
        transpose_rule: transpose_abs,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Sign,
        linearize: linearize_sign,
        transpose_rule: transpose_sign,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Maximum,
        linearize: linearize_maximum,
        transpose_rule: transpose_maximum,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Minimum,
        linearize: linearize_minimum,
        transpose_rule: transpose_minimum,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Compare,
        linearize: linearize_compare,
        transpose_rule: transpose_compare,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Select,
        linearize: linearize_select,
        transpose_rule: transpose_select,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Clamp,
        linearize: linearize_clamp,
        transpose_rule: transpose_clamp,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Exp,
        linearize: linearize_exp,
        transpose_rule: transpose_exp,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Log,
        linearize: linearize_log,
        transpose_rule: transpose_log,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Sin,
        linearize: linearize_sin,
        transpose_rule: transpose_sin,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Cos,
        linearize: linearize_cos,
        transpose_rule: transpose_cos,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Tanh,
        linearize: linearize_tanh,
        transpose_rule: transpose_tanh,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Sqrt,
        linearize: linearize_sqrt,
        transpose_rule: transpose_sqrt,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Rsqrt,
        linearize: linearize_rsqrt,
        transpose_rule: transpose_rsqrt,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Pow,
        linearize: linearize_pow,
        transpose_rule: transpose_pow,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Expm1,
        linearize: linearize_expm1,
        transpose_rule: transpose_expm1,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Log1p,
        linearize: linearize_log1p,
        transpose_rule: transpose_log1p,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DotGeneral,
        linearize: linearize_dot_general,
        transpose_rule: transpose_dot_general,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceSum,
        linearize: linearize_reduce_sum,
        transpose_rule: transpose_reduce_sum,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceSumSquares,
        linearize: linearize_reduce_sum_squares,
        transpose_rule: transpose_reduce_sum_squares,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceProd,
        linearize: linearize_reduce_prod,
        transpose_rule: transpose_reduce_prod,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceMax,
        linearize: linearize_reduce_max,
        transpose_rule: transpose_reduce_max,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceMin,
        linearize: linearize_reduce_min,
        transpose_rule: transpose_reduce_min,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Transpose,
        linearize: linearize_transpose,
        transpose_rule: transpose_transpose,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Reshape,
        linearize: linearize_reshape,
        transpose_rule: transpose_reshape,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::BroadcastInDim,
        linearize: linearize_broadcast_in_dim,
        transpose_rule: transpose_broadcast_in_dim,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Convert,
        linearize: linearize_convert,
        transpose_rule: transpose_convert,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ExtractDiag,
        linearize: linearize_extract_diag,
        transpose_rule: transpose_extract_diag,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::EmbedDiag,
        linearize: linearize_embed_diag,
        transpose_rule: transpose_embed_diag,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Tril,
        linearize: linearize_tril,
        transpose_rule: transpose_tril,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Triu,
        linearize: linearize_triu,
        transpose_rule: transpose_triu,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Gather,
        linearize: linearize_gather,
        transpose_rule: transpose_gather,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::GatherDynamicSliceSizes,
        linearize: linearize_gather_dynamic_slice_sizes,
        transpose_rule: transpose_gather_dynamic_slice_sizes,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Scatter,
        linearize: linearize_scatter,
        transpose_rule: transpose_scatter,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Slice,
        linearize: linearize_slice,
        transpose_rule: transpose_slice,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicSlice,
        linearize: linearize_dynamic_slice,
        transpose_rule: transpose_dynamic_slice,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicUpdateSlice,
        linearize: linearize_dynamic_update_slice,
        transpose_rule: transpose_dynamic_update_slice,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Pad,
        linearize: linearize_pad,
        transpose_rule: transpose_pad,
        residual_mask: ResidualSpec::input(0),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Concatenate,
        linearize: linearize_concatenate,
        transpose_rule: transpose_concatenate,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Reverse,
        linearize: linearize_reverse,
        transpose_rule: transpose_reverse,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ShapeOf,
        linearize: linearize_shape_of,
        transpose_rule: transpose_shape_of,
        residual_mask: ResidualSpec::none(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicTruncate,
        linearize: linearize_dynamic_truncate,
        transpose_rule: transpose_dynamic_truncate,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::PadToMatch,
        linearize: linearize_pad_to_match,
        transpose_rule: transpose_pad_to_match,
        residual_mask: ResidualSpec::all_inputs(),
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Constant,
        linearize: linearize_constant,
        transpose_rule: transpose_constant,
        residual_mask: ResidualSpec::none(),
    },
];

fn linearize_add(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::linearize_add(builder, primal_in, tangent_in, ctx))
}

fn transpose_add(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = metadata_value_refs(inputs);
    semiring::transpose_add(builder, cotangent_out, &inputs, mode, ctx)
}

fn linearize_sub(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::linearize_sub(builder, primal_in, tangent_in, ctx))
}

fn transpose_sub(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = metadata_value_refs(inputs);
    semiring::transpose_sub(builder, cotangent_out, &inputs, mode, ctx)
}

fn linearize_mul(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::linearize_mul(builder, primal_in, tangent_in, ctx))
}

fn transpose_mul(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    semiring::transpose_mul(builder, cotangent_out, inputs, mode, ctx)
}

fn linearize_neg(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::linearize_neg(builder, tangent_in))
}

fn transpose_neg(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::transpose_neg(builder, cotangent_out, mode))
}

fn linearize_conj(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    semiring::linearize_conj(builder, primal_in, tangent_in, ctx)
}

fn transpose_conj(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = metadata_value_refs(inputs);
    semiring::transpose_conj(builder, cotangent_out, &inputs, mode, ctx)
}

fn linearize_div(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_div(
        builder, primal_in, primal_out, tangent_in, ctx,
    ))
}

fn linearize_abs(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_abs(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn linearize_sign(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_sign(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn linearize_maximum(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_maximum(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn linearize_minimum(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_minimum(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn linearize_compare(
    _op: &StdTensorOp,
    _builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(vec![None])
}

fn linearize_select(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_select(
        builder, primal_in, tangent_in,
    ))
}

fn linearize_clamp(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_clamp(builder, primal_in, tangent_in))
}

macro_rules! transpose_elementwise {
    ($name:ident, $callee:path) => {
        fn $name(
            _op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            cotangent_out: &[Option<LocalValueId>],
            inputs: &[TransposeInputRef<'_>],
            mode: &OperationRole,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let inputs = fixed_value_refs(stringify!($name), inputs)?;
            Ok($callee(builder, cotangent_out, &inputs, mode))
        }
    };
}

fn transpose_div(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    elementwise::transpose_div(builder, cotangent_out, inputs, mode, ctx)
}
fn transpose_abs(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("abs", inputs)?;
    Ok(elementwise::transpose_abs(
        builder,
        cotangent_out,
        &inputs,
        mode,
        ctx,
    ))
}
fn transpose_sign(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::transpose_sign(builder, cotangent_out, mode))
}
fn transpose_maximum(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("maximum", inputs)?;
    Ok(elementwise::transpose_maximum(
        builder,
        cotangent_out,
        &inputs,
        mode,
        ctx,
    ))
}

fn transpose_minimum(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("minimum", inputs)?;
    Ok(elementwise::transpose_minimum(
        builder,
        cotangent_out,
        &inputs,
        mode,
        ctx,
    ))
}
fn transpose_select(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    elementwise::transpose_select(builder, cotangent_out, inputs, mode)
}
transpose_elementwise!(transpose_clamp, elementwise::transpose_clamp);
fn transpose_compare(
    _op: &StdTensorOp,
    _builder: &mut dyn PrimitiveRuleBuilder,
    _cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(vec![None, None])
}

macro_rules! analytic_linearize {
    ($name:ident, $callee:path, primal_in) => {
        fn $name(
            _op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            primal_in: &[ValueKey<StdTensorOp>],
            _primal_out: &[ValueKey<StdTensorOp>],
            tangent_in: &[Option<LocalValueId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            Ok($callee(builder, primal_in, tangent_in))
        }
    };
    ($name:ident, $callee:path, primal_out) => {
        fn $name(
            _op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            _primal_in: &[ValueKey<StdTensorOp>],
            primal_out: &[ValueKey<StdTensorOp>],
            tangent_in: &[Option<LocalValueId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            Ok($callee(builder, primal_out, tangent_in))
        }
    };
}

analytic_linearize!(linearize_exp, analytic::linearize_exp, primal_out);
analytic_linearize!(linearize_log, analytic::linearize_log, primal_in);
analytic_linearize!(linearize_sin, analytic::linearize_sin, primal_in);
analytic_linearize!(linearize_cos, analytic::linearize_cos, primal_in);
fn linearize_tanh(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    analytic::linearize_tanh(builder, primal_out, tangent_in, ctx)
}
analytic_linearize!(linearize_sqrt, analytic::linearize_sqrt, primal_out);
fn linearize_rsqrt(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(analytic::linearize_rsqrt(
        builder, primal_in, primal_out, tangent_in,
    ))
}
fn linearize_pow(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    analytic::linearize_pow(builder, primal_in, primal_out, tangent_in, ctx)
}
fn linearize_expm1(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    analytic::linearize_expm1(builder, primal_out, tangent_in, ctx)
}
fn linearize_log1p(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    analytic::linearize_log1p(builder, primal_in, tangent_in, ctx)
}

macro_rules! analytic_transpose {
    ($name:ident, $callee:path) => {
        fn $name(
            _op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            cotangent_out: &[Option<LocalValueId>],
            inputs: &[TransposeInputRef<'_>],
            mode: &OperationRole,
            ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let inputs = fixed_value_refs(stringify!($name), inputs)?;
            Ok($callee(builder, cotangent_out, &inputs, mode, ctx))
        }
    };
}

analytic_transpose!(transpose_exp, analytic::transpose_exp);
analytic_transpose!(transpose_log, analytic::transpose_log);
analytic_transpose!(transpose_sin, analytic::transpose_sin);
analytic_transpose!(transpose_cos, analytic::transpose_cos);
fn transpose_tanh(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("tanh", inputs)?;
    analytic::transpose_tanh(builder, cotangent_out, &inputs, mode, ctx)
}
analytic_transpose!(transpose_sqrt, analytic::transpose_sqrt);
analytic_transpose!(transpose_rsqrt, analytic::transpose_rsqrt);
fn transpose_pow(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("pow", inputs)?;
    analytic::transpose_pow(builder, cotangent_out, &inputs, mode, ctx)
}
analytic_transpose!(transpose_expm1, analytic::transpose_expm1);
fn transpose_log1p(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("log1p", inputs)?;
    analytic::transpose_log1p(builder, cotangent_out, &inputs, mode, ctx)
}

fn linearize_dot_general(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config =
        catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::DotGeneral { config } => config);
    contraction::linearize_dot_general(builder, primal_in, tangent_in, config, ctx)
}

fn transpose_dot_general(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config =
        catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::DotGeneral { config } => config);
    contraction::transpose_dot_general(builder, cotangent_out, inputs, mode, config, ctx)
}

fn linearize_reduce_sum(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::ReduceSum { axes } => axes);
    Ok(contraction::linearize_reduce_sum(
        builder, tangent_in, op, axes,
    ))
}

fn transpose_reduce_sum(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    contraction::transpose_reduce_sum_input(builder, cotangent_out, op, &inputs[0], ctx)
}

fn linearize_reduce_sum_squares(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(
        op,
        ADRuleKind::Jvp,
        StdTensorOp::ReduceSumSquares { axes } => axes
    );
    let square_tangent = semiring::linearize_mul(
        builder,
        &[primal_in[0].clone(), primal_in[0].clone()],
        &[tangent_in[0], tangent_in[0]],
        ctx,
    )[0];
    Ok(contraction::linearize_reduce_sum(
        builder,
        &[square_tangent],
        &StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
        },
        axes,
    ))
}

fn transpose_reduce_sum_squares(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(
        op,
        ADRuleKind::Transpose,
        StdTensorOp::ReduceSumSquares { axes } => axes
    );
    let broadcast = contraction::transpose_reduce_sum_input(
        builder,
        cotangent_out,
        &StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
        },
        &inputs[0],
        ctx,
    )?[0];
    let Some(broadcast) = broadcast else {
        return Ok(vec![None]);
    };
    let product = builder.add_operation(
        StdTensorOp::Mul,
        vec![
            inputs[0].fixed_value("reduce_sum_squares", 0)?,
            ValueRef::Local(broadcast),
        ],
        OperationRole::Linearized {
            active_mask: vec![false, true],
        },
    )[0];
    let doubled = builder.add_operation(
        StdTensorOp::Add,
        vec![ValueRef::Local(product), ValueRef::Local(product)],
        OperationRole::Linearized {
            active_mask: vec![true, true],
        },
    )[0];
    Ok(vec![Some(doubled)])
}

fn linearize_reduce_prod(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::ReduceProd { axes } => axes);
    contraction::linearize_reduce_prod(builder, primal_in, primal_out, tangent_in, axes, ctx)
}

fn transpose_reduce_prod(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("reduce_prod", inputs)?;
    contraction::transpose_reduce_prod(builder, cotangent_out, &inputs, op, ctx)
}

fn linearize_reduce_max(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::ReduceMax { axes } => axes);
    contraction::linearize_reduce_chooser(builder, primal_in, primal_out, tangent_in, axes, ctx)
}

fn linearize_reduce_min(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::ReduceMin { axes } => axes);
    contraction::linearize_reduce_chooser(builder, primal_in, primal_out, tangent_in, axes, ctx)
}

fn transpose_reduce_max(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("reduce_max", inputs)?;
    contraction::transpose_reduce_chooser(builder, cotangent_out, &inputs, op, ctx)
}

fn transpose_reduce_min(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("reduce_min", inputs)?;
    contraction::transpose_reduce_chooser(builder, cotangent_out, &inputs, op, ctx)
}

fn linearize_transpose(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let perm = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::Transpose { perm } => perm);
    Ok(structural::linearize_transpose(builder, tangent_in, perm))
}

fn transpose_transpose(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let perm = catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::Transpose { perm } => perm);
    structural::transpose_transpose(builder, cotangent_out, mode, perm)
}

fn linearize_reshape(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(structural::linearize_reshape(
        builder, primal_in, tangent_in, op, ctx,
    ))
}

fn transpose_reshape(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    structural::transpose_reshape_input(builder, cotangent_out, op, inputs, mode, ctx)
}

fn linearize_broadcast_in_dim(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (shape, dims) = catalog_payload!(
        op,
        ADRuleKind::Jvp,
        StdTensorOp::BroadcastInDim { shape, dims } => (shape, dims)
    );
    Ok(structural::linearize_broadcast_in_dim(
        builder, primal_in, tangent_in, shape, dims, ctx,
    ))
}

fn transpose_broadcast_in_dim(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (shape, dims) = catalog_payload!(
        op,
        ADRuleKind::Transpose,
        StdTensorOp::BroadcastInDim { shape, dims } => (shape, dims)
    );
    structural::transpose_broadcast_in_dim_input(
        builder,
        cotangent_out,
        shape,
        dims,
        inputs,
        mode,
        ctx,
    )
}

fn linearize_convert(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (from, to) =
        catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::Convert { from, to } => (from, to));
    Ok(structural::linearize_convert(
        builder, tangent_in, *from, *to,
    ))
}

fn transpose_convert(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (from, to) = catalog_payload!(
        op,
        ADRuleKind::Transpose,
        StdTensorOp::Convert { from, to } => (from, to)
    );
    Ok(structural::transpose_convert(
        builder,
        cotangent_out,
        mode,
        *from,
        *to,
    ))
}

macro_rules! diagonal_rule {
    ($lin:ident, $trans:ident, $variant:ident, $lin_call:path, $trans_call:path) => {
        fn $lin(
            op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            _primal_in: &[ValueKey<StdTensorOp>],
            _primal_out: &[ValueKey<StdTensorOp>],
            tangent_in: &[Option<LocalValueId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let (axis_a, axis_b) = catalog_payload!(
                op,
                ADRuleKind::Jvp,
                StdTensorOp::$variant { axis_a, axis_b } => (axis_a, axis_b)
            );
            $lin_call(builder, tangent_in, *axis_a, *axis_b)
        }

        fn $trans(
            op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            cotangent_out: &[Option<LocalValueId>],
            inputs: &[TransposeInputRef<'_>],
            mode: &OperationRole,
            ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let (axis_a, axis_b) = catalog_payload!(
                op,
                ADRuleKind::Transpose,
                StdTensorOp::$variant { axis_a, axis_b } => (axis_a, axis_b)
            );
            $trans_call(builder, cotangent_out, inputs, mode, *axis_a, *axis_b, ctx)
        }
    };
}

diagonal_rule!(
    linearize_extract_diag,
    transpose_extract_diag,
    ExtractDiag,
    diagonal::linearize_extract_diag,
    diagonal::transpose_extract_diag
);
diagonal_rule!(
    linearize_embed_diag,
    transpose_embed_diag,
    EmbedDiag,
    diagonal::linearize_embed_diag,
    diagonal::transpose_embed_diag
);

macro_rules! triangular_rule {
    ($lin:ident, $trans:ident, $variant:ident, $lin_call:path, $trans_call:path) => {
        fn $lin(
            op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            _primal_in: &[ValueKey<StdTensorOp>],
            _primal_out: &[ValueKey<StdTensorOp>],
            tangent_in: &[Option<LocalValueId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let k = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::$variant { k } => k);
            Ok($lin_call(builder, tangent_in, *k))
        }

        fn $trans(
            op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            cotangent_out: &[Option<LocalValueId>],
            _inputs: &[TransposeInputRef<'_>],
            mode: &OperationRole,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let k = catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::$variant { k } => k);
            Ok($trans_call(builder, cotangent_out, mode, *k))
        }
    };
}

triangular_rule!(
    linearize_tril,
    transpose_tril,
    Tril,
    structural::linearize_tril,
    structural::transpose_tril
);
triangular_rule!(
    linearize_triu,
    transpose_triu,
    Triu,
    structural::linearize_triu,
    structural::transpose_triu
);

fn linearize_gather(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::Gather(config) => config);
    Ok(indexing::linearize_gather(
        builder, primal_in, tangent_in, config,
    ))
}

fn transpose_gather(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config = catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::Gather(config) => config);
    let inputs = fixed_value_refs("gather", inputs)?;
    indexing::transpose_gather(builder, cotangent_out, &inputs, mode, config, ctx)
}

fn linearize_gather_dynamic_slice_sizes(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim, slice_sizes) = catalog_payload!(
        op,
        ADRuleKind::Jvp,
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        } => (
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        )
    );
    Ok(indexing::linearize_gather_dynamic_slice_sizes(
        builder,
        primal_in,
        tangent_in,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        *index_vector_dim,
        slice_sizes,
    ))
}

fn transpose_gather_dynamic_slice_sizes(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim) = catalog_payload!(
        op,
        ADRuleKind::Transpose,
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            ..
        } => (offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim)
    );
    let inputs = fixed_value_refs("gather_dynamic_slice_sizes", inputs)?;
    indexing::transpose_gather_dynamic_slice_sizes(
        builder,
        cotangent_out,
        &inputs,
        mode,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        *index_vector_dim,
        ctx,
    )
}

fn linearize_scatter(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::Scatter(config) => config);
    indexing::linearize_scatter(builder, primal_in, tangent_in, config, ctx)
}

fn transpose_scatter(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config =
        catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::Scatter(config) => config);
    let inputs = fixed_value_refs("scatter", inputs)?;
    indexing::transpose_scatter(builder, cotangent_out, &inputs, mode, config, ctx)
}

fn linearize_slice(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::Slice(config) => config);
    Ok(structural::linearize_slice(builder, tangent_in, config))
}

fn transpose_slice(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config = catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::Slice(config) => config);
    let inputs = fixed_value_refs("slice", inputs)?;
    structural::transpose_slice(builder, cotangent_out, &inputs, mode, config, ctx)
}

fn linearize_dynamic_slice(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let slice_sizes = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::DynamicSlice { slice_sizes } => slice_sizes);
    Ok(indexing::linearize_dynamic_slice(
        builder,
        primal_in,
        tangent_in,
        slice_sizes,
    ))
}

fn transpose_dynamic_slice(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("dynamic_slice", inputs)?;
    indexing::transpose_dynamic_slice(builder, cotangent_out, &inputs, mode, ctx)
}

fn linearize_dynamic_update_slice(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    indexing::linearize_dynamic_update_slice(builder, primal_in, tangent_in, ctx)
}

fn transpose_dynamic_update_slice(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let inputs = fixed_value_refs("dynamic_update_slice", inputs)?;
    indexing::transpose_dynamic_update_slice(builder, cotangent_out, &inputs, mode, ctx)
}

fn linearize_pad(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::Pad(config) => config);
    Ok(structural::linearize_pad(builder, tangent_in, config))
}

fn transpose_pad(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let config = catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::Pad(config) => config);
    let inputs = fixed_value_refs("pad", inputs)?;
    structural::transpose_pad(builder, cotangent_out, &inputs, mode, config, ctx)
}

fn linearize_concatenate(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (axis, input_count) = catalog_payload!(
        op,
        ADRuleKind::Jvp,
        StdTensorOp::Concatenate { axis, input_count } => (axis, input_count)
    );
    structural::linearize_concatenate(builder, primal_in, tangent_in, *axis, *input_count, ctx)
}

fn transpose_concatenate(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let (axis, input_count) = catalog_payload!(
        op,
        ADRuleKind::Transpose,
        StdTensorOp::Concatenate { axis, input_count } => (axis, input_count)
    );
    let inputs = fixed_value_refs("concatenate", inputs)?;
    structural::transpose_concatenate(
        builder,
        cotangent_out,
        &inputs,
        mode,
        *axis,
        *input_count,
        ctx,
    )
}

fn linearize_reverse(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::Reverse { axes } => axes);
    Ok(structural::linearize_reverse(builder, tangent_in, axes))
}

fn transpose_reverse(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axes = catalog_payload!(op, ADRuleKind::Transpose, StdTensorOp::Reverse { axes } => axes);
    Ok(structural::transpose_reverse(
        builder,
        cotangent_out,
        mode,
        axes,
    ))
}

fn linearize_shape_of(
    _op: &StdTensorOp,
    _builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(vec![None])
}

fn transpose_shape_of(
    _op: &StdTensorOp,
    _builder: &mut dyn PrimitiveRuleBuilder,
    _cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(vec![None])
}

fn linearize_dynamic_truncate(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axis = catalog_payload!(
        op,
        ADRuleKind::Jvp,
        StdTensorOp::DynamicTruncate { axis } => axis
    );
    Ok(dynamic::linearize_dynamic_truncate(
        builder, primal_in, primal_out, tangent_in, *axis, ctx,
    ))
}

fn transpose_dynamic_truncate(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axis = catalog_payload!(
        op,
        ADRuleKind::Transpose,
        StdTensorOp::DynamicTruncate { axis } => axis
    );
    let inputs = fixed_value_refs("dynamic_truncate", inputs)?;
    Ok(dynamic::transpose_dynamic_truncate(
        builder,
        cotangent_out,
        &inputs,
        mode,
        *axis,
    ))
}

fn linearize_pad_to_match(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axis = catalog_payload!(op, ADRuleKind::Jvp, StdTensorOp::PadToMatch { axis } => axis);
    Ok(dynamic::linearize_pad_to_match(
        builder, primal_in, tangent_in, *axis,
    ))
}

fn transpose_pad_to_match(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let axis = catalog_payload!(
        op,
        ADRuleKind::Transpose,
        StdTensorOp::PadToMatch { axis } => axis
    );
    let inputs = fixed_value_refs("pad_to_match", inputs)?;
    dynamic::transpose_pad_to_match(builder, cotangent_out, &inputs, mode, *axis, ctx)
}

fn linearize_constant(
    _op: &StdTensorOp,
    _builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(vec![None])
}

fn transpose_constant(
    _op: &StdTensorOp,
    _builder: &mut dyn PrimitiveRuleBuilder,
    _cotangent_out: &[Option<LocalValueId>],
    _inputs: &[TransposeInputRef<'_>],
    _mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(vec![])
}
