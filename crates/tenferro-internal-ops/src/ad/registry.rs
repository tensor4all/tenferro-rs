use crate::ad::PrimitiveRuleBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_core_ops::PrimitiveOpKind;
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use super::context::ShapeGuardContext;
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
    &[ValueRef<StdTensorOp>],
    &OperationRole,
    &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>>;

pub(crate) trait PrimitiveAdRule: Send + Sync {
    fn kind(&self) -> PrimitiveOpKind;

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
        inputs: &[ValueRef<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}

struct FunctionPrimitiveAdRule {
    kind: PrimitiveOpKind,
    linearize: LinearizeFn,
    transpose_rule: TransposeFn,
}

impl PrimitiveAdRule for FunctionPrimitiveAdRule {
    fn kind(&self) -> PrimitiveOpKind {
        self.kind
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
        inputs: &[ValueRef<StdTensorOp>],
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

static PRIMITIVE_AD_RULES: [&'static dyn PrimitiveAdRule; PrimitiveOpKind::COUNT] = [
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Add,
        linearize: linearize_add,
        transpose_rule: transpose_add,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Mul,
        linearize: linearize_mul,
        transpose_rule: transpose_mul,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Neg,
        linearize: linearize_neg,
        transpose_rule: transpose_neg,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Conj,
        linearize: linearize_conj,
        transpose_rule: transpose_conj,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Div,
        linearize: linearize_div,
        transpose_rule: transpose_div,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Abs,
        linearize: linearize_abs,
        transpose_rule: transpose_abs,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Sign,
        linearize: linearize_sign,
        transpose_rule: transpose_sign,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Maximum,
        linearize: linearize_maximum,
        transpose_rule: transpose_maximum,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Minimum,
        linearize: linearize_minimum,
        transpose_rule: transpose_minimum,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Compare,
        linearize: linearize_compare,
        transpose_rule: transpose_compare,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Select,
        linearize: linearize_select,
        transpose_rule: transpose_select,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Clamp,
        linearize: linearize_clamp,
        transpose_rule: transpose_clamp,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Exp,
        linearize: linearize_exp,
        transpose_rule: transpose_exp,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Log,
        linearize: linearize_log,
        transpose_rule: transpose_log,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Sin,
        linearize: linearize_sin,
        transpose_rule: transpose_sin,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Cos,
        linearize: linearize_cos,
        transpose_rule: transpose_cos,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Tanh,
        linearize: linearize_tanh,
        transpose_rule: transpose_tanh,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Sqrt,
        linearize: linearize_sqrt,
        transpose_rule: transpose_sqrt,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Rsqrt,
        linearize: linearize_rsqrt,
        transpose_rule: transpose_rsqrt,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Pow,
        linearize: linearize_pow,
        transpose_rule: transpose_pow,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Expm1,
        linearize: linearize_expm1,
        transpose_rule: transpose_expm1,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Log1p,
        linearize: linearize_log1p,
        transpose_rule: transpose_log1p,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DotGeneral,
        linearize: linearize_dot_general,
        transpose_rule: transpose_dot_general,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceSum,
        linearize: linearize_reduce_sum,
        transpose_rule: transpose_reduce_sum,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceProd,
        linearize: linearize_reduce_prod,
        transpose_rule: transpose_reduce_prod,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceMax,
        linearize: linearize_reduce_max,
        transpose_rule: transpose_reduce_max,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceMin,
        linearize: linearize_reduce_min,
        transpose_rule: transpose_reduce_min,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Transpose,
        linearize: linearize_transpose,
        transpose_rule: transpose_transpose,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Reshape,
        linearize: linearize_reshape,
        transpose_rule: transpose_reshape,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::BroadcastInDim,
        linearize: linearize_broadcast_in_dim,
        transpose_rule: transpose_broadcast_in_dim,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Convert,
        linearize: linearize_convert,
        transpose_rule: transpose_convert,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ExtractDiag,
        linearize: linearize_extract_diag,
        transpose_rule: transpose_extract_diag,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::EmbedDiag,
        linearize: linearize_embed_diag,
        transpose_rule: transpose_embed_diag,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Tril,
        linearize: linearize_tril,
        transpose_rule: transpose_tril,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Triu,
        linearize: linearize_triu,
        transpose_rule: transpose_triu,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Gather,
        linearize: linearize_gather,
        transpose_rule: transpose_gather,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::GatherDynamicSliceSizes,
        linearize: linearize_gather_dynamic_slice_sizes,
        transpose_rule: transpose_gather_dynamic_slice_sizes,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Scatter,
        linearize: linearize_scatter,
        transpose_rule: transpose_scatter,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Slice,
        linearize: linearize_slice,
        transpose_rule: transpose_slice,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicSlice,
        linearize: linearize_dynamic_slice,
        transpose_rule: transpose_dynamic_slice,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicUpdateSlice,
        linearize: linearize_dynamic_update_slice,
        transpose_rule: transpose_dynamic_update_slice,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Pad,
        linearize: linearize_pad,
        transpose_rule: transpose_pad,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Concatenate,
        linearize: linearize_concatenate,
        transpose_rule: transpose_concatenate,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Reverse,
        linearize: linearize_reverse,
        transpose_rule: transpose_reverse,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::ShapeOf,
        linearize: linearize_shape_of,
        transpose_rule: transpose_shape_of,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicTruncate,
        linearize: linearize_dynamic_truncate,
        transpose_rule: transpose_dynamic_truncate,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::PadToMatch,
        linearize: linearize_pad_to_match,
        transpose_rule: transpose_pad_to_match,
    },
    &FunctionPrimitiveAdRule {
        kind: PrimitiveOpKind::Constant,
        linearize: linearize_constant,
        transpose_rule: transpose_constant,
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
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::transpose_add(builder, cotangent_out, inputs, ctx))
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
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::transpose_mul(
        builder,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
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
    _inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::transpose_neg(builder, cotangent_out))
}

fn linearize_conj(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::linearize_conj(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn transpose_conj(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(semiring::transpose_conj(
        builder,
        cotangent_out,
        inputs,
        ctx,
    ))
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
    _builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::linearize_sign(_builder, tangent_in))
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
            inputs: &[ValueRef<StdTensorOp>],
            mode: &OperationRole,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            Ok($callee(builder, cotangent_out, inputs, mode))
        }
    };
}

fn transpose_div(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::transpose_div(
        builder,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}
fn transpose_abs(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::transpose_abs(
        builder,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}
fn transpose_sign(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::transpose_sign(builder, cotangent_out, mode))
}
fn transpose_maximum(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::transpose_maximum(
        builder,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}

fn transpose_minimum(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(elementwise::transpose_minimum(
        builder,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}
transpose_elementwise!(transpose_select, elementwise::transpose_select);
transpose_elementwise!(transpose_clamp, elementwise::transpose_clamp);
fn transpose_compare(
    _op: &StdTensorOp,
    _builder: &mut dyn PrimitiveRuleBuilder,
    _cotangent_out: &[Option<LocalValueId>],
    _inputs: &[ValueRef<StdTensorOp>],
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
analytic_linearize!(linearize_tanh, analytic::linearize_tanh, primal_out);
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
    Ok(analytic::linearize_pow(
        builder, primal_in, primal_out, tangent_in, ctx,
    ))
}
analytic_linearize!(linearize_expm1, analytic::linearize_expm1, primal_out);
analytic_linearize!(linearize_log1p, analytic::linearize_log1p, primal_in);

macro_rules! analytic_transpose {
    ($name:ident, $callee:path) => {
        fn $name(
            _op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            cotangent_out: &[Option<LocalValueId>],
            inputs: &[ValueRef<StdTensorOp>],
            mode: &OperationRole,
            ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            Ok($callee(builder, cotangent_out, inputs, mode, ctx))
        }
    };
}

analytic_transpose!(transpose_exp, analytic::transpose_exp);
analytic_transpose!(transpose_log, analytic::transpose_log);
analytic_transpose!(transpose_sin, analytic::transpose_sin);
analytic_transpose!(transpose_cos, analytic::transpose_cos);
analytic_transpose!(transpose_tanh, analytic::transpose_tanh);
analytic_transpose!(transpose_sqrt, analytic::transpose_sqrt);
analytic_transpose!(transpose_rsqrt, analytic::transpose_rsqrt);
analytic_transpose!(transpose_pow, analytic::transpose_pow);
analytic_transpose!(transpose_expm1, analytic::transpose_expm1);
analytic_transpose!(transpose_log1p, analytic::transpose_log1p);

fn linearize_dot_general(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::DotGeneral { config } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_dot_general(
        builder, primal_in, tangent_in, config, ctx,
    ))
}

fn transpose_dot_general(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::DotGeneral { config } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::transpose_dot_general(
        builder,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_reduce_sum(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::ReduceSum { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_sum(
        builder, tangent_in, op, axes,
    ))
}

fn transpose_reduce_sum(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(contraction::transpose_reduce_sum(
        builder,
        cotangent_out,
        op,
        inputs,
        ctx,
    ))
}

fn linearize_reduce_prod(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::ReduceProd { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_prod(
        builder, primal_in, primal_out, tangent_in, axes, ctx,
    ))
}

fn transpose_reduce_prod(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(contraction::transpose_reduce_prod(
        builder,
        cotangent_out,
        inputs,
        op,
        ctx,
    ))
}

fn linearize_reduce_max(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::ReduceMax { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_chooser(
        builder, primal_in, primal_out, tangent_in, axes, ctx,
    ))
}

fn linearize_reduce_min(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::ReduceMin { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_chooser(
        builder, primal_in, primal_out, tangent_in, axes, ctx,
    ))
}

fn transpose_reduce_max(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(contraction::transpose_reduce_chooser(
        builder,
        cotangent_out,
        inputs,
        op,
        ctx,
    ))
}

fn transpose_reduce_min(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(contraction::transpose_reduce_chooser(
        builder,
        cotangent_out,
        inputs,
        op,
        ctx,
    ))
}

fn linearize_transpose(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Transpose { perm } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_transpose(builder, tangent_in, perm))
}

fn transpose_transpose(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Transpose { perm } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_transpose(
        builder,
        cotangent_out,
        perm,
    ))
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
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(structural::transpose_reshape(
        builder,
        cotangent_out,
        op,
        inputs,
        ctx,
    ))
}

fn linearize_broadcast_in_dim(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::BroadcastInDim { shape, dims } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_broadcast_in_dim(
        builder, primal_in, tangent_in, shape, dims, ctx,
    ))
}

fn transpose_broadcast_in_dim(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::BroadcastInDim { shape, dims } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_broadcast_in_dim(
        builder,
        cotangent_out,
        shape,
        dims,
        inputs,
        ctx,
    ))
}

fn linearize_convert(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Convert { from, to } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_convert(
        builder, tangent_in, *from, *to,
    ))
}

fn transpose_convert(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Convert { from, to } = op else {
        unreachable!("catalog kind mismatch")
    };
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
            let StdTensorOp::$variant { axis_a, axis_b } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($lin_call(builder, tangent_in, *axis_a, *axis_b))
        }

        fn $trans(
            op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            cotangent_out: &[Option<LocalValueId>],
            inputs: &[ValueRef<StdTensorOp>],
            _mode: &OperationRole,
            ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let StdTensorOp::$variant { axis_a, axis_b } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($trans_call(
                builder,
                cotangent_out,
                inputs,
                *axis_a,
                *axis_b,
                ctx,
            ))
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
            let StdTensorOp::$variant { k } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($lin_call(builder, tangent_in, *k))
        }

        fn $trans(
            op: &StdTensorOp,
            builder: &mut dyn PrimitiveRuleBuilder,
            cotangent_out: &[Option<LocalValueId>],
            _inputs: &[ValueRef<StdTensorOp>],
            _mode: &OperationRole,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
            let StdTensorOp::$variant { k } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($trans_call(builder, cotangent_out, *k))
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
    let StdTensorOp::Gather(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::linearize_gather(
        builder, primal_in, tangent_in, config,
    ))
}

fn transpose_gather(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Gather(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::transpose_gather(
        builder,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_gather_dynamic_slice_sizes(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::GatherDynamicSliceSizes {
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        index_vector_dim,
        slice_sizes,
    } = op
    else {
        unreachable!("catalog kind mismatch")
    };
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
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::GatherDynamicSliceSizes {
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        index_vector_dim,
        ..
    } = op
    else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::transpose_gather_dynamic_slice_sizes(
        builder,
        cotangent_out,
        inputs,
        mode,
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        *index_vector_dim,
        ctx,
    ))
}

fn linearize_scatter(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Scatter(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::linearize_scatter(
        builder, primal_in, tangent_in, config, ctx,
    ))
}

fn transpose_scatter(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Scatter(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::transpose_scatter(
        builder,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_slice(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Slice(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_slice(builder, tangent_in, config))
}

fn transpose_slice(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Slice(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_slice(
        builder,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_dynamic_slice(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::DynamicSlice { slice_sizes } = op else {
        unreachable!("catalog kind mismatch")
    };
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
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(indexing::transpose_dynamic_slice(
        builder,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}

fn linearize_dynamic_update_slice(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(indexing::linearize_dynamic_update_slice(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn transpose_dynamic_update_slice(
    _op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(indexing::transpose_dynamic_update_slice(
        builder,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}

fn linearize_pad(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Pad(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_pad(builder, tangent_in, config))
}

fn transpose_pad(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Pad(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_pad(
        builder,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_concatenate(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Concatenate { axis, input_count } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_concatenate(
        builder,
        primal_in,
        tangent_in,
        *axis,
        *input_count,
        ctx,
    ))
}

fn transpose_concatenate(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Concatenate { axis, input_count } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_concatenate(
        builder,
        cotangent_out,
        inputs,
        mode,
        *axis,
        *input_count,
        ctx,
    ))
}

fn linearize_reverse(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    _primal_in: &[ValueKey<StdTensorOp>],
    _primal_out: &[ValueKey<StdTensorOp>],
    tangent_in: &[Option<LocalValueId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Reverse { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_reverse(builder, tangent_in, axes))
}

fn transpose_reverse(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    _inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::Reverse { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
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
    _inputs: &[ValueRef<StdTensorOp>],
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
    let StdTensorOp::DynamicTruncate { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::linearize_dynamic_truncate(
        builder, primal_in, primal_out, tangent_in, *axis, ctx,
    ))
}

fn transpose_dynamic_truncate(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::DynamicTruncate { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::transpose_dynamic_truncate(
        builder,
        cotangent_out,
        inputs,
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
    let StdTensorOp::PadToMatch { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::linearize_pad_to_match(
        builder, primal_in, tangent_in, *axis,
    ))
}

fn transpose_pad_to_match(
    op: &StdTensorOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    mode: &OperationRole,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let StdTensorOp::PadToMatch { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::transpose_pad_to_match(
        builder,
        cotangent_out,
        inputs,
        mode,
        *axis,
        ctx,
    ))
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
    _inputs: &[ValueRef<StdTensorOp>],
    _mode: &OperationRole,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    Ok(vec![])
}
