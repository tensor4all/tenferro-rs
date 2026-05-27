use chainrules_core::{ADRuleError, ADRuleKind, ADRuleResult};
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_core_ops::PrimitiveOpKind;

use super::context::ShapeGuardContext;
use super::{
    analytic, contraction, diagonal, dynamic, elementwise, indexing, semiring, structural,
};
use crate::std_tensor_op::StdTensorOp;

pub(crate) type LinearizeFn = fn(
    &StdTensorOp,
    &mut FragmentBuilder<StdTensorOp>,
    &[GlobalValKey<StdTensorOp>],
    &[GlobalValKey<StdTensorOp>],
    &[Option<LocalValId>],
    &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>>;

pub(crate) type TransposeFn = fn(
    &StdTensorOp,
    &mut dyn OpEmitter<StdTensorOp>,
    &[Option<LocalValId>],
    &[ValRef<StdTensorOp>],
    &OpMode,
    &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>>;

pub(crate) struct PrimitiveAdRule {
    pub(crate) kind: PrimitiveOpKind,
    pub(crate) linearize: LinearizeFn,
    pub(crate) transpose: TransposeFn,
}

pub(crate) fn primitive_ad_rule(kind: PrimitiveOpKind) -> Option<&'static PrimitiveAdRule> {
    PRIMITIVE_AD_RULES.iter().find(|rule| rule.kind == kind)
}

pub(crate) fn missing_rule(kind: PrimitiveOpKind, rule: ADRuleKind) -> ADRuleError {
    ADRuleError::unsupported(format!("missing primitive AD rule for {kind:?}"), rule)
}

static PRIMITIVE_AD_RULES: &[PrimitiveAdRule] = &[
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Add,
        linearize: linearize_add,
        transpose: transpose_add,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Mul,
        linearize: linearize_mul,
        transpose: transpose_mul,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Neg,
        linearize: linearize_neg,
        transpose: transpose_neg,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Conj,
        linearize: linearize_conj,
        transpose: transpose_conj,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Div,
        linearize: linearize_div,
        transpose: transpose_div,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Abs,
        linearize: linearize_abs,
        transpose: transpose_abs,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Sign,
        linearize: linearize_sign,
        transpose: transpose_sign,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Maximum,
        linearize: linearize_maximum,
        transpose: transpose_maximum,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Minimum,
        linearize: linearize_minimum,
        transpose: transpose_minimum,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Compare,
        linearize: linearize_compare,
        transpose: transpose_compare,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Select,
        linearize: linearize_select,
        transpose: transpose_select,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Clamp,
        linearize: linearize_clamp,
        transpose: transpose_clamp,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Exp,
        linearize: linearize_exp,
        transpose: transpose_exp,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Log,
        linearize: linearize_log,
        transpose: transpose_log,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Sin,
        linearize: linearize_sin,
        transpose: transpose_sin,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Cos,
        linearize: linearize_cos,
        transpose: transpose_cos,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Tanh,
        linearize: linearize_tanh,
        transpose: transpose_tanh,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Sqrt,
        linearize: linearize_sqrt,
        transpose: transpose_sqrt,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Rsqrt,
        linearize: linearize_rsqrt,
        transpose: transpose_rsqrt,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Pow,
        linearize: linearize_pow,
        transpose: transpose_pow,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Expm1,
        linearize: linearize_expm1,
        transpose: transpose_expm1,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Log1p,
        linearize: linearize_log1p,
        transpose: transpose_log1p,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::DotGeneral,
        linearize: linearize_dot_general,
        transpose: transpose_dot_general,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceSum,
        linearize: linearize_reduce_sum,
        transpose: transpose_reduce_sum,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceProd,
        linearize: linearize_reduce_prod,
        transpose: transpose_reduce_prod,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceMax,
        linearize: linearize_reduce_max,
        transpose: transpose_reduce_max,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::ReduceMin,
        linearize: linearize_reduce_min,
        transpose: transpose_reduce_min,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Transpose,
        linearize: linearize_transpose,
        transpose: transpose_transpose,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Reshape,
        linearize: linearize_reshape,
        transpose: transpose_reshape,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::BroadcastInDim,
        linearize: linearize_broadcast_in_dim,
        transpose: transpose_broadcast_in_dim,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Convert,
        linearize: linearize_convert,
        transpose: transpose_convert,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::ExtractDiag,
        linearize: linearize_extract_diag,
        transpose: transpose_extract_diag,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::EmbedDiag,
        linearize: linearize_embed_diag,
        transpose: transpose_embed_diag,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Tril,
        linearize: linearize_tril,
        transpose: transpose_tril,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Triu,
        linearize: linearize_triu,
        transpose: transpose_triu,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Gather,
        linearize: linearize_gather,
        transpose: transpose_gather,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::GatherDynamicSliceSizes,
        linearize: linearize_gather_dynamic_slice_sizes,
        transpose: transpose_gather_dynamic_slice_sizes,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Scatter,
        linearize: linearize_scatter,
        transpose: transpose_scatter,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Slice,
        linearize: linearize_slice,
        transpose: transpose_slice,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicSlice,
        linearize: linearize_dynamic_slice,
        transpose: transpose_dynamic_slice,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicUpdateSlice,
        linearize: linearize_dynamic_update_slice,
        transpose: transpose_dynamic_update_slice,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Pad,
        linearize: linearize_pad,
        transpose: transpose_pad,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Concatenate,
        linearize: linearize_concatenate,
        transpose: transpose_concatenate,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Reverse,
        linearize: linearize_reverse,
        transpose: transpose_reverse,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::ShapeOf,
        linearize: linearize_shape_of,
        transpose: transpose_shape_of,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::DynamicTruncate,
        linearize: linearize_dynamic_truncate,
        transpose: transpose_dynamic_truncate,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::PadToMatch,
        linearize: linearize_pad_to_match,
        transpose: transpose_pad_to_match,
    },
    PrimitiveAdRule {
        kind: PrimitiveOpKind::Constant,
        linearize: linearize_constant,
        transpose: transpose_constant,
    },
];

fn linearize_add(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::linearize_add(builder, tangent_in))
}

fn transpose_add(
    _op: &StdTensorOp,
    _emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::transpose_add(cotangent_out))
}

fn linearize_mul(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::linearize_mul(builder, primal_in, tangent_in))
}

fn transpose_mul(
    _op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::transpose_mul(
        emitter,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}

fn linearize_neg(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::linearize_neg(builder, tangent_in))
}

fn transpose_neg(
    _op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::transpose_neg(emitter, cotangent_out))
}

fn linearize_conj(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::linearize_conj(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn transpose_conj(
    _op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(semiring::transpose_conj(
        emitter,
        cotangent_out,
        inputs,
        ctx,
    ))
}

fn linearize_div(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::linearize_div(
        builder, primal_in, primal_out, tangent_in,
    ))
}

fn linearize_abs(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::linearize_abs(builder, primal_in, tangent_in))
}

fn linearize_sign(
    _op: &StdTensorOp,
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::linearize_sign(_builder, tangent_in))
}

fn linearize_maximum(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::linearize_maximum(
        builder, primal_in, tangent_in,
    ))
}

fn linearize_minimum(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::linearize_minimum(
        builder, primal_in, tangent_in,
    ))
}

fn linearize_compare(
    _op: &StdTensorOp,
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(vec![None])
}

fn linearize_select(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::linearize_select(
        builder, primal_in, tangent_in,
    ))
}

fn linearize_clamp(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::linearize_clamp(builder, primal_in, tangent_in))
}

macro_rules! transpose_elementwise {
    ($name:ident, $callee:path) => {
        fn $name(
            _op: &StdTensorOp,
            emitter: &mut dyn OpEmitter<StdTensorOp>,
            cotangent_out: &[Option<LocalValId>],
            inputs: &[ValRef<StdTensorOp>],
            mode: &OpMode,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            Ok($callee(emitter, cotangent_out, inputs, mode))
        }
    };
}

transpose_elementwise!(transpose_div, elementwise::transpose_div);
transpose_elementwise!(transpose_abs, elementwise::transpose_abs);
fn transpose_sign(
    _op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(elementwise::transpose_sign(emitter, cotangent_out, mode))
}
transpose_elementwise!(transpose_maximum, elementwise::transpose_maximum);
transpose_elementwise!(transpose_minimum, elementwise::transpose_minimum);
transpose_elementwise!(transpose_select, elementwise::transpose_select);
transpose_elementwise!(transpose_clamp, elementwise::transpose_clamp);
fn transpose_compare(
    _op: &StdTensorOp,
    _emitter: &mut dyn OpEmitter<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(vec![None, None])
}

macro_rules! analytic_linearize {
    ($name:ident, $callee:path, primal_in) => {
        fn $name(
            _op: &StdTensorOp,
            builder: &mut FragmentBuilder<StdTensorOp>,
            primal_in: &[GlobalValKey<StdTensorOp>],
            _primal_out: &[GlobalValKey<StdTensorOp>],
            tangent_in: &[Option<LocalValId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            Ok($callee(builder, primal_in, tangent_in))
        }
    };
    ($name:ident, $callee:path, primal_out) => {
        fn $name(
            _op: &StdTensorOp,
            builder: &mut FragmentBuilder<StdTensorOp>,
            _primal_in: &[GlobalValKey<StdTensorOp>],
            primal_out: &[GlobalValKey<StdTensorOp>],
            tangent_in: &[Option<LocalValId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
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
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(analytic::linearize_rsqrt(
        builder, primal_in, primal_out, tangent_in,
    ))
}
fn linearize_pow(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(analytic::linearize_pow(
        builder, primal_in, primal_out, tangent_in,
    ))
}
analytic_linearize!(linearize_expm1, analytic::linearize_expm1, primal_out);
analytic_linearize!(linearize_log1p, analytic::linearize_log1p, primal_in);

macro_rules! analytic_transpose {
    ($name:ident, $callee:path) => {
        fn $name(
            _op: &StdTensorOp,
            emitter: &mut dyn OpEmitter<StdTensorOp>,
            cotangent_out: &[Option<LocalValId>],
            inputs: &[ValRef<StdTensorOp>],
            mode: &OpMode,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            Ok($callee(emitter, cotangent_out, inputs, mode))
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
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::DotGeneral { config } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_dot_general(
        builder, primal_in, tangent_in, config, ctx,
    ))
}

fn transpose_dot_general(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::DotGeneral { config } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::transpose_dot_general(
        emitter,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_reduce_sum(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::ReduceSum { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_sum(
        builder, tangent_in, op, axes,
    ))
}

fn transpose_reduce_sum(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(contraction::transpose_reduce_sum(
        emitter,
        cotangent_out,
        op,
        inputs,
        ctx,
    ))
}

fn linearize_reduce_prod(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::ReduceProd { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_prod(
        builder, primal_in, primal_out, tangent_in, axes, ctx,
    ))
}

fn transpose_reduce_prod(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(contraction::transpose_reduce_prod(
        emitter,
        cotangent_out,
        inputs,
        op,
        ctx,
    ))
}

fn linearize_reduce_max(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::ReduceMax { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_chooser(
        builder, primal_in, primal_out, tangent_in, axes, ctx,
    ))
}

fn linearize_reduce_min(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::ReduceMin { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(contraction::linearize_reduce_chooser(
        builder, primal_in, primal_out, tangent_in, axes, ctx,
    ))
}

fn transpose_reduce_max(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(contraction::transpose_reduce_chooser(
        emitter,
        cotangent_out,
        inputs,
        op,
        ctx,
    ))
}

fn transpose_reduce_min(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(contraction::transpose_reduce_chooser(
        emitter,
        cotangent_out,
        inputs,
        op,
        ctx,
    ))
}

fn linearize_transpose(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Transpose { perm } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_transpose(builder, tangent_in, perm))
}

fn transpose_transpose(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Transpose { perm } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_transpose(
        emitter,
        cotangent_out,
        perm,
    ))
}

fn linearize_reshape(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(structural::linearize_reshape(
        builder, primal_in, tangent_in, op, ctx,
    ))
}

fn transpose_reshape(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(structural::transpose_reshape(
        emitter,
        cotangent_out,
        op,
        inputs,
        ctx,
    ))
}

fn linearize_broadcast_in_dim(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::BroadcastInDim { shape, dims } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_broadcast_in_dim(
        builder, primal_in, tangent_in, shape, dims, ctx,
    ))
}

fn transpose_broadcast_in_dim(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::BroadcastInDim { shape, dims } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_broadcast_in_dim(
        emitter,
        cotangent_out,
        shape,
        dims,
    ))
}

fn linearize_convert(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Convert { from, to } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_convert(
        builder, tangent_in, *from, *to,
    ))
}

fn transpose_convert(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Convert { from, to } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_convert(
        emitter,
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
            builder: &mut FragmentBuilder<StdTensorOp>,
            _primal_in: &[GlobalValKey<StdTensorOp>],
            _primal_out: &[GlobalValKey<StdTensorOp>],
            tangent_in: &[Option<LocalValId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            let StdTensorOp::$variant { axis_a, axis_b } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($lin_call(builder, tangent_in, *axis_a, *axis_b))
        }

        fn $trans(
            op: &StdTensorOp,
            emitter: &mut dyn OpEmitter<StdTensorOp>,
            cotangent_out: &[Option<LocalValId>],
            _inputs: &[ValRef<StdTensorOp>],
            _mode: &OpMode,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            let StdTensorOp::$variant { axis_a, axis_b } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($trans_call(emitter, cotangent_out, *axis_a, *axis_b))
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
            builder: &mut FragmentBuilder<StdTensorOp>,
            _primal_in: &[GlobalValKey<StdTensorOp>],
            _primal_out: &[GlobalValKey<StdTensorOp>],
            tangent_in: &[Option<LocalValId>],
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            let StdTensorOp::$variant { k } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($lin_call(builder, tangent_in, *k))
        }

        fn $trans(
            op: &StdTensorOp,
            emitter: &mut dyn OpEmitter<StdTensorOp>,
            cotangent_out: &[Option<LocalValId>],
            _inputs: &[ValRef<StdTensorOp>],
            _mode: &OpMode,
            _ctx: &mut ShapeGuardContext,
        ) -> ADRuleResult<Vec<Option<LocalValId>>> {
            let StdTensorOp::$variant { k } = op else {
                unreachable!("catalog kind mismatch")
            };
            Ok($trans_call(emitter, cotangent_out, *k))
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
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Gather(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::linearize_gather(
        builder, primal_in, tangent_in, config,
    ))
}

fn transpose_gather(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Gather(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::transpose_gather(
        emitter,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_gather_dynamic_slice_sizes(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
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
        emitter,
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
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Scatter(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::linearize_scatter(
        builder, primal_in, tangent_in, config, ctx,
    ))
}

fn transpose_scatter(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Scatter(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(indexing::transpose_scatter(
        emitter,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_slice(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Slice(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_slice(builder, tangent_in, config))
}

fn transpose_slice(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Slice(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_slice(
        emitter,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_dynamic_slice(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(indexing::transpose_dynamic_slice(
        emitter,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}

fn linearize_dynamic_update_slice(
    _op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(indexing::linearize_dynamic_update_slice(
        builder, primal_in, tangent_in, ctx,
    ))
}

fn transpose_dynamic_update_slice(
    _op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(indexing::transpose_dynamic_update_slice(
        emitter,
        cotangent_out,
        inputs,
        mode,
        ctx,
    ))
}

fn linearize_pad(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Pad(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_pad(builder, tangent_in, config))
}

fn transpose_pad(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Pad(config) = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_pad(
        emitter,
        cotangent_out,
        inputs,
        mode,
        config,
        ctx,
    ))
}

fn linearize_concatenate(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Concatenate { axis, n_inputs } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_concatenate(
        builder, primal_in, tangent_in, *axis, *n_inputs, ctx,
    ))
}

fn transpose_concatenate(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Concatenate { axis, n_inputs } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_concatenate(
        emitter,
        cotangent_out,
        inputs,
        mode,
        *axis,
        *n_inputs,
        ctx,
    ))
}

fn linearize_reverse(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Reverse { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::linearize_reverse(builder, tangent_in, axes))
}

fn transpose_reverse(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::Reverse { axes } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(structural::transpose_reverse(
        emitter,
        cotangent_out,
        mode,
        axes,
    ))
}

fn linearize_shape_of(
    _op: &StdTensorOp,
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(vec![None])
}

fn transpose_shape_of(
    _op: &StdTensorOp,
    _emitter: &mut dyn OpEmitter<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(vec![None])
}

fn linearize_dynamic_truncate(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::DynamicTruncate { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::linearize_dynamic_truncate(
        builder, primal_in, primal_out, tangent_in, *axis, ctx,
    ))
}

fn transpose_dynamic_truncate(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::DynamicTruncate { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::transpose_dynamic_truncate(
        emitter,
        cotangent_out,
        inputs,
        *axis,
    ))
}

fn linearize_pad_to_match(
    op: &StdTensorOp,
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::PadToMatch { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::linearize_pad_to_match(
        builder, primal_in, tangent_in, *axis,
    ))
}

fn transpose_pad_to_match(
    op: &StdTensorOp,
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    inputs: &[ValRef<StdTensorOp>],
    mode: &OpMode,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    let StdTensorOp::PadToMatch { axis } = op else {
        unreachable!("catalog kind mismatch")
    };
    Ok(dynamic::transpose_pad_to_match(
        emitter,
        cotangent_out,
        inputs,
        mode,
        *axis,
        ctx,
    ))
}

fn linearize_constant(
    _op: &StdTensorOp,
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _primal_out: &[GlobalValKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValId>],
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(vec![None])
}

fn transpose_constant(
    _op: &StdTensorOp,
    _emitter: &mut dyn OpEmitter<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>> {
    Ok(vec![])
}
