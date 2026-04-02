use super::builders::*;
use super::common::*;

macro_rules! define_elementwise_scalar_rule_builder {
    (
        $builder:ident,
        $ctor:ident,
        $doc_op:literal,
        $scalar_primal:path,
        $scalar_frule:path,
        |$input:ident, $primal:ident, $cotangent:ident| $rrule:expr
    ) => {
        define_unary_ad_builder!($builder, $ctor, $doc_op, generic, |builder| {
            run_elementwise_scalar_unary_ad(
                concat!($doc_op, "_ad"),
                concat!($doc_op, "_ad_pullback"),
                builder.tensor,
                $scalar_primal,
                $scalar_frule,
                |$input, $primal, $cotangent| $rrule,
            )
        });
    };
}

define_elementwise_scalar_rule_builder!(
    SqrtAdBuilder,
    sqrt_ad,
    "sqrt",
    chainrules::sqrt,
    chainrules::sqrt_frule,
    |_input, primal, cotangent| chainrules::sqrt_rrule(primal, cotangent)
);

define_elementwise_scalar_rule_builder!(
    ExpAdBuilder,
    exp_ad,
    "exp",
    chainrules::exp,
    chainrules::exp_frule,
    |_input, primal, cotangent| chainrules::exp_rrule(primal, cotangent)
);

define_elementwise_scalar_rule_builder!(
    Expm1AdBuilder,
    expm1_ad,
    "expm1",
    chainrules::expm1,
    chainrules::expm1_frule,
    |_input, primal, cotangent| chainrules::expm1_rrule(primal, cotangent)
);

define_elementwise_scalar_rule_builder!(
    LogAdBuilder,
    log_ad,
    "log",
    chainrules::log,
    chainrules::log_frule,
    |input, _primal, cotangent| chainrules::log_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    Log1pAdBuilder,
    log1p_ad,
    "log1p",
    chainrules::log1p,
    chainrules::log1p_frule,
    |input, _primal, cotangent| chainrules::log1p_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    SinAdBuilder,
    sin_ad,
    "sin",
    chainrules::sin,
    chainrules::sin_frule,
    |input, _primal, cotangent| chainrules::sin_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    CosAdBuilder,
    cos_ad,
    "cos",
    chainrules::cos,
    chainrules::cos_frule,
    |input, _primal, cotangent| chainrules::cos_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    TanhAdBuilder,
    tanh_ad,
    "tanh",
    chainrules::tanh,
    chainrules::tanh_frule,
    |_input, primal, cotangent| chainrules::tanh_rrule(primal, cotangent)
);

define_elementwise_scalar_rule_builder!(
    AsinAdBuilder,
    asin_ad,
    "asin",
    chainrules::asin,
    chainrules::asin_frule,
    |input, _primal, cotangent| chainrules::asin_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    AcosAdBuilder,
    acos_ad,
    "acos",
    chainrules::acos,
    chainrules::acos_frule,
    |input, _primal, cotangent| chainrules::acos_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    AtanAdBuilder,
    atan_ad,
    "atan",
    chainrules::atan,
    chainrules::atan_frule,
    |input, _primal, cotangent| chainrules::atan_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    SinhAdBuilder,
    sinh_ad,
    "sinh",
    chainrules::sinh,
    chainrules::sinh_frule,
    |input, _primal, cotangent| chainrules::sinh_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    CoshAdBuilder,
    cosh_ad,
    "cosh",
    chainrules::cosh,
    chainrules::cosh_frule,
    |input, _primal, cotangent| chainrules::cosh_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    AsinhAdBuilder,
    asinh_ad,
    "asinh",
    chainrules::asinh,
    chainrules::asinh_frule,
    |input, _primal, cotangent| chainrules::asinh_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    AcoshAdBuilder,
    acosh_ad,
    "acosh",
    chainrules::acosh,
    chainrules::acosh_frule,
    |input, _primal, cotangent| chainrules::acosh_rrule(input, cotangent)
);

define_elementwise_scalar_rule_builder!(
    AtanhAdBuilder,
    atanh_ad,
    "atanh",
    chainrules::atanh,
    chainrules::atanh_frule,
    |input, _primal, cotangent| chainrules::atanh_rrule(input, cotangent)
);
