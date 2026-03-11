use super::common::*;

macro_rules! define_conj_factor_unary_builder {
    (
        $builder:ident,
        $ctor:ident,
        $doc_op:literal,
        $ad_name:literal,
        $primal_op:expr,
        |$input:ident, $primal:ident| $factor:expr
    ) => {
        define_unary_ad_builder!($builder, $ctor, $doc_op, generic, |builder| {
            run_scalar_unary_ad(
                concat!($ad_name, "_ad"),
                concat!($ad_name, "_ad_pullback"),
                builder.tensor,
                |input| analytic_unary_primal(concat!($ad_name, "_ad_primal"), $primal_op, input),
                |$input, $primal, tangent| {
                    let factor = $factor?;
                    mul_with_conj_factor(concat!($ad_name, "_ad_tangent"), tangent, &factor)
                },
                |$input, $primal, cotangent| {
                    let factor = $factor?;
                    mul_with_conj_factor(concat!($ad_name, "_ad_pullback"), cotangent, &factor)
                },
            )
        });
    };
}

define_conj_factor_unary_builder!(
    ExpAdBuilder,
    exp_ad,
    "exp",
    "exp",
    AnalyticUnaryOp::Exp,
    |_input, primal| Ok::<Tensor<T>, Error>(primal.clone())
);

define_conj_factor_unary_builder!(
    LogAdBuilder,
    log_ad,
    "log",
    "log",
    AnalyticUnaryOp::Log,
    |input, _primal| scalar_unary_primal("log_ad_factor", ScalarUnaryOp::Reciprocal, input)
);

define_conj_factor_unary_builder!(
    SinAdBuilder,
    sin_ad,
    "sin",
    "sin",
    AnalyticUnaryOp::Sin,
    |input, _primal| analytic_unary_primal("sin_ad_factor", AnalyticUnaryOp::Cos, input)
);

define_conj_factor_unary_builder!(
    CosAdBuilder,
    cos_ad,
    "cos",
    "cos",
    AnalyticUnaryOp::Cos,
    |input, _primal| {
        let sin = analytic_unary_primal("cos_ad_factor_sin", AnalyticUnaryOp::Sin, input)?;
        scalar_unary_primal("cos_ad_factor_neg", ScalarUnaryOp::Neg, &sin)
    }
);

define_conj_factor_unary_builder!(
    TanhAdBuilder,
    tanh_ad,
    "tanh",
    "tanh",
    AnalyticUnaryOp::Tanh,
    |_input, primal| {
        let one = one_tensor_like(primal)?;
        let sq = scalar_unary_primal("tanh_ad_factor_sq", ScalarUnaryOp::Square, primal)?;
        scalar_binary_primal("tanh_ad_factor_sub", ScalarBinaryOp::Sub, &one, &sq)
    }
);
