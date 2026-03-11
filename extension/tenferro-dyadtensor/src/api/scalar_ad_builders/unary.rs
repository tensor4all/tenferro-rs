use super::common::*;

define_unary_ad_builder!(ExpAdBuilder, exp_ad, "exp", generic, |builder| {
    run_scalar_unary_ad(
        "exp_ad",
        "exp_ad_pullback",
        builder.tensor,
        |input| analytic_unary_primal("exp_ad_primal", AnalyticUnaryOp::Exp, input),
        |_input, primal, tangent| {
            let conj_primal =
                scalar_unary_primal("exp_ad_tangent_conj", ScalarUnaryOp::Conj, primal)?;
            scalar_binary_primal(
                "exp_ad_tangent_mul",
                ScalarBinaryOp::Mul,
                tangent,
                &conj_primal,
            )
        },
        |_input, primal, cotangent| {
            let conj_primal =
                scalar_unary_primal("exp_ad_pullback_conj", ScalarUnaryOp::Conj, primal)?;
            scalar_binary_primal(
                "exp_ad_pullback_mul",
                ScalarBinaryOp::Mul,
                cotangent,
                &conj_primal,
            )
        },
    )
});

define_unary_ad_builder!(LogAdBuilder, log_ad, "log", generic, |builder| {
    run_scalar_unary_ad(
        "log_ad",
        "log_ad_pullback",
        builder.tensor,
        |input| analytic_unary_primal("log_ad_primal", AnalyticUnaryOp::Log, input),
        |input, _primal, tangent| {
            let conj_input =
                scalar_unary_primal("log_ad_tangent_conj", ScalarUnaryOp::Conj, input)?;
            scalar_binary_primal(
                "log_ad_tangent_div",
                ScalarBinaryOp::Div,
                tangent,
                &conj_input,
            )
        },
        |input, _primal, cotangent| {
            let conj_input =
                scalar_unary_primal("log_ad_pullback_conj", ScalarUnaryOp::Conj, input)?;
            scalar_binary_primal(
                "log_ad_pullback_div",
                ScalarBinaryOp::Div,
                cotangent,
                &conj_input,
            )
        },
    )
});
