use super::common::*;

define_binary_ad_builder!(AddAdBuilder, add_ad, "add", generic, |builder| {
    run_scalar_binary_ad(
        "add_ad",
        "add_ad_pullback",
        builder.lhs,
        builder.rhs,
        |lhs, rhs| scalar_binary_primal("add_ad_primal", ScalarBinaryOp::Add, lhs, rhs),
        |_lhs, _rhs, _primal, lhs_tangent, rhs_tangent| {
            scalar_binary_primal(
                "add_ad_tangent",
                ScalarBinaryOp::Add,
                lhs_tangent,
                rhs_tangent,
            )
        },
        |_lhs, _rhs, _primal, cotangent| Ok((cotangent.clone(), cotangent.clone())),
    )
});

define_binary_ad_builder!(Atan2AdBuilder, atan2_ad, "atan2", real, |builder| {
    run_scalar_binary_ad(
        "atan2_ad",
        "atan2_ad_pullback",
        builder.lhs,
        builder.rhs,
        |lhs, rhs| analytic_binary_primal("atan2_ad_primal", AnalyticBinaryOp::Atan2, lhs, rhs),
        |lhs, rhs, _primal, lhs_tangent, rhs_tangent| {
            let lhs_sq = scalar_unary_primal("atan2_ad_lhs_sq", ScalarUnaryOp::Square, lhs)?;
            let rhs_sq = scalar_unary_primal("atan2_ad_rhs_sq", ScalarUnaryOp::Square, rhs)?;
            let denom =
                scalar_binary_primal("atan2_ad_denom", ScalarBinaryOp::Add, &lhs_sq, &rhs_sq)?;
            let lhs_coeff =
                scalar_binary_primal("atan2_ad_lhs_coeff", ScalarBinaryOp::Div, rhs, &denom)?;
            let neg_lhs = scalar_unary_primal("atan2_ad_neg_lhs", ScalarUnaryOp::Neg, lhs)?;
            let rhs_coeff =
                scalar_binary_primal("atan2_ad_rhs_coeff", ScalarBinaryOp::Div, &neg_lhs, &denom)?;
            let lhs_term = scalar_binary_primal(
                "atan2_ad_lhs_term",
                ScalarBinaryOp::Mul,
                lhs_tangent,
                &lhs_coeff,
            )?;
            let rhs_term = scalar_binary_primal(
                "atan2_ad_rhs_term",
                ScalarBinaryOp::Mul,
                rhs_tangent,
                &rhs_coeff,
            )?;
            scalar_binary_primal(
                "atan2_ad_tangent_sum",
                ScalarBinaryOp::Add,
                &lhs_term,
                &rhs_term,
            )
        },
        |lhs, rhs, _primal, cotangent| {
            let lhs_sq = scalar_unary_primal("atan2_ad_pb_lhs_sq", ScalarUnaryOp::Square, lhs)?;
            let rhs_sq = scalar_unary_primal("atan2_ad_pb_rhs_sq", ScalarUnaryOp::Square, rhs)?;
            let denom =
                scalar_binary_primal("atan2_ad_pb_denom", ScalarBinaryOp::Add, &lhs_sq, &rhs_sq)?;
            let lhs_coeff =
                scalar_binary_primal("atan2_ad_pb_lhs_coeff", ScalarBinaryOp::Div, rhs, &denom)?;
            let neg_lhs = scalar_unary_primal("atan2_ad_pb_neg_lhs", ScalarUnaryOp::Neg, lhs)?;
            let rhs_coeff = scalar_binary_primal(
                "atan2_ad_pb_rhs_coeff",
                ScalarBinaryOp::Div,
                &neg_lhs,
                &denom,
            )?;
            let lhs_grad = scalar_binary_primal(
                "atan2_ad_pb_lhs_grad",
                ScalarBinaryOp::Mul,
                cotangent,
                &lhs_coeff,
            )?;
            let rhs_grad = scalar_binary_primal(
                "atan2_ad_pb_rhs_grad",
                ScalarBinaryOp::Mul,
                cotangent,
                &rhs_coeff,
            )?;
            Ok((lhs_grad, rhs_grad))
        },
    )
});
