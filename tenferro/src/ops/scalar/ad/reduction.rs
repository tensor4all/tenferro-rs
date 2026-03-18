use super::common::*;

define_unary_ad_builder!(MeanAdBuilder, mean_ad, "mean", generic, |builder| {
    run_scalar_unary_ad(
        "mean_ad",
        "mean_ad_pullback",
        builder.tensor,
        |input| scalar_full_reduction_primal("mean_ad_primal", ScalarReductionOp::Mean, input),
        |_input, _primal, tangent| {
            scalar_full_reduction_primal("mean_ad_tangent", ScalarReductionOp::Mean, tangent)
        },
        |input, _primal, cotangent| {
            let scalar = scalar_from_rank0_tensor(cotangent, "mean_ad")?;
            let denom = scalar_from_usize::<T>(input.len())?;
            let payload = broadcast_scalar_like(scalar / denom, input)?;
            Ok(payload)
        },
    )
});

define_unary_ad_builder!(VarAdBuilder, var_ad, "var", real, |builder| {
    run_scalar_unary_ad(
        "var_ad",
        "var_ad_pullback",
        builder.tensor,
        |input| analytic_full_reduction_primal("var_ad_primal", AnalyticReductionOp::Var, input),
        |input, _primal, tangent| variance_reduction_tangent("var_ad_tangent", input, tangent),
        |input, _primal, cotangent| {
            variance_reduction_pullback("var_ad_pullback", input, cotangent)
        },
    )
});

define_unary_ad_builder!(StdAdBuilder, std_ad, "std", real, |builder| {
    run_scalar_unary_ad(
        "std_ad",
        "std_ad_pullback",
        builder.tensor,
        |input| analytic_full_reduction_primal("std_ad_primal", AnalyticReductionOp::Std, input),
        |input, primal, tangent| {
            if scalar_from_rank0_tensor(primal, "std_ad")? == T::zero() {
                return broadcast_scalar_like(T::zero(), primal);
            }
            let var_tangent = variance_reduction_tangent("std_ad_var_tangent", input, tangent)?;
            let two = two_tensor_like(primal)?;
            let denom = scalar_binary_primal("std_ad_denom", ScalarBinaryOp::Mul, primal, &two)?;
            scalar_binary_primal("std_ad_tangent", ScalarBinaryOp::Div, &var_tangent, &denom)
        },
        |input, primal, cotangent| {
            let centered = centered_input_tensor("std_ad_centered", input)?;
            let cotangent_scalar = scalar_from_rank0_tensor(cotangent, "std_ad")?;
            let primal_scalar = scalar_from_rank0_tensor(primal, "std_ad")?;
            if primal_scalar == T::zero() {
                return broadcast_scalar_like(T::zero(), input);
            }
            let coeff = cotangent_scalar / (scalar_from_usize::<T>(input.len())? * primal_scalar);
            let coeff_tensor = broadcast_scalar_like(coeff, input)?;
            scalar_binary_primal(
                "std_ad_pullback",
                ScalarBinaryOp::Mul,
                &centered,
                &coeff_tensor,
            )
        },
    )
});
