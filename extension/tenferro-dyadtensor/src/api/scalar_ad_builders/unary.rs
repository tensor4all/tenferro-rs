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
    SqrtAdBuilder,
    sqrt_ad,
    "sqrt",
    "sqrt",
    AnalyticUnaryOp::Sqrt,
    |_input, primal| {
        let inv = scalar_unary_primal("sqrt_ad_factor_inv", ScalarUnaryOp::Reciprocal, primal)?;
        let two = two_tensor_like(primal)?;
        scalar_binary_primal("sqrt_ad_factor_half", ScalarBinaryOp::Div, &inv, &two)
    }
);

define_conj_factor_unary_builder!(
    ExpAdBuilder,
    exp_ad,
    "exp",
    "exp",
    AnalyticUnaryOp::Exp,
    |_input, primal| Ok::<Tensor<T>, Error>(primal.clone())
);

define_conj_factor_unary_builder!(
    Expm1AdBuilder,
    expm1_ad,
    "expm1",
    "expm1",
    AnalyticUnaryOp::Expm1,
    |input, _primal| analytic_unary_primal("expm1_ad_factor", AnalyticUnaryOp::Exp, input)
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
    Log1pAdBuilder,
    log1p_ad,
    "log1p",
    "log1p",
    AnalyticUnaryOp::Log1p,
    |input, _primal| {
        let one = one_tensor_like(input)?;
        let denom = scalar_binary_primal("log1p_ad_factor_add", ScalarBinaryOp::Add, &one, input)?;
        scalar_unary_primal("log1p_ad_factor_recip", ScalarUnaryOp::Reciprocal, &denom)
    }
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

define_conj_factor_unary_builder!(
    AsinAdBuilder,
    asin_ad,
    "asin",
    "asin",
    AnalyticUnaryOp::Asin,
    |input, _primal| {
        let sq = scalar_unary_primal("asin_ad_factor_sq", ScalarUnaryOp::Square, input)?;
        let one = one_tensor_like(input)?;
        let inside = scalar_binary_primal("asin_ad_factor_inside", ScalarBinaryOp::Sub, &one, &sq)?;
        let root = analytic_unary_primal("asin_ad_factor_sqrt", AnalyticUnaryOp::Sqrt, &inside)?;
        scalar_unary_primal("asin_ad_factor_recip", ScalarUnaryOp::Reciprocal, &root)
    }
);

define_conj_factor_unary_builder!(
    AcosAdBuilder,
    acos_ad,
    "acos",
    "acos",
    AnalyticUnaryOp::Acos,
    |input, _primal| {
        let sq = scalar_unary_primal("acos_ad_factor_sq", ScalarUnaryOp::Square, input)?;
        let one = one_tensor_like(input)?;
        let inside = scalar_binary_primal("acos_ad_factor_inside", ScalarBinaryOp::Sub, &one, &sq)?;
        let root = analytic_unary_primal("acos_ad_factor_sqrt", AnalyticUnaryOp::Sqrt, &inside)?;
        let recip = scalar_unary_primal("acos_ad_factor_recip", ScalarUnaryOp::Reciprocal, &root)?;
        scalar_unary_primal("acos_ad_factor_neg", ScalarUnaryOp::Neg, &recip)
    }
);

define_conj_factor_unary_builder!(
    AtanAdBuilder,
    atan_ad,
    "atan",
    "atan",
    AnalyticUnaryOp::Atan,
    |input, _primal| {
        let sq = scalar_unary_primal("atan_ad_factor_sq", ScalarUnaryOp::Square, input)?;
        let one = one_tensor_like(input)?;
        let denom = scalar_binary_primal("atan_ad_factor_denom", ScalarBinaryOp::Add, &one, &sq)?;
        scalar_unary_primal("atan_ad_factor_recip", ScalarUnaryOp::Reciprocal, &denom)
    }
);

define_conj_factor_unary_builder!(
    SinhAdBuilder,
    sinh_ad,
    "sinh",
    "sinh",
    AnalyticUnaryOp::Sinh,
    |input, _primal| analytic_unary_primal("sinh_ad_factor", AnalyticUnaryOp::Cosh, input)
);

define_conj_factor_unary_builder!(
    CoshAdBuilder,
    cosh_ad,
    "cosh",
    "cosh",
    AnalyticUnaryOp::Cosh,
    |input, _primal| analytic_unary_primal("cosh_ad_factor", AnalyticUnaryOp::Sinh, input)
);

define_conj_factor_unary_builder!(
    AsinhAdBuilder,
    asinh_ad,
    "asinh",
    "asinh",
    AnalyticUnaryOp::Asinh,
    |input, _primal| {
        let sq = scalar_unary_primal("asinh_ad_factor_sq", ScalarUnaryOp::Square, input)?;
        let one = one_tensor_like(input)?;
        let inside =
            scalar_binary_primal("asinh_ad_factor_inside", ScalarBinaryOp::Add, &one, &sq)?;
        let root = analytic_unary_primal("asinh_ad_factor_sqrt", AnalyticUnaryOp::Sqrt, &inside)?;
        scalar_unary_primal("asinh_ad_factor_recip", ScalarUnaryOp::Reciprocal, &root)
    }
);

define_conj_factor_unary_builder!(
    AcoshAdBuilder,
    acosh_ad,
    "acosh",
    "acosh",
    AnalyticUnaryOp::Acosh,
    |input, _primal| {
        let sq = scalar_unary_primal("acosh_ad_factor_sq", ScalarUnaryOp::Square, input)?;
        let one = one_tensor_like(input)?;
        let inside =
            scalar_binary_primal("acosh_ad_factor_inside", ScalarBinaryOp::Sub, &sq, &one)?;
        let root = analytic_unary_primal("acosh_ad_factor_sqrt", AnalyticUnaryOp::Sqrt, &inside)?;
        scalar_unary_primal("acosh_ad_factor_recip", ScalarUnaryOp::Reciprocal, &root)
    }
);

define_conj_factor_unary_builder!(
    AtanhAdBuilder,
    atanh_ad,
    "atanh",
    "atanh",
    AnalyticUnaryOp::Atanh,
    |input, _primal| {
        let sq = scalar_unary_primal("atanh_ad_factor_sq", ScalarUnaryOp::Square, input)?;
        let one = one_tensor_like(input)?;
        let denom = scalar_binary_primal("atanh_ad_factor_denom", ScalarBinaryOp::Sub, &one, &sq)?;
        scalar_unary_primal("atanh_ad_factor_recip", ScalarUnaryOp::Reciprocal, &denom)
    }
);
