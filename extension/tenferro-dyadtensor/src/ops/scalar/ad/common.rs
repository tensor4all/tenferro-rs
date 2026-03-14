pub(super) use chainrules_scalarops::ScalarAd;
pub(super) use num_traits::NumCast;
pub(super) use tenferro_algebra::Scalar;
pub(super) use tenferro_prims::{
    AnalyticBinaryOp, AnalyticReductionOp, AnalyticUnaryOp, ScalarBinaryOp, ScalarReductionOp,
    ScalarUnaryOp,
};
pub(super) use tenferro_tensor::Tensor;

pub(super) use crate::{tape, AdTensor, Error, Result};

pub(super) use crate::ops::common::{
    broadcast_scalar_like, collect_reverse_input_specs, compress_structured_pullback_like,
    dense_input_snapshot_in_runtime, has_any_tangent, has_forward, scalar_from_rank0_tensor,
    wrap_same_type_dense_ad_output,
};
pub(super) use crate::ops::scalar::primal::{
    analytic_binary_primal, analytic_full_reduction_primal, analytic_unary_primal,
    scalar_binary_primal, scalar_full_reduction_primal, scalar_unary_primal,
};
pub(super) use crate::runtime::contracts::{GenericAdRuntimeValue, RealAdRuntimeValue};

macro_rules! define_unary_ad_builder {
    ($builder:ident, $ctor:ident, $doc_op:literal, generic, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&x).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            tensor: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&x).run()?;\n```")]
        pub fn $ctor<'a, T>(tensor: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            $builder { tensor }
        }
    };
    ($builder:ident, $ctor:ident, $doc_op:literal, real, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&x).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            tensor: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&x).run()?;\n```")]
        pub fn $ctor<'a, T>(tensor: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            $builder { tensor }
        }
    };
}

pub(super) use define_unary_ad_builder;

macro_rules! define_binary_ad_builder {
    ($builder:ident, $ctor:ident, $doc_op:literal, generic, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            lhs: &'a AdTensor<T>,
            rhs: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub fn $ctor<'a, T>(lhs: &'a AdTensor<T>, rhs: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: GenericAdRuntimeValue,
        {
            $builder { lhs, rhs }
        }
    };
    ($builder:ident, $ctor:ident, $doc_op:literal, real, |$self_ident:ident| $run:block) => {
        #[doc = concat!("Builder for AD `", $doc_op, "` on tensors.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub struct $builder<'a, T: Scalar> {
            lhs: &'a AdTensor<T>,
            rhs: &'a AdTensor<T>,
        }

        impl<'a, T> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            #[doc = concat!("Executes AD `", $doc_op, "`.")]
            #[doc = ""]
            #[doc = "# Examples"]
            #[doc = ""]
            #[doc = "```ignore\nlet out = builder.run()?;\n```"]
            pub fn run(self) -> Result<AdTensor<T>> {
                let $self_ident = self;
                $run
            }
        }

        #[doc = concat!("Creates a builder for AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro_dyadtensor::", stringify!($ctor), "(&a, &b).run()?;\n```")]
        pub fn $ctor<'a, T>(lhs: &'a AdTensor<T>, rhs: &'a AdTensor<T>) -> $builder<'a, T>
        where
            T: RealAdRuntimeValue,
        {
            $builder { lhs, rhs }
        }
    };
}

pub(super) use define_binary_ad_builder;

pub(super) fn scalar_from_usize<T: ScalarAd>(value: usize) -> Result<T> {
    let Some(real) = <T::Real as NumCast>::from(value) else {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "cannot represent reduction size {value} for scalar type {}",
                std::any::type_name::<T>()
            ),
        });
    };
    Ok(T::from_real(real))
}

pub(super) fn two_tensor_like<T>(template: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Scalar + ScalarAd + Copy,
{
    broadcast_scalar_like(T::one() + T::one(), template)
}

pub(super) fn one_tensor_like<T>(template: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Scalar + ScalarAd + Copy,
{
    broadcast_scalar_like(T::one(), template)
}

pub(super) fn mul_with_conj_factor<T>(
    op_name: &'static str,
    value: &Tensor<T>,
    factor: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let conj_factor = scalar_unary_primal(op_name, ScalarUnaryOp::Conj, factor)?;
    scalar_binary_primal(op_name, ScalarBinaryOp::Mul, value, &conj_factor)
}

pub(super) fn centered_input_tensor<T>(
    op_name: &'static str,
    input: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let mean = scalar_full_reduction_primal(op_name, ScalarReductionOp::Mean, input)?;
    let mean_scalar = scalar_from_rank0_tensor(&mean, op_name)?;
    let mean_broadcast = broadcast_scalar_like(mean_scalar, input)?;
    scalar_binary_primal(op_name, ScalarBinaryOp::Sub, input, &mean_broadcast)
}

pub(super) fn variance_reduction_tangent<T>(
    op_name: &'static str,
    input: &Tensor<T>,
    tangent: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let centered = centered_input_tensor(op_name, input)?;
    let centered_dt = scalar_binary_primal(op_name, ScalarBinaryOp::Mul, &centered, tangent)?;
    let mean_dt = scalar_full_reduction_primal(op_name, ScalarReductionOp::Mean, &centered_dt)?;
    let two = two_tensor_like(&mean_dt)?;
    scalar_binary_primal(op_name, ScalarBinaryOp::Mul, &mean_dt, &two)
}

pub(super) fn variance_reduction_pullback<T>(
    op_name: &'static str,
    input: &Tensor<T>,
    cotangent: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let centered = centered_input_tensor(op_name, input)?;
    let cotangent_scalar = scalar_from_rank0_tensor(cotangent, op_name)?;
    let coeff = (T::one() + T::one()) * cotangent_scalar / scalar_from_usize::<T>(input.len())?;
    let coeff_tensor = broadcast_scalar_like(coeff, input)?;
    scalar_binary_primal(op_name, ScalarBinaryOp::Mul, &centered, &coeff_tensor)
}

pub(super) fn run_scalar_unary_ad<T, FPrimal, FTangent, FPullback>(
    op_name: &'static str,
    _pullback_op_name: &'static str,
    input: &AdTensor<T>,
    primal_fn: FPrimal,
    tangent_fn: FTangent,
    pullback_fn: FPullback,
) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
    FPrimal: Fn(&Tensor<T>) -> Result<Tensor<T>>,
    FTangent: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FPullback: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>> + 'static,
{
    let operands = [input];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
    let (input_primal, input_tangent) =
        dense_input_snapshot_in_runtime(op_name, input, needs_tangent)?;
    let primal = primal_fn(&input_primal)?;
    let tangent = if needs_tangent {
        let tangent = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized tangent"),
        })?;
        Some(tangent_fn(&input_primal, &primal, &tangent)?)
    } else {
        None
    };

    let out = wrap_same_type_dense_ad_output(op_name, &operands, primal.clone(), tangent)?;

    if let Some((node, tape)) = out.reverse_handle() {
        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();

        tape::register_rule::<T>(
            &tape,
            node,
            Box::new(move |cotangent| {
                let grad = pullback_fn(&input_primal, &primal, cotangent.payload())?;
                let Some(spec) = &input_spec else {
                    return Ok(Vec::new());
                };
                let grad = compress_structured_pullback_like(op_name, grad, &spec.layout)?;
                Ok(vec![(spec.node, grad)])
            }),
        );
    }

    Ok(out)
}

pub(super) fn run_scalar_binary_ad<T, FPrimal, FTangent, FPullback>(
    op_name: &'static str,
    _pullback_op_name: &'static str,
    lhs: &AdTensor<T>,
    rhs: &AdTensor<T>,
    primal_fn: FPrimal,
    tangent_fn: FTangent,
    pullback_fn: FPullback,
) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
    FPrimal: Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FTangent: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FPullback: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)>
        + 'static,
{
    let operands = [lhs, rhs];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
    let (lhs_snapshot, rhs_snapshot) = (
        dense_input_snapshot_in_runtime(op_name, lhs, needs_tangent)?,
        dense_input_snapshot_in_runtime(op_name, rhs, needs_tangent)?,
    );
    let (lhs_primal, lhs_tangent) = lhs_snapshot;
    let (rhs_primal, rhs_tangent) = rhs_snapshot;
    let primal = primal_fn(&lhs_primal, &rhs_primal)?;
    let tangent = if needs_tangent {
        let lhs_tangent = lhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized lhs tangent"),
        })?;
        let rhs_tangent = rhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized rhs tangent"),
        })?;
        Some(tangent_fn(
            &lhs_primal,
            &rhs_primal,
            &primal,
            &lhs_tangent,
            &rhs_tangent,
        )?)
    } else {
        None
    };

    let out = wrap_same_type_dense_ad_output(op_name, &operands, primal.clone(), tangent)?;

    if let Some((node, tape)) = out.reverse_handle() {
        let reverse_specs = collect_reverse_input_specs(&operands);

        tape::register_rule::<T>(
            &tape,
            node,
            Box::new(move |cotangent| {
                let (grad_lhs, grad_rhs) =
                    pullback_fn(&lhs_primal, &rhs_primal, &primal, cotangent.payload())?;
                let mut input_grads = Vec::new();
                if let Some(spec) = &reverse_specs[0] {
                    let grad = compress_structured_pullback_like(op_name, grad_lhs, &spec.layout)?;
                    input_grads.push((spec.node, grad));
                }
                if let Some(spec) = &reverse_specs[1] {
                    let grad = compress_structured_pullback_like(op_name, grad_rhs, &spec.layout)?;
                    input_grads.push((spec.node, grad));
                }
                Ok(input_grads)
            }),
        );
    }

    Ok(out)
}
