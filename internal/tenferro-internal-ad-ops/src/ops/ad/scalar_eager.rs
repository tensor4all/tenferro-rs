use super::*;
use std::marker::PhantomData;

use crate::runtime::contracts::{GenericAdRuntimeValue, RealAdRuntimeValue};
use tenferro_prims::{AnalyticUnaryOp, ScalarBinaryOp};
use tidu::{AdResult, AutodiffError, Op, Schema, SlotSchema, Value};

use crate::ops::common::compress_structured_pullback_like;
use crate::ops::scalar::primal::{analytic_unary_primal, scalar_binary_primal};

fn ad_invalid_argument(err: impl std::fmt::Display) -> AutodiffError {
    AutodiffError::InvalidArgument(err.to_string())
}

struct EdgeExpSaved<T: tenferro_algebra::Scalar> {
    input_layout: StructuredTensor<T>,
    output: StructuredTensor<T>,
}

struct EdgeAddSaved<T: tenferro_algebra::Scalar> {
    lhs_layout: StructuredTensor<T>,
    rhs_layout: StructuredTensor<T>,
}

#[derive(Clone, Copy)]
struct EdgeExpOp<T>(PhantomData<T>);

impl<T> Op<StructuredTensor<T>> for EdgeExpOp<T>
where
    T: GenericAdRuntimeValue,
{
    type SavedBackward = EdgeExpSaved<T>;
    type SavedJvp = StructuredTensor<T>;

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        let input_dense = inputs[0].to_dense().map_err(ad_invalid_argument)?;
        let output_dense =
            analytic_unary_primal("edge_exp_primal", AnalyticUnaryOp::Exp, &input_dense)
                .map_err(ad_invalid_argument)?;
        Ok(vec![StructuredTensor::from(output_dense)])
    }

    fn input_schema(&self, _inputs: &[&StructuredTensor<T>]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        inputs: &[&StructuredTensor<T>],
        outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(EdgeExpSaved {
            input_layout: inputs[0].clone(),
            output: outputs[0].clone(),
        })
    }

    fn save_for_jvp(
        &self,
        _inputs: &[&StructuredTensor<T>],
        outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(outputs[0].clone())
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let Some(grad_out) = grad_outputs[0].as_ref() else {
            return Ok(vec![None]);
        };
        let grad_dense = scalar_binary_primal(
            "edge_exp_pullback",
            ScalarBinaryOp::Mul,
            saved.output.payload(),
            grad_out.payload(),
        )
        .map_err(ad_invalid_argument)?;
        let grad =
            compress_structured_pullback_like("edge_exp_pullback", grad_dense, &saved.input_layout)
                .map_err(ad_invalid_argument)?;
        Ok(vec![Some(grad)])
    }

    fn jvp(
        &self,
        saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        let tangent_dense = tangent.to_dense().map_err(ad_invalid_argument)?;
        let jvp_dense = scalar_binary_primal(
            "edge_exp_jvp",
            ScalarBinaryOp::Mul,
            saved.payload(),
            &tangent_dense,
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![Some(StructuredTensor::from(jvp_dense))])
    }
}

#[derive(Clone, Copy)]
struct EdgeAddOp<T>(PhantomData<T>);

impl<T> Op<StructuredTensor<T>> for EdgeAddOp<T>
where
    T: GenericAdRuntimeValue,
{
    type SavedBackward = EdgeAddSaved<T>;
    type SavedJvp = ();

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        let lhs_dense = inputs[0].to_dense().map_err(ad_invalid_argument)?;
        let rhs_dense = inputs[1].to_dense().map_err(ad_invalid_argument)?;
        let output_dense = scalar_binary_primal(
            "edge_add_primal",
            ScalarBinaryOp::Add,
            &lhs_dense,
            &rhs_dense,
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![StructuredTensor::from(output_dense)])
    }

    fn input_schema(&self, _inputs: &[&StructuredTensor<T>]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
            ],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(EdgeAddSaved {
            lhs_layout: inputs[0].clone(),
            rhs_layout: inputs[1].clone(),
        })
    }

    fn save_for_jvp(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(grad_out) = grad_outputs[0].as_ref() else {
            return Ok(vec![None, None]);
        };
        let lhs_grad = if input_grad_mask[0] {
            Some(
                compress_structured_pullback_like(
                    "edge_add_pullback_lhs",
                    grad_out.payload().clone(),
                    &saved.lhs_layout,
                )
                .map_err(ad_invalid_argument)?,
            )
        } else {
            None
        };
        let rhs_grad = if input_grad_mask[1] {
            Some(
                compress_structured_pullback_like(
                    "edge_add_pullback_rhs",
                    grad_out.payload().clone(),
                    &saved.rhs_layout,
                )
                .map_err(ad_invalid_argument)?,
            )
        } else {
            None
        };
        Ok(vec![lhs_grad, rhs_grad])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        match (tangents[0].as_ref(), tangents[1].as_ref()) {
            (None, None) => Ok(vec![None]),
            (Some(lhs), None) => Ok(vec![Some(StructuredTensor::from(
                lhs.to_dense().map_err(ad_invalid_argument)?,
            ))]),
            (None, Some(rhs)) => Ok(vec![Some(StructuredTensor::from(
                rhs.to_dense().map_err(ad_invalid_argument)?,
            ))]),
            (Some(lhs), Some(rhs)) => {
                let lhs_dense = lhs.to_dense().map_err(ad_invalid_argument)?;
                let rhs_dense = rhs.to_dense().map_err(ad_invalid_argument)?;
                let tangent_dense = scalar_binary_primal(
                    "edge_add_jvp",
                    ScalarBinaryOp::Add,
                    &lhs_dense,
                    &rhs_dense,
                )
                .map_err(ad_invalid_argument)?;
                Ok(vec![Some(StructuredTensor::from(tangent_dense))])
            }
        }
    }
}

fn can_use_edge_unary_reverse<T>(tensor: &AdTensor<T>) -> bool
where
    T: GenericAdRuntimeValue,
{
    tensor.structured_tangent().is_none() && tensor.reverse_edge_value().is_some()
}

fn can_use_edge_binary_reverse<T>(lhs: &AdTensor<T>, rhs: &AdTensor<T>) -> bool
where
    T: GenericAdRuntimeValue,
{
    if lhs.structured_tangent().is_some() || rhs.structured_tangent().is_some() {
        return false;
    }
    let lhs_ok = !lhs.requires_grad() || lhs.reverse_edge_value().is_some();
    let rhs_ok = !rhs.requires_grad() || rhs.reverse_edge_value().is_some();
    lhs_ok && rhs_ok && (lhs.requires_grad() || rhs.requires_grad())
}

fn edge_exp<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let input = tensor
        .reverse_edge_value()
        .ok_or(Error::UnsupportedAdOp { op: "edge_exp" })?;
    let output = EdgeExpOp::<T>(PhantomData)
        .apply_one(&[input.as_ref()])
        .map_err(Error::from)?;
    wrap_reverse_edge_output(output)
}

fn edge_add<T>(lhs: &AdTensor<T>, rhs: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
{
    let lhs_edge = if lhs.requires_grad() {
        Some(lhs.reverse_edge_value().ok_or(Error::UnsupportedAdOp {
            op: "edge_add(lhs)",
        })?)
    } else {
        None
    };
    let rhs_edge = if rhs.requires_grad() {
        Some(rhs.reverse_edge_value().ok_or(Error::UnsupportedAdOp {
            op: "edge_add(rhs)",
        })?)
    } else {
        None
    };
    let lhs_plain = (!lhs.requires_grad()).then(|| Value::new(lhs.structured_primal().clone()));
    let rhs_plain = (!rhs.requires_grad()).then(|| Value::new(rhs.structured_primal().clone()));

    let lhs_value: &Value<StructuredTensor<T>> = match lhs_edge.as_ref() {
        Some(value) => value.as_ref(),
        None => lhs_plain.as_ref().expect("lhs plain value"),
    };
    let rhs_value: &Value<StructuredTensor<T>> = match rhs_edge.as_ref() {
        Some(value) => value.as_ref(),
        None => rhs_plain.as_ref().expect("rhs plain value"),
    };

    let output = EdgeAddOp::<T>(PhantomData)
        .apply_one(&[lhs_value, rhs_value])
        .map_err(Error::from)?;
    wrap_reverse_edge_output(output)
}

macro_rules! define_scalar_unary_eager_ad_fn {
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, generic) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: GenericAdRuntimeValue,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, real) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: RealAdRuntimeValue,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
}

macro_rules! define_scalar_binary_eager_ad_fn {
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, generic) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::ad::", stringify!($fn_name), "(&a, &b)?;\n```")]
        pub fn $fn_name<T>(lhs: &AdTensor<T>, rhs: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: GenericAdRuntimeValue,
        {
            super::super::$builder_fn(lhs, rhs).run()
        }
    };
    ($fn_name:ident, $builder_fn:ident, $doc_op:literal, real) => {
        #[doc = concat!("Eager AD `", $doc_op, "`.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::ad::", stringify!($fn_name), "(&a, &b)?;\n```")]
        pub fn $fn_name<T>(lhs: &AdTensor<T>, rhs: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: RealAdRuntimeValue,
        {
            super::super::$builder_fn(lhs, rhs).run()
        }
    };
}

macro_rules! define_scalar_reduction_eager_ad_fn {
    ($fn_name:ident, $builder_fn:ident, $doc_label:literal, generic) => {
        #[doc = concat!("Eager AD full `", $doc_label, "` reduction.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: GenericAdRuntimeValue,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
    ($fn_name:ident, $builder_fn:ident, $doc_label:literal, real) => {
        #[doc = concat!("Eager AD full `", $doc_label, "` reduction.")]
        #[doc = ""]
        #[doc = concat!("Equivalent to `crate::", stringify!($builder_fn), "(...).run()`.")]
        #[doc = ""]
        #[doc = "# Examples"]
        #[doc = ""]
        #[doc = concat!("```ignore\nlet out = tenferro::ad::", stringify!($fn_name), "(&x)?;\n```")]
        pub fn $fn_name<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
        where
            T: RealAdRuntimeValue,
        {
            super::super::$builder_fn(tensor).run()
        }
    };
}

define_scalar_unary_eager_ad_fn!(sqrt, sqrt_ad, "sqrt", generic);
define_scalar_unary_eager_ad_fn!(expm1, expm1_ad, "expm1", generic);
define_scalar_unary_eager_ad_fn!(log, log_ad, "log", generic);
define_scalar_unary_eager_ad_fn!(log1p, log1p_ad, "log1p", generic);
define_scalar_unary_eager_ad_fn!(sin, sin_ad, "sin", generic);
define_scalar_unary_eager_ad_fn!(cos, cos_ad, "cos", generic);
define_scalar_unary_eager_ad_fn!(tanh, tanh_ad, "tanh", generic);
define_scalar_unary_eager_ad_fn!(asin, asin_ad, "asin", generic);
define_scalar_unary_eager_ad_fn!(acos, acos_ad, "acos", generic);
define_scalar_unary_eager_ad_fn!(atan, atan_ad, "atan", generic);
define_scalar_unary_eager_ad_fn!(sinh, sinh_ad, "sinh", generic);
define_scalar_unary_eager_ad_fn!(cosh, cosh_ad, "cosh", generic);
define_scalar_unary_eager_ad_fn!(asinh, asinh_ad, "asinh", generic);
define_scalar_unary_eager_ad_fn!(acosh, acosh_ad, "acosh", generic);
define_scalar_unary_eager_ad_fn!(atanh, atanh_ad, "atanh", generic);

define_scalar_binary_eager_ad_fn!(atan2, atan2_ad, "atan2", real);
define_scalar_binary_eager_ad_fn!(pow, pow_ad, "pow", generic);
define_scalar_binary_eager_ad_fn!(hypot, hypot_ad, "hypot", real);

define_scalar_reduction_eager_ad_fn!(mean, mean_ad, "mean", generic);
define_scalar_reduction_eager_ad_fn!(var, var_ad, "variance", real);
define_scalar_reduction_eager_ad_fn!(std, std_ad, "standard deviation", real);

#[doc = "Eager AD `exp`."]
#[doc = ""]
#[doc = "Equivalent to `crate::exp_ad(...).run()`, but uses the new edge-based reverse path when possible."]
pub fn exp<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
{
    if can_use_edge_unary_reverse(tensor) {
        return edge_exp(tensor);
    }
    super::super::exp_ad(tensor).run()
}

#[doc = "Eager AD `add`."]
#[doc = ""]
#[doc = "Equivalent to `crate::add_ad(...).run()`, but uses the new edge-based reverse path when possible."]
pub fn add<T>(lhs: &AdTensor<T>, rhs: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: GenericAdRuntimeValue,
{
    if can_use_edge_binary_reverse(lhs, rhs) {
        return edge_add(lhs, rhs);
    }
    super::super::add_ad(lhs, rhs).run()
}
