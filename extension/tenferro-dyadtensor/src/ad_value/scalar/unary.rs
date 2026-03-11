use core::ops::Neg;

use chainrules_scalarops as scalarops;

use crate::reverse_tape;

use super::super::core::AdValue;
use super::shared::fresh_ad_scalar_node_id;
use super::AdScalar;

fn unary_ad_scalar_op<T, P, F, R>(
    value: AdScalar<T>,
    op_name: &'static str,
    primal_rule: P,
    frule: F,
    rrule: R,
) -> AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
    P: Fn(T) -> T,
    F: Fn(T, T) -> (T, T),
    R: Fn(T, T, T) -> T + 'static,
{
    match value.into_value() {
        AdValue::Primal(primal) => AdScalar::new_primal(primal_rule(primal)),
        AdValue::Forward { primal, tangent } => {
            let (primal, tangent) = frule(primal, tangent);
            AdScalar::new_forward(primal, tangent)
        }
        AdValue::Reverse {
            primal: input_primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let (output_primal, tangent) = match tangent {
                Some(tangent) => {
                    let (primal, tangent) = frule(input_primal, tangent);
                    (primal, Some(tangent))
                }
                None => (primal_rule(input_primal), None),
            };
            let output_node = fresh_ad_scalar_node_id();
            reverse_tape::register_scalar_rule(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    Ok(vec![(
                        input_node,
                        rrule(input_primal, output_primal, *cotangent),
                    )])
                }),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdScalar::new_reverse(output_primal, output_node, tape, tangent)
        }
    }
}

impl<T> AdScalar<T> {
    /// Applies scalar conjugation with AD propagation.
    pub fn conj(self) -> Self
    where
        T: scalarops::ScalarAd + 'static,
    {
        unary_ad_scalar_op(
            self,
            "conj",
            scalarops::conj,
            scalarops::conj_frule,
            |_input, _output, cotangent| scalarops::conj_rrule(cotangent),
        )
    }

    /// Applies scalar square-root with AD propagation.
    pub fn sqrt(self) -> Self
    where
        T: scalarops::ScalarAd + 'static,
    {
        unary_ad_scalar_op(
            self,
            "sqrt",
            scalarops::sqrt,
            scalarops::sqrt_frule,
            |_input, output, cotangent| scalarops::sqrt_rrule(output, cotangent),
        )
    }

    /// Applies scalar real-exponent power with AD propagation.
    pub fn powf(self, exponent: <T as scalarops::ScalarAd>::Real) -> Self
    where
        T: scalarops::ScalarAd + 'static,
        <T as scalarops::ScalarAd>::Real: 'static,
    {
        unary_ad_scalar_op(
            self,
            "powf",
            move |primal| scalarops::powf(primal, exponent),
            move |primal, tangent| scalarops::powf_frule(primal, exponent, tangent),
            move |input, _output, cotangent| scalarops::powf_rrule(input, exponent, cotangent),
        )
    }

    /// Applies scalar integer-exponent power with AD propagation.
    pub fn powi(self, exponent: i32) -> Self
    where
        T: scalarops::ScalarAd + 'static,
    {
        unary_ad_scalar_op(
            self,
            "powi",
            move |primal| scalarops::powi(primal, exponent),
            move |primal, tangent| scalarops::powi_frule(primal, exponent, tangent),
            move |input, _output, cotangent| scalarops::powi_rrule(input, exponent, cotangent),
        )
    }
}

impl<T> Neg for AdScalar<T>
where
    T: scalarops::ScalarAd + Neg<Output = T> + 'static,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        unary_ad_scalar_op(
            self,
            "neg",
            |primal| -primal,
            |primal, tangent| (-primal, -tangent),
            |_input, _output, cotangent| -cotangent,
        )
    }
}
