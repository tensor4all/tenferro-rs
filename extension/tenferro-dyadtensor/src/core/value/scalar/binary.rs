use core::ops::{Add, Div, Mul, Sub};

use chainrules_scalarops as scalarops;

use crate::{tape, Error, Result};

use super::super::core::{AdValue, NodeId, TapeId};
use super::shared::fresh_ad_scalar_node_id;
use super::AdScalar;

#[derive(Clone, Copy)]
struct BinaryScalarState<T> {
    primal: T,
    tangent: Option<T>,
    reverse: Option<(NodeId, TapeId)>,
}

fn split_binary_state<T: scalarops::ScalarAd>(value: AdScalar<T>) -> BinaryScalarState<T> {
    match value.into_value() {
        AdValue::Primal(primal) => BinaryScalarState {
            primal,
            tangent: None,
            reverse: None,
        },
        AdValue::Forward { primal, tangent } => BinaryScalarState {
            primal,
            tangent: Some(tangent),
            reverse: None,
        },
        AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => BinaryScalarState {
            primal,
            tangent,
            reverse: Some((node, tape)),
        },
    }
}

fn add_rrule_wrapped<T: scalarops::ScalarAd>(_lhs: T, _rhs: T, cotangent: T) -> (T, T) {
    scalarops::add_rrule(cotangent)
}

fn sub_rrule_wrapped<T: scalarops::ScalarAd>(_lhs: T, _rhs: T, cotangent: T) -> (T, T) {
    scalarops::sub_rrule(cotangent)
}

fn mul_rrule_wrapped<T: scalarops::ScalarAd>(lhs: T, rhs: T, cotangent: T) -> (T, T) {
    scalarops::mul_rrule(lhs, rhs, cotangent)
}

fn div_rrule_wrapped<T: scalarops::ScalarAd>(lhs: T, rhs: T, cotangent: T) -> (T, T) {
    scalarops::div_rrule(lhs, rhs, cotangent)
}

fn binary_ad_scalar_try_op<T: scalarops::ScalarAd + 'static>(
    lhs: AdScalar<T>,
    rhs: AdScalar<T>,
    primal_rule: fn(T, T) -> T,
    frule: fn(T, T, T, T) -> (T, T),
    rrule: fn(T, T, T) -> (T, T),
) -> Result<AdScalar<T>> {
    let lhs_state = split_binary_state(lhs);
    let rhs_state = split_binary_state(rhs);

    let primal = primal_rule(lhs_state.primal, rhs_state.primal);
    let has_tangent = lhs_state.tangent.is_some() || rhs_state.tangent.is_some();
    let tangent = if has_tangent {
        let dx = lhs_state.tangent.unwrap_or_else(|| T::from_i32(0));
        let dy = rhs_state.tangent.unwrap_or_else(|| T::from_i32(0));
        let (_, tangent) = frule(lhs_state.primal, rhs_state.primal, dx, dy);
        Some(tangent)
    } else {
        None
    };

    match (lhs_state.reverse, rhs_state.reverse) {
        (None, None) => {
            if let Some(tangent) = tangent {
                Ok(AdScalar::new_forward(primal, tangent))
            } else {
                Ok(AdScalar::new_primal(primal))
            }
        }
        (lhs_reverse, rhs_reverse) => {
            let tape = match (lhs_reverse, rhs_reverse) {
                (Some((_, lhs_tape)), Some((_, rhs_tape))) if lhs_tape != rhs_tape => {
                    return Err(Error::MixedReverseTape {
                        expected: lhs_tape.0,
                        found: rhs_tape.0,
                    });
                }
                (Some((_, lhs_tape)), Some(_)) => lhs_tape,
                (Some((_, lhs_tape)), None) => lhs_tape,
                (None, Some((_, rhs_tape))) => rhs_tape,
                (None, None) => unreachable!("non-reverse case handled above"),
            };

            let output_node = fresh_ad_scalar_node_id();
            let lhs_primal = lhs_state.primal;
            let rhs_primal = rhs_state.primal;
            let lhs_node = lhs_reverse.map(|(node, _)| node);
            let rhs_node = rhs_reverse.map(|(node, _)| node);

            tape::register_scalar_rule(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    let (dlhs, drhs) = rrule(lhs_primal, rhs_primal, *cotangent);
                    let mut grads = Vec::new();
                    if let Some(node) = lhs_node {
                        grads.push((node, dlhs));
                    }
                    if let Some(node) = rhs_node {
                        grads.push((node, drhs));
                    }
                    Ok(grads)
                }),
            );

            Ok(AdScalar::new_reverse(primal, output_node, tape, tangent))
        }
    }
}

impl<T> AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    /// Checked addition for AD scalars.
    pub fn try_add(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            scalarops::add,
            scalarops::add_frule,
            add_rrule_wrapped,
        )
    }

    /// Checked subtraction for AD scalars.
    pub fn try_sub(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            scalarops::sub,
            scalarops::sub_frule,
            sub_rrule_wrapped,
        )
    }

    /// Checked multiplication for AD scalars.
    pub fn try_mul(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            scalarops::mul,
            scalarops::mul_frule,
            mul_rrule_wrapped,
        )
    }

    /// Checked division for AD scalars.
    pub fn try_div(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            scalarops::div,
            scalarops::div_frule,
            div_rrule_wrapped,
        )
    }
}

impl<T> Add for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(rhs)
    }
}

impl<T> Sub for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Result<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.try_sub(rhs)
    }
}

impl<T> Mul for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Result<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.try_mul(rhs)
    }
}

impl<T> Div for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Result<Self>;

    fn div(self, rhs: Self) -> Self::Output {
        self.try_div(rhs)
    }
}
