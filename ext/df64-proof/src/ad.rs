//! First-order AD for the contribution's externally defined operations.
//!
//! The runtime keys one rule per family and role, so the rules below dispatch on the
//! operation payload. Each rule emits the contribution's own operations for the
//! derivative work: a preset broadcast is not available for a scalar tenferro does
//! not declare, and the factorization's adjoint is the contribution's numerical body
//! too.

use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticAdRuleRole, SemanticLinearizeRequest,
    SemanticLinearizeResult, SemanticLinearizeRule, SemanticPrimalVjpRequest,
    SemanticPrimalVjpRule,
};
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_runtime::program::SemanticProgramBuilder;

use crate::extension::{Df64Expand, Df64FromF64, Df64Qr, Df64ToF64, Df64Total, DF64_OPS_FAMILY};

/// One operation of the contribution's family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Df64Op {
    /// The total sum.
    Total,
    /// The scalar broadcast, which is the total sum's adjoint.
    Expand,
    /// The reduced QR factorization.
    Qr,
    /// The narrowing conversion to `f64`.
    ToF64,
    /// The widening conversion from `f64`.
    FromF64,
}

impl Df64Op {
    /// Classify the payload a rule was asked about.
    fn of(op: &dyn ExtensionOp) -> Option<Self> {
        let any = op.as_any();
        if any.downcast_ref::<Df64Total>().is_some() {
            Some(Self::Total)
        } else if any.downcast_ref::<Df64Expand>().is_some() {
            Some(Self::Expand)
        } else if any.downcast_ref::<Df64Qr>().is_some() {
            Some(Self::Qr)
        } else if any.downcast_ref::<Df64ToF64>().is_some() {
            Some(Self::ToF64)
        } else if any.downcast_ref::<Df64FromF64>().is_some() {
            Some(Self::FromF64)
        } else {
            None
        }
    }
}

/// An operation outside a rule's domain is an error rather than a guess.
fn unsupported(op: Df64Op, role: SemanticAdRuleRole) -> SemanticAdError {
    SemanticAdError::Rule {
        family_id: DF64_OPS_FAMILY,
        role,
        source: Box::new(std::io::Error::other(format!(
            "the Df64 operations family has no {role:?} rule for {op:?}"
        ))),
    }
}

/// Reverse-mode rule for the contribution's operations.
///
/// The total sum's adjoint is a broadcast, and the conversions are linear, so their
/// adjoint swaps the two directions. None of them reads the primal value, so the rule
/// declares no residual.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::semantic_extension::{SemanticExtensionRuleSet, SemanticPrimalVjpRule};
/// use tenferro_df64_proof::ad::Df64VjpRule;
///
/// let rule = Df64VjpRule;
/// assert_eq!(
///     <Df64VjpRule as SemanticPrimalVjpRule>::family_id(&rule),
///     "tenferro-df64-proof.df64_ops.v1"
/// );
///
/// // The rule set a downstream application installs.
/// let rules = SemanticExtensionRuleSet::new()
///     .with_primal_vjp(std::sync::Arc::new(rule))
///     .expect("one rule per family");
/// assert!(rules.lookup_primal_vjp("tenferro-df64-proof.df64_ops.v1").is_some());
/// ```
#[derive(Debug)]
pub struct Df64VjpRule;

impl SemanticPrimalVjpRule for Df64VjpRule {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::none()
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let role = SemanticAdRuleRole::PrimalVjp;
        let op = Df64Op::of(request.op()).ok_or_else(|| SemanticAdError::Rule {
            family_id: DF64_OPS_FAMILY,
            role,
            source: Box::new(std::io::Error::other(
                "the payload is not one of the Df64 operations",
            )),
        })?;
        let inactive = || vec![AdValue::Absent; request.primal_input_count()].into_boxed_slice();
        let Some(AdValue::Value(cotangent)) = request.cotangent_outputs().first().copied() else {
            return Ok(inactive());
        };

        let emitted = match op {
            Df64Op::Total => {
                // The adjoint places the cotangent back into the input's shape, which
                // the broadcast payload carries, so the shape is read from the primal
                // input's metadata.
                let shape = exact_shape(&request)?;
                builder.add_extension(Arc::new(Df64Expand::new(shape)), &[cotangent])
            }
            // A linear conversion's adjoint is the opposite conversion.
            Df64Op::ToF64 => builder.add_extension(Arc::new(Df64FromF64), &[cotangent]),
            Df64Op::FromF64 => builder.add_extension(Arc::new(Df64ToF64), &[cotangent]),
            Df64Op::Expand | Df64Op::Qr => return Err(unsupported(op, role)),
        }
        .map_err(SemanticAdError::Build)?;

        Ok(emitted
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if index < request.primal_input_count() && request.active_inputs()[index] {
                    AdValue::Value(*value)
                } else {
                    AdValue::Absent
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice())
    }
}

/// Forward-mode rule for the contribution's operations.
///
/// The total sum is linear, so its tangent output is the sum of the tangent inputs,
/// and each conversion's tangent passes through the same conversion.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::semantic_extension::{SemanticExtensionRuleSet, SemanticLinearizeRule};
/// use tenferro_df64_proof::ad::Df64LinearizeRule;
///
/// let rules = SemanticExtensionRuleSet::new()
///     .with_linearize(std::sync::Arc::new(Df64LinearizeRule))
///     .expect("one linearize rule per family");
/// assert!(rules.lookup_linearize("tenferro-df64-proof.df64_ops.v1").is_some());
/// ```
#[derive(Debug)]
pub struct Df64LinearizeRule;

impl SemanticLinearizeRule for Df64LinearizeRule {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let role = SemanticAdRuleRole::Linearize;
        let op = Df64Op::of(request.op()).ok_or_else(|| SemanticAdError::Rule {
            family_id: DF64_OPS_FAMILY,
            role,
            source: Box::new(std::io::Error::other(
                "the payload is not one of the Df64 operations",
            )),
        })?;
        let (operation, tangents): (Arc<dyn ExtensionOp>, Vec<_>) = match op {
            Df64Op::Total => (
                Arc::new(Df64Total),
                request
                    .tangent_inputs()
                    .iter()
                    .filter_map(|value| value.value())
                    .collect(),
            ),
            Df64Op::ToF64 | Df64Op::FromF64 => {
                let tangent = request
                    .tangent_inputs()
                    .iter()
                    .find_map(|value| value.value());
                let operation: Arc<dyn ExtensionOp> = if op == Df64Op::ToF64 {
                    Arc::new(Df64ToF64)
                } else {
                    Arc::new(Df64FromF64)
                };
                (operation, tangent.into_iter().collect())
            }
            Df64Op::Expand | Df64Op::Qr => return Err(unsupported(op, role)),
        };
        if tangents.is_empty() {
            let inactive = (0..request.primal_outputs().len())
                .map(|_| AdValue::Absent)
                .collect::<Vec<_>>();
            return Ok(SemanticLinearizeResult::new(inactive, Vec::new()));
        }
        let emitted = builder
            .add_extension(operation, &tangents)
            .map_err(SemanticAdError::Build)?;
        let outputs = emitted
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if request
                    .active_outputs()
                    .get(index)
                    .copied()
                    .unwrap_or(false)
                {
                    AdValue::Value(*value)
                } else {
                    AdValue::Absent
                }
            })
            .collect::<Vec<_>>();
        Ok(SemanticLinearizeResult::new(outputs, Vec::new()))
    }
}

/// Read the primal input's shape as concrete extents.
///
/// # Errors
///
/// Returns [`SemanticAdError::Invariant`] when the shape is not fully concrete,
/// because a broadcast of an unknown extent has no declared shape for the op payload.
fn exact_shape(request: &SemanticPrimalVjpRequest<'_>) -> Result<Vec<usize>, SemanticAdError> {
    let metadata = request.primal_input_meta(0)?;
    metadata
        .shape()
        .iter()
        .map(|extent| match extent.as_exact() {
            Some(tenferro_ops::dim_expr::DimExpr::Const(value)) => Ok(*value),
            _ => Err(SemanticAdError::Invariant {
                family_id: DF64_OPS_FAMILY,
                role: SemanticAdRuleRole::PrimalVjp,
                message: format!(
                    "the adjoint of {DF64_OPS_FAMILY} needs a concrete input shape, got {extent:?}"
                ),
            }),
        })
        .collect()
}
