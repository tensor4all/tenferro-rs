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

use crate::extension::{
    Df64Einsum, Df64EinsumJvp, Df64EinsumVjp, Df64Expand, Df64FromF64, Df64Qr, Df64QrJvp,
    Df64QrVjp, Df64ToF64, Df64Total, DF64_OPS_FAMILY,
};

/// One operation of the contribution's family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Df64Op {
    /// The total sum.
    Total,
    /// The scalar broadcast, which is the total sum's adjoint.
    Expand,
    /// The reduced QR factorization.
    Qr,
    /// A matrix contraction.
    Einsum,
    /// The narrowing conversion to `f64`.
    ToF64,
    /// The widening conversion from `f64`.
    FromF64,
    /// The factorization's adjoint.
    QrVjp,
    /// The factorization's tangent.
    QrJvp,
}

impl Df64Op {
    /// Classify the payload a rule was asked about.
    fn of(op: &dyn ExtensionOp) -> Option<Self> {
        let any = op.as_any();
        if any.downcast_ref::<Df64Total>().is_some() {
            Some(Self::Total)
        } else if any.downcast_ref::<Df64Expand>().is_some() {
            Some(Self::Expand)
        } else if any.downcast_ref::<Df64Einsum>().is_some() {
            Some(Self::Einsum)
        } else if any.downcast_ref::<Df64Qr>().is_some() {
            Some(Self::Qr)
        } else if any.downcast_ref::<Df64ToF64>().is_some() {
            Some(Self::ToF64)
        } else if any.downcast_ref::<Df64FromF64>().is_some() {
            Some(Self::FromF64)
        } else if any.downcast_ref::<Df64QrVjp>().is_some() {
            Some(Self::QrVjp)
        } else if any.downcast_ref::<Df64QrJvp>().is_some() {
            Some(Self::QrJvp)
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
        // The union over the family: the factorization's adjoint reads its primal
        // factors, and the contraction's adjoint reads its operands (the case the
        // residual specification names for an einsum rule).
        ResidualSpec::all_outputs().with_all_inputs()
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
        let emitted = match op {
            Df64Op::Total => {
                let Some(AdValue::Value(cotangent)) = request.cotangent_outputs().first().copied()
                else {
                    return Ok(inactive());
                };
                // The adjoint places the cotangent back into the input's shape, which
                // the broadcast payload carries, so the shape is read from the primal
                // input's metadata.
                let shape = exact_shape(&request)?;
                builder.add_extension(Arc::new(Df64Expand::new(shape)), &[cotangent])
            }
            // A linear conversion's adjoint is the opposite conversion.
            Df64Op::ToF64 | Df64Op::FromF64 => {
                let Some(AdValue::Value(cotangent)) = request.cotangent_outputs().first().copied()
                else {
                    return Ok(inactive());
                };
                let operation = if op == Df64Op::ToF64 {
                    Arc::new(Df64FromF64) as Arc<dyn ExtensionOp>
                } else {
                    Arc::new(Df64ToF64) as Arc<dyn ExtensionOp>
                };
                builder.add_extension(operation, &[cotangent])
            }
            Df64Op::Qr => {
                // A loss need not depend on both factors, so the adjoint is told which
                // cotangents are present and reads the primal factors it needs.
                let has_q = matches!(request.cotangent_outputs().first(), Some(AdValue::Value(_)));
                let has_r = matches!(request.cotangent_outputs().get(1), Some(AdValue::Value(_)));
                if !has_q && !has_r {
                    return Ok(inactive());
                }
                let mut operands = vec![
                    request.primal_output_value(0)?,
                    request.primal_output_value(1)?,
                ];
                for (present, cotangent) in [
                    (has_q, request.cotangent_outputs().first().copied()),
                    (has_r, request.cotangent_outputs().get(1).copied()),
                ] {
                    if !present {
                        continue;
                    }
                    match cotangent {
                        Some(AdValue::Value(value)) => operands.push(value),
                        _ => return Ok(inactive()),
                    }
                }
                builder.add_extension(Arc::new(Df64QrVjp::of(has_q, has_r)), &operands)
            }
            Df64Op::Einsum => {
                // The adjoint of a contraction contracts the output cotangent with the other
                // operand, so the helper reads both operands and the cotangent and produces one
                // cotangent per operand.
                let Some(contraction) = request.op().as_any().downcast_ref::<Df64Einsum>() else {
                    return Err(unsupported(op, role));
                };
                let Some(AdValue::Value(cotangent)) = request.cotangent_outputs().first().copied()
                else {
                    return Ok(inactive());
                };
                // The helpers are defined for the pairwise case, so a wider pattern is refused
                // rather than differentiated as if it were pairwise.
                let Some((lhs, rhs, out)) = contraction.labels() else {
                    return Err(unsupported(op, role));
                };
                // The primal operation accepted this pattern, so the adjoint's validation is a
                // rule-level invariant rather than a user error.
                let Ok(adjoint) = Df64EinsumVjp::of(lhs, rhs, out) else {
                    return Err(unsupported(op, role));
                };
                builder.add_extension(
                    Arc::new(adjoint),
                    &[
                        request.primal_input_value(0)?,
                        request.primal_input_value(1)?,
                        cotangent,
                    ],
                )
            }
            Df64Op::Expand | Df64Op::QrVjp | Df64Op::QrJvp => {
                return Err(unsupported(op, role));
            }
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
            Df64Op::Qr => {
                // The tangent of the factorization is its own body: it needs both primal
                // factors and the input tangent, so the rule emits one operation.
                let Some(tangent) = request
                    .tangent_inputs()
                    .first()
                    .and_then(|value| value.value())
                else {
                    let inactive = (0..request.primal_outputs().len())
                        .map(|_| AdValue::Absent)
                        .collect::<Vec<_>>();
                    return Ok(SemanticLinearizeResult::new(inactive, Vec::new()));
                };
                (
                    Arc::new(Df64QrJvp) as Arc<dyn ExtensionOp>,
                    vec![
                        request.primal_outputs()[0],
                        request.primal_outputs()[1],
                        tangent,
                    ],
                )
            }
            Df64Op::Einsum => {
                // The tangent of a contraction contracts each tangent with the other operand, so
                // the rule emits one helper carrying both operands and whichever tangents exist.
                let Some(contraction) = request.op().as_any().downcast_ref::<Df64Einsum>() else {
                    return Err(unsupported(op, role));
                };
                let has_lhs = request
                    .tangent_inputs()
                    .first()
                    .and_then(|value| value.value())
                    .is_some();
                let has_rhs = request
                    .tangent_inputs()
                    .get(1)
                    .and_then(|value| value.value())
                    .is_some();
                if !has_lhs && !has_rhs {
                    let inactive = (0..request.primal_inputs().len())
                        .map(|_| AdValue::Absent)
                        .collect::<Vec<_>>();
                    return Ok(SemanticLinearizeResult::new(inactive, Vec::new()));
                }
                // The helpers are defined for the pairwise case, so a wider pattern is refused
                // rather than differentiated as if it were pairwise.
                let Some((lhs, rhs, out)) = contraction.labels() else {
                    return Err(unsupported(op, role));
                };
                let Ok(tangent) = Df64EinsumJvp::of(lhs, rhs, out, has_lhs, has_rhs) else {
                    return Err(unsupported(op, role));
                };
                let mut operands = vec![request.primal_inputs()[0], request.primal_inputs()[1]];
                for (present, value) in [
                    (has_lhs, request.tangent_inputs().first()),
                    (has_rhs, request.tangent_inputs().get(1)),
                ] {
                    if !present {
                        continue;
                    }
                    match value.and_then(|value| value.value()) {
                        Some(value) => operands.push(value),
                        None => return Err(unsupported(op, role)),
                    }
                }
                (Arc::new(tangent) as Arc<dyn ExtensionOp>, operands)
            }
            Df64Op::Expand | Df64Op::QrVjp | Df64Op::QrJvp => {
                return Err(unsupported(op, role));
            }
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
