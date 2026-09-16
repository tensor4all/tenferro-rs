//! First-order AD for the extension-owned operations.
//!
//! The adjoint of a total sum is a broadcast of the output cotangent, and the
//! broadcast for a scalar tenferro does not declare is the contribution's own
//! operation, so the rule emits that node instead of asking for a preset one.

use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticAdRuleRole, SemanticPrimalVjpRequest,
    SemanticPrimalVjpRule,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::SemanticProgramBuilder;

use crate::extension::{Df64Expand, Df64Total, DF64_OPS_FAMILY};

/// Adjoint of the extension-owned total sum.
///
/// The sum's derivative does not depend on the primal value, so the rule reads no
/// residual and only needs the output cotangent.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::ad::Df64TotalVjpRule;
/// use tenferro_ad::semantic_extension::{SemanticExtensionRuleSet, SemanticPrimalVjpRule};
///
/// let rule = Df64TotalVjpRule;
/// assert_eq!(<Df64TotalVjpRule as SemanticPrimalVjpRule>::family_id(&rule), "tenferro-df64-proof.df64_ops.v1");
///
/// // The rule set a downstream application installs.
/// let rules = SemanticExtensionRuleSet::new()
///     .with_primal_vjp(std::sync::Arc::new(rule))
///     .expect("one rule per family");
/// assert!(rules.lookup_primal_vjp("tenferro-df64-proof.df64_ops.v1").is_some());
/// ```
#[derive(Debug)]
pub struct Df64TotalVjpRule;

impl SemanticPrimalVjpRule for Df64TotalVjpRule {
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
        // The family holds the contribution's operations and only the total sum in it
        // is differentiable, so any other payload is outside this rule's domain.
        if request.op().as_any().downcast_ref::<Df64Total>().is_none() {
            return Err(SemanticAdError::Rule {
                family_id: DF64_OPS_FAMILY,
                role: SemanticAdRuleRole::PrimalVjp,
                source: Box::new(std::io::Error::other(
                    "the Df64 ops family adjoint is defined for the total sum",
                )),
            });
        }
        let inactive = || vec![AdValue::Absent; request.primal_input_count()].into_boxed_slice();
        let Some(AdValue::Value(cotangent)) = request.cotangent_outputs().first().copied() else {
            return Ok(inactive());
        };

        // The adjoint places the cotangent back into the input's shape, which the op
        // payload carries, so the shape has to be exact here.
        let shape = exact_shape(&request)?;
        let outputs = builder
            .add_extension(Arc::new(Df64Expand::new(shape)), &[cotangent])
            .map_err(SemanticAdError::Build)?;
        let mut cotangents = Vec::with_capacity(request.primal_input_count());
        for (index, value) in outputs.iter().enumerate() {
            if index < request.primal_input_count() && request.active_inputs()[index] {
                cotangents.push(AdValue::Value(*value));
            } else {
                cotangents.push(AdValue::Absent);
            }
        }
        Ok(cotangents.into_boxed_slice())
    }
}

/// Read the primal input's shape as concrete extents.
///
/// # Errors
///
/// Returns [`SemanticAdError::Unsupported`] when the shape is not fully concrete,
/// because a broadcast of an unknown extent has no declared shape for the op
/// payload.
fn exact_shape(request: &SemanticPrimalVjpRequest<'_>) -> Result<Vec<usize>, SemanticAdError> {
    let metadata = request.primal_input_meta(0)?;
    metadata
        .shape()
        .iter()
        .map(|extent| match extent.as_exact() {
            Some(DimExpr::Const(value)) => Ok(*value),
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
