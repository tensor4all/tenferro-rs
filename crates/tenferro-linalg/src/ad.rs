//! Automatic differentiation support for `tenferro-linalg`.
//!
//! This module is enabled by the `autodiff` feature. It provides the linalg
//! extension rule set used by explicit `tenferro_ad::AdContext` values.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_ad::AdContext;
//! use tenferro_runtime::TracedTensor;
//!
//! let ad = AdContext::builder()
//!     .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
//!     .build()
//!     .unwrap();
//! let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
//! let (_u, s, _vt) = tenferro_linalg::svd(&x).unwrap();
//! let loss = s.reduce_sum(&[0]);
//! let grad = ad.grad(&loss, &x).unwrap();
//! assert_eq!(grad.rank, 2);
//! ```

use std::sync::Arc;

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ad::extension::{
    is_extension_rule_registered, register_extension_rule as register_rule, ExtensionAdRuleTrait,
    ExtensionOpTrait, ExtensionRegistryError, ExtensionRuleSet,
};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use crate::extension::{LinalgExtensionOp, LinalgOp};
use crate::LINALG_EXTENSION_FAMILY_ID;

mod rules;

/// Return the explicit linalg extension AD rule set.
///
/// # Examples
///
/// ```rust
/// let rules = tenferro_linalg::ad_rules().unwrap();
/// assert!(rules.is_rule_registered(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID));
/// ```
pub fn ad_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    ExtensionRuleSet::new().with_rule(Arc::new(LinalgAdRule))
}

/// Register the `tenferro-linalg` extension AD rule.
///
/// This process-global registration API is retained as a compatibility bridge.
/// Prefer explicit [`ad_rules`] ownership through `tenferro_ad::AdContext`.
///
/// # Examples
///
/// ```rust
/// tenferro_linalg::register_extension_rule().unwrap();
/// ```
pub fn register_extension_rule() -> Result<(), ExtensionRegistryError> {
    if is_extension_rule_registered(LINALG_EXTENSION_FAMILY_ID) {
        return Ok(());
    }
    let rules = ad_rules()?;
    let Some(rule) = rules.lookup_rule(LINALG_EXTENSION_FAMILY_ID) else {
        return Ok(());
    };
    match register_rule(rule) {
        Ok(()) | Err(ExtensionRegistryError::DuplicateRule { .. }) => Ok(()),
        Err(err) => Err(err),
    }
}

#[derive(Debug)]
struct LinalgAdRule;

impl ExtensionAdRuleTrait for LinalgAdRule {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOpTrait,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Jvp)?;
        let tangents = match op.op() {
            LinalgOp::Lu => rules::linearize_lu(builder, primal_in, primal_out, tangent_in, ctx),
            LinalgOp::LuFactor => vec![None; op.output_count()],
            LinalgOp::LuSolvePrepared {
                transpose_a,
                conjugate_a,
            } => rules::linearize_lu_solve_prepared(
                builder,
                primal_in,
                primal_out,
                tangent_in,
                transpose_a,
                conjugate_a,
                ctx,
            ),
            LinalgOp::FullPivLu => {
                rules::linearize_full_piv_lu(builder, primal_in, primal_out, tangent_in, ctx)
            }
            LinalgOp::FullPivLuSolve { transpose_a } => rules::linearize_full_piv_lu_solve(
                builder,
                primal_in,
                primal_out,
                tangent_in,
                transpose_a,
                ctx,
            ),
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => rules::linearize_triangular_solve(
                builder,
                primal_in,
                primal_out,
                tangent_in,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                ctx,
            ),
            LinalgOp::Cholesky => {
                rules::linearize_cholesky(builder, primal_in, primal_out, tangent_in, ctx)
            }
            LinalgOp::Svd { eps } => {
                rules::linearize_svd(builder, primal_in, primal_out, tangent_in, eps, ctx)
            }
            LinalgOp::SvdVals { eps } => {
                rules::linearize_svd_values(builder, primal_in, tangent_in, eps, ctx)
            }
            LinalgOp::Qr => rules::linearize_qr(builder, primal_in, primal_out, tangent_in, ctx),
            LinalgOp::Eigh { eps } => {
                rules::linearize_eigh(builder, primal_in, primal_out, tangent_in, eps, ctx)
            }
            LinalgOp::EighVals { eps } => {
                rules::linearize_eigh_values(builder, primal_in, tangent_in, eps, ctx)
            }
            LinalgOp::Eig { input_dtype } => {
                rules::linearize_eig(builder, primal_in, primal_out, tangent_in, input_dtype, ctx)
            }
        };
        Ok(tangents)
    }

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOpTrait,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Transpose)?;
        let mut builder = DynBuilder(builder);
        let cotangents = match op.op() {
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => rules::transpose_triangular_solve(
                &mut builder,
                cotangent_out,
                inputs,
                mode,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                ctx,
            ),
            LinalgOp::LuSolvePrepared {
                transpose_a,
                conjugate_a,
            } => rules::transpose_lu_solve_prepared(
                &mut builder,
                cotangent_out,
                inputs,
                mode,
                transpose_a,
                conjugate_a,
                ctx,
            ),
            LinalgOp::FullPivLuSolve { transpose_a } => rules::transpose_full_piv_lu_solve(
                &mut builder,
                cotangent_out,
                inputs,
                mode,
                transpose_a,
                ctx,
            ),
            LinalgOp::Cholesky
            | LinalgOp::Lu
            | LinalgOp::LuFactor
            | LinalgOp::FullPivLu
            | LinalgOp::Svd { .. }
            | LinalgOp::SvdVals { .. }
            | LinalgOp::Qr
            | LinalgOp::Eigh { .. }
            | LinalgOp::EighVals { .. }
            | LinalgOp::Eig { .. } => vec![None; op.input_count()],
        };
        Ok(cotangents)
    }
}

struct DynBuilder<'a>(&'a mut dyn PrimitiveRuleBuilder);

impl PrimitiveRuleBuilder for DynBuilder<'_> {
    fn add_operation(
        &mut self,
        op: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
        mode: OperationRole,
    ) -> Vec<LocalValueId> {
        self.0.add_operation(op, inputs, mode)
    }
}

fn downcast_ad_op<'a>(
    op: &'a dyn ExtensionOpTrait,
    kind: ADRuleKind,
) -> ADRuleResult<&'a LinalgExtensionOp> {
    op.as_any()
        .downcast_ref::<LinalgExtensionOp>()
        .ok_or_else(|| {
            ADRuleError::unsupported("tenferro-linalg.linalg.v1 payload type mismatch", kind)
        })
}
