//! Automatic differentiation support for `tenferro-linalg`.
//!
//! This crate is the explicit opt-in AD companion for the primal
//! `tenferro-linalg` extension runtime.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_ad::TracedTensorAdExt;
//! use tenferro_runtime::TracedTensor;
//!
//! tenferro_linalg_ad::register_extension_rule().unwrap();
//! let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
//! let (_u, s, _vt) = tenferro_linalg::svd(&x);
//! let loss = s.reduce_sum(&[0]);
//! let grad = loss.grad(&x).unwrap();
//! assert_eq!(grad.rank, 2);
//! ```

use std::sync::Arc;

use chainrules_core::{ADRuleError, ADRuleKind, ADRuleResult};
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_ad::extension::{
    is_extension_rule_registered, register_extension_rule as register_rule, ExtensionAdRuleTrait,
    ExtensionOpTrait, ExtensionRegistryError,
};
use tenferro_linalg::ad_support::{LinalgExtensionOp, LinalgOp};
use tenferro_linalg::LINALG_EXTENSION_FAMILY_ID;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;

pub mod eager_tensor;
mod rules;

/// Register the `tenferro-linalg` extension AD rule.
///
/// Registration is process-local and idempotent: duplicate registrations are
/// treated as success.
///
/// # Examples
///
/// ```rust
/// tenferro_linalg_ad::register_extension_rule().unwrap();
/// ```
pub fn register_extension_rule() -> Result<(), ExtensionRegistryError> {
    if is_extension_rule_registered(LINALG_EXTENSION_FAMILY_ID) {
        return Ok(());
    }
    match register_rule(Arc::new(LinalgAdRule)) {
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
        builder: &mut FragmentBuilder<StdTensorOp>,
        primal_in: &[GlobalValKey<StdTensorOp>],
        primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Linearize)?;
        let tangents = match op.op() {
            LinalgOp::Lu => rules::linearize_lu(builder, primal_in, primal_out, tangent_in, ctx),
            LinalgOp::FullPivLu => {
                rules::linearize_full_piv_lu(builder, primal_in, primal_out, tangent_in, ctx)
            }
            LinalgOp::Solve { transpose_a } => {
                rules::linearize_solve(builder, primal_in, primal_out, tangent_in, transpose_a, ctx)
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
            LinalgOp::Qr => rules::linearize_qr(builder, primal_in, primal_out, tangent_in, ctx),
            LinalgOp::Eigh { eps } => {
                rules::linearize_eigh(builder, primal_in, primal_out, tangent_in, eps, ctx)
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
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        mode: &OpMode,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Transpose)?;
        let mut emitter = DynEmitter(emitter);
        let cotangents = match op.op() {
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => rules::transpose_triangular_solve(
                &mut emitter,
                cotangent_out,
                inputs,
                mode,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                ctx,
            ),
            LinalgOp::Solve { transpose_a } => {
                rules::transpose_solve(&mut emitter, cotangent_out, inputs, mode, transpose_a, ctx)
            }
            LinalgOp::FullPivLuSolve { transpose_a } => rules::transpose_full_piv_lu_solve(
                &mut emitter,
                cotangent_out,
                inputs,
                mode,
                transpose_a,
                ctx,
            ),
            LinalgOp::Cholesky
            | LinalgOp::Lu
            | LinalgOp::FullPivLu
            | LinalgOp::Svd { .. }
            | LinalgOp::Qr
            | LinalgOp::Eigh { .. }
            | LinalgOp::Eig { .. } => vec![None; op.n_inputs()],
        };
        Ok(cotangents)
    }
}

struct DynEmitter<'a>(&'a mut dyn OpEmitter<StdTensorOp>);

impl OpEmitter<StdTensorOp> for DynEmitter<'_> {
    fn add_op(
        &mut self,
        op: StdTensorOp,
        inputs: Vec<ValRef<StdTensorOp>>,
        mode: OpMode,
    ) -> Vec<LocalValId> {
        self.0.add_op(op, inputs, mode)
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
