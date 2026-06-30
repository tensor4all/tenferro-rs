//! Automatic differentiation support for `tenferro-linalg`.
//!
//! This module is enabled by the `autodiff` feature. It provides the linalg
//! extension rule set used by explicit `tenferro_ad::AdContext` values.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_ad::AdContext;
//! use tenferro_linalg::TracedTensorLinalgExt;
//! use tenferro_runtime::TracedTensor;
//!
//! let ad = AdContext::builder()
//!     .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
//!     .build()
//!     .unwrap();
//! let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
//! let (_u, s, _vt) = x.svd().unwrap();
//! let loss = s.reduce_sum(&[0]).unwrap();
//! let grad = ad.grad(&loss, &x).unwrap();
//! assert_eq!(grad.rank, 2);
//! ```

use std::sync::Arc;

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ad::extension::{
    ExtensionAdRule, ExtensionOp, ExtensionRegistryError, ExtensionRuleSet,
};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use crate::extension::{LinalgExtensionOp, LinalgOp};
use crate::LINALG_EXTENSION_FAMILY_ID;

mod rules;
pub mod support;

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

#[derive(Debug)]
struct LinalgAdRule;

impl ExtensionAdRule for LinalgAdRule {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Jvp)?;
        match op.op() {
            LinalgOp::Lu => rules::linearize_lu(builder, primal_in, primal_out, tangent_in, ctx),
            LinalgOp::LuFactor => Ok(vec![None; op.output_count()]),
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
                rules::TriangularSolveFlags::new(left_side, lower, transpose_a, unit_diagonal),
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
            LinalgOp::EigVals { input_dtype } => {
                rules::linearize_eig_values(builder, primal_in, tangent_in, input_dtype, ctx)
            }
        }
    }

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Transpose)?;
        let mut builder = DynBuilder(builder);
        match op.op() {
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
                rules::TriangularSolveFlags::new(left_side, lower, transpose_a, unit_diagonal),
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
            LinalgOp::Eigh { eps } => {
                rules::transpose_eigh(&mut builder, cotangent_out, inputs, mode, eps, ctx)
            }
            LinalgOp::EighVals { eps } => {
                rules::transpose_eigh_values(&mut builder, cotangent_out, inputs, mode, eps, ctx)
            }
            LinalgOp::Cholesky
            | LinalgOp::Lu
            | LinalgOp::LuFactor
            | LinalgOp::FullPivLu
            | LinalgOp::Svd { .. }
            | LinalgOp::SvdVals { .. }
            | LinalgOp::Qr
            | LinalgOp::Eig { .. }
            | LinalgOp::EigVals { .. } => Ok(vec![None; op.input_count()]),
        }
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

fn downcast_ad_op(op: &dyn ExtensionOp, kind: ADRuleKind) -> ADRuleResult<&LinalgExtensionOp> {
    op.as_any()
        .downcast_ref::<LinalgExtensionOp>()
        .ok_or_else(|| {
            ADRuleError::invalid_input("tenferro-linalg.linalg.v1", kind, "payload type mismatch")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::DEFAULT_DECOMPOSITION_AD_EPS;
    use computegraph::graph::GraphBuilder;
    use tenferro_ops::input_key::TensorInputKey;
    use tenferro_ops::{ShapeExtent, SymDim, TensorMeta};
    use tenferro_tensor::DType;

    fn input_key(id: u64) -> ValueKey<StdTensorOp> {
        ValueKey::Input(TensorInputKey::User { id })
    }

    fn insert_meta(ctx: &mut ShapeGuardContext, key: ValueKey<StdTensorOp>, shape: &[usize]) {
        ctx.insert_metadata(
            key,
            TensorMeta::exact(
                DType::F64,
                shape.iter().copied().map(SymDim::from).collect(),
            ),
        );
    }

    #[test]
    fn full_piv_lu_jvp_returns_inactive_outputs_for_non_square_input() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let mut ctx = ShapeGuardContext::default();
        let primal = input_key(1);
        insert_meta(&mut ctx, primal.clone(), &[2, 3]);
        let tangent = builder.add_input(TensorInputKey::User { id: 2 });
        let outputs = [
            input_key(10),
            input_key(11),
            input_key(12),
            input_key(13),
            input_key(14),
        ];
        let op = LinalgExtensionOp::new(LinalgOp::FullPivLu);

        let result = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[primal],
                &outputs,
                &[Some(tangent)],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result, vec![None, None, None, None, None]);
        assert!(builder.build().operations().is_empty());
    }

    #[test]
    fn triangular_solve_jvp_rejects_non_matrix_operands() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let mut ctx = ShapeGuardContext::default();
        let lhs = input_key(20);
        let rhs = input_key(21);
        insert_meta(&mut ctx, lhs.clone(), &[2, 2]);
        insert_meta(&mut ctx, rhs.clone(), &[2]);
        let rhs_tangent = builder.add_input(TensorInputKey::User { id: 22 });
        let op = LinalgExtensionOp::new(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        });

        let err = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[lhs, rhs],
                &[input_key(23)],
                &[None, Some(rhs_tangent)],
                &mut ctx,
            )
            .unwrap_err();

        assert_eq!(err.rule(), ADRuleKind::Jvp);
        assert!(err
            .to_string()
            .contains("expected matrix operands with rank >= 2"));
        assert!(builder.build().operations().is_empty());
    }

    #[test]
    fn triangular_solve_jvp_accepts_upper_bound_matrix_metadata() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let mut ctx = ShapeGuardContext::default();
        let lhs = input_key(30);
        let rhs = input_key(31);
        ctx.insert_metadata(
            lhs.clone(),
            TensorMeta::with_extents(
                DType::F64,
                vec![
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                ],
            ),
        );
        ctx.insert_metadata(
            rhs.clone(),
            TensorMeta::with_extents(
                DType::F64,
                vec![
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                    ShapeExtent::upper_bound(SymDim::from(2usize)),
                ],
            ),
        );
        let rhs_tangent = builder.add_input(TensorInputKey::User { id: 32 });
        let op = LinalgExtensionOp::new(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        });

        let result = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[lhs.clone(), rhs],
                &[input_key(33)],
                &[None, Some(rhs_tangent)],
                &mut ctx,
            )
            .unwrap();

        assert!(result[0].is_some());
        let graph = builder.build();
        assert_eq!(graph.operations().len(), 1);
        let solve = &graph.operations()[0];
        assert_eq!(solve.inputs[0], ValueRef::External(lhs));
        assert_eq!(solve.inputs[1], ValueRef::Local(rhs_tangent));
    }

    #[test]
    fn triangular_solve_transpose_accepts_upper_bound_matrix_metadata() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let mut ctx = ShapeGuardContext::default();
        let lhs = input_key(40);
        let rhs = input_key(41);
        ctx.insert_metadata(
            lhs.clone(),
            TensorMeta::with_extents(
                DType::F64,
                vec![
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                ],
            ),
        );
        ctx.insert_metadata(
            rhs.clone(),
            TensorMeta::with_extents(
                DType::F64,
                vec![
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                    ShapeExtent::upper_bound(SymDim::from(2usize)),
                ],
            ),
        );
        let cotangent = builder.add_input(TensorInputKey::User { id: 42 });
        let op = LinalgExtensionOp::new(LinalgOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        });

        let result = LinalgAdRule
            .transpose_rule(
                &op,
                &mut builder,
                &[Some(cotangent)],
                &[ValueRef::External(lhs.clone()), ValueRef::External(rhs)],
                &OperationRole::Linearized {
                    active_mask: vec![false, true],
                },
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result[0], None);
        assert!(result[1].is_some());
        let graph = builder.build();
        assert_eq!(graph.operations().len(), 1);
        assert_eq!(graph.operations()[0].inputs[0], ValueRef::External(lhs));
        assert_eq!(graph.operations()[0].inputs[1], ValueRef::Local(cotangent));
    }

    #[test]
    fn cholesky_jvp_uses_rank_when_input_metadata_is_upper_bound() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let mut ctx = ShapeGuardContext::default();
        let primal = input_key(50);
        ctx.insert_metadata(
            primal.clone(),
            TensorMeta::with_extents(
                DType::F64,
                vec![
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                    ShapeExtent::upper_bound(SymDim::from(4usize)),
                ],
            ),
        );
        let tangent = builder.add_input(TensorInputKey::User { id: 51 });
        let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

        let result = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[primal],
                &[input_key(52)],
                &[Some(tangent)],
                &mut ctx,
            )
            .unwrap();

        assert!(result[0].is_some());
        assert!(!builder.build().operations().is_empty());
    }

    #[test]
    fn cholesky_jvp_propagates_missing_input_metadata() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let mut ctx = ShapeGuardContext::default();
        let primal = input_key(55);
        let tangent = builder.add_input(TensorInputKey::User { id: 56 });
        let op = LinalgExtensionOp::new(LinalgOp::Cholesky);

        let err = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[primal],
                &[input_key(57)],
                &[Some(tangent)],
                &mut ctx,
            )
            .unwrap_err();

        assert_eq!(err.rule(), ADRuleKind::Jvp);
        assert!(err.to_string().contains("missing TensorMeta"));
        assert!(builder.build().operations().is_empty());
    }

    #[test]
    fn one_input_linalg_jvps_return_inactive_for_non_matrix_input() {
        let cases = [
            LinalgOp::Cholesky,
            LinalgOp::Lu,
            LinalgOp::FullPivLu,
            LinalgOp::Svd {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
            },
            LinalgOp::SvdVals {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
            },
            LinalgOp::Qr,
            LinalgOp::Eigh {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
            },
            LinalgOp::EighVals {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
            },
            LinalgOp::Eig {
                input_dtype: DType::F64,
            },
            LinalgOp::EigVals {
                input_dtype: DType::F64,
            },
        ];

        for (case_index, kind) in cases.into_iter().enumerate() {
            let mut builder = GraphBuilder::<StdTensorOp>::new();
            let mut ctx = ShapeGuardContext::default();
            let primal = input_key(100 + case_index as u64);
            insert_meta(&mut ctx, primal.clone(), &[3]);
            let tangent = builder.add_input(TensorInputKey::User {
                id: 200 + case_index as u64,
            });
            let op = LinalgExtensionOp::new(kind);
            let outputs: Vec<_> = (0..op.output_count())
                .map(|offset| input_key(300 + case_index as u64 * 10 + offset as u64))
                .collect();

            let result = LinalgAdRule
                .linearize(
                    &op,
                    &mut builder,
                    &[primal],
                    &outputs,
                    &[Some(tangent)],
                    &mut ctx,
                )
                .unwrap();

            assert_eq!(result, vec![None; op.output_count()], "{kind:?}");
            assert!(
                builder.build().operations().is_empty(),
                "{kind:?} should not emit a malformed matrix AD graph"
            );
        }
    }
}
