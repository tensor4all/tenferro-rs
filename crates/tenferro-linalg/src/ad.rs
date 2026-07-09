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
    ExtensionLinearTransposeRule, ExtensionLinearizeRule, ExtensionOp, ExtensionRegistryError,
    ExtensionRuleSet,
};
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;
use tidu::{ADRuleError, ADRuleKind, ADRuleResult, PrimitiveTransposeInput};

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
/// assert!(rules.is_linearize_registered(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID));
/// assert!(rules.is_linear_transpose_registered(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID));
/// ```
pub fn ad_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    ExtensionRuleSet::new()
        .with_linearize(Arc::new(LinalgAdRule))?
        .with_linear_transpose(Arc::new(LinalgAdRule))
}

#[derive(Debug)]
struct LinalgAdRule;

impl ExtensionLinearizeRule for LinalgAdRule {
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
            LinalgOp::Svd { derivative_eps, .. } => rules::linearize_svd(
                builder,
                primal_in,
                primal_out,
                tangent_in,
                derivative_eps,
                ctx,
            ),
            LinalgOp::SvdVals { derivative_eps } => {
                rules::linearize_svd_values(builder, primal_in, tangent_in, derivative_eps, ctx)
            }
            LinalgOp::Qr { .. } => {
                rules::linearize_qr(builder, primal_in, primal_out, tangent_in, ctx)
            }
            LinalgOp::Eigh { derivative_eps, .. } => rules::linearize_eigh(
                builder,
                primal_in,
                primal_out,
                tangent_in,
                derivative_eps,
                ctx,
            ),
            LinalgOp::EighVals { derivative_eps } => {
                rules::linearize_eigh_values(builder, primal_in, tangent_in, derivative_eps, ctx)
            }
            LinalgOp::Eig { input_dtype } => {
                rules::linearize_eig(builder, primal_in, primal_out, tangent_in, input_dtype, ctx)
            }
            LinalgOp::EigVals { input_dtype } => {
                rules::linearize_eig_values(builder, primal_in, tangent_in, input_dtype, ctx)
            }
        }
    }
}

impl ExtensionLinearTransposeRule for LinalgAdRule {
    fn family_id(&self) -> &'static str {
        LINALG_EXTENSION_FAMILY_ID
    }

    fn linear_transpose(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[PrimitiveTransposeInput<StdTensorOp>],
        active_mask: &[bool],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Transpose)?;
        let mut builder = DynBuilder(builder);
        let mode = OperationRole::Linearized {
            active_mask: active_mask.to_vec(),
        };
        match op.op() {
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => {
                let value_inputs =
                    linear_solve_transpose_inputs("triangular_solve", inputs, active_mask)?;
                rules::transpose_triangular_solve(
                    &mut builder,
                    cotangent_out,
                    &value_inputs,
                    &mode,
                    rules::TriangularSolveFlags::new(left_side, lower, transpose_a, unit_diagonal),
                    ctx,
                )
            }
            LinalgOp::LuSolvePrepared {
                transpose_a,
                conjugate_a,
            } => {
                let value_inputs = lu_solve_prepared_transpose_inputs(inputs, active_mask)?;
                rules::transpose_lu_solve_prepared(
                    &mut builder,
                    cotangent_out,
                    &value_inputs,
                    &mode,
                    transpose_a,
                    conjugate_a,
                    ctx,
                )
            }
            LinalgOp::FullPivLuSolve { transpose_a } => {
                let value_inputs =
                    linear_solve_transpose_inputs("full_piv_lu_solve", inputs, active_mask)?;
                rules::transpose_full_piv_lu_solve(
                    &mut builder,
                    cotangent_out,
                    &value_inputs,
                    &mode,
                    transpose_a,
                    ctx,
                )
            }
            LinalgOp::Cholesky
            | LinalgOp::Lu
            | LinalgOp::LuFactor
            | LinalgOp::FullPivLu
            | LinalgOp::Svd { .. }
            | LinalgOp::SvdVals { .. }
            | LinalgOp::Qr { .. }
            | LinalgOp::Eigh { .. }
            | LinalgOp::EighVals { .. }
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

fn linear_solve_transpose_inputs(
    op: &str,
    inputs: &[PrimitiveTransposeInput<StdTensorOp>],
    active_mask: &[bool],
) -> ADRuleResult<Vec<ValueRef<StdTensorOp>>> {
    let matrix_active = active_mask.first().copied().unwrap_or(false);
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            if index == 0 || matrix_active {
                fixed_transpose_value(op, index, input)
            } else {
                Ok(metadata_transpose_value(input))
            }
        })
        .collect()
}

fn lu_solve_prepared_transpose_inputs(
    inputs: &[PrimitiveTransposeInput<StdTensorOp>],
    active_mask: &[bool],
) -> ADRuleResult<Vec<ValueRef<StdTensorOp>>> {
    let matrix_active = active_mask.first().copied().unwrap_or(false);
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            if index <= 2 || matrix_active {
                fixed_transpose_value("lu_solve_prepared", index, input)
            } else {
                Ok(metadata_transpose_value(input))
            }
        })
        .collect()
}

fn metadata_transpose_value(input: &PrimitiveTransposeInput<StdTensorOp>) -> ValueRef<StdTensorOp> {
    ValueRef::External(input.key().clone())
}

fn fixed_transpose_value(
    op: &str,
    index: usize,
    input: &PrimitiveTransposeInput<StdTensorOp>,
) -> ADRuleResult<ValueRef<StdTensorOp>> {
    match input {
        PrimitiveTransposeInput::Residual(key) => Ok(ValueRef::External(key.clone())),
        PrimitiveTransposeInput::Linear {
            primal: Some(primal),
            ..
        } => Ok(ValueRef::External(primal.clone())),
        PrimitiveTransposeInput::Linear { key, primal: None } => {
            Err(ADRuleError::invalid_input(
                op,
                ADRuleKind::Transpose,
                format!(
                    "transpose input {index} is linear-only and cannot be retained as a tensor operand: {key:?}"
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{EighGauge, QrGauge, SvdGauge, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS};
    use computegraph::graph::GraphBuilder;
    use std::collections::HashSet;
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

    fn insert_typed_meta(
        ctx: &mut ShapeGuardContext,
        key: ValueKey<StdTensorOp>,
        dtype: DType,
        shape: &[usize],
    ) {
        ctx.insert_metadata(
            key,
            TensorMeta::exact(dtype, shape.iter().copied().map(SymDim::from).collect()),
        );
    }

    fn eigh_context() -> (
        ShapeGuardContext,
        ValueKey<StdTensorOp>,
        Vec<ValueKey<StdTensorOp>>,
    ) {
        let mut ctx = ShapeGuardContext::default();
        let a = input_key(1);
        let w = input_key(2);
        let v = input_key(3);
        insert_typed_meta(&mut ctx, a.clone(), DType::F64, &[2, 2]);
        insert_typed_meta(&mut ctx, w.clone(), DType::F64, &[2]);
        insert_typed_meta(&mut ctx, v.clone(), DType::F64, &[2, 2]);
        (ctx, a, vec![w, v])
    }

    fn eig_context() -> (
        ShapeGuardContext,
        ValueKey<StdTensorOp>,
        Vec<ValueKey<StdTensorOp>>,
    ) {
        let mut ctx = ShapeGuardContext::default();
        let a = input_key(114);
        let w = input_key(115);
        let v = input_key(116);
        insert_typed_meta(&mut ctx, a.clone(), DType::F64, &[2, 2]);
        insert_typed_meta(&mut ctx, w.clone(), DType::C64, &[2]);
        insert_typed_meta(&mut ctx, v.clone(), DType::C64, &[2, 2]);
        (ctx, a, vec![w, v])
    }

    fn lu_context(
        shape: &[usize],
    ) -> (
        ShapeGuardContext,
        ValueKey<StdTensorOp>,
        Vec<ValueKey<StdTensorOp>>,
    ) {
        let mut ctx = ShapeGuardContext::default();
        let a = input_key(4);
        let p = input_key(5);
        let l = input_key(6);
        let u = input_key(7);
        let parity = input_key(8);
        let k = shape[0].min(shape[1]);
        insert_typed_meta(&mut ctx, a.clone(), DType::F64, shape);
        insert_typed_meta(&mut ctx, p.clone(), DType::F64, &[shape[0], shape[0]]);
        insert_typed_meta(&mut ctx, l.clone(), DType::F64, &[shape[0], k]);
        insert_typed_meta(&mut ctx, u.clone(), DType::F64, &[k, shape[1]]);
        insert_typed_meta(&mut ctx, parity.clone(), DType::F64, &[]);
        (ctx, a, vec![p, l, u, parity])
    }

    fn svd_context(
        shape: &[usize],
    ) -> (
        ShapeGuardContext,
        ValueKey<StdTensorOp>,
        Vec<ValueKey<StdTensorOp>>,
    ) {
        let mut ctx = ShapeGuardContext::default();
        let a = input_key(120);
        let u = input_key(121);
        let s = input_key(122);
        let vt = input_key(123);
        let k = shape[0].min(shape[1]);
        insert_typed_meta(&mut ctx, a.clone(), DType::F64, shape);
        insert_typed_meta(&mut ctx, u.clone(), DType::F64, &[shape[0], k]);
        insert_typed_meta(&mut ctx, s.clone(), DType::F64, &[k]);
        insert_typed_meta(&mut ctx, vt.clone(), DType::F64, &[k, shape[1]]);
        (ctx, a, vec![u, s, vt])
    }

    fn qr_context(
        shape: &[usize],
    ) -> (
        ShapeGuardContext,
        ValueKey<StdTensorOp>,
        Vec<ValueKey<StdTensorOp>>,
    ) {
        let mut ctx = ShapeGuardContext::default();
        let a = input_key(9);
        let q = input_key(10);
        let r = input_key(11);
        let k = shape[0].min(shape[1]);
        insert_typed_meta(&mut ctx, a.clone(), DType::F64, shape);
        insert_typed_meta(&mut ctx, q.clone(), DType::F64, &[shape[0], k]);
        insert_typed_meta(&mut ctx, r.clone(), DType::F64, &[k, shape[1]]);
        (ctx, a, vec![q, r])
    }

    fn with_active_values(
        ctx: ShapeGuardContext,
        values: impl IntoIterator<Item = ValueKey<StdTensorOp>>,
    ) -> ShapeGuardContext {
        ctx.with_linearize_active_values(Arc::new(values.into_iter().collect::<HashSet<_>>()))
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
    fn lu_linearize_prunes_inactive_factor_outputs() {
        let op = LinalgExtensionOp::new(LinalgOp::Lu);
        for (case, active_slot, expected_active) in [
            ("l only", 1_usize, vec![false, true, false, false]),
            ("u only", 2_usize, vec![false, false, true, false]),
        ] {
            let (ctx, a, outputs) = lu_context(&[2, 2]);
            let mut ctx = with_active_values(ctx, [outputs[active_slot].clone()]);
            let mut builder = GraphBuilder::<StdTensorOp>::new();
            let tangent = builder.add_input(TensorInputKey::User { id: 130 });

            let result = LinalgAdRule
                .linearize(
                    &op,
                    &mut builder,
                    &[a],
                    &outputs,
                    &[Some(tangent)],
                    &mut ctx,
                )
                .unwrap();

            assert_eq!(
                result.iter().map(Option::is_some).collect::<Vec<_>>(),
                expected_active,
                "{case}"
            );
            let pruned_count = builder.build().operations().len();

            let (full_ctx, full_a, full_outputs) = lu_context(&[2, 2]);
            let mut full_ctx =
                with_active_values(full_ctx, [full_outputs[1].clone(), full_outputs[2].clone()]);
            let mut full_builder = GraphBuilder::<StdTensorOp>::new();
            let full_tangent = full_builder.add_input(TensorInputKey::User { id: 131 });
            let full_result = LinalgAdRule
                .linearize(
                    &op,
                    &mut full_builder,
                    &[full_a],
                    &full_outputs,
                    &[Some(full_tangent)],
                    &mut full_ctx,
                )
                .unwrap();

            assert_eq!(
                full_result.iter().map(Option::is_some).collect::<Vec<_>>(),
                vec![false, true, true, false],
                "{case}"
            );
            let full_count = full_builder.build().operations().len();
            assert!(
                pruned_count < full_count,
                "{case} should not emit both LU factor tangent branches: {pruned_count} >= {full_count}"
            );
        }
    }

    #[test]
    fn one_input_linalg_jvps_prune_when_all_outputs_are_inactive() {
        let cases = [
            (
                LinalgOp::Lu,
                lu_context(&[2, 2]),
                vec![None, None, None, None],
            ),
            (
                LinalgOp::Svd {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: SvdGauge::Raw,
                },
                svd_context(&[2, 2]),
                vec![None, None, None],
            ),
            (
                LinalgOp::Eigh {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: EighGauge::Raw,
                },
                eigh_context(),
                vec![None, None],
            ),
            (
                LinalgOp::Eig {
                    input_dtype: DType::F64,
                },
                eig_context(),
                vec![None, None],
            ),
            (
                LinalgOp::Qr {
                    gauge: QrGauge::Raw,
                },
                qr_context(&[3, 2]),
                vec![None, None],
            ),
        ];

        for (kind, (ctx, a, outputs), expected) in cases {
            let mut ctx = with_active_values(ctx, []);
            let mut builder = GraphBuilder::<StdTensorOp>::new();
            let tangent = builder.add_input(TensorInputKey::User { id: 132 });
            let op = LinalgExtensionOp::new(kind);

            let result = LinalgAdRule
                .linearize(
                    &op,
                    &mut builder,
                    &[a],
                    &outputs,
                    &[Some(tangent)],
                    &mut ctx,
                )
                .unwrap();

            assert_eq!(result, expected, "{kind:?}");
            assert!(
                builder.build().operations().is_empty(),
                "{kind:?} should not emit AD graph operations for inactive outputs"
            );
        }
    }

    #[test]
    fn svd_linearize_prunes_inactive_vector_outputs() {
        let (ctx, a, outputs) = svd_context(&[2, 2]);
        let mut ctx = with_active_values(ctx, [outputs[1].clone()]);
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let tangent = builder.add_input(TensorInputKey::User { id: 133 });
        let op = LinalgExtensionOp::new(LinalgOp::Svd {
            derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
            gauge: SvdGauge::Raw,
        });

        let result = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[a],
                &outputs,
                &[Some(tangent)],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(
            result.iter().map(Option::is_some).collect::<Vec<_>>(),
            vec![false, true, false]
        );
        assert!(
            builder.build().operations().len() <= 5,
            "singular-value-only SVD JVP should not emit the vector F-matrix chain"
        );
    }

    #[test]
    fn eigh_linearize_prunes_inactive_eigenvalue_output() {
        let (ctx, a, outputs) = eigh_context();
        let mut ctx = with_active_values(ctx, [outputs[1].clone()]);
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let tangent = builder.add_input(TensorInputKey::User { id: 134 });
        let op = LinalgExtensionOp::new(LinalgOp::Eigh {
            derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
            gauge: EighGauge::Raw,
        });

        let result = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[a],
                &outputs,
                &[Some(tangent)],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(
            result.iter().map(Option::is_some).collect::<Vec<_>>(),
            vec![false, true]
        );
    }

    #[test]
    fn eig_linearize_prunes_unsupported_inactive_eigenvalue_output() {
        let (ctx, a, outputs) = eig_context();
        let mut ctx = with_active_values(ctx, [outputs[1].clone()]);
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let tangent = builder.add_input(TensorInputKey::User { id: 137 });
        let op = LinalgExtensionOp::new(LinalgOp::Eig {
            input_dtype: DType::F64,
        });

        let result = LinalgAdRule
            .linearize(
                &op,
                &mut builder,
                &[a],
                &outputs,
                &[Some(tangent)],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result, vec![None, None]);
        assert!(
            builder.build().operations().is_empty(),
            "eigenvectors-only Eig JVP is unsupported and should not emit eigenvalue tangent work"
        );
    }

    #[test]
    fn qr_linearize_prunes_inactive_factor_outputs() {
        let op = LinalgExtensionOp::new(LinalgOp::Qr {
            gauge: QrGauge::Raw,
        });
        for (case, active_slot, expected_active) in [
            ("q only", 0_usize, vec![true, false]),
            ("r only", 1_usize, vec![false, true]),
        ] {
            let (ctx, a, outputs) = qr_context(&[3, 2]);
            let mut ctx = with_active_values(ctx, [outputs[active_slot].clone()]);
            let mut builder = GraphBuilder::<StdTensorOp>::new();
            let tangent = builder.add_input(TensorInputKey::User { id: 135 });

            let result = LinalgAdRule
                .linearize(
                    &op,
                    &mut builder,
                    &[a],
                    &outputs,
                    &[Some(tangent)],
                    &mut ctx,
                )
                .unwrap();

            assert_eq!(
                result.iter().map(Option::is_some).collect::<Vec<_>>(),
                expected_active,
                "{case}"
            );
            let pruned_count = builder.build().operations().len();

            let (full_ctx, full_a, full_outputs) = qr_context(&[3, 2]);
            let mut full_ctx =
                with_active_values(full_ctx, [full_outputs[0].clone(), full_outputs[1].clone()]);
            let mut full_builder = GraphBuilder::<StdTensorOp>::new();
            let full_tangent = full_builder.add_input(TensorInputKey::User { id: 136 });
            let full_result = LinalgAdRule
                .linearize(
                    &op,
                    &mut full_builder,
                    &[full_a],
                    &full_outputs,
                    &[Some(full_tangent)],
                    &mut full_ctx,
                )
                .unwrap();

            assert_eq!(
                full_result.iter().map(Option::is_some).collect::<Vec<_>>(),
                vec![true, true],
                "{case}"
            );
            let full_count = full_builder.build().operations().len();
            assert!(
                pruned_count < full_count,
                "{case} should not emit both QR factor tangent branches: {pruned_count} >= {full_count}"
            );
        }
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
            .linear_transpose(
                &op,
                &mut builder,
                &[Some(cotangent)],
                &[
                    PrimitiveTransposeInput::Residual(lhs.clone()),
                    PrimitiveTransposeInput::Residual(rhs),
                ],
                &[false, true],
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
    fn eigh_values_has_no_handwritten_direct_transpose() {
        let (mut ctx, a, _primal_outputs) = eigh_context();
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let cotangent = builder.add_input(TensorInputKey::User { id: 85 });
        let op = LinalgExtensionOp::new(LinalgOp::EighVals {
            derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
        });

        let result = LinalgAdRule
            .linear_transpose(
                &op,
                &mut builder,
                &[Some(cotangent)],
                &[PrimitiveTransposeInput::Residual(a)],
                &[true],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result, vec![None]);
        assert!(
            builder.build().operations().is_empty(),
            "EighVals reverse support should come from linearize + generic transpose"
        );
    }

    #[test]
    fn full_eigh_has_no_handwritten_direct_transpose() {
        let (mut ctx, a, _primal_outputs) = eigh_context();
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let g_w = builder.add_input(TensorInputKey::User { id: 86 });
        let g_v = builder.add_input(TensorInputKey::User { id: 87 });
        let op = LinalgExtensionOp::new(LinalgOp::Eigh {
            derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
            gauge: EighGauge::Raw,
        });

        let result = LinalgAdRule
            .linear_transpose(
                &op,
                &mut builder,
                &[Some(g_w), Some(g_v)],
                &[PrimitiveTransposeInput::Residual(a)],
                &[true],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result, vec![None]);
        assert!(
            builder.build().operations().is_empty(),
            "Eigh reverse support should come from linearize + generic transpose"
        );
    }

    #[test]
    fn qr_has_no_handwritten_direct_transpose() {
        let (mut ctx, a, _primal_outputs) = qr_context(&[3, 2]);
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let g_q = builder.add_input(TensorInputKey::User { id: 88 });
        let g_r = builder.add_input(TensorInputKey::User { id: 89 });

        let result = LinalgAdRule
            .linear_transpose(
                &LinalgExtensionOp::new(LinalgOp::Qr {
                    gauge: QrGauge::Raw,
                }),
                &mut builder,
                &[Some(g_q), Some(g_r)],
                &[PrimitiveTransposeInput::Residual(a)],
                &[true],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result, vec![None]);
        assert!(
            builder.build().operations().is_empty(),
            "QR reverse support should come from linearize + generic transpose"
        );
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
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                gauge: SvdGauge::Raw,
            },
            LinalgOp::SvdVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
            },
            LinalgOp::Qr {
                gauge: QrGauge::Raw,
            },
            LinalgOp::Eigh {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                gauge: EighGauge::Raw,
            },
            LinalgOp::EighVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
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
