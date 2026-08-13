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
//!     .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
//!     .unwrap()
//!     .build()
//!     .unwrap();
//! let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
//! let (_u, s, _vt) = x.svd().unwrap();
//! let loss = s.reduce_sum(Some(&[0])).unwrap();
//! let grad = ad.grad(&loss, &x).unwrap();
//! assert_eq!(grad.rank, 2);
//! ```

#[cfg(test)]
use std::sync::Arc;

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ad::extension::ExtensionOp;
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::ad::{ADRuleError, ADRuleKind, ADRuleResult, PrimitiveTransposeInput};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeGuardContext;

use crate::extension::{LinalgExtensionOp, LinalgOp};
#[cfg(test)]
use crate::LINALG_EXTENSION_FAMILY_ID;

mod rules;
mod semantic;
pub mod support;

pub use semantic::semantic_ad_rules;

#[derive(Debug)]
struct LinalgAdRule;

impl LinalgAdRule {
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
            LinalgOp::SignDetFromLuFactor => rules::linearize_signdet_from_lu_factor(
                builder, primal_in, primal_out, tangent_in, ctx,
            ),
            LinalgOp::LogAbsDetFromLuFactor => {
                rules::linearize_logabsdet_from_lu_factor(builder, primal_in, tangent_in, ctx)
            }
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
            LinalgOp::Solve => {
                rules::linearize_solve(builder, primal_in, primal_out, tangent_in, false, ctx)
            }
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
            // Full-matrices SVD is value-only; AD is intentionally unsupported
            // (recorded as such in the linalg AD support manifest). Emit no
            // tangent rather than a silent thin-SVD derivative, matching the
            // LuFactor unsupported precedent above.
            LinalgOp::SvdFull => Ok(vec![None; op.output_count()]),
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
            LinalgOp::Solve => {
                let value_inputs = linear_solve_transpose_inputs("solve", inputs, active_mask)?;
                rules::transpose_solve(
                    &mut builder,
                    cotangent_out,
                    &value_inputs,
                    &mode,
                    false,
                    ctx,
                )
            }
            LinalgOp::Cholesky
            | LinalgOp::Lu
            | LinalgOp::LuFactor
            | LinalgOp::SignDetFromLuFactor
            | LinalgOp::LogAbsDetFromLuFactor
            | LinalgOp::FullPivLu
            | LinalgOp::Svd { .. }
            | LinalgOp::SvdFull
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

    fn cpu_runtime_with_linalg(backend: &tenferro_cpu::CpuBackend) -> tenferro_runtime::Runtime {
        let mut builder = tenferro_runtime::Runtime::builder();
        builder
            .register_engine(tenferro_cpu::runtime_engine_registration(backend).unwrap())
            .unwrap();
        builder
            .install_extension_module(
                crate::extension::extension_module::<tenferro_cpu::CpuBackend>(
                    tenferro_cpu::runtime_engine_id().unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        builder.build().unwrap()
    }

    fn run_one(
        runtime: &tenferro_runtime::Runtime,
        program: &tenferro_runtime::CompiledGraph,
        inputs: &[&tenferro_tensor::Tensor],
    ) -> tenferro_runtime::Result<tenferro_tensor::Tensor> {
        let mut outputs = runtime.run_compiled(program, inputs)?;
        assert_eq!(outputs.len(), 1);
        Ok(outputs.remove(0))
    }

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
            LinalgOp::SvdFull,
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

    #[test]
    fn triangular_solve_semantic_rules_execute_jvp_and_vjp_numerically() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
        use tenferro_runtime::GraphCompiler;
        use tenferro_tensor::Tensor;

        let mut builder = SemanticProgramBuilder::new();
        let matrix = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::Const(2), DimExpr::Const(2)],
            ))
            .unwrap();
        let rhs = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::Const(2), DimExpr::Const(1)],
            ))
            .unwrap();
        let solution = builder
            .add_extension(
                Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                })),
                &[matrix, rhs],
            )
            .unwrap()[0];
        let source = builder.finish(&[solution]).unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 4.0]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 8.0]).unwrap();
        let rhs_tangent = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 4.0]).unwrap();
        let jvp = ad.jvp_program(&source, &[false, true]).unwrap();
        let compiled = GraphCompiler::new()
            .compile_frozen_program(jvp.frozen())
            .unwrap();
        let backend = tenferro_cpu::CpuBackend::new();
        let runtime = cpu_runtime_with_linalg(&backend);
        let tangent = run_one(&runtime, &compiled, &[&matrix, &rhs, &rhs_tangent]).unwrap();
        assert_eq!(tangent.as_slice::<f64>().unwrap(), &[1.0, 0.75]);

        let output_cotangent = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 1.0]).unwrap();
        let vjp = ad.vjp_program(&source, &[false, true], &[true]).unwrap();
        let compiled = GraphCompiler::new()
            .compile_frozen_program(vjp.frozen())
            .unwrap();
        let cotangent = run_one(&runtime, &compiled, &[&matrix, &rhs, &output_cotangent]).unwrap();
        assert_eq!(cotangent.as_slice::<f64>().unwrap(), &[0.375, 0.25]);
    }

    #[test]
    fn triangular_solve_semantic_vjp_reuses_imported_primal_solution() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

        let mut builder = SemanticProgramBuilder::new();
        let matrix = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::Const(4), DimExpr::Const(4)],
            ))
            .unwrap();
        let rhs = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::Const(4), DimExpr::Const(2)],
            ))
            .unwrap();
        let solution = builder
            .add_extension(
                Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                })),
                &[matrix, rhs],
            )
            .unwrap()[0];
        let source = builder.finish(&[solution]).unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
        let triangular_solve_count = vjp
            .frozen()
            .program
            .operations()
            .filter(|operation| match operation.op() {
                SemanticOpRef::Extension(extension) => extension
                    .as_any()
                    .downcast_ref::<LinalgExtensionOp>()
                    .is_some_and(|op| matches!(op.op(), LinalgOp::TriangularSolve { .. })),
                _ => false,
            })
            .count();

        assert_eq!(
            triangular_solve_count, 2,
            "semantic VJP should contain only the imported primal solve and the adjoint solve"
        );
    }

    #[test]
    fn eigh_values_semantic_rules_execute_jvp_and_vjp_numerically() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
        use tenferro_runtime::GraphCompiler;
        use tenferro_tensor::Tensor;

        let mut builder = SemanticProgramBuilder::new();
        let matrix = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::Const(2), DimExpr::Const(2)],
            ))
            .unwrap();
        let eigenvalues = builder
            .add_extension(
                Arc::new(LinalgExtensionOp::new(LinalgOp::EighVals {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                })),
                &[matrix],
            )
            .unwrap()[0];
        let source = builder.finish(&[eigenvalues]).unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();
        let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]).unwrap();
        let tangent = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
        let backend = tenferro_cpu::CpuBackend::new();
        let runtime = cpu_runtime_with_linalg(&backend);

        let jvp = ad.jvp_program(&source, &[true]).unwrap();
        let compiled = GraphCompiler::new()
            .compile_frozen_program(jvp.frozen())
            .unwrap();
        let tangent_output = run_one(&runtime, &compiled, &[&matrix, &tangent]).unwrap();
        assert_eq!(tangent_output.as_slice::<f64>().unwrap(), &[2.0, 4.0]);

        let output_cotangent = Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap();
        let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
        let compiled = GraphCompiler::new()
            .compile_frozen_program(vjp.frozen())
            .unwrap();
        let cotangent = run_one(&runtime, &compiled, &[&matrix, &output_cotangent]).unwrap();
        assert_eq!(cotangent.as_slice::<f64>().unwrap(), &[5.0, 0.0, 0.0, 7.0]);
    }

    fn semantic_real_inner_product(
        lhs: &tenferro_tensor::Tensor,
        rhs: &tenferro_tensor::Tensor,
    ) -> f64 {
        use num_complex::Complex64;
        use tenferro_tensor::Tensor;

        assert_eq!(lhs.dtype(), rhs.dtype());
        assert_eq!(lhs.shape(), rhs.shape());
        match (lhs, rhs) {
            (Tensor::F64(lhs), Tensor::F64(rhs)) => lhs
                .as_slice()
                .unwrap()
                .iter()
                .zip(rhs.as_slice().unwrap())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum(),
            (Tensor::C64(lhs), Tensor::C64(rhs)) => lhs
                .as_slice()
                .unwrap()
                .iter()
                .zip(rhs.as_slice().unwrap())
                .map(|(lhs, rhs): (&Complex64, &Complex64)| (lhs.conj() * rhs).re)
                .sum(),
            _ => panic!("semantic linalg parity helper received {:?}", lhs.dtype()),
        }
    }

    #[test]
    fn supported_one_input_semantic_rules_execute_jvp_vjp_adjoint_parity() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::ad_support::ones_tensor;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
        use tenferro_runtime::GraphCompiler;
        use tenferro_tensor::Tensor;

        struct Case {
            name: &'static str,
            op: LinalgOp,
            active_outputs: &'static [usize],
            matrix: [f64; 4],
            tangent: [f64; 4],
        }

        let general = [3.0, 0.5, 1.0, 2.0];
        let general_tangent = [0.2, -0.1, 0.3, 0.4];
        let symmetric = [4.0, 0.5, 0.5, 2.0];
        let symmetric_tangent = [0.2, 0.1, 0.1, 0.4];
        let cases = [
            Case {
                name: "cholesky",
                op: LinalgOp::Cholesky,
                active_outputs: &[0],
                matrix: symmetric,
                tangent: symmetric_tangent,
            },
            Case {
                name: "lu",
                op: LinalgOp::Lu,
                active_outputs: &[1, 2],
                matrix: general,
                tangent: general_tangent,
            },
            Case {
                name: "full_piv_lu",
                op: LinalgOp::FullPivLu,
                active_outputs: &[1, 2],
                matrix: general,
                tangent: general_tangent,
            },
            Case {
                name: "svd",
                op: LinalgOp::Svd {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: SvdGauge::Raw,
                },
                active_outputs: &[0, 1, 2],
                matrix: general,
                tangent: general_tangent,
            },
            Case {
                name: "svd_values",
                op: LinalgOp::SvdVals {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                },
                active_outputs: &[0],
                matrix: general,
                tangent: general_tangent,
            },
            Case {
                name: "qr",
                op: LinalgOp::Qr {
                    gauge: QrGauge::Raw,
                },
                active_outputs: &[0, 1],
                matrix: general,
                tangent: general_tangent,
            },
            Case {
                name: "eigh",
                op: LinalgOp::Eigh {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: EighGauge::Raw,
                },
                active_outputs: &[0, 1],
                matrix: symmetric,
                tangent: symmetric_tangent,
            },
            Case {
                name: "eigh_values",
                op: LinalgOp::EighVals {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                },
                active_outputs: &[0],
                matrix: symmetric,
                tangent: symmetric_tangent,
            },
            Case {
                name: "eig",
                op: LinalgOp::Eig {
                    input_dtype: DType::F64,
                },
                active_outputs: &[0],
                matrix: general,
                tangent: general_tangent,
            },
            Case {
                name: "eig_values",
                op: LinalgOp::EigVals {
                    input_dtype: DType::F64,
                },
                active_outputs: &[0],
                matrix: general,
                tangent: general_tangent,
            },
        ];
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        for case in cases {
            let mut builder = SemanticProgramBuilder::new();
            let matrix_value = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(2), DimExpr::Const(2)],
                ))
                .unwrap();
            let all_outputs = builder
                .add_extension(Arc::new(LinalgExtensionOp::new(case.op)), &[matrix_value])
                .unwrap();
            let roots: Vec<_> = case
                .active_outputs
                .iter()
                .map(|index| all_outputs[*index])
                .collect();
            let source = builder.finish(&roots).unwrap();
            let matrix = Tensor::from_vec_col_major(vec![2, 2], case.matrix.to_vec()).unwrap();
            let tangent = Tensor::from_vec_col_major(vec![2, 2], case.tangent.to_vec()).unwrap();
            let backend = tenferro_cpu::CpuBackend::new();
            let runtime = cpu_runtime_with_linalg(&backend);

            let primal = GraphCompiler::new()
                .compile_frozen_program(&source)
                .unwrap();
            let primal_outputs = runtime.run_compiled(&primal, &[&matrix]).unwrap();
            let cotangents: Vec<_> = primal_outputs
                .iter()
                .map(|output| ones_tensor(output.dtype(), output.shape().to_vec()).unwrap())
                .collect();

            let jvp = ad.jvp_program(&source, &[true]).unwrap();
            let jvp = GraphCompiler::new()
                .compile_frozen_program(jvp.frozen())
                .unwrap();
            let tangent_outputs = runtime.run_compiled(&jvp, &[&matrix, &tangent]).unwrap();
            assert_eq!(
                tangent_outputs.len(),
                cotangents.len(),
                "{} JVP output order",
                case.name
            );

            let vjp = ad
                .vjp_program(&source, &[true], &vec![true; roots.len()])
                .unwrap();
            let vjp = GraphCompiler::new()
                .compile_frozen_program(vjp.frozen())
                .unwrap();
            let mut vjp_inputs = vec![&matrix];
            vjp_inputs.extend(cotangents.iter());
            let input_cotangents = runtime
                .run_compiled(&vjp, &vjp_inputs)
                .unwrap_or_else(|error| panic!("{} semantic VJP execution: {error}", case.name));
            assert_eq!(input_cotangents.len(), 1, "{} VJP input order", case.name);

            let lhs: f64 = tangent_outputs
                .iter()
                .zip(&cotangents)
                .map(|(tangent, cotangent)| semantic_real_inner_product(cotangent, tangent))
                .sum();
            let rhs = semantic_real_inner_product(&tangent, &input_cotangents[0]);
            let tolerance = 1.0e-8 * lhs.abs().max(rhs.abs()).max(1.0);
            assert!(
                (lhs - rhs).abs() <= tolerance,
                "{} semantic adjoint parity failed: <ct,J dx>={lhs}, <J^T ct,dx>={rhs}, tolerance={tolerance}",
                case.name
            );
        }
    }

    #[test]
    fn semantic_solve_rules_cover_flags_and_complex_adjoint_parity() {
        use num_complex::Complex64;
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::ad_support::ones_tensor;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
        use tenferro_runtime::GraphCompiler;
        use tenferro_tensor::Tensor;

        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        for complex in [false, true] {
            let dtype = if complex { DType::C64 } else { DType::F64 };
            for left_side in [false, true] {
                for lower in [false, true] {
                    for transpose_a in [false, true] {
                        for unit_diagonal in [false, true] {
                            let rhs_shape = if left_side { [2, 1] } else { [1, 2] };
                            let mut builder = SemanticProgramBuilder::new();
                            let matrix_value = builder
                                .input(ProgramInputSpec::new(
                                    dtype,
                                    [DimExpr::Const(2), DimExpr::Const(2)],
                                ))
                                .unwrap();
                            let rhs_value = builder
                                .input(ProgramInputSpec::new(
                                    dtype,
                                    rhs_shape.into_iter().map(DimExpr::Const),
                                ))
                                .unwrap();
                            let solution = builder
                                .add_extension(
                                    Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
                                        left_side,
                                        lower,
                                        transpose_a,
                                        unit_diagonal,
                                    })),
                                    &[matrix_value, rhs_value],
                                )
                                .unwrap()[0];
                            let source = builder.finish(&[solution]).unwrap();
                            let (matrix, matrix_tangent, rhs, rhs_tangent) = if complex {
                                (
                                    Tensor::from_vec_col_major(
                                        vec![2, 2],
                                        vec![
                                            Complex64::new(2.0, 0.5),
                                            Complex64::new(0.75, -0.2),
                                            Complex64::new(-0.5, 0.3),
                                            Complex64::new(3.0, -0.25),
                                        ],
                                    )
                                    .unwrap(),
                                    Tensor::from_vec_col_major(
                                        vec![2, 2],
                                        vec![
                                            Complex64::new(0.2, 0.1),
                                            Complex64::new(0.05, -0.03),
                                            Complex64::new(-0.08, 0.04),
                                            Complex64::new(0.4, -0.1),
                                        ],
                                    )
                                    .unwrap(),
                                    Tensor::from_vec_col_major(
                                        rhs_shape.to_vec(),
                                        vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)],
                                    )
                                    .unwrap(),
                                    Tensor::from_vec_col_major(
                                        rhs_shape.to_vec(),
                                        vec![Complex64::new(0.3, -0.2), Complex64::new(-0.1, 0.4)],
                                    )
                                    .unwrap(),
                                )
                            } else {
                                (
                                    Tensor::from_vec_col_major(
                                        vec![2, 2],
                                        vec![2.0_f64, 0.75, -0.5, 3.0],
                                    )
                                    .unwrap(),
                                    Tensor::from_vec_col_major(
                                        vec![2, 2],
                                        vec![0.2_f64, 0.05, -0.08, 0.4],
                                    )
                                    .unwrap(),
                                    Tensor::from_vec_col_major(
                                        rhs_shape.to_vec(),
                                        vec![1.0_f64, 2.0],
                                    )
                                    .unwrap(),
                                    Tensor::from_vec_col_major(
                                        rhs_shape.to_vec(),
                                        vec![0.3_f64, -0.1],
                                    )
                                    .unwrap(),
                                )
                            };
                            let backend = tenferro_cpu::CpuBackend::new();
                            let runtime = cpu_runtime_with_linalg(&backend);

                            let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
                            let jvp = GraphCompiler::new()
                                .compile_frozen_program(jvp.frozen())
                                .unwrap();
                            let tangent_output = run_one(
                                &runtime,
                                &jvp,
                                &[&matrix, &rhs, &matrix_tangent, &rhs_tangent],
                            )
                            .unwrap();
                            let cotangent =
                                ones_tensor(dtype, tangent_output.shape().to_vec()).unwrap();

                            let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
                            let vjp = GraphCompiler::new()
                                .compile_frozen_program(vjp.frozen())
                                .unwrap();
                            let input_cotangents = runtime
                                .run_compiled(&vjp, &[&matrix, &rhs, &cotangent])
                                .unwrap();
                            assert_eq!(input_cotangents.len(), 2);

                            let lhs = semantic_real_inner_product(&cotangent, &tangent_output);
                            let rhs =
                                semantic_real_inner_product(&matrix_tangent, &input_cotangents[0])
                                    + semantic_real_inner_product(
                                        &rhs_tangent,
                                        &input_cotangents[1],
                                    );
                            let tolerance = 1.0e-8 * lhs.abs().max(rhs.abs()).max(1.0);
                            assert!(
                                (lhs - rhs).abs() <= tolerance,
                                "triangular solve semantic adjoint parity failed for complex={complex}, left={left_side}, lower={lower}, transpose={transpose_a}, unit={unit_diagonal}: lhs={lhs}, rhs={rhs}, tolerance={tolerance}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn semantic_prepared_and_full_pivot_solve_execute_all_flags_with_adjoint_parity() {
        use num_complex::Complex64;
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::ad_support::ones_tensor;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
        use tenferro_runtime::GraphCompiler;
        use tenferro_tensor::Tensor;

        #[derive(Clone, Copy)]
        enum SolveCase {
            Prepared {
                transpose_a: bool,
                conjugate_a: bool,
            },
            FullPivot {
                transpose_a: bool,
            },
        }

        let mut cases = Vec::new();
        for transpose_a in [false, true] {
            for conjugate_a in [false, true] {
                cases.push(SolveCase::Prepared {
                    transpose_a,
                    conjugate_a,
                });
            }
            cases.push(SolveCase::FullPivot { transpose_a });
        }

        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        for complex in [false, true] {
            let dtype = if complex { DType::C64 } else { DType::F64 };
            for case in &cases {
                let mut builder = SemanticProgramBuilder::new();
                let matrix_value = builder
                    .input(ProgramInputSpec::new(
                        dtype,
                        [DimExpr::Const(2), DimExpr::Const(2)],
                    ))
                    .unwrap();
                let rhs_value = builder
                    .input(ProgramInputSpec::new(
                        dtype,
                        [DimExpr::Const(2), DimExpr::Const(1)],
                    ))
                    .unwrap();
                let (name, solution) = match *case {
                    SolveCase::Prepared {
                        transpose_a,
                        conjugate_a,
                    } => {
                        let packed_lu = builder
                            .input(ProgramInputSpec::new(
                                dtype,
                                [DimExpr::Const(2), DimExpr::Const(2)],
                            ))
                            .unwrap();
                        let pivots = builder
                            .input(ProgramInputSpec::new(DType::I32, [DimExpr::Const(2)]))
                            .unwrap();
                        let solution = builder
                            .add_extension(
                                Arc::new(LinalgExtensionOp::new(LinalgOp::LuSolvePrepared {
                                    transpose_a,
                                    conjugate_a,
                                })),
                                &[matrix_value, packed_lu, pivots, rhs_value],
                            )
                            .unwrap()[0];
                        (
                            format!("prepared transpose={transpose_a} conjugate={conjugate_a}"),
                            solution,
                        )
                    }
                    SolveCase::FullPivot { transpose_a } => {
                        let solution = builder
                            .add_extension(
                                Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLuSolve {
                                    transpose_a,
                                })),
                                &[matrix_value, rhs_value],
                            )
                            .unwrap()[0];
                        (format!("full-pivot transpose={transpose_a}"), solution)
                    }
                };
                let source = builder.finish(&[solution]).unwrap();

                let (matrix, matrix_tangent, rhs, rhs_tangent) = if complex {
                    (
                        Tensor::from_vec_col_major(
                            vec![2, 2],
                            vec![
                                Complex64::new(2.0, 0.5),
                                Complex64::new(0.75, -0.2),
                                Complex64::new(-0.5, 0.3),
                                Complex64::new(3.0, -0.25),
                            ],
                        )
                        .unwrap(),
                        Tensor::from_vec_col_major(
                            vec![2, 2],
                            vec![
                                Complex64::new(0.2, 0.1),
                                Complex64::new(0.05, -0.03),
                                Complex64::new(-0.08, 0.04),
                                Complex64::new(0.4, -0.1),
                            ],
                        )
                        .unwrap(),
                        Tensor::from_vec_col_major(
                            vec![2, 1],
                            vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)],
                        )
                        .unwrap(),
                        Tensor::from_vec_col_major(
                            vec![2, 1],
                            vec![Complex64::new(0.3, -0.2), Complex64::new(-0.1, 0.4)],
                        )
                        .unwrap(),
                    )
                } else {
                    (
                        Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.75, -0.5, 3.0])
                            .unwrap(),
                        Tensor::from_vec_col_major(vec![2, 2], vec![0.2_f64, 0.05, -0.08, 0.4])
                            .unwrap(),
                        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap(),
                        Tensor::from_vec_col_major(vec![2, 1], vec![0.3_f64, -0.1]).unwrap(),
                    )
                };

                let backend = tenferro_cpu::CpuBackend::new();
                let runtime = cpu_runtime_with_linalg(&backend);
                let factor_outputs = if matches!(case, SolveCase::Prepared { .. }) {
                    let mut factor_builder = SemanticProgramBuilder::new();
                    let factor_matrix = factor_builder
                        .input(ProgramInputSpec::new(
                            dtype,
                            [DimExpr::Const(2), DimExpr::Const(2)],
                        ))
                        .unwrap();
                    let factor = factor_builder
                        .add_extension(
                            Arc::new(LinalgExtensionOp::new(LinalgOp::LuFactor)),
                            &[factor_matrix],
                        )
                        .unwrap();
                    let factor_source = factor_builder.finish(&factor).unwrap();
                    let factor_program = GraphCompiler::new()
                        .compile_frozen_program(&factor_source)
                        .unwrap();
                    runtime.run_compiled(&factor_program, &[&matrix]).unwrap()
                } else {
                    Vec::new()
                };
                let active_inputs: &[bool] = if matches!(case, SolveCase::Prepared { .. }) {
                    &[true, true, false, false]
                } else {
                    &[true, true]
                };
                let mut primal_inputs = vec![&matrix, &rhs];
                if matches!(case, SolveCase::Prepared { .. }) {
                    primal_inputs.extend([&factor_outputs[0], &factor_outputs[1]]);
                }

                let jvp = ad.jvp_program(&source, active_inputs).unwrap();
                let jvp = GraphCompiler::new()
                    .compile_frozen_program(jvp.frozen())
                    .unwrap();
                let mut jvp_inputs = primal_inputs.clone();
                jvp_inputs.extend([&matrix_tangent, &rhs_tangent]);
                let tangent_output = run_one(&runtime, &jvp, &jvp_inputs).unwrap_or_else(|error| {
                    panic!("{name} complex={complex} semantic JVP execution: {error}")
                });
                let cotangent = ones_tensor(dtype, tangent_output.shape().to_vec()).unwrap();

                let vjp = ad.vjp_program(&source, active_inputs, &[true]).unwrap();
                let vjp = GraphCompiler::new()
                    .compile_frozen_program(vjp.frozen())
                    .unwrap();
                let mut vjp_inputs = primal_inputs;
                vjp_inputs.push(&cotangent);
                let input_cotangents =
                    runtime
                        .run_compiled(&vjp, &vjp_inputs)
                        .unwrap_or_else(|error| {
                            panic!("{name} complex={complex} semantic VJP execution: {error}")
                        });
                assert_eq!(input_cotangents.len(), 2, "{name}");

                let lhs = semantic_real_inner_product(&cotangent, &tangent_output);
                let rhs = semantic_real_inner_product(&matrix_tangent, &input_cotangents[0])
                    + semantic_real_inner_product(&rhs_tangent, &input_cotangents[1]);
                let tolerance = 1.0e-8 * lhs.abs().max(rhs.abs()).max(1.0);
                assert!(
                    (lhs - rhs).abs() <= tolerance,
                    "{name} complex={complex} semantic adjoint parity failed: lhs={lhs}, rhs={rhs}, tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn symbolic_rectangular_lu_qr_execute_tall_and_wide_semantic_adjoint_parity() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::ad_support::ones_tensor;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
        use tenferro_runtime::GraphCompiler;
        use tenferro_tensor::Tensor;

        let cases = [
            ("lu", LinalgOp::Lu, vec![1, 2]),
            (
                "qr",
                LinalgOp::Qr {
                    gauge: QrGauge::Raw,
                },
                vec![0, 1],
            ),
        ];
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        for (name, op, active_output_indices) in cases {
            let mut builder = SemanticProgramBuilder::new();
            let _rows = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    }],
                ))
                .unwrap();
            let _cols = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    }],
                ))
                .unwrap();
            let matrix_value = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [
                        DimExpr::InputDim {
                            input_idx: 0,
                            axis: 0,
                        },
                        DimExpr::InputDim {
                            input_idx: 1,
                            axis: 0,
                        },
                    ],
                ))
                .unwrap();
            let all_outputs = builder
                .add_extension(Arc::new(LinalgExtensionOp::new(op)), &[matrix_value])
                .unwrap();
            let roots: Vec<_> = active_output_indices
                .iter()
                .map(|index| all_outputs[*index])
                .collect();
            let source = builder.finish(&roots).unwrap();
            let primal = GraphCompiler::new()
                .compile_frozen_program(&source)
                .unwrap();
            let jvp = ad.jvp_program(&source, &[false, false, true]).unwrap();
            let jvp = GraphCompiler::new()
                .compile_frozen_program(jvp.frozen())
                .unwrap();
            let vjp = ad
                .vjp_program(&source, &[false, false, true], &vec![true; roots.len()])
                .unwrap();
            let vjp = GraphCompiler::new()
                .compile_frozen_program(vjp.frozen())
                .unwrap();
            let backend = tenferro_cpu::CpuBackend::new();
            let runtime = cpu_runtime_with_linalg(&backend);

            for (rows, cols) in [(3, 2), (2, 3)] {
                let row_anchor =
                    Tensor::from_vec_col_major(vec![rows], vec![0.0_f64; rows]).unwrap();
                let col_anchor =
                    Tensor::from_vec_col_major(vec![cols], vec![0.0_f64; cols]).unwrap();
                let matrix_data: Vec<_> = (0..cols)
                    .flat_map(|col| {
                        (0..rows).map(move |row| {
                            if row == col {
                                3.0 + row as f64
                            } else {
                                0.05 * (1 + row + col) as f64
                            }
                        })
                    })
                    .collect();
                let tangent_data: Vec<_> = (0..cols)
                    .flat_map(|col| (0..rows).map(move |row| 0.01 * (1 + row + 2 * col) as f64))
                    .collect();
                let matrix = Tensor::from_vec_col_major(vec![rows, cols], matrix_data).unwrap();
                let tangent = Tensor::from_vec_col_major(vec![rows, cols], tangent_data).unwrap();
                let primal_outputs = runtime
                    .run_compiled(&primal, &[&row_anchor, &col_anchor, &matrix])
                    .unwrap_or_else(|error| {
                        panic!("{name} {rows}x{cols} symbolic primal execution: {error}")
                    });
                let cotangents: Vec<_> = primal_outputs
                    .iter()
                    .map(|output| ones_tensor(output.dtype(), output.shape().to_vec()).unwrap())
                    .collect();
                let tangent_outputs = runtime
                    .run_compiled(&jvp, &[&row_anchor, &col_anchor, &matrix, &tangent])
                    .unwrap_or_else(|error| {
                        panic!("{name} {rows}x{cols} symbolic JVP execution: {error}")
                    });
                let mut vjp_inputs = vec![&row_anchor, &col_anchor, &matrix];
                vjp_inputs.extend(&cotangents);
                let input_cotangents =
                    runtime
                        .run_compiled(&vjp, &vjp_inputs)
                        .unwrap_or_else(|error| {
                            panic!("{name} {rows}x{cols} symbolic VJP execution: {error}")
                        });
                assert_eq!(input_cotangents.len(), 1);
                for (index, output) in tangent_outputs.iter().enumerate() {
                    let values = output.as_slice::<f64>().unwrap();
                    assert!(
                        values.iter().all(|value| value.is_finite()),
                        "{name} {rows}x{cols} symbolic JVP output {index} is non-finite: {values:?}"
                    );
                }
                let vjp_values = input_cotangents[0].as_slice::<f64>().unwrap();
                assert!(
                    vjp_values.iter().all(|value| value.is_finite()),
                    "{name} {rows}x{cols} symbolic VJP is non-finite: {vjp_values:?}"
                );

                let lhs: f64 = tangent_outputs
                    .iter()
                    .zip(&cotangents)
                    .map(|(tangent, cotangent)| semantic_real_inner_product(cotangent, tangent))
                    .sum();
                let rhs = semantic_real_inner_product(&tangent, &input_cotangents[0]);
                let tolerance = 1.0e-8 * lhs.abs().max(rhs.abs()).max(1.0);
                assert!(
                    (lhs - rhs).abs() <= tolerance,
                    "{name} {rows}x{cols} symbolic semantic adjoint parity failed: lhs={lhs}, rhs={rhs}, tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn semantic_svd_output_pruning_reduces_graph_and_retained_pool_allocation() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

        fn svd_source(values_only: bool) -> tenferro_runtime::program::FrozenProgram {
            let mut builder = SemanticProgramBuilder::new();
            let matrix = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(3), DimExpr::Const(2)],
                ))
                .unwrap();
            let outputs = builder
                .add_extension(
                    Arc::new(LinalgExtensionOp::new(LinalgOp::Svd {
                        derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                        gauge: SvdGauge::Raw,
                    })),
                    &[matrix],
                )
                .unwrap();
            let roots = if values_only {
                vec![outputs[1]]
            } else {
                outputs.into_vec()
            };
            builder.finish(&roots).unwrap()
        }

        fn transform_sizes(
            source: &tenferro_runtime::program::FrozenProgram,
        ) -> (usize, usize, usize) {
            use tenferro_runtime::GraphCompiler;
            use tenferro_tensor::Tensor;

            let ad = AdContext::builder()
                .with_semantic_extension_rules(
                    semantic_ad_rules().expect("linalg semantic AD rules"),
                )
                .unwrap()
                .build()
                .unwrap();
            let jvp = ad.jvp_program(source, &[true]).unwrap();
            let vjp = ad
                .vjp_program(source, &[true], &vec![true; source.program.outputs().len()])
                .unwrap();
            let jvp_operations = jvp.frozen().program.operations().count();
            let vjp_operations = vjp.frozen().program.operations().count();
            let compiled = GraphCompiler::new()
                .compile_frozen_program(jvp.frozen())
                .unwrap();
            let backend = tenferro_cpu::CpuBackend::new();
            let runtime = cpu_runtime_with_linalg(&backend);
            let matrix =
                Tensor::from_vec_col_major(vec![3, 2], vec![3.0_f64, 0.5, 0.25, 1.0, 4.0, 0.75])
                    .unwrap();
            let tangent =
                Tensor::from_vec_col_major(vec![3, 2], vec![0.2_f64, -0.1, 0.05, 0.3, 0.4, -0.2])
                    .unwrap();
            let outputs = runtime
                .run_compiled(&compiled, &[&matrix, &tangent])
                .unwrap();
            drop(outputs);
            (
                jvp_operations,
                vjp_operations,
                backend.buffer_pool_stats().unwrap().capacity_bytes,
            )
        }

        let full = transform_sizes(&svd_source(false));
        let values_only = transform_sizes(&svd_source(true));
        assert!(
            values_only.0 < full.0,
            "values-only semantic SVD JVP must be smaller: values={values_only:?}, full={full:?}"
        );
        assert!(
            values_only.1 < full.1,
            "values-only semantic SVD VJP must be smaller: values={values_only:?}, full={full:?}"
        );
        assert!(
            values_only.2 < full.2,
            "values-only semantic SVD execution must retain less pooled allocation capacity: values={values_only:?}, full={full:?}"
        );
    }

    #[test]
    fn semantic_rules_cover_every_one_input_linalg_manifest_route() {
        use crate::LinalgAdOpKind;
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

        let cases = [
            (
                LinalgAdOpKind::Cholesky,
                LinalgOp::Cholesky,
                vec![true],
                true,
            ),
            (
                LinalgAdOpKind::Lu,
                LinalgOp::Lu,
                vec![false, true, true, false],
                true,
            ),
            (
                LinalgAdOpKind::LuFactor,
                LinalgOp::LuFactor,
                vec![true, false, false],
                false,
            ),
            (
                LinalgAdOpKind::FullPivLu,
                LinalgOp::FullPivLu,
                vec![false, true, true, false, false],
                true,
            ),
            (
                LinalgAdOpKind::Svd,
                LinalgOp::Svd {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: SvdGauge::Raw,
                },
                vec![true, true, true],
                true,
            ),
            (
                LinalgAdOpKind::SvdFull,
                LinalgOp::SvdFull,
                vec![true, false, false],
                false,
            ),
            (
                LinalgAdOpKind::SvdVals,
                LinalgOp::SvdVals {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                },
                vec![true],
                true,
            ),
            (
                LinalgAdOpKind::Qr,
                LinalgOp::Qr {
                    gauge: QrGauge::Raw,
                },
                vec![true, true],
                true,
            ),
            (
                LinalgAdOpKind::Eigh,
                LinalgOp::Eigh {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: EighGauge::Raw,
                },
                vec![true, true],
                true,
            ),
            (
                LinalgAdOpKind::EighVals,
                LinalgOp::EighVals {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                },
                vec![true],
                true,
            ),
            (
                LinalgAdOpKind::Eig,
                LinalgOp::Eig {
                    input_dtype: DType::F64,
                },
                vec![true, false],
                true,
            ),
            (
                LinalgAdOpKind::EigVals,
                LinalgOp::EigVals {
                    input_dtype: DType::F64,
                },
                vec![true],
                true,
            ),
        ];
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        for (kind, op, active_outputs, supported) in cases {
            let mut builder = SemanticProgramBuilder::new();
            let matrix = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(2), DimExpr::Const(2)],
                ))
                .unwrap();
            let outputs = builder
                .add_extension(Arc::new(LinalgExtensionOp::new(op)), &[matrix])
                .unwrap();
            let source = builder.finish(&outputs).unwrap();

            let jvp = ad.jvp_program(&source, &[true]);
            let vjp = ad.vjp_program(&source, &[true], &active_outputs);
            if supported {
                assert!(jvp.is_ok(), "{kind:?} semantic JVP failed: {jvp:?}");
                assert!(vjp.is_ok(), "{kind:?} semantic VJP failed: {vjp:?}");
            } else {
                let jvp = jvp.unwrap_or_else(|error| {
                    panic!("{kind:?} value-only semantic JVP failed unexpectedly: {error}")
                });
                assert!(
                    jvp.derivative_output_indices().iter().all(Option::is_none),
                    "{kind:?} value-only semantic JVP must not expose derivatives"
                );
                assert!(vjp.is_err(), "{kind:?} semantic VJP must be unsupported");
            }
        }
    }

    #[test]
    fn semantic_rules_cover_every_solve_manifest_route() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();
        let matrix_spec =
            || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)]);
        let rhs_spec = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(1)]);
        let cases = [
            (
                LinalgOp::FullPivLuSolve { transpose_a: false },
                vec![matrix_spec(), rhs_spec()],
                vec![true, true],
            ),
            (
                LinalgOp::TriangularSolve {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                },
                vec![matrix_spec(), rhs_spec()],
                vec![true, true],
            ),
            (
                LinalgOp::LuSolvePrepared {
                    transpose_a: false,
                    conjugate_a: false,
                },
                vec![
                    matrix_spec(),
                    matrix_spec(),
                    ProgramInputSpec::new(DType::I32, [DimExpr::Const(2)]),
                    rhs_spec(),
                ],
                vec![true, false, false, true],
            ),
        ];

        for (op, input_specs, active_inputs) in cases {
            let mut builder = SemanticProgramBuilder::new();
            let inputs: Vec<_> = input_specs
                .into_iter()
                .map(|spec| builder.input(spec).unwrap())
                .collect();
            let outputs = builder
                .add_extension(Arc::new(LinalgExtensionOp::new(op)), &inputs)
                .unwrap();
            let source = builder.finish(&outputs).unwrap();

            let jvp = ad.jvp_program(&source, &active_inputs);
            let vjp = ad.vjp_program(&source, &active_inputs, &[true]);
            assert!(jvp.is_ok(), "{op:?} semantic JVP failed: {jvp:?}");
            assert!(vjp.is_ok(), "{op:?} semantic VJP failed: {vjp:?}");
        }

        let mut builder = SemanticProgramBuilder::new();
        let inputs = [
            builder.input(matrix_spec()).unwrap(),
            builder.input(matrix_spec()).unwrap(),
            builder
                .input(ProgramInputSpec::new(DType::I32, [DimExpr::Const(2)]))
                .unwrap(),
            builder.input(rhs_spec()).unwrap(),
        ];
        let outputs = builder
            .add_extension(
                Arc::new(LinalgExtensionOp::new(LinalgOp::LuSolvePrepared {
                    transpose_a: false,
                    conjugate_a: false,
                })),
                &inputs,
            )
            .unwrap();
        let source = builder.finish(&outputs).unwrap();
        let vjp = ad
            .vjp_program(&source, &[false, true, false, false], &[true])
            .expect("prepared LU pivot/parity slots remain residual-only");
        assert_eq!(
            vjp.derivative_output_indices(),
            &[None, None, None, None],
            "prepared LU pivot/parity inputs must not produce cotangent outputs"
        );
    }

    #[test]
    fn semantic_rule_registration_matches_linalg_support_manifest_routes() {
        let rules = semantic_ad_rules().expect("linalg semantic AD rules");

        assert!(rules.lookup_linearize(LINALG_EXTENSION_FAMILY_ID).is_some());
        assert!(rules
            .lookup_linear_transpose(LINALG_EXTENSION_FAMILY_ID)
            .is_some());
        assert!(
            rules
                .lookup_primal_vjp(LINALG_EXTENSION_FAMILY_ID)
                .is_none(),
            "the linalg manifest declares no custom primal-VJP route"
        );
    }

    #[test]
    fn semantic_decomposition_rules_preserve_symbolic_matrix_shapes() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

        let cases = [
            (LinalgOp::Cholesky, vec![true]),
            (LinalgOp::Lu, vec![false, true, true, false]),
            (LinalgOp::FullPivLu, vec![false, true, true, false, false]),
            (
                LinalgOp::Svd {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: SvdGauge::Raw,
                },
                vec![true, true, true],
            ),
            (
                LinalgOp::SvdVals {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                },
                vec![true],
            ),
            (
                LinalgOp::Qr {
                    gauge: QrGauge::Raw,
                },
                vec![true, true],
            ),
            (
                LinalgOp::Eigh {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: EighGauge::Raw,
                },
                vec![true, true],
            ),
            (
                LinalgOp::EighVals {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                },
                vec![true],
            ),
            (
                LinalgOp::Eig {
                    input_dtype: DType::F64,
                },
                vec![true, false],
            ),
            (
                LinalgOp::EigVals {
                    input_dtype: DType::F64,
                },
                vec![true],
            ),
        ];
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        for (op, active_outputs) in cases {
            let mut builder = SemanticProgramBuilder::new();
            let matrix = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [
                        DimExpr::InputDim {
                            input_idx: 0,
                            axis: 0,
                        },
                        DimExpr::InputDim {
                            input_idx: 0,
                            axis: 0,
                        },
                    ],
                ))
                .unwrap();
            let outputs = builder
                .add_extension(Arc::new(LinalgExtensionOp::new(op)), &[matrix])
                .unwrap();
            let source = builder.finish(&outputs).unwrap();

            let jvp = ad.jvp_program(&source, &[true]);
            let vjp = ad.vjp_program(&source, &[true], &active_outputs);
            assert!(jvp.is_ok(), "{op:?} symbolic semantic JVP failed: {jvp:?}");
            assert!(vjp.is_ok(), "{op:?} symbolic semantic VJP failed: {vjp:?}");
        }
    }

    #[test]
    fn semantic_rectangular_rules_accept_distinct_symbolic_extents() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

        let cases = [
            (LinalgOp::Lu, vec![false, true, true, false]),
            (
                LinalgOp::Svd {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: SvdGauge::Raw,
                },
                vec![true, true, true],
            ),
            (
                LinalgOp::Qr {
                    gauge: QrGauge::Raw,
                },
                vec![true, true],
            ),
        ];
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        for (op, active_outputs) in cases {
            let mut builder = SemanticProgramBuilder::new();
            let _rows = builder
                .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
                .unwrap();
            let _cols = builder
                .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
                .unwrap();
            let matrix = builder
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [
                        DimExpr::InputDim {
                            input_idx: 0,
                            axis: 0,
                        },
                        DimExpr::InputDim {
                            input_idx: 1,
                            axis: 0,
                        },
                    ],
                ))
                .unwrap();
            let outputs = builder
                .add_extension(Arc::new(LinalgExtensionOp::new(op)), &[matrix])
                .unwrap();
            let source = builder.finish(&outputs).unwrap();

            let jvp = ad.jvp_program(&source, &[false, false, true]);
            let vjp = ad.vjp_program(&source, &[false, false, true], &active_outputs);
            assert!(
                jvp.is_ok(),
                "{op:?} distinct-symbolic semantic JVP failed: {jvp:?}"
            );
            assert!(
                vjp.is_ok(),
                "{op:?} distinct-symbolic semantic VJP failed: {vjp:?}"
            );
        }
    }

    #[test]
    fn semantic_full_piv_lu_rejects_unconstrained_rectangular_symbolic_shape() {
        use tenferro_ad::semantic_extension::{SemanticAdError, SemanticAdRuleRole};
        use tenferro_ad::semantic_transform::SemanticAdTransformError;
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

        let mut builder = SemanticProgramBuilder::new();
        builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
            .unwrap();
        builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let matrix = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                ],
            ))
            .unwrap();
        let outputs = builder
            .add_extension(
                Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLu)),
                &[matrix],
            )
            .unwrap();
        let source = builder.finish(&[outputs[1], outputs[2]]).unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        let jvp = ad
            .jvp_program(&source, &[false, false, true])
            .expect_err("unconstrained symbolic full-pivot LU must not assume a square matrix");
        let SemanticAdTransformError::Extension(SemanticAdError::Rule {
            family_id,
            role,
            source,
        }) = jvp
        else {
            panic!("expected a typed linalg rule failure, got {jvp:?}");
        };
        assert_eq!(family_id, LINALG_EXTENSION_FAMILY_ID);
        assert_eq!(role, SemanticAdRuleRole::Linearize);
        assert!(matches!(
            source.downcast_ref::<ADRuleError>(),
            Some(ADRuleError::InvalidInput {
                rule: ADRuleKind::Jvp,
                ..
            })
        ));
    }

    #[test]
    fn symbolic_svd_semantic_derivatives_preserve_primal_output_and_input_metadata() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

        let mut builder = SemanticProgramBuilder::new();
        builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
            .unwrap();
        builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let matrix = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0,
                    },
                ],
            ))
            .unwrap();
        let outputs = builder
            .add_extension(
                Arc::new(LinalgExtensionOp::new(LinalgOp::Svd {
                    derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                    gauge: SvdGauge::Raw,
                })),
                &[matrix],
            )
            .unwrap();
        let source = builder.finish(&[outputs[0]]).unwrap();
        let expected_jvp = source.program.value_metadata(outputs[0]).unwrap();
        let expected_vjp = source.program.value_metadata(matrix).unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().expect("linalg semantic AD rules"))
            .unwrap()
            .build()
            .unwrap();

        let jvp = ad.jvp_program(&source, &[false, false, true]).unwrap();
        let jvp_output = jvp.frozen().program.outputs()[0];
        let actual_jvp = jvp.frozen().program.value_metadata(jvp_output).unwrap();
        assert_eq!(actual_jvp.dtype(), expected_jvp.dtype());
        assert_eq!(actual_jvp.shape(), expected_jvp.shape());

        let vjp = ad
            .vjp_program(&source, &[false, false, true], &[true])
            .unwrap();
        let vjp_output = vjp.frozen().program.outputs()[0];
        let actual_vjp = vjp.frozen().program.value_metadata(vjp_output).unwrap();
        assert_eq!(actual_vjp.dtype(), expected_vjp.dtype());
        assert_eq!(actual_vjp.shape(), expected_vjp.shape());
    }
}
