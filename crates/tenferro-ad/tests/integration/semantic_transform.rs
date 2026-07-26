use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use num_complex::Complex64;
use tenferro_ad::semantic_extension::{
    AdValue, SemanticAdError, SemanticExtensionRuleSet, SemanticLinearTransposeRequest,
    SemanticLinearTransposeRule, SemanticLinearizeRequest, SemanticLinearizeResult,
    SemanticLinearizeRule, SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
use tenferro_ad::semantic_transform::SemanticAdTransformError;
use tenferro_ad::AdContext;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::program::{CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder};
use tenferro_runtime::GraphCompiler;
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
};

use crate::support::{cpu_runtime, RunCompiledTestExt};

const FAMILY: &str = "tenferro-ad.semantic-transform-test.v1";

#[derive(Clone, Debug)]
struct AddInputsExtension;

impl ExtensionOp for AddInputsExtension {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(1);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        context: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        context.require_same_shape(0, 1)?;
        Ok(vec![(
            context.input_dtype(0)?,
            context.input_shape(0)?.to_vec(),
        )])
    }
}

#[derive(Debug)]
struct AddInputsRule;

impl SemanticLinearizeRule for AddInputsRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        assert_eq!(request.active_outputs(), &[true]);
        assert_eq!(request.provenance().label(), Some(FAMILY));
        let tangent = match request.tangent_inputs() {
            [AdValue::Value(lhs), AdValue::Value(rhs)] => {
                AdValue::Value(builder.add_op(CoreSemanticOp::Add, &[*lhs, *rhs])?[0])
            }
            [AdValue::Value(value), AdValue::Absent] | [AdValue::Absent, AdValue::Value(value)] => {
                AdValue::Value(*value)
            }
            [AdValue::Absent, AdValue::Absent] => AdValue::Absent,
            _ => unreachable!(),
        };
        Ok(SemanticLinearizeResult::new([tangent], []))
    }
}

impl SemanticLinearTransposeRule for AddInputsRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        assert_eq!(request.provenance().label(), Some(FAMILY));
        let cotangent = request.cotangent_outputs()[0];
        Ok(request
            .active_inputs()
            .iter()
            .map(|active| if *active { cotangent } else { AdValue::Absent })
            .collect())
    }
}

impl SemanticPrimalVjpRule for AddInputsRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        assert_eq!(request.provenance().label(), Some(FAMILY));
        let cotangent = request.cotangent_outputs()[0];
        Ok(request
            .active_inputs()
            .iter()
            .map(|active| if *active { cotangent } else { AdValue::Absent })
            .collect())
    }
}

fn repeated_input_program() -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    builder
        .bind_input(
            input,
            Arc::new(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0_f64]).unwrap()),
        )
        .unwrap();
    let output = builder
        .add_extension(Arc::new(AddInputsExtension), &[input, input])
        .unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn core_square_program() -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let output = builder
        .add_op(CoreSemanticOp::Mul, &[input, input])
        .unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn bound_core_square_program(values: Vec<f64>) -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    builder
        .bind_input(
            input,
            Arc::new(Tensor::from_vec_col_major(vec![2], values).unwrap()),
        )
        .unwrap();
    let output = builder
        .add_op(CoreSemanticOp::Mul, &[input, input])
        .unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn unary_core_program(
    input_shape: impl IntoIterator<Item = DimExpr>,
    op: CoreSemanticOp,
) -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, input_shape))
        .unwrap();
    let output = builder.add_op(op, &[input]).unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn binary_core_program(
    lhs_dtype: DType,
    lhs_shape: impl IntoIterator<Item = DimExpr>,
    rhs_dtype: DType,
    rhs_shape: impl IntoIterator<Item = DimExpr>,
    op: CoreSemanticOp,
) -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(lhs_dtype, lhs_shape))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(rhs_dtype, rhs_shape))
        .unwrap();
    let output = builder.add_op(op, &[lhs, rhs]).unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn rules() -> SemanticExtensionRuleSet {
    let mut rules = SemanticExtensionRuleSet::new();
    rules.register_linearize(Arc::new(AddInputsRule)).unwrap();
    rules.register_primal_vjp(Arc::new(AddInputsRule)).unwrap();
    rules
}

fn transpose_rules() -> SemanticExtensionRuleSet {
    let mut rules = SemanticExtensionRuleSet::new();
    rules.register_linearize(Arc::new(AddInputsRule)).unwrap();
    rules
        .register_linear_transpose(Arc::new(AddInputsRule))
        .unwrap();
    rules
}

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(rules())
        .unwrap()
        .build()
        .unwrap()
}

#[test]
fn semantic_jvp_appends_ordered_tangent_inputs_and_returns_only_tangents() {
    let source = repeated_input_program();
    let transformed = ad_context().jvp_program(&source, &[true]).unwrap();

    assert_eq!(transformed.frozen().program.inputs().len(), 2);
    assert_eq!(transformed.frozen().program.outputs().len(), 1);
    assert_eq!(transformed.derivative_input_indices(), &[Some(1)]);
    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    assert_eq!(transformed.frozen().program.operations().count(), 1);
    assert!(matches!(
        transformed
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Add)
    ));
    assert_eq!(transformed.frozen().bindings.len(), 1);
}

#[test]
fn semantic_vjp_accumulates_repeated_input_cotangents() {
    let source = repeated_input_program();
    let transformed = ad_context().vjp_program(&source, &[true], &[true]).unwrap();

    assert_eq!(transformed.frozen().program.inputs().len(), 2);
    assert_eq!(transformed.derivative_input_indices(), &[Some(1)]);
    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    assert_eq!(transformed.frozen().program.operations().count(), 1);
    assert!(matches!(
        transformed
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Add)
    ));
    assert_eq!(transformed.frozen().bindings.len(), 1);
}

#[test]
fn semantic_vjp_falls_back_to_linearize_then_transpose_explicitly() {
    let source = repeated_input_program();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(transpose_rules())
        .unwrap()
        .build()
        .unwrap();
    let transformed = ad.vjp_program(&source, &[true], &[true]).unwrap();

    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    assert!(matches!(
        transformed
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Add)
    ));
}

#[test]
fn semantic_activity_is_ordered_typed_and_preserves_inactive_values() {
    let source = repeated_input_program();
    let ad = ad_context();

    let inactive = ad.jvp_program(&source, &[false]).unwrap();
    assert_eq!(inactive.derivative_input_indices(), &[None]);
    assert_eq!(inactive.derivative_output_indices(), &[None]);
    assert!(inactive.frozen().program.outputs().is_empty());

    assert!(matches!(
        ad.jvp_program(&source, &[]),
        Err(SemanticAdTransformError::ActivityArity {
            field: "active_inputs",
            expected: 1,
            actual: 0,
            ..
        })
    ));
    assert!(matches!(
        ad.vjp_program(&source, &[true], &[]),
        Err(SemanticAdTransformError::ActivityArity {
            field: "active_outputs",
            expected: 1,
            actual: 0,
            ..
        })
    ));
}

#[test]
fn semantic_core_jvp_linearizes_product_rule_and_accumulates_terms() {
    let transformed = ad_context()
        .jvp_program(&core_square_program(), &[true])
        .unwrap();

    assert_eq!(transformed.derivative_input_indices(), &[Some(1)]);
    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    let operations: Vec<_> = transformed.frozen().program.operations().collect();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations.last().unwrap().op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Add)
    ));
}

#[test]
fn semantic_core_vjp_applies_hermitian_product_rule_and_accumulates_aliases() {
    let transformed = ad_context()
        .vjp_program(&core_square_program(), &[true], &[true])
        .unwrap();

    assert_eq!(transformed.derivative_input_indices(), &[Some(1)]);
    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    assert!(matches!(
        transformed
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Add)
    ));
}

#[test]
fn semantic_core_transpose_jvp_and_vjp_use_forward_and_inverse_permutations() {
    let source = unary_core_program(
        [DimExpr::Const(2), DimExpr::Const(3), DimExpr::Const(4)],
        CoreSemanticOp::Transpose {
            perm: vec![2, 0, 1],
        },
    );
    let ad = ad_context();

    let jvp = ad.jvp_program(&source, &[true]).unwrap();
    assert!(matches!(
        jvp.frozen().program.operations().last().unwrap().op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Transpose { perm })
            if perm.as_slice() == [2, 0, 1]
    ));

    let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
    assert!(matches!(
        vjp.frozen().program.operations().last().unwrap().op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Transpose { perm })
            if perm.as_slice() == [1, 2, 0]
    ));
}

#[test]
fn semantic_core_reshape_and_reduce_sum_transpose_restore_input_shapes() {
    let reshape = unary_core_program(
        [DimExpr::Const(2), DimExpr::Const(3)],
        CoreSemanticOp::Reshape {
            to_shape: vec![DimExpr::Const(6)],
        },
    );
    let ad = ad_context();
    let reshape_vjp = ad.vjp_program(&reshape, &[true], &[true]).unwrap();
    assert!(matches!(
        reshape_vjp
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Reshape { to_shape })
            if to_shape.as_slice() == [DimExpr::Const(2), DimExpr::Const(3)]
    ));

    let reduce = unary_core_program(
        [DimExpr::Const(2), DimExpr::Const(3), DimExpr::Const(4)],
        CoreSemanticOp::ReduceSum { axes: vec![0, 2] },
    );
    let reduce_vjp = ad.vjp_program(&reduce, &[true], &[true]).unwrap();
    assert!(matches!(
        reduce_vjp
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::BroadcastInDim {
            shape,
            dims
        }) if shape.as_slice()
            == [
                DimExpr::Const(2),
                DimExpr::Const(3),
                DimExpr::Const(4)
            ] && dims.as_slice() == [1]
    ));
}

#[test]
fn semantic_core_broadcast_vjp_reduces_inserted_and_singleton_axes() {
    let source = unary_core_program(
        [DimExpr::Const(2), DimExpr::Const(1)],
        CoreSemanticOp::BroadcastInDim {
            shape: vec![DimExpr::Const(3), DimExpr::Const(2), DimExpr::Const(4)],
            dims: vec![1, 2],
        },
    );

    let transformed = ad_context().vjp_program(&source, &[true], &[true]).unwrap();
    let operations: Vec<_> = transformed.frozen().program.operations().collect();
    assert!(matches!(
        operations[operations.len() - 2].op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::ReduceSum { axes })
            if axes.as_slice() == [0, 2]
    ));
    assert!(matches!(
        operations.last().unwrap().op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Reshape { to_shape })
            if to_shape.as_slice() == [DimExpr::Const(2), DimExpr::Const(1)]
    ));
}

#[test]
fn semantic_core_self_adjoint_structural_ops_reapply_their_payloads() {
    let cases = [
        CoreSemanticOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        CoreSemanticOp::Tril { k: -1 },
        CoreSemanticOp::Triu { k: 2 },
        CoreSemanticOp::Reverse { axes: vec![0, 1] },
    ];

    for op in cases {
        let source = unary_core_program([DimExpr::Const(3), DimExpr::Const(3)], op.clone());
        let ad = ad_context();
        assert!(ad.jvp_program(&source, &[true]).is_ok(), "JVP for {op:?}");
        assert!(
            ad.vjp_program(&source, &[true], &[true]).is_ok(),
            "VJP for {op:?}"
        );
    }
}

#[test]
fn semantic_core_analytic_unary_rules_support_real_and_complex_jvp_vjp() {
    let operations = [
        CoreSemanticOp::Exp,
        CoreSemanticOp::Log,
        CoreSemanticOp::Sin,
        CoreSemanticOp::Cos,
        CoreSemanticOp::Tanh,
        CoreSemanticOp::Sqrt,
        CoreSemanticOp::Rsqrt,
        CoreSemanticOp::Expm1,
        CoreSemanticOp::Log1p,
    ];

    for dtype in [DType::F64, DType::C64] {
        for op in &operations {
            let mut builder = SemanticProgramBuilder::new();
            let input = builder
                .input(ProgramInputSpec::new(dtype, [DimExpr::Const(2)]))
                .unwrap();
            let output = builder.add_op(op.clone(), &[input]).unwrap()[0];
            let source = builder.finish(&[output]).unwrap();
            let ad = ad_context();
            assert!(
                ad.jvp_program(&source, &[true]).is_ok(),
                "JVP for {dtype:?} {op:?}"
            );
            let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
            assert_eq!(vjp.derivative_output_indices(), &[Some(0)]);
            if dtype == DType::C64 {
                assert!(vjp.frozen().program.operations().any(|operation| matches!(
                    operation.op(),
                    tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Conj)
                )));
            }
        }
    }
}

#[test]
fn semantic_core_div_jvp_vjp_handle_broadcast_and_hermitian_coefficients() {
    let source = binary_core_program(
        DType::F64,
        [DimExpr::Const(2), DimExpr::Const(1)],
        DType::C64,
        [DimExpr::Const(3)],
        CoreSemanticOp::Div,
    );
    let ad = ad_context();

    let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
    assert_eq!(jvp.derivative_input_indices(), &[Some(2), Some(3)]);
    assert!(matches!(
        jvp.frozen().program.operations().last().unwrap().op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Sub)
    ));

    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    assert_eq!(vjp.derivative_output_indices(), &[Some(0), Some(1)]);
    assert!(vjp.frozen().program.operations().any(|operation| matches!(
        operation.op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Conj)
    )));
    let lhs_cotangent = vjp.frozen().program.outputs()[0];
    let rhs_cotangent = vjp.frozen().program.outputs()[1];
    assert_eq!(
        vjp.frozen()
            .program
            .value_metadata(lhs_cotangent)
            .unwrap()
            .dtype(),
        DType::F64
    );
    assert_eq!(
        vjp.frozen()
            .program
            .value_metadata(rhs_cotangent)
            .unwrap()
            .dtype(),
        DType::C64
    );
}

#[test]
fn semantic_core_pow_abs_sign_and_select_follow_activity_and_dtype_contracts() {
    let pow = binary_core_program(
        DType::C64,
        [DimExpr::Const(2)],
        DType::F64,
        [],
        CoreSemanticOp::Pow,
    );
    let ad = ad_context();
    assert!(ad.jvp_program(&pow, &[true, true]).is_ok());
    let pow_vjp = ad.vjp_program(&pow, &[true, true], &[true]).unwrap();
    assert_eq!(pow_vjp.derivative_output_indices(), &[Some(0), Some(1)]);
    assert!(pow_vjp
        .frozen()
        .program
        .operations()
        .any(|operation| matches!(
            operation.op(),
            tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Conj)
        )));
    let lhs_only_jvp = ad.jvp_program(&pow, &[true, false]).unwrap();
    assert!(!lhs_only_jvp
        .frozen()
        .program
        .operations()
        .any(|operation| matches!(
            operation.op(),
            tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Log)
        )));
    let lhs_only_vjp = ad.vjp_program(&pow, &[true, false], &[true]).unwrap();
    assert!(!lhs_only_vjp
        .frozen()
        .program
        .operations()
        .any(|operation| matches!(
            operation.op(),
            tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Log)
        )));

    let mut builder = SemanticProgramBuilder::new();
    let abs_input = builder
        .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(2)]))
        .unwrap();
    let abs_output = builder.add_op(CoreSemanticOp::Abs, &[abs_input]).unwrap()[0];
    let abs = builder.finish(&[abs_output]).unwrap();
    let abs_jvp = ad.jvp_program(&abs, &[true]).unwrap();
    assert_eq!(
        abs_jvp
            .frozen()
            .program
            .value_metadata(abs_jvp.frozen().program.outputs()[0])
            .unwrap()
            .dtype(),
        DType::F64
    );
    assert!(ad.vjp_program(&abs, &[true], &[true]).is_ok());

    let sign = unary_core_program([DimExpr::Const(2)], CoreSemanticOp::Sign);
    let sign_jvp = ad.jvp_program(&sign, &[true]).unwrap();
    assert_eq!(sign_jvp.derivative_output_indices(), &[None]);
    let sign_vjp = ad.vjp_program(&sign, &[true], &[true]).unwrap();
    assert_eq!(sign_vjp.derivative_output_indices(), &[None]);

    let mut builder = SemanticProgramBuilder::new();
    let condition = builder
        .input(ProgramInputSpec::new(DType::Bool, [DimExpr::Const(2)]))
        .unwrap();
    let on_true = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let on_false = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let selected = builder
        .add_op(CoreSemanticOp::Select, &[condition, on_true, on_false])
        .unwrap()[0];
    let select = builder.finish(&[selected]).unwrap();
    let select_jvp = ad.jvp_program(&select, &[false, true, false]).unwrap();
    assert_eq!(
        select_jvp.derivative_input_indices(),
        &[None, Some(3), None]
    );
    assert!(matches!(
        select_jvp
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Select)
    ));
    let select_vjp = ad
        .vjp_program(&select, &[false, true, true], &[true])
        .unwrap();
    assert_eq!(
        select_vjp.derivative_output_indices(),
        &[None, Some(0), Some(1)]
    );
}

#[test]
fn semantic_core_dot_general_supports_ordered_jvp_and_hermitian_vjp() {
    let source = binary_core_program(
        DType::C64,
        [DimExpr::Const(2), DimExpr::Const(3)],
        DType::C64,
        [DimExpr::Const(3), DimExpr::Const(4)],
        CoreSemanticOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        },
    );
    let ad = ad_context();

    let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
    assert_eq!(jvp.derivative_input_indices(), &[Some(2), Some(3)]);
    assert_eq!(jvp.derivative_output_indices(), &[Some(0)]);
    assert_eq!(
        jvp.frozen()
            .program
            .operations()
            .filter(|operation| matches!(
                operation.op(),
                tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::DotGeneral { .. })
            ))
            .count(),
        2
    );

    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    assert_eq!(vjp.derivative_output_indices(), &[Some(0), Some(1)]);
    assert_eq!(
        vjp.frozen()
            .program
            .operations()
            .filter(|operation| matches!(
                operation.op(),
                tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::DotGeneral { .. })
            ))
            .count(),
        2
    );
    assert!(
        vjp.frozen()
            .program
            .operations()
            .filter(|operation| matches!(
                operation.op(),
                tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Conj)
            ))
            .count()
            >= 2
    );
}

#[test]
fn semantic_core_complex_sign_jvp_executes_numerically() {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(3)]))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Sign, &[input]).unwrap()[0];
    let source = builder.finish(&[output]).unwrap();

    let jvp = ad_context().jvp_program(&source, &[true]).unwrap();
    assert_eq!(jvp.derivative_output_indices(), &[Some(0)]);

    let input_values = vec![
        Complex64::new(3.0, 4.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(0.0, 0.0),
    ];
    let tangent_values = vec![
        Complex64::new(0.5, -0.25),
        Complex64::new(-0.3, 0.8),
        Complex64::new(1.0, -2.0),
    ];
    let input_tensor = Tensor::from_vec_col_major(vec![3], input_values.clone()).unwrap();
    let tangent_tensor = Tensor::from_vec_col_major(vec![3], tangent_values.clone()).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let actual = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input_tensor, &tangent_tensor])
        .unwrap();

    let expected = input_values
        .iter()
        .zip(tangent_values.iter())
        .map(|(&x, &dx)| {
            let abs = x.norm();
            if abs == 0.0 {
                return Complex64::new(0.0, 0.0);
            }
            let sign = x / abs;
            let abs_tangent = (sign.conj() * dx).re;
            dx / abs - sign * (abs_tangent / abs)
        })
        .collect::<Vec<_>>();
    for (actual, expected) in actual.as_slice::<Complex64>().unwrap().iter().zip(expected) {
        assert!(
            (*actual - expected).norm() <= 1e-12,
            "actual={actual:?} expected={expected:?}"
        );
    }
}

#[test]
fn semantic_core_dot_general_jvp_and_vjp_execute_numerically() {
    let source = binary_core_program(
        DType::F64,
        [DimExpr::Const(2), DimExpr::Const(3)],
        DType::F64,
        [DimExpr::Const(3), DimExpr::Const(2)],
        CoreSemanticOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        },
    );
    let ad = ad_context();
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0]).unwrap();
    let lhs_tangent =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let rhs_tangent =
        Tensor::from_vec_col_major(vec![3, 2], vec![0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();

    let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let jvp_value = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&lhs, &rhs, &lhs_tangent, &rhs_tangent])
        .unwrap();
    assert_eq!(jvp_value.as_slice::<f64>().unwrap(), &[7.0, 0.0, 9.0, 4.0]);

    let output_cotangent =
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 1.0, 1.0, 1.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let cotangents = cpu_runtime()
        .run_compiled(&compiled, &[&lhs, &rhs, &output_cotangent])
        .unwrap();
    assert_eq!(
        cotangents[0].as_slice::<f64>().unwrap(),
        &[15.0, 15.0, 19.0, 19.0, 23.0, 23.0]
    );
    assert_eq!(
        cotangents[1].as_slice::<f64>().unwrap(),
        &[5.0, 7.0, 9.0, 5.0, 7.0, 9.0]
    );
}

#[test]
fn semantic_core_extrema_split_ties_and_clamp_routes_active_values() {
    let ad = ad_context();
    for op in [CoreSemanticOp::Maximum, CoreSemanticOp::Minimum] {
        let source = binary_core_program(
            DType::F64,
            [DimExpr::Const(3)],
            DType::F64,
            [DimExpr::Const(3)],
            op,
        );
        assert!(ad.jvp_program(&source, &[true, true]).is_ok());
        assert!(ad.vjp_program(&source, &[true, true], &[true]).is_ok());
    }

    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let lower = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let upper = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let clamped = builder
        .add_op(CoreSemanticOp::Clamp, &[input, lower, upper])
        .unwrap()[0];
    let source = builder.finish(&[clamped]).unwrap();
    assert!(ad.jvp_program(&source, &[true, true, true]).is_ok());
    assert!(ad
        .vjp_program(&source, &[true, true, true], &[true])
        .is_ok());
}

#[test]
fn semantic_core_maximum_jvp_and_vjp_execute_with_balanced_ties() {
    let source = binary_core_program(
        DType::F64,
        [DimExpr::Const(3)],
        DType::F64,
        [DimExpr::Const(3)],
        CoreSemanticOp::Maximum,
    );
    let ad = ad_context();
    let lhs = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 2.0, 1.0]).unwrap();
    let lhs_tangent = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();
    let rhs_tangent = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let tangent = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&lhs, &rhs, &lhs_tangent, &rhs_tangent])
        .unwrap();
    assert_eq!(tangent.as_slice::<f64>().unwrap(), &[1.0, 11.0, 30.0]);

    let output_cotangent = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 4.0, 6.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let cotangents = cpu_runtime()
        .run_compiled(&compiled, &[&lhs, &rhs, &output_cotangent])
        .unwrap();
    assert_eq!(cotangents[0].as_slice::<f64>().unwrap(), &[0.0, 2.0, 6.0]);
    assert_eq!(cotangents[1].as_slice::<f64>().unwrap(), &[2.0, 2.0, 0.0]);
}

#[test]
fn semantic_core_nonlinear_reductions_transform_product_and_balanced_extrema() {
    let ad = ad_context();
    for op in [
        CoreSemanticOp::ReduceProd { axes: vec![0] },
        CoreSemanticOp::ReduceMax { axes: vec![0] },
        CoreSemanticOp::ReduceMin { axes: vec![0] },
    ] {
        let source = unary_core_program([DimExpr::Const(2), DimExpr::Const(3)], op);
        assert!(ad.jvp_program(&source, &[true]).is_ok());
        assert!(ad.vjp_program(&source, &[true], &[true]).is_ok());
    }
}

#[test]
fn semantic_core_reduce_prod_handles_zero_multiplicity_numerically() {
    let source = unary_core_program(
        [DimExpr::Const(2), DimExpr::Const(3)],
        CoreSemanticOp::ReduceProd { axes: vec![0] },
    );
    let ad = ad_context();
    let input =
        Tensor::from_vec_col_major(vec![2, 3], vec![2.0_f64, 3.0, 0.0, 4.0, 0.0, 0.0]).unwrap();
    let tangent = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();

    let jvp = ad.jvp_program(&source, &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &tangent])
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[5.0, 4.0, 0.0]);

    let output_cotangent = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &output_cotangent])
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[3.0, 2.0, 8.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn semantic_core_reduce_max_balances_ties_numerically() {
    let source = unary_core_program(
        [DimExpr::Const(2), DimExpr::Const(3)],
        CoreSemanticOp::ReduceMax { axes: vec![0] },
    );
    let ad = ad_context();
    let input =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 4.0, 4.0, 3.0, 1.0]).unwrap();
    let tangent =
        Tensor::from_vec_col_major(vec![2, 3], vec![10.0_f64, 20.0, 30.0, 50.0, 70.0, 80.0])
            .unwrap();

    let jvp = ad.jvp_program(&source, &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &tangent])
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[20.0, 40.0, 70.0]);

    let output_cotangent = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 4.0, 6.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &output_cotangent])
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[0.0, 2.0, 2.0, 2.0, 6.0, 0.0]
    );
}

#[test]
fn semantic_core_discrete_ops_are_explicitly_inactive() {
    let ad = ad_context();
    let rem = binary_core_program(
        DType::F64,
        [DimExpr::Const(2)],
        DType::F64,
        [DimExpr::Const(2)],
        CoreSemanticOp::Rem,
    );
    assert_eq!(
        ad.jvp_program(&rem, &[true, true])
            .unwrap()
            .derivative_output_indices(),
        &[None]
    );
    assert_eq!(
        ad.vjp_program(&rem, &[true, true], &[true])
            .unwrap()
            .derivative_output_indices(),
        &[None, None]
    );

    let compare = binary_core_program(
        DType::F64,
        [DimExpr::Const(2)],
        DType::F64,
        [DimExpr::Const(2)],
        CoreSemanticOp::Compare(tenferro_runtime::CompareDir::Eq),
    );
    assert_eq!(
        ad.jvp_program(&compare, &[true, true])
            .unwrap()
            .derivative_output_indices(),
        &[None]
    );

    let shape = unary_core_program(
        [DimExpr::Const(2), DimExpr::Const(3)],
        CoreSemanticOp::ShapeOf { axis: 1 },
    );
    assert_eq!(
        ad.jvp_program(&shape, &[true])
            .unwrap()
            .derivative_output_indices(),
        &[None]
    );

    let mut builder = SemanticProgramBuilder::new();
    let constant = builder
        .add_op(
            CoreSemanticOp::Constant {
                dtype: DType::F64,
                bytes: 2.0_f64.to_le_bytes().to_vec(),
            },
            &[],
        )
        .unwrap()[0];
    let constant = builder.finish(&[constant]).unwrap();
    assert_eq!(
        ad.jvp_program(&constant, &[])
            .unwrap()
            .derivative_output_indices(),
        &[None]
    );
    assert!(ad
        .vjp_program(&constant, &[], &[true])
        .unwrap()
        .derivative_output_indices()
        .is_empty());
}

#[test]
fn semantic_core_slice_pad_and_concatenate_transform_structurally() {
    let ad = ad_context();
    for op in [
        CoreSemanticOp::Slice(SliceConfig {
            starts: vec![1],
            limits: vec![5],
            strides: vec![2],
        }),
        CoreSemanticOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![2],
            interior_padding: vec![1],
        }),
    ] {
        let source = unary_core_program([DimExpr::Const(5)], op);
        assert!(ad.jvp_program(&source, &[true]).is_ok());
        assert!(ad.vjp_program(&source, &[true], &[true]).is_ok());
    }

    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(1)]))
        .unwrap();
    let output = builder
        .add_op(
            CoreSemanticOp::Concatenate {
                axis: 0,
                input_count: 2,
            },
            &[lhs, rhs],
        )
        .unwrap()[0];
    let source = builder.finish(&[output]).unwrap();
    assert!(ad.jvp_program(&source, &[true, false]).is_ok());
    assert!(ad.vjp_program(&source, &[true, true], &[true]).is_ok());
}

#[test]
fn semantic_core_strided_slice_jvp_and_vjp_execute_numerically() {
    let source = unary_core_program(
        [DimExpr::Const(5)],
        CoreSemanticOp::Slice(SliceConfig {
            starts: vec![1],
            limits: vec![5],
            strides: vec![2],
        }),
    );
    let ad = ad_context();
    let input = Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let tangent =
        Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]).unwrap();

    let jvp = ad.jvp_program(&source, &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &tangent])
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[20.0, 40.0]);

    let output_cotangent = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &output_cotangent])
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[0.0, 2.0, 0.0, 4.0, 0.0]
    );
}

#[test]
fn semantic_core_concatenate_jvp_zero_fills_and_vjp_splits_numerically() {
    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(1)]))
        .unwrap();
    let output = builder
        .add_op(
            CoreSemanticOp::Concatenate {
                axis: 0,
                input_count: 2,
            },
            &[lhs, rhs],
        )
        .unwrap()[0];
    let source = builder.finish(&[output]).unwrap();
    let ad = ad_context();
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    let lhs_tangent = Tensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0]).unwrap();

    let jvp = ad.jvp_program(&source, &[true, false]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&lhs, &rhs, &lhs_tangent])
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[10.0, 20.0, 0.0]);

    let output_cotangent = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled(&compiled, &[&lhs, &rhs, &output_cotangent])
        .unwrap();
    assert_eq!(result[0].as_slice::<f64>().unwrap(), &[4.0, 5.0]);
    assert_eq!(result[1].as_slice::<f64>().unwrap(), &[6.0]);
}

#[test]
fn semantic_core_pad_jvp_and_vjp_execute_numerically() {
    let source = unary_core_program(
        [DimExpr::Const(2)],
        CoreSemanticOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![2],
            interior_padding: vec![1],
        }),
    );
    let ad = ad_context();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let tangent = Tensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0]).unwrap();

    let jvp = ad.jvp_program(&source, &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &tangent])
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[0.0, 10.0, 0.0, 20.0, 0.0, 0.0]
    );

    let output_cotangent =
        Tensor::from_vec_col_major(vec![6], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input, &output_cotangent])
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

fn rank1_gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

fn rank1_scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}

#[test]
fn semantic_core_gather_and_scatter_jvp_vjp_execute_numerically() {
    let ad = ad_context();
    let mut builder = SemanticProgramBuilder::new();
    let operand = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(5)]))
        .unwrap();
    let indices = builder
        .input(ProgramInputSpec::new(
            DType::I64,
            [DimExpr::Const(3), DimExpr::Const(1)],
        ))
        .unwrap();
    let gathered = builder
        .add_op(
            CoreSemanticOp::Gather(rank1_gather_config()),
            &[operand, indices],
        )
        .unwrap()[0];
    let gather = builder.finish(&[gathered]).unwrap();
    let operand_value =
        Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let indices_value = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 2, 2]).unwrap();
    let operand_tangent =
        Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]).unwrap();

    let jvp = ad.jvp_program(&gather, &[true, false]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(
            &compiled,
            &[&operand_value, &indices_value, &operand_tangent],
        )
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[10.0, 30.0, 30.0]);

    let gather_cotangent = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let vjp = ad.vjp_program(&gather, &[true, false], &[true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(
            &compiled,
            &[&operand_value, &indices_value, &gather_cotangent],
        )
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[1.0, 0.0, 5.0, 0.0, 0.0]
    );

    let mut builder = SemanticProgramBuilder::new();
    let operand = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(5)]))
        .unwrap();
    let indices = builder
        .input(ProgramInputSpec::new(
            DType::I64,
            [DimExpr::Const(3), DimExpr::Const(1)],
        ))
        .unwrap();
    let updates = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let scattered = builder
        .add_op(
            CoreSemanticOp::Scatter(rank1_scatter_config()),
            &[operand, indices, updates],
        )
        .unwrap()[0];
    let scatter = builder.finish(&[scattered]).unwrap();
    let updates_value = Tensor::from_vec_col_major(vec![3], vec![6.0_f64, 7.0, 8.0]).unwrap();
    let updates_tangent = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    let jvp = ad.jvp_program(&scatter, &[true, false, true]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(
            &compiled,
            &[
                &operand_value,
                &indices_value,
                &updates_value,
                &operand_tangent,
                &updates_tangent,
            ],
        )
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[11.0, 20.0, 35.0, 40.0, 50.0]
    );

    let scatter_cotangent =
        Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let vjp = ad
        .vjp_program(&scatter, &[true, false, true], &[true])
        .unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled(
            &compiled,
            &[
                &operand_value,
                &indices_value,
                &updates_value,
                &scatter_cotangent,
            ],
        )
        .unwrap();
    assert_eq!(
        result[0].as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(result[1].as_slice::<f64>().unwrap(), &[1.0, 3.0, 3.0]);
}

#[test]
fn semantic_core_dynamic_slice_and_update_jvp_vjp_execute_numerically() {
    let ad = ad_context();
    let mut builder = SemanticProgramBuilder::new();
    let operand = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(5)]))
        .unwrap();
    let starts = builder
        .input(ProgramInputSpec::new(DType::I64, [DimExpr::Const(1)]))
        .unwrap();
    let sliced = builder
        .add_op(
            CoreSemanticOp::DynamicSlice {
                slice_sizes: vec![2],
            },
            &[operand, starts],
        )
        .unwrap()[0];
    let dynamic_slice = builder.finish(&[sliced]).unwrap();
    let operand_value =
        Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let starts_value = Tensor::from_vec_col_major(vec![1], vec![2_i64]).unwrap();
    let operand_tangent =
        Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]).unwrap();

    let jvp = ad.jvp_program(&dynamic_slice, &[true, false]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(
            &compiled,
            &[&operand_value, &starts_value, &operand_tangent],
        )
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[30.0, 40.0]);

    let slice_cotangent = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let vjp = ad
        .vjp_program(&dynamic_slice, &[true, false], &[true])
        .unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(
            &compiled,
            &[&operand_value, &starts_value, &slice_cotangent],
        )
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[0.0, 0.0, 1.0, 2.0, 0.0]
    );

    let mut builder = SemanticProgramBuilder::new();
    let operand = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(5)]))
        .unwrap();
    let update = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let starts = builder
        .input(ProgramInputSpec::new(DType::I64, [DimExpr::Const(1)]))
        .unwrap();
    let updated = builder
        .add_op(
            CoreSemanticOp::DynamicUpdateSlice,
            &[operand, update, starts],
        )
        .unwrap()[0];
    let dynamic_update = builder.finish(&[updated]).unwrap();
    let update_value = Tensor::from_vec_col_major(vec![2], vec![8.0_f64, 9.0]).unwrap();
    let update_tangent = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let jvp = ad
        .jvp_program(&dynamic_update, &[true, true, false])
        .unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(
            &compiled,
            &[
                &operand_value,
                &update_value,
                &starts_value,
                &operand_tangent,
                &update_tangent,
            ],
        )
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[10.0, 20.0, 1.0, 2.0, 50.0]
    );

    let update_cotangent =
        Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let vjp = ad
        .vjp_program(&dynamic_update, &[true, true, false], &[true])
        .unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled(
            &compiled,
            &[
                &operand_value,
                &update_value,
                &starts_value,
                &update_cotangent,
            ],
        )
        .unwrap();
    assert_eq!(
        result[0].as_slice::<f64>().unwrap(),
        &[1.0, 2.0, 0.0, 0.0, 5.0]
    );
    assert_eq!(result[1].as_slice::<f64>().unwrap(), &[3.0, 4.0]);
}

#[test]
fn semantic_core_dynamic_truncate_and_pad_to_match_execute_jvp_vjp() {
    let ad = ad_context();
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(5)]))
        .unwrap();
    let size = builder
        .input(ProgramInputSpec::new(DType::F64, []))
        .unwrap();
    let truncated = builder
        .add_op(CoreSemanticOp::DynamicTruncate { axis: 0 }, &[input, size])
        .unwrap()[0];
    let truncate = builder.finish(&[truncated]).unwrap();
    let input_value =
        Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let size_value = Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    let tangent =
        Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]).unwrap();

    let jvp = ad.jvp_program(&truncate, &[true, false]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&input_value, &size_value, &tangent])
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[10.0, 20.0, 30.0]);

    let vjp = ad.vjp_program(&truncate, &[true, false], &[true]).unwrap();
    assert_eq!(vjp.derivative_output_indices(), &[Some(0), None]);
    assert!(vjp.frozen().program.operations().any(|operation| matches!(
        operation.op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::PadToMatch { axis: 0 })
    )));

    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(3)]))
        .unwrap();
    let reference = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(5)]))
        .unwrap();
    let padded = builder
        .add_op(CoreSemanticOp::PadToMatch { axis: 0 }, &[input, reference])
        .unwrap()[0];
    let pad_to_match = builder.finish(&[padded]).unwrap();
    let short_input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let reference_value = Tensor::from_vec_col_major(vec![5], vec![0.0_f64; 5]).unwrap();
    let short_tangent = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();

    let jvp = ad.jvp_program(&pad_to_match, &[true, false]).unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(jvp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&short_input, &reference_value, &short_tangent])
        .unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[10.0, 20.0, 30.0, 0.0, 0.0]
    );

    let pad_cotangent =
        Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let vjp = ad
        .vjp_program(&pad_to_match, &[true, false], &[true])
        .unwrap();
    let compiled = GraphCompiler::new()
        .compile_frozen_program(vjp.frozen())
        .unwrap();
    let result = cpu_runtime()
        .run_compiled_one_output(&compiled, &[&short_input, &reference_value, &pad_cotangent])
        .unwrap();
    assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn semantic_program_transforms_use_activity_keyed_collision_checked_cache() {
    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Mul, &[lhs, rhs]).unwrap()[0];
    let source = builder.finish(&[output]).unwrap();
    let ad = AdContext::builder().build().unwrap();

    let first = ad.jvp_program(&source, &[true, false]).unwrap();
    let second = ad.jvp_program(&source, &[true, false]).unwrap();
    assert!(first
        .frozen()
        .program
        .semantic_eq(second.frozen().program.as_ref()));
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 1);

    ad.jvp_program(&source, &[false, true]).unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 2);
    ad.vjp_program(&source, &[true, false], &[true]).unwrap();
    ad.vjp_program(&source, &[true, false], &[true]).unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 3);

    ad.clear_ad_transform_caches().unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 0);
}

#[test]
fn semantic_program_transform_cache_reuses_bound_programs_without_stale_bindings() {
    fn only_bound_f64_values(program: &tenferro_runtime::program::FrozenProgram) -> Vec<f64> {
        let mut bindings = program.bindings.iter();
        let (_, tensor) = bindings.next().expect("one source input binding");
        assert!(
            bindings.next().is_none(),
            "derivative seed inputs must not inherit source bindings"
        );
        tensor.as_slice::<f64>().unwrap().to_vec()
    }

    let first_source = bound_core_square_program(vec![2.0, 3.0]);
    let second_source = bound_core_square_program(vec![5.0, 7.0]);
    assert!(first_source
        .program
        .semantic_eq(second_source.program.as_ref()));

    let ad = AdContext::builder().build().unwrap();

    let first_jvp = ad.jvp_program(&first_source, &[true]).unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 1);
    assert_eq!(only_bound_f64_values(first_jvp.frozen()), vec![2.0, 3.0]);

    let second_jvp = ad.jvp_program(&second_source, &[true]).unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 1);
    assert!(first_jvp
        .frozen()
        .program
        .semantic_eq(second_jvp.frozen().program.as_ref()));
    assert_eq!(only_bound_f64_values(second_jvp.frozen()), vec![5.0, 7.0]);

    let first_vjp = ad.vjp_program(&first_source, &[true], &[true]).unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 2);
    assert_eq!(only_bound_f64_values(first_vjp.frozen()), vec![2.0, 3.0]);

    let second_vjp = ad.vjp_program(&second_source, &[true], &[true]).unwrap();
    assert_eq!(ad.ad_transform_cache_stats().unwrap().entries, 2);
    assert!(first_vjp
        .frozen()
        .program
        .semantic_eq(second_vjp.frozen().program.as_ref()));
    assert_eq!(only_bound_f64_values(second_vjp.frozen()), vec![5.0, 7.0]);
}
