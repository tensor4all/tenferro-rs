use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use super::*;
use crate::optimize::EinsumPlanSpec;
use tenferro_cpu::CpuBackend;
use tenferro_ops::ext_op::invoke_extension_shape_inference;
use tenferro_runtime::{ExtensionCacheSelector, ExtensionCacheStore, ExtensionExecutionContext};
use tenferro_tensor::TensorValue;

#[cfg(feature = "autodiff")]
#[test]
fn semantic_rules_run_through_whole_program_jvp_and_vjp() {
    use tenferro_ad::AdContext;
    use tenferro_ops::dim_expr::DimExpr;
    use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(3)],
        ))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(3), DimExpr::Const(4)],
        ))
        .unwrap();
    let output = builder
        .add_extension(
            Arc::new(EinsumExtensionOp::new(EinsumSubscripts::new(
                &[&[0, 1], &[1, 2]],
                &[0, 2],
            ))),
            &[lhs, rhs],
        )
        .unwrap()[0];
    let source = builder.finish(&[output]).unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();

    let jvp = ad.jvp_program(&source, &[true, true]).unwrap();
    assert_eq!(jvp.derivative_input_indices(), &[Some(2), Some(3)]);
    assert!(matches!(
        jvp.frozen().program.operations().last().unwrap().op(),
        SemanticOpRef::Core(CoreSemanticOp::Add)
    ));

    let vjp = ad.vjp_program(&source, &[true, true], &[true]).unwrap();
    assert_eq!(vjp.derivative_output_indices(), &[Some(0), Some(1)]);
    assert_eq!(
        vjp.frozen()
            .program
            .operations()
            .filter(|operation| matches!(operation.op(), SemanticOpRef::Extension(_)))
            .count(),
        2
    );
}

#[test]
fn infer_output_meta_uses_output_labels_and_promotes_dtype() {
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    let lhs_shape = [SymDim::from(2usize), SymDim::from(3usize)];
    let rhs_shape = [SymDim::from(3usize), SymDim::from(4usize)];

    let meta = invoke_extension_shape_inference(
        &op,
        &[DType::F32, DType::F64],
        &[lhs_shape.as_slice(), rhs_shape.as_slice()],
    )
    .unwrap()
    .output_metas;

    assert_eq!(meta[0].0, DType::F64);
    assert_eq!(meta[0].1, vec![SymDim::from(2usize), SymDim::from(4usize)]);
}

#[test]
fn extension_dtype_promotion_delegates_to_canonical_tensor_rules() {
    let source = include_str!("../extension.rs");
    assert!(
        !source.contains("fn promote_dtype("),
        "einsum extension metadata must not duplicate the canonical dtype promotion lattice"
    );

    let dtypes = [
        DType::Bool,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::C32,
        DType::C64,
    ];
    for lhs in dtypes {
        for rhs in dtypes {
            assert_eq!(
                promote_dtypes([lhs, rhs]),
                tenferro_tensor::validate::promote_dtype(lhs, rhs),
                "promotion mismatch for {lhs:?}, {rhs:?}"
            );
        }
    }
}

#[test]
fn infer_output_meta_keeps_structural_errors_and_records_extent_equality() {
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    let lhs_shape = [SymDim::from(2usize), SymDim::from(3usize)];
    let bad_rhs_rank = [SymDim::from(3usize)];
    let bad_rhs_extent = [SymDim::from(5usize), SymDim::from(4usize)];

    assert!(invoke_extension_shape_inference(
        &op,
        &[DType::F64],
        &[lhs_shape.as_slice(), bad_rhs_rank.as_slice()]
    )
    .is_err());
    assert!(invoke_extension_shape_inference(
        &op,
        &[DType::F64, DType::F64],
        &[lhs_shape.as_slice(), bad_rhs_rank.as_slice()]
    )
    .is_err());
    let inferred = invoke_extension_shape_inference(
        &op,
        &[DType::F64, DType::F64],
        &[lhs_shape.as_slice(), bad_rhs_extent.as_slice()],
    )
    .expect("extent mismatch is represented as a shape equality");
    assert_eq!(inferred.constraints.len(), 1);
}

#[test]
fn semantic_payload_does_not_store_static_tree_execution_hint() {
    let source = include_str!("../extension.rs");
    let payload_start = source
        .find("pub(crate) struct EinsumExtensionOp")
        .expect("einsum extension payload should exist");
    let impl_start = source[payload_start..]
        .find("impl std::fmt::Debug for EinsumExtensionOp")
        .expect("debug impl should follow payload")
        + payload_start;
    let payload_source = &source[payload_start..impl_start];

    assert!(
        !payload_source.contains("ContractionTree") && !payload_source.contains("static_tree"),
        "semantic einsum payload must carry plan identity only; runtime trees belong in prepared/runtime caches"
    );
}

#[test]
fn payload_identity_includes_plan_spec() {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let left_to_right =
        EinsumExtensionOp::with_plan_spec(subscripts.clone(), EinsumPlanSpec::LeftToRight);
    let explicit_path =
        EinsumExtensionOp::with_plan_spec(subscripts, EinsumPlanSpec::Path(vec![(1, 2), (0, 1)]));

    assert!(!left_to_right.payload_eq(&explicit_path));
    assert_ne!(payload_hash(&left_to_right), payload_hash(&explicit_path));
}

#[test]
fn payload_identity_includes_output_shape_hint() {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let plan_spec = EinsumPlanSpec::LeftToRight;
    let without_hint = EinsumExtensionOp::with_plan_spec(subscripts.clone(), plan_spec.clone());
    let with_hint = EinsumExtensionOp::with_output_shape_hint(
        subscripts,
        vec![SymDim::from(2usize), SymDim::from(4usize)],
        plan_spec,
    );

    assert!(!without_hint.payload_eq(&with_hint));
    assert_ne!(payload_hash(&without_hint), payload_hash(&with_hint));
}

#[test]
fn einsum_extension_caches_verify_exact_key_data_after_hash_lookup() {
    let extension_source = include_str!("../extension.rs");
    assert!(extension_source.contains("struct RuntimeTreeCacheKeyData"));
    assert!(extension_source.contains("struct CachedRuntimeTree"));
    assert!(extension_source.contains("key_data.matches_runtime_tree("));
    assert!(!extension_source.contains("get::<Arc<ContractionTree>>(&key)"));
    assert!(!extension_source.contains("RuntimeExecProgram"));

    let traced_source = include_str!("../traced.rs");
    assert!(traced_source.contains("struct ParsedEinsumCacheEntry"));
    assert!(!traced_source.contains("get::<Arc<ParsedEinsum>>(&key)"));

    let eager_source = include_str!("../eager_ad.rs");
    assert!(eager_source.contains("struct ExpandedEagerProgramCacheKeyData"));
    assert!(eager_source.contains("struct CachedExpandedEagerProgram"));
    assert!(eager_source.contains("key_data.matches_expanded_eager_program("));
    assert!(!eager_source.contains("get::<Arc<ExpandedEagerProgram>>(&key)"));
}

#[test]
fn execute_einsum_extension_reads_consumes_strided_view_inputs() {
    let base = Arc::new(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let view =
        TensorValue::from_parts((*base).duplicate().unwrap(), vec![3, 2], vec![2, 1], 0).unwrap();
    let input = view.tensor_read();
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1]], &[0, 1]));
    let mut backend = CpuBackend::new();
    let mut caches = ExtensionCacheStore::new();
    let mut ctx = ExtensionExecutionContext::new(&mut backend, &mut caches);

    let outputs = execute_einsum_extension_reads(&op, &[input], &mut ctx)
        .expect("read-capable einsum extension execution");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(
        outputs[0].as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn runtime_einsum_changing_shapes_track_native_plan_cache_stats() {
    let mut backend = CpuBackend::new();
    let mut caches = ExtensionCacheStore::new();
    let op = EinsumExtensionOp::with_plan_spec(
        EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]),
        EinsumPlanSpec::LeftToRight,
    );
    let cases = [(2, 3, 4, 5), (3, 2, 5, 4), (4, 3, 2, 6)];

    for &(m, k, n, p) in &cases {
        let lhs = Tensor::from_vec_col_major(vec![m, k], sequential_f64(m * k, 1.0)).unwrap();
        let mid = Tensor::from_vec_col_major(vec![k, n], sequential_f64(k * n, 10.0)).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![n, p], sequential_f64(n * p, 100.0)).unwrap();

        let mut ctx = ExtensionExecutionContext::new(&mut backend, &mut caches);
        let outputs = execute_einsum_extension(&op, &[&lhs, &mid, &rhs], &mut ctx).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape(), &[m, p]);
        assert_einsum_matches_matmul_chain(&outputs[0], &lhs, &mid, &rhs);
    }

    let (m, k, n, p) = cases[0];
    let lhs = Tensor::from_vec_col_major(vec![m, k], sequential_f64(m * k, 1.0)).unwrap();
    let mid = Tensor::from_vec_col_major(vec![k, n], sequential_f64(k * n, 10.0)).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![n, p], sequential_f64(n * p, 100.0)).unwrap();
    let mut ctx = ExtensionExecutionContext::new(&mut backend, &mut caches);
    let outputs = execute_einsum_extension(&op, &[&lhs, &mid, &rhs], &mut ctx).unwrap();
    assert_einsum_matches_matmul_chain(&outputs[0], &lhs, &mid, &rhs);

    let stats = caches.stats(ExtensionCacheSelector::All);
    assert_eq!(stats.entries, cases.len());
    assert_eq!(stats.misses, cases.len() as u64);
    assert_eq!(stats.hits, 1);
}

#[test]
#[cfg(feature = "autodiff")]
fn vjp_einsum_op_inherits_plan_spec_without_storing_concrete_tree() {
    let primal_op = EinsumExtensionOp::with_plan_spec(
        EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]),
        EinsumPlanSpec::Path(vec![(1, 2), (0, 1)]),
    );
    let vjp_subscripts = EinsumSubscripts {
        inputs: vec![vec![0, 3], vec![1, 2], vec![2, 3]],
        output: vec![0, 1],
    };
    let vjp_shapes = vec![
        vec![DimExpr::Const(2), DimExpr::Const(5)],
        vec![DimExpr::Const(3), DimExpr::Const(4)],
        vec![DimExpr::Const(4), DimExpr::Const(5)],
    ];

    let op = semantic_vjp_einsum_op(&primal_op, 0, vjp_subscripts, &vjp_shapes).unwrap();

    assert!(matches!(
        op.plan_spec(),
        EinsumPlanSpec::FixedPairs(pairs) if pairs == &vec![(1, 2), (0, 3)]
    ));
}

#[test]
#[cfg(feature = "autodiff")]
fn vjp_einsum_op_derives_plan_for_nonfirst_active_input() {
    let primal_op = EinsumExtensionOp::with_plan_spec(
        EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]),
        EinsumPlanSpec::Path(vec![(1, 2), (0, 1)]),
    );
    let vjp_subscripts = EinsumSubscripts {
        inputs: vec![vec![0, 3], vec![0, 1], vec![2, 3]],
        output: vec![1, 2],
    };
    let vjp_shapes = vec![
        vec![DimExpr::Const(2), DimExpr::Const(5)],
        vec![DimExpr::Const(2), DimExpr::Const(3)],
        vec![DimExpr::Const(4), DimExpr::Const(5)],
    ];

    let op = semantic_vjp_einsum_op(&primal_op, 1, vjp_subscripts, &vjp_shapes).unwrap();

    assert!(matches!(
        op.plan_spec(),
        EinsumPlanSpec::FixedPairs(pairs) if pairs == &vec![(0, 1), (3, 2)]
    ));
}

#[test]
#[cfg(feature = "autodiff")]
fn repeated_label_projection_projects_each_extra_occurrence() {
    use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

    let mut builder = SemanticProgramBuilder::new();
    let cotangent = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [
                DimExpr::Const(2),
                DimExpr::Const(3),
                DimExpr::Const(3),
                DimExpr::Const(3),
            ],
        ))
        .unwrap();
    let result = semantic_project_repeated_labels(&mut builder, cotangent, &[0, 1, 1, 1]).unwrap();
    let frozen = builder.finish(&[result]).unwrap();
    let ops: Vec<_> = frozen
        .program
        .operations()
        .map(|operation| operation.op())
        .collect();

    assert!(matches!(
        ops.as_slice(),
        [
            SemanticOpRef::Core(CoreSemanticOp::ExtractDiag {
                axis_a: 1,
                axis_b: 2
            }),
            SemanticOpRef::Core(CoreSemanticOp::EmbedDiag {
                axis_a: 1,
                axis_b: 2
            }),
            SemanticOpRef::Core(CoreSemanticOp::ExtractDiag {
                axis_a: 1,
                axis_b: 3
            }),
            SemanticOpRef::Core(CoreSemanticOp::EmbedDiag {
                axis_a: 1,
                axis_b: 3
            })
        ]
    ));
}

#[test]
#[cfg(feature = "autodiff")]
fn vjp_broadcast_remap_failure_returns_error() {
    use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};

    let mut builder = SemanticProgramBuilder::new();
    let cotangent = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(4)],
        ))
        .unwrap();
    let err = semantic_broadcast_einsum_vjp(
        &mut builder,
        cotangent,
        &[0, 2],
        &[0, 1],
        vec![DimExpr::Const(2), DimExpr::Const(3)],
    )
    .expect_err("unmappable VJP labels should be an AD rule error");

    let message = err.to_string();
    assert!(message.contains("einsum VJP cannot remap labels"));
}

#[test]
#[cfg(feature = "autodiff")]
fn semantic_rules_preserve_nary_order_absent_tangents_and_active_vjp_mask() {
    use tenferro_ad::semantic_extension::AdValue;
    use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

    let subscripts = EinsumSubscripts {
        inputs: vec![
            vec![b'i' as u32, b'j' as u32],
            vec![b'j' as u32, b'k' as u32],
        ],
        output: vec![b'i' as u32, b'k' as u32],
    };
    let make_op =
        || EinsumExtensionOp::with_plan_spec(subscripts.clone(), EinsumPlanSpec::LeftToRight);

    let mut source = SemanticProgramBuilder::new();
    let source_lhs = source
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(3)],
        ))
        .unwrap();
    let source_rhs = source
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(3), DimExpr::Const(4)],
        ))
        .unwrap();
    let source_output = source
        .add_extension(Arc::new(make_op()), &[source_lhs, source_rhs])
        .unwrap()[0];
    let source = source.finish(&[source_output]).unwrap();
    let operation = source.program.operations().next().unwrap();

    let mut builder = SemanticProgramBuilder::new();
    let lhs = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(3)],
        ))
        .unwrap();
    let rhs = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(3), DimExpr::Const(4)],
        ))
        .unwrap();
    let dlhs = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(3)],
        ))
        .unwrap();
    let cotangent = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::Const(2), DimExpr::Const(4)],
        ))
        .unwrap();
    let primal_output = builder
        .add_extension(Arc::new(make_op()), &[lhs, rhs])
        .unwrap()[0];
    let rules = semantic_ad_rules().unwrap();
    let linearized = rules
        .linearize_operation(
            operation,
            &[lhs, rhs],
            &[primal_output],
            &[AdValue::Value(dlhs), AdValue::Absent],
            &[true],
            &mut builder,
        )
        .unwrap();
    let AdValue::Value(tangent_output) = linearized.tangent_outputs()[0] else {
        panic!("einsum tangent must be active");
    };
    let input_cotangents = rules
        .linear_transpose_operation(
            operation,
            &[lhs, rhs],
            &[primal_output],
            &[AdValue::Value(cotangent)],
            &[true, false],
            linearized.residuals(),
            &mut builder,
        )
        .unwrap();
    assert!(matches!(input_cotangents[0], AdValue::Value(_)));
    assert_eq!(input_cotangents[1], AdValue::Absent);

    let frozen = builder.finish(&[tangent_output]).unwrap();
    let tangent_operation = frozen
        .program
        .operations()
        .filter(|operation| matches!(operation.op(), SemanticOpRef::Extension(_)))
        .nth(1)
        .unwrap();
    assert_eq!(tangent_operation.inputs(), &[dlhs, rhs]);
}

fn payload_hash(op: &EinsumExtensionOp) -> u64 {
    let mut hasher = DefaultHasher::new();
    op.payload_hash(&mut hasher);
    hasher.finish()
}

fn sequential_f64(len: usize, offset: f64) -> Vec<f64> {
    (0..len).map(|index| offset + index as f64).collect()
}

fn assert_einsum_matches_matmul_chain(output: &Tensor, lhs: &Tensor, mid: &Tensor, rhs: &Tensor) {
    let &[m, k] = lhs.shape() else {
        panic!("lhs must be rank-2");
    };
    let &[mid_k, n] = mid.shape() else {
        panic!("mid must be rank-2");
    };
    let &[rhs_n, p] = rhs.shape() else {
        panic!("rhs must be rank-2");
    };
    assert_eq!(mid_k, k);
    assert_eq!(rhs_n, n);
    assert_eq!(output.shape(), &[m, p]);
    let lhs = lhs.as_slice::<f64>().unwrap();
    let mid = mid.as_slice::<f64>().unwrap();
    let rhs = rhs.as_slice::<f64>().unwrap();
    let output = output.as_slice::<f64>().unwrap();

    let mut expected = vec![0.0; m * p];
    for row in 0..m {
        for col in 0..p {
            let mut total = 0.0;
            for inner_n in 0..n {
                let mut lhs_mid = 0.0;
                for inner_k in 0..k {
                    lhs_mid += lhs[row + inner_k * m] * mid[inner_k + inner_n * k];
                }
                total += lhs_mid * rhs[inner_n + col * n];
            }
            expected[row + col * m] = total;
        }
    }

    let max_abs = output
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 1.0e-8,
        "max_abs={max_abs}, actual={output:?}, expected={expected:?}"
    );
}
