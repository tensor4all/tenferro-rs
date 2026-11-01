use std::collections::BTreeMap;

use computegraph::graph::{Graph, GraphBuilder};
use computegraph::types::ValueRef;

use super::{
    all_primitive_ad_support, identity_matrix, one_like, primitive_ad_support, promote_dtype,
    AdRuleSupport,
};
use crate::dim_expr::DimExpr;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};
use tenferro_tensor::DType;

fn op_kind_name(op: &StdTensorOp) -> &'static str {
    match op {
        StdTensorOp::Constant { .. } => "Constant",
        StdTensorOp::BroadcastInDim { .. } => "BroadcastInDim",
        StdTensorOp::EmbedDiag { .. } => "EmbedDiag",
        StdTensorOp::Exp => "Exp",
        StdTensorOp::Log => "Log",
        StdTensorOp::Sin => "Sin",
        StdTensorOp::Cos => "Cos",
        StdTensorOp::Tanh => "Tanh",
        StdTensorOp::Sqrt => "Sqrt",
        StdTensorOp::Rsqrt => "Rsqrt",
        StdTensorOp::Expm1 => "Expm1",
        StdTensorOp::Log1p => "Log1p",
        _ => "Other",
    }
}

fn op_kind_histogram(graph: &Graph<StdTensorOp>) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for op in graph.operations() {
        *counts.entry(op_kind_name(&op.operation)).or_insert(0) += 1;
    }
    counts
}

fn assert_no_analytic_constant_shortcuts(graph: &Graph<StdTensorOp>) {
    for op in graph.operations() {
        assert!(
            !matches!(
                op.operation,
                StdTensorOp::Exp
                    | StdTensorOp::Log
                    | StdTensorOp::Sin
                    | StdTensorOp::Cos
                    | StdTensorOp::Tanh
                    | StdTensorOp::Sqrt
                    | StdTensorOp::Rsqrt
                    | StdTensorOp::Expm1
                    | StdTensorOp::Log1p
            ),
            "AD semantic constant helpers must not use analytic shortcuts; saw {:?}",
            op.operation
        );
    }
}

#[test]
fn primitive_ad_support_manifest_covers_core_catalog_order() {
    let manifest = all_primitive_ad_support();
    assert_eq!(manifest.len(), PrimitiveOpKind::COUNT);

    for descriptor in all_primitive_descriptors() {
        let entry = primitive_ad_support(descriptor.kind);
        assert_eq!(entry.kind, descriptor.kind);
        assert_eq!(manifest[descriptor.kind.as_index()], *entry);
    }
}

#[test]
fn primitive_ad_support_manifest_has_no_unsupported_entries() {
    for entry in all_primitive_ad_support() {
        assert_ne!(entry.linearize, AdRuleSupport::Unsupported);
        assert_ne!(entry.transpose, AdRuleSupport::Unsupported);
    }
}

#[test]
fn primitive_ad_support_manifest_marks_known_non_differentiable_ops() {
    for kind in [
        PrimitiveOpKind::Compare,
        PrimitiveOpKind::ShapeOf,
        PrimitiveOpKind::Constant,
    ] {
        let entry = primitive_ad_support(kind);
        assert_eq!(entry.linearize, AdRuleSupport::NonDifferentiable);
        assert_eq!(entry.transpose, AdRuleSupport::NonDifferentiable);
    }
}

#[test]
fn promote_dtype_covers_supported_pairs_without_runtime_unreachable() {
    let source = include_str!("../support.rs");
    assert!(
        !source.contains("promote_dtype: unhandled pair"),
        "dtype promotion should use exhaustive DType matching, not a runtime unreachable"
    );

    assert_eq!(promote_dtype(DType::Bool, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::I32, DType::I64), DType::I64);
    assert_eq!(promote_dtype(DType::I64, DType::F32), DType::F64);
    assert_eq!(promote_dtype(DType::F32, DType::C32), DType::C32);
    assert_eq!(promote_dtype(DType::F64, DType::C32), DType::C64);
    assert_eq!(promote_dtype(DType::C32, DType::C64), DType::C64);
}

#[test]
fn promote_dtype_delegates_to_canonical_tensor_validation_rules() {
    let source = include_str!("../support.rs");
    assert!(
        source.contains("tenferro_tensor::validate::promote_dtype"),
        "AD support must delegate to the canonical tensor dtype promotion lattice"
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
                promote_dtype(lhs, rhs),
                tenferro_tensor::validate::promote_dtype(lhs, rhs),
                "promotion mismatch for {lhs:?}, {rhs:?}"
            );
        }
    }
}

#[test]
fn one_like_helper_emits_semantic_constant_not_analytic_shortcut() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let anchor = builder.add_input(TensorInputKey::User { id: 901 });

    let one = one_like(&mut builder, DType::F64, ValueRef::Local(anchor), 2);
    let graph = builder.build();

    assert!(graph.values()[one].producer.is_some());
    assert_no_analytic_constant_shortcuts(&graph);
    assert_eq!(
        op_kind_histogram(&graph),
        BTreeMap::from([("BroadcastInDim", 1), ("Constant", 1)])
    );
}

#[test]
fn identity_matrix_helper_emits_semantic_constant_and_remaps_shape_source() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let anchor = builder.add_input(TensorInputKey::User { id: 900 });

    let identity = identity_matrix(
        &mut builder,
        DType::F64,
        3,
        &[DimExpr::InputDim {
            input_idx: 0,
            axis: 2,
        }],
        ValueRef::Local(anchor),
        0,
    );
    let graph = builder.build();

    assert_no_analytic_constant_shortcuts(&graph);
    assert_eq!(
        op_kind_histogram(&graph),
        BTreeMap::from([("BroadcastInDim", 1), ("Constant", 1), ("EmbedDiag", 1)])
    );

    let (embed_diag_id, _) = graph.values()[identity]
        .producer
        .expect("identity output must be produced by embed diag");
    assert!(matches!(
        graph.operations()[embed_diag_id].operation,
        StdTensorOp::EmbedDiag { .. }
    ));
    let broadcast_id = match graph.operations()[embed_diag_id].inputs[0] {
        ValueRef::Local(id) => id,
        ref other => panic!("embed diag should consume broadcast output, got {other:?}"),
    };
    let (broadcast_op_id, _) = graph.values()[broadcast_id]
        .producer
        .expect("identity diagonal must be produced by broadcast");
    let broadcast = &graph.operations()[broadcast_op_id];
    assert_eq!(
        broadcast.operation,
        StdTensorOp::BroadcastInDim {
            shape: vec![
                DimExpr::Const(3),
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 2,
                },
            ],
            dims: vec![],
        }
    );
    assert_eq!(broadcast.inputs[1], ValueRef::Local(anchor));
}

#[test]
#[should_panic(
    expected = "identity_matrix shape expressions must reference only the provided shape_source_idx"
)]
fn identity_matrix_helper_rejects_unbound_shape_sources() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let anchor = builder.add_input(TensorInputKey::User { id: 902 });

    let _ = identity_matrix(
        &mut builder,
        DType::F64,
        3,
        &[DimExpr::InputDim {
            input_idx: 1,
            axis: 0,
        }],
        ValueRef::Local(anchor),
        0,
    );
}

/// Every preset dtype the AD catalog can seed, with the exact little-endian
/// bytes the symbolic zero and one constants must carry.
fn preset_scalar_cases() -> Vec<(DType, Vec<u8>, Vec<u8>)> {
    fn c32(re: f32, im: f32) -> Vec<u8> {
        let mut bytes = re.to_le_bytes().to_vec();
        bytes.extend_from_slice(&im.to_le_bytes());
        bytes
    }
    fn c64(re: f64, im: f64) -> Vec<u8> {
        let mut bytes = re.to_le_bytes().to_vec();
        bytes.extend_from_slice(&im.to_le_bytes());
        bytes
    }
    vec![
        (
            DType::F32,
            0.0_f32.to_le_bytes().to_vec(),
            1.0_f32.to_le_bytes().to_vec(),
        ),
        (
            DType::F64,
            0.0_f64.to_le_bytes().to_vec(),
            1.0_f64.to_le_bytes().to_vec(),
        ),
        (
            DType::I32,
            0_i32.to_le_bytes().to_vec(),
            1_i32.to_le_bytes().to_vec(),
        ),
        (
            DType::I64,
            0_i64.to_le_bytes().to_vec(),
            1_i64.to_le_bytes().to_vec(),
        ),
        (DType::Bool, vec![0], vec![1]),
        (DType::C32, c32(0.0, 0.0), c32(1.0, 0.0)),
        (DType::C64, c64(0.0, 0.0), c64(1.0, 0.0)),
    ]
}

fn sole_constant_bytes(
    graph: &Graph<StdTensorOp>,
    value: computegraph::types::LocalValueId,
) -> (DType, Vec<u8>) {
    let (op_id, _) = graph.values()[value]
        .producer
        .expect("scalar seed must be produced by a constant");
    match &graph.operations()[op_id].operation {
        StdTensorOp::Constant { dtype, bytes } => (*dtype, bytes.clone()),
        other => panic!("expected a Constant, saw {other:?}"),
    }
}

#[test]
fn zero_like_emits_exact_zero_bytes_for_every_preset_dtype() {
    for (idx, (dtype, zero, _)) in preset_scalar_cases().into_iter().enumerate() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let anchor = builder.add_input(TensorInputKey::User {
            id: 920 + idx as u64,
        });
        let value = super::zero_like(&mut builder, dtype, ValueRef::Local(anchor), 0);
        let graph = builder.build();

        assert_eq!(
            op_kind_histogram(&graph),
            BTreeMap::from([("Constant", 1)]),
            "a rank-0 seed for {dtype:?} must be one constant and nothing else"
        );
        assert_eq!(sole_constant_bytes(&graph, value), (dtype, zero));
    }
}

#[test]
fn one_like_emits_exact_one_bytes_for_every_preset_dtype() {
    for (idx, (dtype, _, one)) in preset_scalar_cases().into_iter().enumerate() {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let anchor = builder.add_input(TensorInputKey::User {
            id: 930 + idx as u64,
        });
        let value = super::one_like(&mut builder, dtype, ValueRef::Local(anchor), 0);
        let graph = builder.build();

        assert_eq!(
            op_kind_histogram(&graph),
            BTreeMap::from([("Constant", 1)]),
            "a rank-0 seed for {dtype:?} must be one constant and nothing else"
        );
        assert_eq!(sole_constant_bytes(&graph, value), (dtype, one));
    }
}

#[test]
fn scalar_seeds_broadcast_once_per_anchor_axis() {
    for (idx, (dtype, zero, one)) in preset_scalar_cases().into_iter().enumerate() {
        for (label, expected_bytes) in [("zero", zero), ("one", one)] {
            let mut builder = GraphBuilder::<StdTensorOp>::new();
            let anchor = builder.add_input(TensorInputKey::User {
                id: 940 + idx as u64,
            });
            let value = if label == "zero" {
                super::zero_like(&mut builder, dtype, ValueRef::Local(anchor), 3)
            } else {
                super::one_like(&mut builder, dtype, ValueRef::Local(anchor), 3)
            };
            let graph = builder.build();

            assert_eq!(
                op_kind_histogram(&graph),
                BTreeMap::from([("BroadcastInDim", 1), ("Constant", 1)]),
                "{label} seed on a rank-3 anchor must lift one scalar once"
            );

            let (broadcast_id, _) = graph.values()[value]
                .producer
                .expect("non-scalar seed must be produced by a broadcast");
            let broadcast = &graph.operations()[broadcast_id];
            assert_eq!(
                broadcast.operation,
                StdTensorOp::BroadcastInDim {
                    shape: (0..3)
                        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
                        .collect(),
                    dims: vec![],
                }
            );
            assert_eq!(broadcast.inputs[1], ValueRef::Local(anchor));
            let scalar = match broadcast.inputs[0] {
                ValueRef::Local(id) => id,
                ref other => panic!("broadcast must consume a local scalar, got {other:?}"),
            };
            assert_eq!(sole_constant_bytes(&graph, scalar), (dtype, expected_bytes));
        }
    }
}
