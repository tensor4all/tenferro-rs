use std::cmp::Ordering;

use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::ad::context::{resolve_and_guard, resolve_dim, ShapeGuard, ShapeGuardContext};
use crate::dim_expr::DimExpr;
use crate::input_key::TensorInputKey;
use crate::shape_extent::ShapeExtent;
use crate::std_tensor_op::StdTensorOp;
use crate::{SymDim, TensorMeta};

#[cfg(feature = "autodiff")]
mod contraction_tests;
#[cfg(feature = "autodiff")]
mod elementwise_tests;
#[cfg(feature = "autodiff")]
mod indexing_tests;
#[cfg(feature = "autodiff")]
mod registry_tests;
#[cfg(feature = "autodiff")]
mod structural_tests;

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn input_key(id: u64) -> ValueKey<StdTensorOp> {
    ValueKey::Input(tensor_input(id))
}

fn meta(dtype: DType, shape: &[usize]) -> TensorMeta {
    TensorMeta::exact(dtype, shape.iter().copied().map(SymDim::from).collect())
}

#[test]
fn resolve_dim_const() {
    assert_eq!(resolve_dim(&DimExpr::Const(7)).unwrap(), 7);
}

#[test]
fn resolve_dim_const_expr() {
    let expr = DimExpr::min(DimExpr::Const(3), DimExpr::Const(5));
    assert_eq!(resolve_dim(&expr).unwrap(), 3);
}

#[test]
fn resolve_and_guard_records_greater() {
    let mut ctx = ShapeGuardContext::default();
    let (m, n) = resolve_and_guard(&DimExpr::Const(5), &DimExpr::Const(3), &mut ctx).unwrap();
    assert_eq!((m, n), (5, 3));
    assert_eq!(
        ctx.guards(),
        &[ShapeGuard {
            dim_a: 5,
            dim_b: 3,
            ordering: Ordering::Greater,
        }]
    );
}

#[test]
fn resolve_and_guard_records_less() {
    let mut ctx = ShapeGuardContext::default();
    let (m, n) = resolve_and_guard(&DimExpr::Const(2), &DimExpr::Const(4), &mut ctx).unwrap();
    assert_eq!((m, n), (2, 4));
    assert_eq!(ctx.guards()[0].ordering, Ordering::Less);
}

#[test]
fn resolve_and_guard_records_equal() {
    let mut ctx = ShapeGuardContext::default();
    let (m, n) = resolve_and_guard(&DimExpr::Const(3), &DimExpr::Const(3), &mut ctx).unwrap();
    assert_eq!((m, n), (3, 3));
    assert_eq!(ctx.guards()[0].ordering, Ordering::Equal);
}

#[test]
fn guards_accumulate() {
    let mut ctx = ShapeGuardContext::default();
    resolve_and_guard(&DimExpr::Const(5), &DimExpr::Const(3), &mut ctx).unwrap();
    resolve_and_guard(&DimExpr::Const(2), &DimExpr::Const(4), &mut ctx).unwrap();
    assert_eq!(ctx.guards().len(), 2);
    assert_eq!(ctx.guards()[0].ordering, Ordering::Greater);
    assert_eq!(ctx.guards()[1].ordering, Ordering::Less);
}

#[test]
fn clear_guards_empties() {
    let mut ctx = ShapeGuardContext::default();
    resolve_and_guard(&DimExpr::Const(5), &DimExpr::Const(3), &mut ctx).unwrap();
    ctx.clear_guards();
    assert!(ctx.guards().is_empty());
}

#[test]
fn shape_and_dtype_queries_work_for_concrete_input_values() {
    let mut ctx = ShapeGuardContext::default();
    let key = input_key(1);
    ctx.insert_metadata(key.clone(), meta(DType::F64, &[2, 3]));

    let val = ValueRef::External(key);
    assert_eq!(ctx.dtype_of(&val).unwrap(), DType::F64);
    assert_eq!(
        ctx.shape_of(&val).unwrap(),
        &[SymDim::from(2usize), SymDim::from(3usize)]
    );
}

#[test]
fn global_metadata_registry_does_not_read_poisoned_inner_state_contract() {
    let source = include_str!("../context.rs");

    assert!(
        !source.contains("into_inner()"),
        "global AD metadata registry must fail closed on mutex poison instead of reading poisoned state"
    );
    assert!(
        source.contains("MetadataRegistryError::LockPoisoned"),
        "global AD metadata registry should expose an explicit poison error"
    );
}

#[test]
fn shape_queries_work_for_symbolic_input_values() {
    let mut ctx = ShapeGuardContext::default();
    let key = input_key(2);
    let symbolic_shape = vec![SymDim::tensor_axis(41, 0), SymDim::tensor_axis(41, 1)];
    ctx.insert_metadata(
        key.clone(),
        TensorMeta::exact(DType::F32, symbolic_shape.clone()),
    );

    assert_eq!(
        ctx.shape_of(&ValueRef::External(key)).unwrap(),
        symbolic_shape.as_slice()
    );
}

#[test]
fn metadata_exposes_exact_extents() {
    let key = input_key(700);
    let mut ctx = ShapeGuardContext::default();
    let meta = TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]);
    ctx.insert_metadata(key.clone(), meta.clone());

    let val = ValueRef::External(key);
    assert_eq!(ctx.extents_of(&val).unwrap(), meta.extents());
    assert_eq!(ctx.exact_shape_of(&val).unwrap(), meta.exact_shape());
}

#[test]
fn metadata_exact_shape_rejects_upper_bound() {
    let key = input_key(701);
    let mut ctx = ShapeGuardContext::default();
    let meta = TensorMeta::with_extents(
        DType::F64,
        vec![ShapeExtent::upper_bound(SymDim::from(4usize))],
    );
    ctx.insert_metadata(key.clone(), meta);

    let value = ValueRef::External(key);
    assert_eq!(ctx.exact_shape_of(&value).unwrap(), None);
    assert!(matches!(
        ctx.shape_of(&value),
        Err(err)
            if matches!(
                err.typed_source(),
                crate::ad::context::ShapeGuardError::NonExactShape { .. }
            )
    ));
}

#[test]
fn metadata_if_available_returns_none_for_unattached_or_bad_local_values() {
    let mut ctx = ShapeGuardContext::default();
    assert!(ctx.metadata_if_available(&ValueRef::Local(0)).is_none());

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let input = builder.add_input(tensor_input(710));
    builder.set_outputs(vec![input]);
    let graph = builder.build();
    ctx.attach_graph(&graph);

    assert!(ctx
        .metadata_if_available(&ValueRef::Local(usize::MAX))
        .is_none());
}

#[test]
fn metadata_queries_resolve_local_values_through_attached_graph() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let lhs = builder.add_input(tensor_input(10));
    let rhs = builder.add_input(tensor_input(11));
    let sum = builder.add_operation(
        StdTensorOp::Add,
        vec![ValueRef::Local(lhs), ValueRef::Local(rhs)],
        OperationRole::Primary,
    )[0];
    builder.set_outputs(vec![sum]);
    let graph = builder.build();

    let mut ctx = ShapeGuardContext::default();
    ctx.attach_graph(&graph);
    ctx.insert_metadata(graph.values()[lhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(graph.values()[rhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(graph.values()[sum].key.clone(), meta(DType::F64, &[2, 3]));

    let local_sum = ValueRef::Local(sum);
    assert_eq!(ctx.dtype_of(&local_sum).unwrap(), DType::F64);
    assert_eq!(
        ctx.shape_of(&local_sum).unwrap(),
        &[SymDim::from(2usize), SymDim::from(3usize)]
    );
}

#[test]
fn representative_graph_values_all_have_queryable_tensor_metadata() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let lhs = builder.add_input(tensor_input(20));
    let rhs = builder.add_input(tensor_input(21));
    let add = builder.add_operation(
        StdTensorOp::Add,
        vec![ValueRef::Local(lhs), ValueRef::Local(rhs)],
        OperationRole::Primary,
    )[0];
    let mul = builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(add), ValueRef::Local(rhs)],
        OperationRole::Primary,
    )[0];
    let dot_rhs = builder.add_input(tensor_input(22));
    let dot = builder.add_operation(
        StdTensorOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        },
        vec![ValueRef::Local(mul), ValueRef::Local(dot_rhs)],
        OperationRole::Primary,
    )[0];
    let reduce = builder.add_operation(
        StdTensorOp::ReduceSum { axes: vec![1] },
        vec![ValueRef::Local(dot)],
        OperationRole::Primary,
    )[0];
    let reshape = builder.add_operation(
        StdTensorOp::Reshape {
            to_shape: DimExpr::from_concrete(&[1, 2]),
        },
        vec![ValueRef::Local(reduce)],
        OperationRole::Primary,
    )[0];
    let broadcast = builder.add_operation(
        StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(&[3, 1, 2]),
            dims: vec![1, 2],
        },
        vec![ValueRef::Local(reshape)],
        OperationRole::Primary,
    )[0];
    builder.set_outputs(vec![broadcast]);
    let graph = builder.build();

    let mut ctx = ShapeGuardContext::default();
    ctx.attach_graph(&graph);
    ctx.insert_metadata(graph.values()[lhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(graph.values()[rhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(graph.values()[add].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(graph.values()[mul].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(
        graph.values()[dot_rhs].key.clone(),
        meta(DType::F64, &[3, 4]),
    );
    ctx.insert_metadata(graph.values()[dot].key.clone(), meta(DType::F64, &[2, 4]));
    ctx.insert_metadata(graph.values()[reduce].key.clone(), meta(DType::F64, &[2]));
    ctx.insert_metadata(
        graph.values()[reshape].key.clone(),
        meta(DType::F64, &[1, 2]),
    );
    ctx.insert_metadata(
        graph.values()[broadcast].key.clone(),
        meta(DType::F64, &[3, 1, 2]),
    );

    for local_id in 0..graph.values().len() {
        let value = ValueRef::Local(local_id);
        let metadata = ctx.metadata_of(&value).unwrap();
        assert_eq!(metadata.dtype, DType::F64);
        assert!(
            metadata.rank() > 0 || local_id == reduce,
            "missing shape metadata for local value {local_id}"
        );
    }
}
