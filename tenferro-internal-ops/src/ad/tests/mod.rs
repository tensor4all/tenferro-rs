use std::cmp::Ordering;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::ad::context::{resolve_and_guard, resolve_dim, ShapeGuard, ShapeGuardContext};
use crate::dim_expr::DimExpr;
use crate::input_key::TensorInputKey;
use crate::shape_extent::ShapeExtent;
use crate::std_tensor_op::StdTensorOp;
use crate::{SymDim, TensorMeta};

#[cfg(feature = "autodiff")]
mod elementwise_tests;
#[cfg(feature = "autodiff")]
mod indexing_tests;
#[cfg(feature = "autodiff")]
mod registry_tests;

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn input_key(id: u64) -> GlobalValKey<StdTensorOp> {
    GlobalValKey::Input(tensor_input(id))
}

fn meta(dtype: DType, shape: &[usize]) -> TensorMeta {
    TensorMeta::exact(dtype, shape.iter().copied().map(SymDim::from).collect())
}

#[test]
fn resolve_dim_const() {
    assert_eq!(resolve_dim(&DimExpr::Const(7)), 7);
}

#[test]
fn resolve_dim_const_expr() {
    let expr = DimExpr::min(DimExpr::Const(3), DimExpr::Const(5));
    assert_eq!(resolve_dim(&expr), 3);
}

#[test]
fn resolve_and_guard_records_greater() {
    let mut ctx = ShapeGuardContext::default();
    let (m, n) = resolve_and_guard(&DimExpr::Const(5), &DimExpr::Const(3), &mut ctx);
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
    let (m, n) = resolve_and_guard(&DimExpr::Const(2), &DimExpr::Const(4), &mut ctx);
    assert_eq!((m, n), (2, 4));
    assert_eq!(ctx.guards()[0].ordering, Ordering::Less);
}

#[test]
fn resolve_and_guard_records_equal() {
    let mut ctx = ShapeGuardContext::default();
    let (m, n) = resolve_and_guard(&DimExpr::Const(3), &DimExpr::Const(3), &mut ctx);
    assert_eq!((m, n), (3, 3));
    assert_eq!(ctx.guards()[0].ordering, Ordering::Equal);
}

#[test]
fn guards_accumulate() {
    let mut ctx = ShapeGuardContext::default();
    resolve_and_guard(&DimExpr::Const(5), &DimExpr::Const(3), &mut ctx);
    resolve_and_guard(&DimExpr::Const(2), &DimExpr::Const(4), &mut ctx);
    assert_eq!(ctx.guards().len(), 2);
    assert_eq!(ctx.guards()[0].ordering, Ordering::Greater);
    assert_eq!(ctx.guards()[1].ordering, Ordering::Less);
}

#[test]
fn clear_guards_empties() {
    let mut ctx = ShapeGuardContext::default();
    resolve_and_guard(&DimExpr::Const(5), &DimExpr::Const(3), &mut ctx);
    ctx.clear_guards();
    assert!(ctx.guards().is_empty());
}

#[test]
fn shape_and_dtype_queries_work_for_concrete_input_values() {
    let mut ctx = ShapeGuardContext::default();
    let key = input_key(1);
    ctx.insert_metadata(key.clone(), meta(DType::F64, &[2, 3]));

    let val = ValRef::External(key);
    assert_eq!(ctx.dtype_of(&val), DType::F64);
    assert_eq!(
        ctx.shape_of(&val),
        &[SymDim::from(2usize), SymDim::from(3usize)]
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
        ctx.shape_of(&ValRef::External(key)),
        symbolic_shape.as_slice()
    );
}

#[test]
fn metadata_exposes_exact_extents() {
    let key = input_key(700);
    let mut ctx = ShapeGuardContext::default();
    let meta = TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]);
    ctx.insert_metadata(key.clone(), meta.clone());

    let val = ValRef::External(key);
    assert_eq!(ctx.extents_of(&val), meta.extents());
    assert_eq!(ctx.exact_shape_of(&val), Some(meta.shape.clone()));
}

#[test]
fn metadata_exact_shape_rejects_upper_bound() {
    let key = input_key(701);
    let mut ctx = ShapeGuardContext::default();
    let meta = TensorMeta {
        dtype: DType::F64,
        shape: vec![SymDim::from(4usize)],
        extents: vec![ShapeExtent::upper_bound(SymDim::from(4usize))],
    };
    ctx.insert_metadata(key.clone(), meta);

    assert_eq!(ctx.exact_shape_of(&ValRef::External(key)), None);
}

#[test]
fn metadata_queries_resolve_local_values_through_attached_fragment() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let lhs = builder.add_input(tensor_input(10));
    let rhs = builder.add_input(tensor_input(11));
    let sum = builder.add_op(
        StdTensorOp::Add,
        vec![ValRef::Local(lhs), ValRef::Local(rhs)],
        OpMode::Primal,
    )[0];
    builder.set_outputs(vec![sum]);
    let fragment = builder.build();

    let mut ctx = ShapeGuardContext::default();
    ctx.attach_fragment(&fragment);
    ctx.insert_metadata(fragment.vals()[lhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(fragment.vals()[rhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(fragment.vals()[sum].key.clone(), meta(DType::F64, &[2, 3]));

    let local_sum = ValRef::Local(sum);
    assert_eq!(ctx.dtype_of(&local_sum), DType::F64);
    assert_eq!(
        ctx.shape_of(&local_sum),
        &[SymDim::from(2usize), SymDim::from(3usize)]
    );
}

#[test]
fn representative_fragment_values_all_have_queryable_tensor_metadata() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let lhs = builder.add_input(tensor_input(20));
    let rhs = builder.add_input(tensor_input(21));
    let add = builder.add_op(
        StdTensorOp::Add,
        vec![ValRef::Local(lhs), ValRef::Local(rhs)],
        OpMode::Primal,
    )[0];
    let mul = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(add), ValRef::Local(rhs)],
        OpMode::Primal,
    )[0];
    let dot_rhs = builder.add_input(tensor_input(22));
    let dot = builder.add_op(
        StdTensorOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        },
        vec![ValRef::Local(mul), ValRef::Local(dot_rhs)],
        OpMode::Primal,
    )[0];
    let reduce = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![1] },
        vec![ValRef::Local(dot)],
        OpMode::Primal,
    )[0];
    let reshape = builder.add_op(
        StdTensorOp::Reshape {
            to_shape: DimExpr::from_concrete(&[1, 2]),
        },
        vec![ValRef::Local(reduce)],
        OpMode::Primal,
    )[0];
    let broadcast = builder.add_op(
        StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(&[3, 1, 2]),
            dims: vec![1, 2],
        },
        vec![ValRef::Local(reshape)],
        OpMode::Primal,
    )[0];
    builder.set_outputs(vec![broadcast]);
    let fragment = builder.build();

    let mut ctx = ShapeGuardContext::default();
    ctx.attach_fragment(&fragment);
    ctx.insert_metadata(fragment.vals()[lhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(fragment.vals()[rhs].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(fragment.vals()[add].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(fragment.vals()[mul].key.clone(), meta(DType::F64, &[2, 3]));
    ctx.insert_metadata(
        fragment.vals()[dot_rhs].key.clone(),
        meta(DType::F64, &[3, 4]),
    );
    ctx.insert_metadata(fragment.vals()[dot].key.clone(), meta(DType::F64, &[2, 4]));
    ctx.insert_metadata(fragment.vals()[reduce].key.clone(), meta(DType::F64, &[2]));
    ctx.insert_metadata(
        fragment.vals()[reshape].key.clone(),
        meta(DType::F64, &[1, 2]),
    );
    ctx.insert_metadata(
        fragment.vals()[broadcast].key.clone(),
        meta(DType::F64, &[3, 1, 2]),
    );

    for local_id in 0..fragment.vals().len() {
        let value = ValRef::Local(local_id);
        let metadata = ctx.metadata_of(&value);
        assert_eq!(metadata.dtype, DType::F64);
        assert!(
            !metadata.shape.is_empty() || local_id == reduce,
            "missing shape metadata for local value {local_id}"
        );
    }
}
