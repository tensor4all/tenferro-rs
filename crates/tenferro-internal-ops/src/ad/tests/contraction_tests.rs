use crate::ad::ADRuleKind;
use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueRef};
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::{ShapeExtent, SymDim, TensorMeta};

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

#[test]
fn dot_general_transpose_rejects_out_of_bounds_dims_without_panicking() {
    let lhs = super::input_key(1);
    let rhs = super::input_key(2);
    let mut ctx = ShapeGuardContext::default();
    ctx.insert_metadata(lhs.clone(), super::meta(DType::F64, &[2, 2]));
    ctx.insert_metadata(rhs.clone(), super::meta(DType::F64, &[2, 2]));

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let cotangent = builder.add_input(tensor_input(3));
    let op = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![2],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };

    let err = op
        .transpose_rule(
            &mut builder,
            &[Some(cotangent)],
            &[ValueRef::External(lhs), ValueRef::External(rhs)],
            &OperationRole::Linearized {
                active_mask: vec![true, true],
            },
            &mut ctx,
        )
        .unwrap_err();

    assert_eq!(err.rule(), ADRuleKind::Transpose);
    assert!(err.to_string().contains("lhs_rank=2"));
}

#[test]
fn reduction_transposes_accept_upper_bound_input_metadata() {
    let ops = [
        StdTensorOp::ReduceSum { axes: vec![0] },
        StdTensorOp::ReduceProd { axes: vec![0] },
        StdTensorOp::ReduceMax { axes: vec![0] },
        StdTensorOp::ReduceMin { axes: vec![0] },
    ];

    for (idx, op) in ops.into_iter().enumerate() {
        let input = super::input_key(100 + idx as u64);
        let mut ctx = ShapeGuardContext::default();
        ctx.insert_metadata(
            input.clone(),
            TensorMeta::with_extents(
                DType::F64,
                vec![ShapeExtent::upper_bound(SymDim::from(8usize))],
            ),
        );

        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let cotangent = builder.add_input(tensor_input(200 + idx as u64));
        let result = op
            .transpose_rule(
                &mut builder,
                &[Some(cotangent)],
                &[ValueRef::External(input.clone())],
                &OperationRole::Linearized {
                    active_mask: vec![true],
                },
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());

        let graph = builder.build();
        assert!(
            graph.operations().iter().any(|operation| {
                operation.inputs.contains(&ValueRef::External(input.clone()))
                    && matches!(
                        &operation.operation,
                        StdTensorOp::BroadcastInDim { shape, dims }
                            if shape == &vec![DimExpr::InputDim { input_idx: 1, axis: 0 }]
                                && dims.is_empty()
                    )
            }),
            "{op:?} transpose should restore the runtime input extent instead of requiring an exact shape"
        );
    }
}

#[test]
fn reduction_jvps_accept_upper_bound_input_metadata() {
    let ops = [
        StdTensorOp::ReduceProd { axes: vec![0] },
        StdTensorOp::ReduceMax { axes: vec![0] },
        StdTensorOp::ReduceMin { axes: vec![0] },
    ];

    for (idx, op) in ops.into_iter().enumerate() {
        let input = super::input_key(300 + idx as u64);
        let output = super::input_key(400 + idx as u64);
        let mut ctx = ShapeGuardContext::default();
        ctx.insert_metadata(
            input.clone(),
            TensorMeta::with_extents(
                DType::F64,
                vec![ShapeExtent::upper_bound(SymDim::from(8usize))],
            ),
        );

        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let tangent = builder.add_input(tensor_input(500 + idx as u64));
        let result = op
            .jvp_rule(
                &mut builder,
                std::slice::from_ref(&input),
                &[output],
                &[Some(tangent)],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());

        let graph = builder.build();
        assert!(
            graph.operations().iter().any(|operation| {
                operation.inputs.contains(&ValueRef::External(input.clone()))
                    && matches!(
                        &operation.operation,
                        StdTensorOp::BroadcastInDim { shape, dims }
                            if shape == &vec![DimExpr::InputDim { input_idx: 1, axis: 0 }]
                                && dims.is_empty()
                    )
            }),
            "{op:?} JVP should restore the runtime input extent instead of requiring an exact shape"
        );
    }
}
