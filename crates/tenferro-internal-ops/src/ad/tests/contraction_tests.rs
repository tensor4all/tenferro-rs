use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueRef};
use tenferro_tensor::{DType, DotGeneralConfig};
use tidu::ADRuleKind;

use crate::ad::context::ShapeGuardContext;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;

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
