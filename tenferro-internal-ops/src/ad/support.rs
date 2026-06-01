use computegraph::types::{LocalValueId, OperationRole, ValueRef};
use tenferro_tensor::DType;

use crate::ad::context::ShapeGuardContext;
use crate::ad::PrimitiveRuleBuilder;
use crate::std_tensor_op::StdTensorOp;

pub fn is_real_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
}

pub fn conjugate_primal_if_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ValueRef<StdTensorOp> {
    let dtype = ctx.dtype_of(&input);
    conjugate_primal_if_dtype_complex(builder, input, dtype)
}

pub fn conjugate_primal_if_dtype_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    dtype: DType,
) -> ValueRef<StdTensorOp> {
    if is_real_dtype(dtype) {
        input
    } else {
        ValueRef::Local(
            builder.add_operation(StdTensorOp::Conj, vec![input], OperationRole::Primary)[0],
        )
    }
}

pub fn conjugate_linear_if_dtype_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: LocalValueId,
    dtype: DType,
) -> LocalValueId {
    if is_real_dtype(dtype) {
        input
    } else {
        builder.add_operation(
            StdTensorOp::Conj,
            vec![ValueRef::Local(input)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0]
    }
}
