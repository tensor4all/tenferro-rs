use computegraph::types::{LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::DType;

use crate::ad::context::ShapeGuardContext;
use crate::std_tensor_op::StdTensorOp;

pub(crate) fn is_real_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
}

pub(crate) fn conjugate_primal_if_complex(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ValRef<StdTensorOp> {
    let dtype = ctx.dtype_of(&input);
    conjugate_primal_if_dtype_complex(emitter, input, dtype)
}

pub(crate) fn conjugate_primal_if_dtype_complex(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    dtype: DType,
) -> ValRef<StdTensorOp> {
    if is_real_dtype(dtype) {
        input
    } else {
        ValRef::Local(emitter.add_op(StdTensorOp::Conj, vec![input], OpMode::Primal)[0])
    }
}

pub(crate) fn conjugate_linear_if_dtype_complex(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    input: LocalValId,
    dtype: DType,
) -> LocalValId {
    if is_real_dtype(dtype) {
        input
    } else {
        emitter.add_op(
            StdTensorOp::Conj,
            vec![ValRef::Local(input)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        )[0]
    }
}
