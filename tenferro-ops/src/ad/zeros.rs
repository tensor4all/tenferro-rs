use computegraph::types::{LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_tensor::DType;

use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;

pub(super) fn build_zero_like(
    emitter: &mut impl OpEmitter<StdTensorOp>,
    dtype: DType,
    anchor: ValRef<StdTensorOp>,
    anchor_rank: usize,
) -> LocalValId {
    let zero_scalar = emitter.add_op(
        StdTensorOp::Constant {
            dtype,
            bytes: zero_bytes(dtype),
        },
        vec![],
        OpMode::Primal,
    )[0];
    if anchor_rank == 0 {
        return zero_scalar;
    }

    let shape: Vec<DimExpr> = (0..anchor_rank)
        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    let out = emitter.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![],
        },
        vec![ValRef::Local(zero_scalar), anchor],
        OpMode::Primal,
    );
    out[0]
}

fn zero_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 0.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 0.0_f64.to_le_bytes().to_vec(),
        DType::I64 => 0_i64.to_le_bytes().to_vec(),
        DType::C32 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
    }
}
