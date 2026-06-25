use crate::ad::PrimitiveRuleBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueRef};
use tenferro_tensor::DType;

use crate::dim_expr::DimExpr;
use crate::std_tensor_op::StdTensorOp;

pub(super) fn build_zero_like(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    anchor: ValueRef<StdTensorOp>,
    anchor_rank: usize,
) -> LocalValueId {
    build_scalar_like(builder, dtype, zero_bytes(dtype), anchor, anchor_rank)
}

pub(super) fn build_one_like(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    anchor: ValueRef<StdTensorOp>,
    anchor_rank: usize,
) -> LocalValueId {
    build_scalar_like(builder, dtype, one_bytes(dtype), anchor, anchor_rank)
}

fn build_scalar_like(
    builder: &mut dyn PrimitiveRuleBuilder,
    dtype: DType,
    bytes: Vec<u8>,
    anchor: ValueRef<StdTensorOp>,
    anchor_rank: usize,
) -> LocalValueId {
    let scalar = builder.add_operation(
        StdTensorOp::Constant { dtype, bytes },
        vec![],
        OperationRole::Primary,
    )[0];
    if anchor_rank == 0 {
        return scalar;
    }

    let shape: Vec<DimExpr> = (0..anchor_rank)
        .map(|axis| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    let out = builder.add_operation(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: vec![],
        },
        vec![ValueRef::Local(scalar), anchor],
        OperationRole::Primary,
    );
    out[0]
}

fn zero_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 0.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 0.0_f64.to_le_bytes().to_vec(),
        DType::I32 => 0_i32.to_le_bytes().to_vec(),
        DType::I64 => 0_i64.to_le_bytes().to_vec(),
        DType::Bool => vec![0],
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

fn one_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 1.0_f32.to_le_bytes().to_vec(),
        DType::F64 => 1.0_f64.to_le_bytes().to_vec(),
        DType::I32 => 1_i32.to_le_bytes().to_vec(),
        DType::I64 => 1_i64.to_le_bytes().to_vec(),
        DType::Bool => vec![1],
        DType::C32 => {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            bytes
        }
        DType::C64 => {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&1.0_f64.to_le_bytes());
            bytes.extend_from_slice(&0.0_f64.to_le_bytes());
            bytes
        }
    }
}
