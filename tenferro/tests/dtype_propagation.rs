use tenferro::shape_infer::infer_output_dtype;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig};

#[test]
fn test_add_preserves_f32() {
    let op = StdTensorOp::Add;
    assert_eq!(
        infer_output_dtype(&op, &[DType::F32, DType::F32]),
        DType::F32
    );
}

#[test]
fn test_add_preserves_c32() {
    let op = StdTensorOp::Add;
    assert_eq!(
        infer_output_dtype(&op, &[DType::C32, DType::C32]),
        DType::C32
    );
}

#[test]
fn test_convert_uses_target_dtype() {
    let op = StdTensorOp::Convert {
        from: DType::F32,
        to: DType::F64,
    };
    assert_eq!(infer_output_dtype(&op, &[DType::F32]), DType::F64);
}

#[test]
fn test_constant_uses_variant_dtype() {
    let op = StdTensorOp::Constant {
        dtype: DType::C64,
        bytes: vec![],
    };
    assert_eq!(infer_output_dtype(&op, &[]), DType::C64);
}

#[test]
fn test_eig_f32_produces_c32() {
    let op = StdTensorOp::Eig {
        input_dtype: DType::F32,
        input_shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
    };
    assert_eq!(infer_output_dtype(&op, &[DType::F32]), DType::C32);
}

#[test]
fn test_eig_f64_produces_c64() {
    let op = StdTensorOp::Eig {
        input_dtype: DType::F64,
        input_shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
    };
    assert_eq!(infer_output_dtype(&op, &[DType::F64]), DType::C64);
}

#[test]
fn test_reduce_sum_preserves_dtype() {
    let op = StdTensorOp::ReduceSum {
        axes: vec![0],
        input_shape: vec![DimExpr::Const(4), DimExpr::Const(3)],
    };
    assert_eq!(infer_output_dtype(&op, &[DType::F32]), DType::F32);
}

#[test]
fn test_dot_general_preserves_lhs_dtype() {
    let op = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            lhs_rank: 2,
            rhs_rank: 2,
        },
        lhs_rank: 2,
        rhs_rank: 2,
    };
    assert_eq!(
        infer_output_dtype(&op, &[DType::F32, DType::F32]),
        DType::F32
    );
}
