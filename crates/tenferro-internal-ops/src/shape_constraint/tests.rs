use tenferro_tensor::DType;

use super::{ExtensionShapeContext, ExtensionShapeError, ShapeRelation};
use crate::SymDim;

fn inference_context<'a>(
    input_dtypes: &'a [DType],
    input_shapes: &'a [&'a [SymDim]],
) -> ExtensionShapeContext<'a> {
    ExtensionShapeContext::new_for_inference("test.shape.v1", input_dtypes, input_shapes)
}

#[test]
fn expression_equality_is_recorded_without_solving() {
    let lhs_shape = [SymDim::tensor_axis(10, 0)];
    let rhs_shape = [SymDim::tensor_axis(11, 0)];
    let input_dtypes = [DType::F64, DType::F32];
    let input_shapes: [&[SymDim]; 2] = [&lhs_shape, &rhs_shape];
    let mut ctx = inference_context(&input_dtypes, &input_shapes);

    assert_eq!(ctx.input_dtype(0), Ok(DType::F64));
    assert_eq!(ctx.input_shape(1), Ok(rhs_shape.as_slice()));
    let lhs = ctx.input_axis(0, 0).unwrap();
    let rhs = ctx.input_axis(1, 0).unwrap();
    ctx.require_equal(lhs.clone(), 2 * rhs.clone()).unwrap();

    let constraints = ctx.constraints();
    assert_eq!(constraints.len(), 1);
    assert_eq!(constraints[0].relation(), ShapeRelation::Equal);
    assert_eq!(constraints[0].lhs(), &lhs);
    assert_eq!(constraints[0].rhs(), &(2 * rhs));
    let expected = constraints.to_vec();
    assert_eq!(ctx.into_constraints(), expected);
}

#[test]
fn require_same_shape_records_one_equality_per_axis() {
    let lhs_shape = [SymDim::tensor_axis(10, 0), SymDim::tensor_axis(10, 1)];
    let rhs_shape = [SymDim::tensor_axis(11, 0), SymDim::tensor_axis(11, 1)];
    let input_dtypes = [DType::F64, DType::F64];
    let input_shapes: [&[SymDim]; 2] = [&lhs_shape, &rhs_shape];
    let mut ctx = inference_context(&input_dtypes, &input_shapes);

    ctx.require_same_shape(0, 1).unwrap();

    assert_eq!(ctx.constraints().len(), 2);
    assert_eq!(ctx.constraints()[0].lhs(), &lhs_shape[0]);
    assert_eq!(ctx.constraints()[0].rhs(), &rhs_shape[0]);
    assert_eq!(ctx.constraints()[1].lhs(), &lhs_shape[1]);
    assert_eq!(ctx.constraints()[1].rhs(), &rhs_shape[1]);
}

#[test]
fn require_same_shape_rank_mismatch_is_typed() {
    let lhs_shape = [SymDim::from(2)];
    let rhs_shape = [SymDim::from(2), SymDim::from(3)];
    let input_dtypes = [DType::F64, DType::F64];
    let input_shapes: [&[SymDim]; 2] = [&lhs_shape, &rhs_shape];
    let mut ctx = inference_context(&input_dtypes, &input_shapes);

    assert_eq!(
        ctx.require_same_shape(0, 1),
        Err(ExtensionShapeError::RankMismatch {
            family_id: "test.shape.v1",
            lhs_input: 0,
            lhs_rank: 1,
            rhs_input: 1,
            rhs_rank: 2,
        })
    );
    assert!(ctx.constraints().is_empty());
}

#[test]
fn input_and_axis_bounds_errors_are_typed() {
    let shape = [SymDim::from(2)];
    let input_dtypes = [DType::F64];
    let input_shapes: [&[SymDim]; 1] = [&shape];
    let mut ctx = inference_context(&input_dtypes, &input_shapes);

    let input_error = ExtensionShapeError::InputOutOfBounds {
        family_id: "test.shape.v1",
        input: 1,
        input_count: 1,
    };
    assert_eq!(ctx.input_dtype(1), Err(input_error.clone()));
    assert_eq!(ctx.input_shape(1), Err(input_error.clone()));
    assert_eq!(ctx.require_same_shape(0, 1), Err(input_error));

    let axis_error = ExtensionShapeError::AxisOutOfBounds {
        family_id: "test.shape.v1",
        input: 0,
        axis: 1,
        rank: 1,
    };
    assert_eq!(ctx.input_axis(0, 1), Err(axis_error.clone()));
    assert_eq!(ctx.require_axes_equal((0, 0), (0, 1)), Err(axis_error));
    assert!(ctx.constraints().is_empty());
}
