use super::{map_static_output_shape_result, static_output_shape};
use crate::Error;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::{GraphCompiler, GraphProgramLoweringShapeError, TracedTensor};

#[test]
fn missing_output_shape_metadata_maps_to_invalid_program() {
    let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&x.neg()).unwrap();
    let inst = program.lowering_view().instructions().next().unwrap();

    let err = static_output_shape(inst, 1, &[&[1]]).unwrap_err();

    let Error::InvalidProgram { message } = err else {
        panic!("expected InvalidProgram, got {err:?}");
    };
    assert!(message.contains("ExecOp::Negate missing output_extents for output 1"));
}

#[test]
fn exact_shape_mapping_returns_static_shape() {
    let shape = map_static_output_shape_result(Ok(vec![2, 3, 5])).unwrap();

    assert_eq!(shape, vec![2, 3, 5]);
}

#[test]
fn non_static_shape_metadata_maps_to_xla_error() {
    let err = map_static_output_shape_result(Err(GraphProgramLoweringShapeError::NonStatic {
        op: "DynamicTruncate",
        output_index: 0,
        axis: 1,
        kind: "an upper bound",
    }))
    .unwrap_err();

    let Error::NonStaticShape {
        op,
        output_index,
        axis,
        kind,
    } = err
    else {
        panic!("expected NonStaticShape, got {err:?}");
    };
    assert_eq!(op, "DynamicTruncate");
    assert_eq!(output_index, 0);
    assert_eq!(axis, 1);
    assert_eq!(kind, "an upper bound");
}

#[test]
fn invalid_dim_expr_maps_to_invalid_program() {
    let source = DimExpr::InputDim {
        input_idx: 1,
        axis: 0,
    }
    .eval(&[&[2]])
    .unwrap_err();
    let err = map_static_output_shape_result(Err(GraphProgramLoweringShapeError::InvalidDimExpr {
        op: "BroadcastInDim",
        output_index: 0,
        axis: 1,
        source,
    }))
    .unwrap_err();

    let Error::InvalidProgram { message } = err else {
        panic!("expected InvalidProgram, got {err:?}");
    };
    assert!(
        message.contains("ExecOp::BroadcastInDim output 0 axis 1 has invalid dimension expression"),
        "{message}"
    );
}
