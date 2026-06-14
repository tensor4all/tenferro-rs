use super::static_output_shape;
use crate::Error;
use tenferro_runtime::{GraphCompiler, TracedTensor};

#[test]
fn missing_output_shape_metadata_maps_to_invalid_program() {
    let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]);
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&x.neg()).unwrap();
    let inst = program.lowering_view().instructions().next().unwrap();

    let err = static_output_shape(inst, 1, &[&[1]]).unwrap_err();

    let Error::InvalidProgram { message } = err else {
        panic!("expected InvalidProgram, got {err:?}");
    };
    assert!(message.contains("ExecOp::Negate missing output_extents for output 1"));
}
