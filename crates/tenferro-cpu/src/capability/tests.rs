use super::*;

#[test]
fn capability_descriptor_helpers_set_axes_and_output_dtypes() {
    let owned = owned_read(PrimitiveOpKind::Abs, DType::C64, DType::F64, F);
    assert_eq!(owned.backend, BackendId::Cpu);
    assert_eq!(owned.op, PrimitiveOpKind::Abs);
    assert_eq!(owned.dtype, DType::C64);
    assert_eq!(owned.output_dtype, DType::F64);
    assert_eq!(owned.result, F);
    assert_eq!(owned.read_inputs, F);
    assert_eq!(owned.write_output, U);
    assert_eq!(owned.strided_output, U);
    assert_eq!(owned.accumulation, U);

    let same = same_dtype(PrimitiveOpKind::Rem, DType::I64, N);
    assert_eq!(same.output_dtype, DType::I64);
    assert_eq!(same.write_output, U);

    let write = same_dtype_write(PrimitiveOpKind::Div, DType::I32, N, F);
    assert_eq!(write.output_dtype, DType::I32);
    assert_eq!(write.result, N);
    assert_eq!(write.read_inputs, N);
    assert_eq!(write.write_output, F);
    assert_eq!(write.strided_output, U);
    assert_eq!(write.accumulation, U);

    let dot = dot_native(DType::C32);
    assert_eq!(dot.op, PrimitiveOpKind::DotGeneral);
    assert_eq!(dot.output_dtype, DType::C32);
    assert_eq!(dot.result, N);
    assert_eq!(dot.read_inputs, N);
    assert_eq!(dot.write_output, N);
    assert_eq!(dot.strided_output, N);
    assert_eq!(dot.accumulation, N);
}
