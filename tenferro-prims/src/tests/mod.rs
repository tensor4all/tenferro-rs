use tenferro_algebra::Standard;

use crate::{
    CpuBackend, CpuContext, SemiringBinaryOp, SemiringCoreDescriptor,
    SemiringFastPathDescriptor, TensorSemiringCore, TensorSemiringFastPath,
};

#[test]
fn protocol_smoke_semiring_core_can_plan_make_contiguous() {
    let mut ctx = CpuContext::new(1);
    let desc = SemiringCoreDescriptor::MakeContiguous;
    let result = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 3]],
    );
    assert!(result.is_ok());
}

#[test]
fn protocol_smoke_semiring_fast_path_can_plan_elementwise_mul() {
    let mut ctx = CpuContext::new(1);
    let desc = SemiringFastPathDescriptor::ElementwiseBinary {
        op: SemiringBinaryOp::Mul,
    };
    let result = <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 3], &[2, 3]],
    );
    assert!(result.is_ok());
}
