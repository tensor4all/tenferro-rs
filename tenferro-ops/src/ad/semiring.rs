use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_add(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn linearize_mul(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn linearize_neg(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn linearize_conj(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_add(_cotangent_out: &[Option<LocalValId>]) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_mul(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_neg(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_conj(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    todo!()
}
