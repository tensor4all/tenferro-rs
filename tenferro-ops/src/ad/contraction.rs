use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use tenferro_tensor::DotGeneralConfig;

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_dot_general(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _primal_in: &[GlobalValKey<StdTensorOp>],
    _tangent_in: &[Option<LocalValId>],
    _config: &DotGeneralConfig,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn linearize_reduce_sum(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
    _op: &StdTensorOp,
    _axes: &[usize],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_dot_general(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _inputs: &[ValRef<StdTensorOp>],
    _mode: &OpMode,
    _config: &DotGeneralConfig,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_reduce_sum(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    todo!()
}
