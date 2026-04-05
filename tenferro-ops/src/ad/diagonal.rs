use computegraph::fragment::FragmentBuilder;
use computegraph::types::LocalValId;

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_extract_diag(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
    _axis_a: usize,
    _axis_b: usize,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn linearize_embed_diag(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
    _axis_a: usize,
    _axis_b: usize,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_extract_diag(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _axis_a: usize,
    _axis_b: usize,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_embed_diag(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _axis_a: usize,
    _axis_b: usize,
) -> Vec<Option<LocalValId>> {
    todo!()
}
