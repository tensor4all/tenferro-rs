use computegraph::fragment::FragmentBuilder;
use computegraph::types::{LocalValId, ValRef};

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_transpose(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
    _perm: &[usize],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn linearize_reshape(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
    _op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn linearize_broadcast_in_dim(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _tangent_in: &[Option<LocalValId>],
    _shape: &[usize],
    _dims: &[usize],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_transpose(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _perm: &[usize],
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_reshape(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    todo!()
}

pub fn transpose_broadcast_in_dim(
    _builder: &mut FragmentBuilder<StdTensorOp>,
    _cotangent_out: &[Option<LocalValId>],
    _shape: &[usize],
    _dims: &[usize],
) -> Vec<Option<LocalValId>> {
    let _ = (_builder, _cotangent_out, _shape, _dims);
    let _unused: Option<ValRef<StdTensorOp>> = None;
    todo!()
}
