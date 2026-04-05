use computegraph::fragment::FragmentBuilder;
use computegraph::types::{LocalValId, OpMode, ValRef};

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_transpose(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    perm: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Transpose {
                    perm: perm.to_vec(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape {
        from_shape,
        to_shape,
    } = op
    else {
        unreachable!("linearize_reshape expects Reshape");
    };

    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    from_shape: from_shape.clone(),
                    to_shape: to_shape.clone(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn linearize_broadcast_in_dim(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    shape: &[usize],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape: shape.to_vec(),
                    dims: dims.to_vec(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_transpose(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    perm: &[usize],
) -> Vec<Option<LocalValId>> {
    let mut inv = vec![0; perm.len()];
    for (index, &value) in perm.iter().enumerate() {
        inv[value] = index;
    }

    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::Transpose { perm: inv },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape {
        from_shape,
        to_shape,
    } = op
    else {
        unreachable!("transpose_reshape expects Reshape");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    from_shape: to_shape.clone(),
                    to_shape: from_shape.clone(),
                },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}

pub fn transpose_broadcast_in_dim(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    shape: &[usize],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    let broadcast_axes: Vec<usize> = (0..shape.len()).filter(|dim| !dims.contains(dim)).collect();

    match cotangent_out[0] {
        Some(ct) if broadcast_axes.is_empty() => vec![Some(ct)],
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::ReduceSum {
                    axes: broadcast_axes,
                    input_shape: shape.to_vec(),
                },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
