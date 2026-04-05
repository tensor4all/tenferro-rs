use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use tenferro_tensor::DotGeneralConfig;

use crate::std_tensor_op::StdTensorOp;

pub fn linearize_svd(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None, None];
    };

    let uh = adjoint_2d(builder, ValRef::External(primal_out[0].clone()));
    let v = adjoint_2d(builder, ValRef::External(primal_out[2].clone()));
    let tmp = matmul(
        builder,
        ValRef::Local(uh),
        ValRef::Local(da),
        vec![false, true],
    );
    let ds_mat = matmul(
        builder,
        ValRef::Local(tmp),
        ValRef::Local(v),
        vec![true, false],
    );
    let ds = builder.add_op(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValRef::Local(ds_mat)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );

    vec![None, Some(ds[0]), None]
}

pub fn linearize_eigh(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
) -> Vec<Option<LocalValId>> {
    let Some(da) = tangent_in[0] else {
        return vec![None, None];
    };

    let vh = adjoint_2d(builder, ValRef::External(primal_out[1].clone()));
    let tmp = matmul(
        builder,
        ValRef::Local(vh),
        ValRef::Local(da),
        vec![false, true],
    );
    let projected = matmul(
        builder,
        ValRef::Local(tmp),
        ValRef::External(primal_out[1].clone()),
        vec![true, false],
    );
    let dw = builder.add_op(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValRef::Local(projected)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    );

    vec![Some(dw[0]), None]
}

fn adjoint_2d(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: ValRef<StdTensorOp>,
) -> LocalValId {
    let conjugated = builder.add_op(StdTensorOp::Conj, vec![input], OpMode::Primal);
    let transposed = builder.add_op(
        StdTensorOp::Transpose { perm: vec![1, 0] },
        vec![ValRef::Local(conjugated[0])],
        OpMode::Primal,
    );
    transposed[0]
}

fn matmul(
    builder: &mut FragmentBuilder<StdTensorOp>,
    lhs: ValRef<StdTensorOp>,
    rhs: ValRef<StdTensorOp>,
    active_mask: Vec<bool>,
) -> LocalValId {
    let out = builder.add_op(
        StdTensorOp::DotGeneral(matrix_multiply_config()),
        vec![lhs, rhs],
        OpMode::Linear { active_mask },
    );
    out[0]
}

fn matrix_multiply_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    }
}
