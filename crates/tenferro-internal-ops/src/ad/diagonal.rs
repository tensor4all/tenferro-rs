use crate::ad::context::ShapeGuardContext;
use crate::ad::support::linear_transpose_input_active;
use crate::ad::transpose_input::TransposeInputRef;
use crate::ad::PrimitiveRuleBuilder;
use crate::ad::{ADRuleError, ADRuleKind, ADRuleResult};
use computegraph::types::{LocalValueId, OperationRole, ValueRef};
use tenferro_tensor::PadConfig;

use crate::std_tensor_op::StdTensorOp;
use crate::sym_dim::SymDim;

pub fn linearize_extract_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    axis_a: usize,
    axis_b: usize,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::ExtractDiag { axis_a, axis_b },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            Ok(vec![Some(out[0])])
        }
        None => Ok(vec![None]),
    }
}

pub fn linearize_embed_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    tangent_in: &[Option<LocalValueId>],
    axis_a: usize,
    axis_b: usize,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_operation(
                StdTensorOp::EmbedDiag { axis_a, axis_b },
                vec![ValueRef::Local(dx)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            Ok(vec![Some(out[0])])
        }
        None => Ok(vec![None]),
    }
}

pub fn transpose_extract_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    axis_a: usize,
    axis_b: usize,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !linear_transpose_input_active(mode, 0) {
        return Ok(vec![None]);
    }

    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match cotangent_out[0] {
        Some(ct) => {
            let source_axis = if axis_a < axis_b { axis_a } else { axis_a - 1 };
            let out = builder.add_operation(
                StdTensorOp::EmbedDiag {
                    axis_a: source_axis,
                    axis_b,
                },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            let padded =
                pad_embedded_diag_to_input(builder, out[0], &inputs[0], axis_a, axis_b, ctx)?;
            Ok(vec![Some(padded)])
        }
        None => Ok(vec![None]),
    }
}

pub fn transpose_embed_diag(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    mode: &OperationRole,
    axis_a: usize,
    axis_b: usize,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    if !linear_transpose_input_active(mode, 0) {
        return Ok(vec![None]);
    }

    // TODO: ExtractDiag/EmbedDiag could be replaced by Gather/Scatter
    match cotangent_out[0] {
        Some(ct) => {
            let source_axis = if axis_b <= axis_a { axis_a + 1 } else { axis_a };
            let out = builder.add_operation(
                StdTensorOp::ExtractDiag {
                    axis_a: source_axis,
                    axis_b,
                },
                vec![ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![true],
                },
            );
            if axis_b < axis_a {
                let rank = ctx.rank_of(&inputs[0].metadata_value())?;
                let mut perm: Vec<usize> = (0..rank).collect();
                let diag_axis = perm.remove(axis_b);
                perm.insert(axis_a, diag_axis);
                let transposed = builder.add_operation(
                    StdTensorOp::Transpose { perm },
                    vec![ValueRef::Local(out[0])],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                return Ok(vec![Some(transposed[0])]);
            }
            Ok(vec![Some(out[0])])
        }
        None => Ok(vec![None]),
    }
}

fn pad_embedded_diag_to_input(
    builder: &mut dyn PrimitiveRuleBuilder,
    embedded: LocalValueId,
    input: &TransposeInputRef<'_>,
    axis_a: usize,
    axis_b: usize,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<LocalValueId> {
    let input_ref = input.metadata_value();
    if let Some(input_shape) = ctx.shape_if_available(&input_ref) {
        if let Some(padded) =
            static_pad_embedded_diag_to_input(builder, embedded, &input_shape, axis_a, axis_b)?
        {
            return Ok(padded);
        }
    }

    let shape_source = input.shape_source_value("ExtractDiag", 0)?;
    let padded_axis_a = builder.add_operation(
        StdTensorOp::PadToMatch { axis: axis_a },
        vec![ValueRef::Local(embedded), shape_source.clone()],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    );
    let padded_axis_b = builder.add_operation(
        StdTensorOp::PadToMatch { axis: axis_b },
        vec![ValueRef::Local(padded_axis_a[0]), shape_source],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    );
    Ok(padded_axis_b[0])
}

fn static_pad_embedded_diag_to_input(
    builder: &mut dyn PrimitiveRuleBuilder,
    embedded: LocalValueId,
    input_shape: &[SymDim],
    axis_a: usize,
    axis_b: usize,
) -> ADRuleResult<Option<LocalValueId>> {
    if axis_a >= input_shape.len() || axis_b >= input_shape.len() || axis_a == axis_b {
        return Err(ADRuleError::invalid_input(
            "ExtractDiag",
            ADRuleKind::Transpose,
            format!(
                "diagonal axes ({axis_a}, {axis_b}) out of bounds or not distinct for input rank {}",
                input_shape.len()
            ),
        ));
    }

    let Some(size_a) = input_shape[axis_a].constant_value() else {
        return Ok(None);
    };
    let Some(size_b) = input_shape[axis_b].constant_value() else {
        return Ok(None);
    };
    let diag_size = size_a.min(size_b);
    let mut high = vec![0_i64; input_shape.len()];
    high[axis_a] = i64::try_from(size_a - diag_size).map_err(|_| {
        ADRuleError::invalid_input(
            "ExtractDiag",
            ADRuleKind::Transpose,
            "axis_a static diagonal padding does not fit in i64",
        )
    })?;
    high[axis_b] = i64::try_from(size_b - diag_size).map_err(|_| {
        ADRuleError::invalid_input(
            "ExtractDiag",
            ADRuleKind::Transpose,
            "axis_b static diagonal padding does not fit in i64",
        )
    })?;

    if high.iter().all(|&pad| pad == 0) {
        return Ok(Some(embedded));
    }
    let rank = input_shape.len();
    let out = builder.add_operation(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![0_i64; rank],
            edge_padding_high: high,
            interior_padding: vec![0_i64; rank],
        }),
        vec![ValueRef::Local(embedded)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    Ok(Some(out[0]))
}
