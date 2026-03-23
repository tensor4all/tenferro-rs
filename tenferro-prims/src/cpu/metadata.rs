use strided_view::{StridedView, StridedViewMut};
use tenferro_device::{Error, Result};

use crate::cpu::common::plan_reduction;
use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::{
    validate_shape_eq, CpuBackend, CpuContext, MetadataBinaryOp, MetadataDType, MetadataGenerateOp,
    MetadataPrimsDescriptor, MetadataReductionOp, MetadataTensorMut, MetadataTensorRef,
    MetadataTernaryOp, TensorMetadataPrims,
};

fn tensor_dims_ref<'a>(tensor: &'a MetadataTensorRef<'a>) -> &'a [usize] {
    match tensor {
        MetadataTensorRef::I32(tensor) => tensor.dims(),
        MetadataTensorRef::Bool(tensor) => tensor.dims(),
    }
}

fn tensor_dims_mut<'a>(tensor: &'a MetadataTensorMut<'a>) -> &'a [usize] {
    match tensor {
        MetadataTensorMut::I32(tensor) => tensor.dims(),
        MetadataTensorMut::Bool(tensor) => tensor.dims(),
    }
}

fn validate_supported_generate(output_dtype: MetadataDType) -> Result<()> {
    if output_dtype == MetadataDType::I32 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "metadata iota currently supports I32 output only".into(),
        ))
    }
}

fn validate_supported_binary(
    op: MetadataBinaryOp,
    lhs_dtype: MetadataDType,
    rhs_dtype: MetadataDType,
    output_dtype: MetadataDType,
) -> Result<()> {
    if !matches!(op, MetadataBinaryOp::Equal | MetadataBinaryOp::NotEqual) {
        return Err(Error::InvalidArgument(format!(
            "metadata binary operation {op:?} is not supported on CpuBackend"
        )));
    }
    match (lhs_dtype, rhs_dtype, output_dtype) {
        (MetadataDType::I32, MetadataDType::I32, MetadataDType::Bool)
        | (MetadataDType::Bool, MetadataDType::Bool, MetadataDType::Bool) => Ok(()),
        _ => Err(Error::InvalidArgument(format!(
            "unsupported metadata binary dtype combination: lhs={lhs_dtype:?} rhs={rhs_dtype:?} dst={output_dtype:?}"
        ))),
    }
}

fn validate_supported_ternary(
    op: MetadataTernaryOp,
    cond_dtype: MetadataDType,
    lhs_dtype: MetadataDType,
    rhs_dtype: MetadataDType,
    output_dtype: MetadataDType,
) -> Result<()> {
    if !matches!(op, MetadataTernaryOp::Where) {
        return Err(Error::InvalidArgument(format!(
            "metadata ternary operation {op:?} is not supported on CpuBackend"
        )));
    }
    match (cond_dtype, lhs_dtype, rhs_dtype, output_dtype) {
        (MetadataDType::Bool, MetadataDType::I32, MetadataDType::I32, MetadataDType::I32)
        | (MetadataDType::Bool, MetadataDType::Bool, MetadataDType::Bool, MetadataDType::Bool) =>
        {
            Ok(())
        }
        _ => Err(Error::InvalidArgument(format!(
            "unsupported metadata ternary dtype combination: cond={cond_dtype:?} lhs={lhs_dtype:?} rhs={rhs_dtype:?} dst={output_dtype:?}"
        ))),
    }
}

fn validate_supported_reduction(
    op: MetadataReductionOp,
    input_dtype: MetadataDType,
    output_dtype: MetadataDType,
) -> Result<()> {
    match (op, input_dtype, output_dtype) {
        (MetadataReductionOp::Sum, MetadataDType::Bool, MetadataDType::I32)
        | (MetadataReductionOp::Sum, MetadataDType::I32, MetadataDType::I32)
        | (MetadataReductionOp::All, MetadataDType::Bool, MetadataDType::Bool)
        | (MetadataReductionOp::Any, MetadataDType::Bool, MetadataDType::Bool) => Ok(()),
        _ => Err(Error::InvalidArgument(format!(
            "unsupported metadata reduction dtype combination: op={op:?} input={input_dtype:?} dst={output_dtype:?}"
        ))),
    }
}

fn validate_metadata_handle_count(
    inputs: &[MetadataTensorRef<'_>],
    expected: usize,
    op_name: &str,
) -> Result<()> {
    if inputs.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects {expected} input(s) (got {})",
            inputs.len()
        )));
    }
    Ok(())
}

fn execute_metadata_generate_i32(output: &mut StridedViewMut<i32>) -> Result<()> {
    let dims = output.dims().to_vec();
    let mut value = 0i32;
    crate::for_each_index(&dims, |idx| {
        output.set(idx, value);
        value = value
            .checked_add(1)
            .expect("metadata iota overflow should be validated at plan time");
    });
    Ok(())
}

fn execute_metadata_binary_map<Lhs, Rhs, F>(
    lhs: &StridedView<Lhs>,
    rhs: &StridedView<Rhs>,
    output: &mut StridedViewMut<u8>,
    f: F,
) -> Result<()>
where
    Lhs: Copy,
    Rhs: Copy,
    F: Fn(Lhs, Rhs) -> u8 + Copy,
{
    let dims = output.dims().to_vec();
    crate::for_each_index(&dims, |idx| {
        output.set(idx, f(lhs.get(idx), rhs.get(idx)));
    });
    Ok(())
}

fn execute_metadata_ternary_map<Lhs, F>(
    cond: &StridedView<u8>,
    on_true: &StridedView<Lhs>,
    on_false: &StridedView<Lhs>,
    output: &mut StridedViewMut<Lhs>,
    f: F,
) -> Result<()>
where
    Lhs: Copy,
    F: Fn(u8, Lhs, Lhs) -> Lhs + Copy,
{
    let dims = output.dims().to_vec();
    crate::for_each_index(&dims, |idx| {
        output.set(idx, f(cond.get(idx), on_true.get(idx), on_false.get(idx)));
    });
    Ok(())
}

fn execute_metadata_reduce_sum_i32(
    input: &StridedView<i32>,
    output: &mut StridedViewMut<i32>,
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let (kept_axis_positions, reduced_axis_positions) =
        build_metadata_reduction_axis_positions(in_dims.len(), kept_axes, reduced_axes);
    let reduced_total: usize = reduced_dims.iter().product();
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    crate::for_each_index(&out_dims, |out_idx| {
        let mut sum = 0i32;
        for red_flat in 0..reduced_total {
            crate::cpu::common::unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_metadata_reduction_input_index(
                out_idx,
                &red_idx,
                &kept_axis_positions,
                &reduced_axis_positions,
                &mut in_idx,
            );
            sum += input.get(&in_idx);
        }
        output.set(out_idx, sum);
    });

    Ok(())
}

fn execute_metadata_reduce_sum_bool(
    input: &StridedView<u8>,
    output: &mut StridedViewMut<i32>,
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let (kept_axis_positions, reduced_axis_positions) =
        build_metadata_reduction_axis_positions(in_dims.len(), kept_axes, reduced_axes);
    let reduced_total: usize = reduced_dims.iter().product();
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    crate::for_each_index(&out_dims, |out_idx| {
        let mut sum = 0i32;
        for red_flat in 0..reduced_total {
            crate::cpu::common::unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_metadata_reduction_input_index(
                out_idx,
                &red_idx,
                &kept_axis_positions,
                &reduced_axis_positions,
                &mut in_idx,
            );
            sum += if input.get(&in_idx) != 0 { 1 } else { 0 };
        }
        output.set(out_idx, sum);
    });

    Ok(())
}

fn execute_metadata_reduce_all_bool(
    input: &StridedView<u8>,
    output: &mut StridedViewMut<u8>,
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let (kept_axis_positions, reduced_axis_positions) =
        build_metadata_reduction_axis_positions(in_dims.len(), kept_axes, reduced_axes);
    let reduced_total: usize = reduced_dims.iter().product();
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    crate::for_each_index(&out_dims, |out_idx| {
        let mut all_true = true;
        for red_flat in 0..reduced_total {
            crate::cpu::common::unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_metadata_reduction_input_index(
                out_idx,
                &red_idx,
                &kept_axis_positions,
                &reduced_axis_positions,
                &mut in_idx,
            );
            if input.get(&in_idx) == 0 {
                all_true = false;
                break;
            }
        }
        output.set(out_idx, if all_true { 1 } else { 0 });
    });

    Ok(())
}

fn execute_metadata_reduce_any_bool(
    input: &StridedView<u8>,
    output: &mut StridedViewMut<u8>,
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let (kept_axis_positions, reduced_axis_positions) =
        build_metadata_reduction_axis_positions(in_dims.len(), kept_axes, reduced_axes);
    let reduced_total: usize = reduced_dims.iter().product();
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    crate::for_each_index(&out_dims, |out_idx| {
        let mut any_true = false;
        for red_flat in 0..reduced_total {
            crate::cpu::common::unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_metadata_reduction_input_index(
                out_idx,
                &red_idx,
                &kept_axis_positions,
                &reduced_axis_positions,
                &mut in_idx,
            );
            if input.get(&in_idx) != 0 {
                any_true = true;
                break;
            }
        }
        output.set(out_idx, if any_true { 1 } else { 0 });
    });

    Ok(())
}

fn build_metadata_reduction_axis_positions(
    rank: usize,
    kept_axes: &[usize],
    reduced_axes: &[usize],
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut kept_axis_positions = vec![None; rank];
    for (output_axis, &input_axis) in kept_axes.iter().enumerate() {
        kept_axis_positions[input_axis] = Some(output_axis);
    }

    let mut reduced_axis_positions = vec![None; rank];
    for (reduced_axis, &input_axis) in reduced_axes.iter().enumerate() {
        reduced_axis_positions[input_axis] = Some(reduced_axis);
    }

    (kept_axis_positions, reduced_axis_positions)
}

fn build_metadata_reduction_input_index(
    out_idx: &[usize],
    red_idx: &[usize],
    kept_axis_positions: &[Option<usize>],
    reduced_axis_positions: &[Option<usize>],
    in_idx: &mut [usize],
) {
    for (axis, slot) in in_idx.iter_mut().enumerate() {
        if let Some(output_axis) = kept_axis_positions[axis] {
            *slot = out_idx[output_axis];
        } else if let Some(reduced_axis) = reduced_axis_positions[axis] {
            *slot = red_idx[reduced_axis];
        } else {
            unreachable!("metadata reduction axis {axis} missing from kept/reduced axis lists");
        }
    }
}

fn plan_metadata_reduction(
    input_dims: &[usize],
    output_dims: &[usize],
    desc: &MetadataPrimsDescriptor,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let MetadataPrimsDescriptor::Reduction {
        modes_a, modes_c, ..
    } = desc
    else {
        return Err(Error::InvalidArgument(
            "expected metadata reduction descriptor".into(),
        ));
    };
    let reduction = plan_reduction(
        modes_a,
        modes_c,
        &[input_dims, output_dims],
        "CpuMetadataReduction",
    )?;
    let kept_axes = modes_c
        .iter()
        .map(|mode| {
            modes_a
                .iter()
                .position(|&candidate| candidate == *mode)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                    "CpuMetadataReduction: output mode {mode} not found in input modes {modes_a:?}"
                ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((kept_axes, reduction.reduced_axes))
}

impl TensorMetadataPrims for CpuBackend {
    type Plan = MetadataPrimsDescriptor;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &MetadataPrimsDescriptor,
        inputs: &[MetadataTensorRef<'_>],
        output: MetadataTensorMut<'_>,
    ) -> Result<Self::Plan> {
        match desc {
            MetadataPrimsDescriptor::Generate { op, output_dtype } => {
                validate_supported_generate(*output_dtype)?;
                if !inputs.is_empty() {
                    return Err(Error::InvalidArgument(
                        "metadata generate expects no inputs".into(),
                    ));
                }
                match output {
                    MetadataTensorMut::I32(_) => {}
                    MetadataTensorMut::Bool(_) => {
                        return Err(Error::InvalidArgument(
                            "metadata iota currently supports I32 output only".into(),
                        ));
                    }
                }
                if !matches!(*op, MetadataGenerateOp::IotaStartZero) {
                    return Err(Error::InvalidArgument(format!(
                        "metadata generate operation {op:?} is not supported on CpuBackend"
                    )));
                }
                Ok(desc.clone())
            }
            MetadataPrimsDescriptor::Binary {
                op,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                validate_supported_binary(*op, *lhs_dtype, *rhs_dtype, *output_dtype)?;
                validate_metadata_handle_count(inputs, 2, "CpuMetadataBinary")?;
                if !matches!(&output, MetadataTensorMut::Bool(_)) {
                    return Err(Error::InvalidArgument(
                        "metadata binary currently writes bool metadata outputs".into(),
                    ));
                }
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_ref(&inputs[1]),
                    "CpuMetadataBinary input",
                )?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_mut(&output),
                    "CpuMetadataBinary output",
                )?;
                Ok(desc.clone())
            }
            MetadataPrimsDescriptor::Ternary {
                op,
                cond_dtype,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                validate_supported_ternary(
                    *op,
                    *cond_dtype,
                    *lhs_dtype,
                    *rhs_dtype,
                    *output_dtype,
                )?;
                validate_metadata_handle_count(inputs, 3, "CpuMetadataTernary")?;
                if !matches!(*output_dtype, MetadataDType::I32 | MetadataDType::Bool) {
                    return Err(Error::InvalidArgument(
                        "unsupported metadata ternary output dtype".into(),
                    ));
                }
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_ref(&inputs[1]),
                    "CpuMetadataTernary input",
                )?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_ref(&inputs[2]),
                    "CpuMetadataTernary input",
                )?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_mut(&output),
                    "CpuMetadataTernary output",
                )?;
                Ok(desc.clone())
            }
            MetadataPrimsDescriptor::Reduction {
                input_dtype,
                output_dtype,
                op,
                ..
            } => {
                validate_supported_reduction(*op, *input_dtype, *output_dtype)?;
                validate_metadata_handle_count(inputs, 1, "CpuMetadataReduction")?;
                let _ = plan_metadata_reduction(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_mut(&output),
                    desc,
                )?;
                Ok(desc.clone())
            }
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        inputs: &[MetadataTensorRef<'_>],
        output: MetadataTensorMut<'_>,
    ) -> Result<()> {
        match plan {
            MetadataPrimsDescriptor::Generate {
                op: MetadataGenerateOp::IotaStartZero,
                output_dtype: MetadataDType::I32,
            } => match output {
                MetadataTensorMut::I32(output) => {
                    let mut output = tensor_to_view_mut(output)?;
                    execute_metadata_generate_i32(&mut output)?;
                    Ok(())
                }
                MetadataTensorMut::Bool(_) => Err(Error::InvalidArgument(
                    "metadata iota currently supports I32 output only".into(),
                )),
            },
            MetadataPrimsDescriptor::Binary {
                op,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                validate_metadata_handle_count(inputs, 2, "CpuMetadataBinary")?;
                match (
                    *lhs_dtype,
                    *rhs_dtype,
                    *output_dtype,
                    inputs[0],
                    inputs[1],
                    output,
                ) {
                    (
                        MetadataDType::I32,
                        MetadataDType::I32,
                        MetadataDType::Bool,
                        MetadataTensorRef::I32(lhs),
                        MetadataTensorRef::I32(rhs),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        let lhs = tensor_to_view(lhs)?;
                        let rhs = tensor_to_view(rhs)?;
                        let mut dst = tensor_to_view_mut(dst)?;
                        execute_metadata_binary_map(&lhs, &rhs, &mut dst, |x, y| {
                            let equal = x == y;
                            let mapped = match *op {
                                MetadataBinaryOp::Equal => equal,
                                MetadataBinaryOp::NotEqual => !equal,
                                _ => unreachable!("unsupported metadata binary op"),
                            };
                            if mapped {
                                1
                            } else {
                                0
                            }
                        })
                    }
                    (
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataTensorRef::Bool(lhs),
                        MetadataTensorRef::Bool(rhs),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        let lhs = tensor_to_view(lhs)?;
                        let rhs = tensor_to_view(rhs)?;
                        let mut dst = tensor_to_view_mut(dst)?;
                        execute_metadata_binary_map(&lhs, &rhs, &mut dst, |x, y| {
                            let equal = (x != 0) == (y != 0);
                            let mapped = match *op {
                                MetadataBinaryOp::Equal => equal,
                                MetadataBinaryOp::NotEqual => !equal,
                                _ => unreachable!("unsupported metadata binary op"),
                            };
                            if mapped {
                                1
                            } else {
                                0
                            }
                        })
                    }
                    _ => Err(Error::InvalidArgument(
                        "unsupported metadata binary execution dtype combination".into(),
                    )),
                }
            }
            MetadataPrimsDescriptor::Ternary {
                op: MetadataTernaryOp::Where,
                cond_dtype,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                validate_metadata_handle_count(inputs, 3, "CpuMetadataTernary")?;
                match (
                    *cond_dtype,
                    *lhs_dtype,
                    *rhs_dtype,
                    *output_dtype,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    output,
                ) {
                    (
                        MetadataDType::Bool,
                        MetadataDType::I32,
                        MetadataDType::I32,
                        MetadataDType::I32,
                        MetadataTensorRef::Bool(cond),
                        MetadataTensorRef::I32(on_true),
                        MetadataTensorRef::I32(on_false),
                        MetadataTensorMut::I32(dst),
                    ) => {
                        let cond = tensor_to_view(cond)?;
                        let on_true = tensor_to_view(on_true)?;
                        let on_false = tensor_to_view(on_false)?;
                        let mut dst = tensor_to_view_mut(dst)?;
                        execute_metadata_ternary_map(
                            &cond,
                            &on_true,
                            &on_false,
                            &mut dst,
                            |c, t, f| {
                                if c != 0 {
                                    t
                                } else {
                                    f
                                }
                            },
                        )
                    }
                    (
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataTensorRef::Bool(cond),
                        MetadataTensorRef::Bool(on_true),
                        MetadataTensorRef::Bool(on_false),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        let cond = tensor_to_view(cond)?;
                        let on_true = tensor_to_view(on_true)?;
                        let on_false = tensor_to_view(on_false)?;
                        let mut dst = tensor_to_view_mut(dst)?;
                        execute_metadata_ternary_map(
                            &cond,
                            &on_true,
                            &on_false,
                            &mut dst,
                            |c, t, f| {
                                if c != 0 {
                                    t
                                } else {
                                    f
                                }
                            },
                        )
                    }
                    _ => Err(Error::InvalidArgument(
                        "unsupported metadata ternary execution dtype combination".into(),
                    )),
                }
            }
            MetadataPrimsDescriptor::Reduction {
                op,
                input_dtype,
                output_dtype,
                ..
            } => {
                validate_metadata_handle_count(inputs, 1, "CpuMetadataReduction")?;
                let (kept_axes, reduced_axes) = plan_metadata_reduction(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_mut(&output),
                    plan,
                )?;
                match (*op, *input_dtype, *output_dtype, inputs[0], output) {
                    (
                        MetadataReductionOp::Sum,
                        MetadataDType::I32,
                        MetadataDType::I32,
                        MetadataTensorRef::I32(input),
                        MetadataTensorMut::I32(output),
                    ) => {
                        let input = tensor_to_view(input)?;
                        let mut output = tensor_to_view_mut(output)?;
                        execute_metadata_reduce_sum_i32(
                            &input,
                            &mut output,
                            &kept_axes,
                            &reduced_axes,
                        )
                    }
                    (
                        MetadataReductionOp::Sum,
                        MetadataDType::Bool,
                        MetadataDType::I32,
                        MetadataTensorRef::Bool(input),
                        MetadataTensorMut::I32(output),
                    ) => {
                        let input = tensor_to_view(input)?;
                        let mut output = tensor_to_view_mut(output)?;
                        execute_metadata_reduce_sum_bool(
                            &input,
                            &mut output,
                            &kept_axes,
                            &reduced_axes,
                        )
                    }
                    (
                        MetadataReductionOp::All,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataTensorRef::Bool(input),
                        MetadataTensorMut::Bool(output),
                    ) => {
                        let input = tensor_to_view(input)?;
                        let mut output = tensor_to_view_mut(output)?;
                        execute_metadata_reduce_all_bool(
                            &input,
                            &mut output,
                            &kept_axes,
                            &reduced_axes,
                        )
                    }
                    (
                        MetadataReductionOp::Any,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataTensorRef::Bool(input),
                        MetadataTensorMut::Bool(output),
                    ) => {
                        let input = tensor_to_view(input)?;
                        let mut output = tensor_to_view_mut(output)?;
                        execute_metadata_reduce_any_bool(
                            &input,
                            &mut output,
                            &kept_axes,
                            &reduced_axes,
                        )
                    }
                    _ => Err(Error::InvalidArgument(
                        "unsupported metadata reduction execution dtype combination".into(),
                    )),
                }
            }
            _ => Err(Error::InvalidArgument(
                "unsupported metadata metadata descriptor on CpuBackend".into(),
            )),
        }
    }

    fn has_metadata_support(desc: MetadataPrimsDescriptor) -> bool {
        match desc {
            MetadataPrimsDescriptor::Generate {
                op: MetadataGenerateOp::IotaStartZero,
                output_dtype: MetadataDType::I32,
            } => true,
            MetadataPrimsDescriptor::Binary {
                op,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                matches!(op, MetadataBinaryOp::Equal | MetadataBinaryOp::NotEqual)
                    && matches!(
                        (lhs_dtype, rhs_dtype, output_dtype),
                        (MetadataDType::I32, MetadataDType::I32, MetadataDType::Bool)
                            | (
                                MetadataDType::Bool,
                                MetadataDType::Bool,
                                MetadataDType::Bool
                            )
                    )
            }
            MetadataPrimsDescriptor::Ternary {
                op: MetadataTernaryOp::Where,
                cond_dtype: MetadataDType::Bool,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => matches!(
                (lhs_dtype, rhs_dtype, output_dtype),
                (MetadataDType::I32, MetadataDType::I32, MetadataDType::I32)
                    | (
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataDType::Bool
                    )
            ),
            MetadataPrimsDescriptor::Reduction {
                op,
                input_dtype,
                output_dtype,
                ..
            } => matches!(
                (op, input_dtype, output_dtype),
                (
                    MetadataReductionOp::Sum,
                    MetadataDType::Bool,
                    MetadataDType::I32
                ) | (
                    MetadataReductionOp::Sum,
                    MetadataDType::I32,
                    MetadataDType::I32
                ) | (
                    MetadataReductionOp::All,
                    MetadataDType::Bool,
                    MetadataDType::Bool
                ) | (
                    MetadataReductionOp::Any,
                    MetadataDType::Bool,
                    MetadataDType::Bool
                )
            ),
            _ => false,
        }
    }
}
