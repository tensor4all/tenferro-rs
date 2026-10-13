// Portions of this file are adapted from cubek-reduce:
// https://github.com/tracel-ai/cubek/tree/9cf90b797107d46829e1c9d9355ce801c3dd4a7d/crates/cubek-reduce
//
// Original copyright:
// Copyright (c) 2022 Nathaniel Simard & CubeCL Framework Contributors
//
// Original source paths:
// - crates/cubek-reduce/src/launch/base.rs
// - crates/cubek-reduce/src/launch/strategy.rs
// - crates/cubek-reduce/src/routines/unit.rs
// - crates/cubek-reduce/src/routines/blueprint.rs
//
// Original license: MIT OR Apache-2.0.
// See tenferro-gpu/THIRD_PARTY_NOTICES.md for license notice text.
// Tenferro changes: narrowed to tenferro reduction ops, current CubeCL fork,
// single-axis keepdims output, and explicit tenferro column-major bindings.

use cubecl::{features::Plane, prelude::*};

use super::{
    definition::validate_keepdims_output_shape,
    kernels,
    routines::{plan, unit_launch_settings, ReduceProblem},
};
use crate::kernels::{CubeclKernelError, Result};

#[cfg(test)]
mod tests;

/// Launch strategy for a single-axis reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReduceStrategy {
    /// Let the launcher choose the kernel strategy.
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedReduceStrategy {
    Unit,
    Plane,
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedReduceLaunch {
    pub(crate) kind: ResolvedReduceStrategy,
    pub(crate) cube_count: CubeCount,
    pub(crate) cube_dim: CubeDim,
    pub(crate) axis: usize,
    pub(crate) output_len: usize,
}

pub(crate) fn validate_launch(
    input: &TensorBinding,
    output: &TensorBinding,
    axis: usize,
) -> Result<ReduceProblem> {
    validate_reduce_problem(&input.shape, &output.shape, axis)
}

fn validate_reduce_problem(
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
) -> Result<ReduceProblem> {
    validate_keepdims_output_shape(input_shape, output_shape, axis)?;

    let reduce_len = input_shape[axis];
    if reduce_len == 0 {
        return Err(CubeclKernelError::InvalidStrategy {
            reason: format!("cannot reduce zero-length axis {axis}"),
        });
    }

    let input_len = input_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| CubeclKernelError::InvalidStrategy {
                reason: format!(
                    "reduction input element count overflows usize for shape {input_shape:?}"
                ),
            })
    })?;
    let reduce_count = input_len / reduce_len;

    Ok(ReduceProblem {
        reduce_len,
        reduce_count,
        axis,
    })
}

pub(crate) fn launch_with_unit_settings(
    client: &Client,
    problem: ReduceProblem,
) -> Result<ResolvedReduceLaunch> {
    let settings = unit_launch_settings(client, problem)?;
    let _has_idle_units = settings.blueprint.idle_units;
    Ok(ResolvedReduceLaunch {
        kind: ResolvedReduceStrategy::Unit,
        cube_count: settings.cube_count,
        cube_dim: settings.cube_dim,
        axis: problem.axis,
        output_len: problem.reduce_count,
    })
}

pub(crate) fn launch_with_plane_settings(
    client: &Client,
    problem: ReduceProblem,
) -> Result<ResolvedReduceLaunch> {
    let plane_width = client.properties().hardware.plane_size_max.max(1);
    validate_plane_strategy(
        problem,
        plane_width,
        client.features().plane.contains(Plane::Ops),
    )?;

    let cubes = u32::try_from(problem.reduce_count.max(1)).map_err(|_| {
        CubeclKernelError::InvalidStrategy {
            reason: format!(
                "reduction output element count {} exceeds static CubeCL launch limit",
                problem.reduce_count
            ),
        }
    })?;
    Ok(ResolvedReduceLaunch {
        kind: ResolvedReduceStrategy::Plane,
        cube_count: CubeCount::Static(cubes, 1, 1),
        cube_dim: CubeDim::new_1d(plane_width),
        axis: problem.axis,
        output_len: problem.reduce_count,
    })
}

fn validate_plane_strategy(
    problem: ReduceProblem,
    plane_width: u32,
    has_plane_ops: bool,
) -> Result<()> {
    if !has_plane_ops {
        return Err(CubeclKernelError::InvalidStrategy {
            reason: "plane reduction requires backend plane operations".to_owned(),
        });
    }
    if problem.reduce_len < plane_width as usize {
        return Err(CubeclKernelError::InvalidStrategy {
            reason: format!(
                "plane reduction requires reduce axis length {} to be at least plane width {}",
                problem.reduce_len, plane_width
            ),
        });
    }
    Ok(())
}

pub(crate) fn auto_reduce_strategy(
    client: &Client,
    problem: ReduceProblem,
) -> Result<ResolvedReduceStrategy> {
    let unit_axis_limit = client.properties().hardware.plane_size_max.max(1) as usize;
    auto_reduce_strategy_for_capabilities(
        problem.reduce_len,
        unit_axis_limit,
        client.features().plane.contains(Plane::Ops),
    )
}

fn auto_reduce_strategy_for_capabilities(
    reduce_len: usize,
    unit_axis_limit: usize,
    has_plane_ops: bool,
) -> Result<ResolvedReduceStrategy> {
    if reduce_len <= unit_axis_limit {
        return Ok(ResolvedReduceStrategy::Unit);
    }
    if has_plane_ops {
        Ok(ResolvedReduceStrategy::Plane)
    } else {
        Err(CubeclKernelError::InvalidStrategy {
            reason: format!(
                "Auto reduction cannot reduce axis length {} without plane operations",
                reduce_len
            ),
        })
    }
}

/// Launch a floating-point sum reduction.
pub(crate) fn launch_sum_float<F: Float + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    partials: Option<TensorBinding>,
) -> Result<()> {
    let plan = plan(client, &input, &output, axis, strategy)?;

    if let Some(split) = plan.split {
        // Pass 1: one cube per (output element, chunk) writes a partial.
        let partials = partials.ok_or_else(|| CubeclKernelError::InvalidStrategy {
            reason: "split float-sum reduction requires a partials buffer".to_owned(),
        })?;
        let pass1_cubes =
            u32::try_from(plan.output_len.checked_mul(split.splits).ok_or_else(|| {
                CubeclKernelError::InvalidStrategy {
                    reason: format!(
                        "split reduction grid {} exceeds static CubeCL launch limit",
                        plan.output_len
                    ),
                }
            })?)
            .map_err(|_| CubeclKernelError::InvalidStrategy {
                reason: format!(
                    "split reduction grid {} exceeds static CubeCL launch limit",
                    plan.output_len
                ),
            })?;

        // Pass 2: reduce the `splits`-long partials axis with the kernel the
        // existing unit/plane selection gives for `reduce_len = splits`.
        let combine_problem = ReduceProblem {
            reduce_len: split.splits,
            reduce_count: plan.output_len,
            axis: plan.axis,
        };
        let combine = match auto_reduce_strategy(client, combine_problem)? {
            ResolvedReduceStrategy::Unit => launch_with_unit_settings(client, combine_problem)?,
            ResolvedReduceStrategy::Plane => launch_with_plane_settings(client, combine_problem)?,
        };

        unsafe {
            // SAFETY: `plan` validated the input/keepdims-output contract and
            // the non-empty reduce axis. Pass 1 covers exactly
            // `plan.output_len * splits` guarded cubes over the axis slices and
            // writes each partial once; `combine` was validated for the
            // `splits`-long partials axis. Both launches enqueue in program
            // order on the client stream, so pass 2 reads pass 1's partials.
            kernels::reduce_sum_float_chunked::launch_unchecked::<F>(
                client,
                CubeCount::Static(pass1_cubes, 1, 1),
                plan.cube_dim,
                input.into_tensor_arg(),
                partials.clone().into_tensor_arg(),
                plan.axis,
                plan.output_len,
                split.splits,
                split.chunk_len,
            );

            match combine.kind {
                ResolvedReduceStrategy::Unit => kernels::reduce_sum_float::launch_unchecked::<F>(
                    client,
                    combine.cube_count,
                    combine.cube_dim,
                    partials.into_tensor_arg(),
                    output.into_tensor_arg(),
                    plan.axis,
                    plan.output_len,
                ),
                ResolvedReduceStrategy::Plane => {
                    kernels::reduce_sum_float_plane::launch_unchecked::<F>(
                        client,
                        combine.cube_count,
                        combine.cube_dim,
                        partials.into_tensor_arg(),
                        output.into_tensor_arg(),
                        plan.axis,
                        plan.output_len,
                    )
                }
            }
        }

        return Ok(());
    }

    unsafe {
        // SAFETY: `plan` validated the input/keepdims-output contract and the
        // non-empty reduce axis; its resolved launch covers exactly
        // `plan.output_len` guarded output elements.
        match plan.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_sum_float::launch_unchecked::<F>(
                client,
                plan.cube_count,
                plan.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                plan.axis,
                plan.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_sum_float_plane::launch_unchecked::<F>(
                    client,
                    plan.cube_count,
                    plan.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    plan.axis,
                    plan.output_len,
                )
            }
        }
    }

    Ok(())
}

/// Launch a floating-point sum-of-squares reduction.
pub(crate) fn launch_sum_squares_float<F: Float + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` proves the input and keepdims output
        // layouts agree and the reduced axis is non-empty. The resolved launch
        // covers exactly `problem.reduce_count` guarded output elements.
        match launch.kind {
            ResolvedReduceStrategy::Unit => {
                kernels::reduce_sum_squares_float::launch_unchecked::<F>(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    launch.axis,
                    launch.output_len,
                )
            }
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_sum_squares_float_plane::launch_unchecked::<F>(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    launch.axis,
                    launch.output_len,
                )
            }
        }
    }

    Ok(())
}

/// Launch an integer sum reduction.
pub(crate) fn launch_sum_int<I: Int + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_sum_int::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => kernels::reduce_sum_int_plane::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
        }
    }

    Ok(())
}

/// Launch a complex sum reduction.
pub(crate) fn launch_sum_complex<C: ComplexCore + CubeElement + PlaneReduce>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_sum_complex::launch_unchecked::<C>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_sum_complex_plane::launch_unchecked::<C>(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    launch.axis,
                    launch.output_len,
                )
            }
        }
    }

    Ok(())
}

/// Launch a floating-point product reduction.
pub(crate) fn launch_prod_float<F: Float + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_prod_float::launch_unchecked::<F>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_prod_float_plane::launch_unchecked::<F>(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    launch.axis,
                    launch.output_len,
                )
            }
        }
    }

    Ok(())
}

/// Launch an integer product reduction.
pub(crate) fn launch_prod_int<I: Int + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_prod_int::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => kernels::reduce_prod_int_plane::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
        }
    }

    Ok(())
}

/// Launch a complex product reduction.
pub(crate) fn launch_prod_complex<C: ComplexCore + CubeElement + PlaneReduce>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_prod_complex::launch_unchecked::<C>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_prod_complex_plane::launch_unchecked::<C>(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    launch.axis,
                    launch.output_len,
                )
            }
        }
    }

    Ok(())
}

/// Launch a floating-point maximum reduction.
pub(crate) fn launch_max_float<F: Float + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_max_float::launch_unchecked::<F>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_max_float_plane::launch_unchecked::<F>(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    launch.axis,
                    launch.output_len,
                )
            }
        }
    }

    Ok(())
}

/// Launch an integer maximum reduction.
pub(crate) fn launch_max_int<I: Int + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_max_int::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => kernels::reduce_max_int_plane::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
        }
    }

    Ok(())
}

/// Launch a floating-point minimum reduction.
pub(crate) fn launch_min_float<F: Float + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_min_float::launch_unchecked::<F>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_min_float_plane::launch_unchecked::<F>(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    input.into_tensor_arg(),
                    output.into_tensor_arg(),
                    launch.axis,
                    launch.output_len,
                )
            }
        }
    }

    Ok(())
}

/// Launch an integer minimum reduction.
pub(crate) fn launch_min_int<I: Int + CubeElement>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
    _partials: Option<TensorBinding>,
) -> Result<()> {
    let launch = plan(client, &input, &output, axis, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `plan` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_min_int::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => kernels::reduce_min_int_plane::launch_unchecked::<I>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
        }
    }

    Ok(())
}
