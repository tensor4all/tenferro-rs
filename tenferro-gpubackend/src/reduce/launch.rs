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
// See tenferro-gpubackend/THIRD_PARTY_NOTICES.md for license notice text.
// Tenferro changes: narrowed to tenferro reduction ops, current CubeCL fork,
// single-axis keepdims output, and explicit tenferro column-major bindings.

use cubecl::{features::Plane, prelude::*};

use super::{
    kernels,
    routines::{unit_launch_settings, ReduceProblem},
    validate_keepdims_output_shape,
};
use crate::{CubeclKernelError, Result};

#[cfg(test)]
mod tests;

/// Launch strategy for a single-axis reduction.
///
/// # Examples
///
/// ```
/// use tenferro_gpubackend::reduce::ReduceStrategy;
///
/// let strategy = ReduceStrategy::Auto;
/// assert_eq!(format!("{strategy:?}"), "Auto");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceStrategy {
    /// Let the launcher choose the kernel strategy.
    Auto,
    /// Use one worker per keepdims output element.
    Unit,
    /// Use one hardware plane/subgroup per keepdims output element.
    Plane,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedReduceStrategy {
    Unit,
    Plane,
}

#[derive(Clone, Debug)]
struct ResolvedReduceLaunch {
    kind: ResolvedReduceStrategy,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    axis: usize,
    output_len: usize,
}

fn validate_launch<R: Runtime>(
    input: &TensorBinding<R>,
    output: &TensorBinding<R>,
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

    let input_len = input_shape.iter().product::<usize>();
    let reduce_count = input_len / reduce_len;

    Ok(ReduceProblem {
        reduce_len,
        reduce_count,
        axis,
    })
}

fn launch_with_unit_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
) -> ResolvedReduceLaunch {
    let settings = unit_launch_settings(client, problem);
    let _has_idle_units = settings.blueprint.idle_units;
    ResolvedReduceLaunch {
        kind: ResolvedReduceStrategy::Unit,
        cube_count: settings.cube_count,
        cube_dim: settings.cube_dim,
        axis: problem.axis,
        output_len: problem.reduce_count,
    }
}

fn launch_with_plane_settings<R: Runtime>(
    client: &ComputeClient<R>,
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

fn auto_reduce_strategy<R: Runtime>(
    client: &ComputeClient<R>,
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

fn resolve_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    strategy: ReduceStrategy,
) -> Result<ResolvedReduceLaunch> {
    match strategy {
        ReduceStrategy::Unit => Ok(launch_with_unit_settings(client, problem)),
        ReduceStrategy::Plane => launch_with_plane_settings(client, problem),
        ReduceStrategy::Auto => match auto_reduce_strategy(client, problem)? {
            ResolvedReduceStrategy::Unit => Ok(launch_with_unit_settings(client, problem)),
            ResolvedReduceStrategy::Plane => launch_with_plane_settings(client, problem),
        },
    }
}

/// Launch a floating-point sum reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_gpubackend::reduce::{launch_sum_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_sum_float::<R, f32>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_sum_float<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_sum_float::launch_unchecked::<F, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_sum_float_plane::launch_unchecked::<F, R>(
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
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_gpubackend::reduce::{launch_sum_int, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_sum_int::<R, i64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_sum_int<R: Runtime, I: Int + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_sum_int::launch_unchecked::<I, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_sum_int_plane::launch_unchecked::<I, R>(
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

/// Launch a complex sum reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use num_complex::Complex64;
/// # use tenferro_gpubackend::reduce::{launch_sum_complex, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_sum_complex::<R, Complex64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_sum_complex<R: Runtime, C: ComplexCore + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_sum_complex::launch_unchecked::<C, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_sum_complex_plane::launch_unchecked::<C, R>(
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
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_gpubackend::reduce::{launch_prod_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_prod_float::<R, f64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_prod_float<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_prod_float::launch_unchecked::<F, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_prod_float_plane::launch_unchecked::<F, R>(
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
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_gpubackend::reduce::{launch_prod_int, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_prod_int::<R, i64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_prod_int<R: Runtime, I: Int + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_prod_int::launch_unchecked::<I, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_prod_int_plane::launch_unchecked::<I, R>(
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

/// Launch a complex product reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use num_complex::Complex32;
/// # use tenferro_gpubackend::reduce::{launch_prod_complex, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_prod_complex::<R, Complex32>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_prod_complex<R: Runtime, C: ComplexCore + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_prod_complex::launch_unchecked::<C, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_prod_complex_plane::launch_unchecked::<C, R>(
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
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_gpubackend::reduce::{launch_max_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_max_float::<R, f32>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_max_float<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_max_float::launch_unchecked::<F, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_max_float_plane::launch_unchecked::<F, R>(
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

/// Launch a floating-point minimum reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_gpubackend::reduce::{launch_min_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_gpubackend::Result<()> {
/// launch_min_float::<R, f64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_min_float<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let launch = resolve_launch_settings(client, problem, strategy)?;

    unsafe {
        // SAFETY: `validate_launch` produced a `ReduceProblem` proving input
        // and keepdims output shapes are compatible and the reduce axis is
        // non-empty. `resolve_launch_settings` derives the launched output
        // domain from that validated problem; the reduction kernel uses
        // `output_len == problem.reduce_count` to guard output indexing.
        match launch.kind {
            ResolvedReduceStrategy::Unit => kernels::reduce_min_float::launch_unchecked::<F, R>(
                client,
                launch.cube_count,
                launch.cube_dim,
                input.into_tensor_arg(),
                output.into_tensor_arg(),
                launch.axis,
                launch.output_len,
            ),
            ResolvedReduceStrategy::Plane => {
                kernels::reduce_min_float_plane::launch_unchecked::<F, R>(
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
