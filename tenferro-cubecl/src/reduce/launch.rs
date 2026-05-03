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
// See tenferro-cubecl/THIRD_PARTY_NOTICES.md for license notice text.
// Tenferro changes: narrowed to tenferro reduction ops, current CubeCL fork,
// single-axis keepdims output, and explicit tenferro column-major bindings.

use cubecl::prelude::*;

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
/// use tenferro_cubecl::reduce::ReduceStrategy;
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
    strategy: ReduceStrategy,
) -> (CubeCount, CubeDim, usize, usize) {
    match strategy {
        ReduceStrategy::Auto | ReduceStrategy::Unit => {
            let settings = unit_launch_settings(client, problem);
            let _has_idle_units = settings.blueprint.idle_units;
            (
                settings.cube_count,
                settings.cube_dim,
                problem.axis,
                problem.reduce_count,
            )
        }
    }
}

/// Launch a floating-point sum reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_sum_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
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
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_sum_float::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
    }

    Ok(())
}

/// Launch an integer sum reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_sum_int, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
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
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_sum_int::launch_unchecked::<I, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
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
/// # use tenferro_cubecl::reduce::{launch_sum_complex, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_sum_complex::<R, Complex64>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_sum_complex<R: Runtime, C: Complex + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_sum_complex::launch_unchecked::<C, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
    }

    Ok(())
}

/// Launch a floating-point product reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_prod_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
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
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_prod_float::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
    }

    Ok(())
}

/// Launch an integer product reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_prod_int, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
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
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_prod_int::launch_unchecked::<I, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
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
/// # use tenferro_cubecl::reduce::{launch_prod_complex, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
/// launch_prod_complex::<R, Complex32>(client, input, output, 0, ReduceStrategy::Auto)
/// # }
/// ```
pub fn launch_prod_complex<R: Runtime, C: Complex + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()> {
    let problem = validate_launch(&input, &output, axis)?;
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_prod_complex::launch_unchecked::<C, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
    }

    Ok(())
}

/// Launch a floating-point maximum reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_max_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
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
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_max_float::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
    }

    Ok(())
}

/// Launch a floating-point minimum reduction.
///
/// # Examples
///
/// ```
/// # use cubecl::prelude::*;
/// # use tenferro_cubecl::reduce::{launch_min_float, ReduceStrategy};
/// # fn example<R: Runtime>(
/// #     client: &ComputeClient<R>,
/// #     input: TensorBinding<R>,
/// #     output: TensorBinding<R>,
/// # ) -> tenferro_cubecl::Result<()> {
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
    let (cube_count, cube_dim, axis, output_len) =
        launch_with_unit_settings(client, problem, strategy);

    unsafe {
        kernels::reduce_min_float::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            axis,
            output_len,
        );
    }

    Ok(())
}
