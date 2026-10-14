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

//! Shared reduction launch routines.

// INVARIANT: CubeCL's kernel DSL lowers modulo checks differently from host
// Rust; the reduction dimensions were validated before launch.
#![allow(clippy::manual_is_multiple_of)]

use cubecl::prelude::*;

use super::launch::{
    auto_reduce_strategy, launch_with_plane_settings, launch_with_unit_settings, validate_launch,
    ReduceStrategy, ResolvedReduceStrategy,
};
use crate::kernels::{CubeclKernelError, Result};

/// Validated single-axis reduction dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReduceProblem {
    /// Length of the axis being reduced.
    pub reduce_len: usize,
    /// Number of keepdims output elements.
    pub reduce_count: usize,
    /// Axis being reduced.
    pub axis: usize,
}

/// Starting cap on the number of axis chunks. It exists only to bound the
/// launch grid and the partials buffer; this value is a starting point and is
/// tuned by the acceptance measurement, not by the geometry rule itself.
pub(crate) const MAX_SPLITS: usize = 4096;

/// A split of the reduce axis into equal-length chunk slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ReduceSplit {
    /// Number of chunk slices the axis is cut into.
    pub splits: usize,
    /// Elements in each chunk; the last chunk may hold the division remainder.
    pub chunk_len: usize,
}

/// Decide whether the reduce axis is long enough to split across cubes.
///
/// The floor rule keeps every chunk at least `MIN_CHUNK = plane_width * 8`
/// elements, so each thread of a cube always has its own first element and
/// the kernels need no per-thread identity value.
pub(crate) fn split_plane(problem: ReduceProblem, plane_width: u32) -> Option<ReduceSplit> {
    let min_chunk = plane_width as usize * 8;
    let splits = (problem.reduce_len / min_chunk).clamp(1, MAX_SPLITS);
    if splits <= 1 {
        return None;
    }
    Some(ReduceSplit {
        splits,
        chunk_len: problem.reduce_len / splits,
    })
}

/// A resolved reduction launch, including the optional axis split.
#[derive(Clone, Debug)]
pub(crate) struct ReduceLaunchPlan {
    /// Kernel family selected for the (unsplit) reduction.
    pub kind: ResolvedReduceStrategy,
    /// Cubes to dispatch for the unsplit reduction.
    pub cube_count: CubeCount,
    /// Units per cube for the unsplit reduction.
    pub cube_dim: CubeDim,
    /// Axis being reduced.
    pub axis: usize,
    /// Number of keepdims output elements.
    pub output_len: usize,
    /// Axis split for pass 1, or `None` to run the existing single pass.
    pub split: Option<ReduceSplit>,
}

/// Validate, resolve the strategy and decide the split, without launching.
pub(crate) fn plan(
    client: &Client,
    input: &TensorBinding,
    output: &TensorBinding,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<ReduceLaunchPlan> {
    let problem = validate_launch(input, output, axis)?;
    let resolved = match strategy {
        ReduceStrategy::Auto => match auto_reduce_strategy(client, problem)? {
            ResolvedReduceStrategy::Unit => launch_with_unit_settings(client, problem)?,
            ResolvedReduceStrategy::Plane => launch_with_plane_settings(client, problem)?,
        },
    };
    let plane_width = client.properties().hardware.plane_size_max.max(1);
    Ok(ReduceLaunchPlan {
        kind: resolved.kind,
        cube_count: resolved.cube_count,
        cube_dim: resolved.cube_dim,
        axis: resolved.axis,
        output_len: resolved.output_len,
        split: split_plane(problem, plane_width),
    })
}

/// The partials tensor a split plan needs: the keepdims shape with the reduce
/// axis set to `splits`. `None` when the plan runs unsplit.
pub(crate) fn partials_shape(
    plan: &ReduceLaunchPlan,
    output_shape: &[usize],
    axis: usize,
) -> Option<Vec<usize>> {
    let split = plan.split?;
    let mut shape = output_shape.to_vec();
    shape[axis] = split.splits;
    Some(shape)
}

/// Unit-kernel launch blueprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnitReduceBlueprint {
    /// Whether some units in the last cube do not map to an output element.
    pub idle_units: bool,
}

/// Concrete CubeCL launch settings for a reduction kernel.
#[derive(Clone, Debug)]
pub struct ReduceLaunchSettings {
    /// Number of cubes to dispatch.
    pub cube_count: CubeCount,
    /// Number of units per cube.
    pub cube_dim: CubeDim,
    /// Unit-kernel metadata.
    pub blueprint: UnitReduceBlueprint,
}

/// Choose launch settings for the first unit-kernel implementation.
pub(crate) fn unit_launch_settings(
    client: &Client,
    problem: ReduceProblem,
) -> Result<ReduceLaunchSettings> {
    unit_launch_settings_for_plane_size(client.properties().hardware.plane_size_max, problem)
}

fn unit_launch_settings_for_plane_size(
    plane_size: u32,
    problem: ReduceProblem,
) -> Result<ReduceLaunchSettings> {
    let cube_dim = CubeDim::new_1d(plane_size);
    let units_per_cube = cube_dim.num_elems() as usize;
    let cubes =
        u32::try_from(problem.reduce_count.div_ceil(units_per_cube).max(1)).map_err(|_| {
            CubeclKernelError::InvalidStrategy {
                reason: format!(
                    "reduction output element count {} exceeds static CubeCL launch limit",
                    problem.reduce_count
                ),
            }
        })?;

    Ok(ReduceLaunchSettings {
        cube_count: CubeCount::Static(cubes, 1, 1),
        cube_dim,
        blueprint: UnitReduceBlueprint {
            idle_units: problem.reduce_count % units_per_cube != 0,
        },
    })
}

#[cfg(test)]
mod tests;
