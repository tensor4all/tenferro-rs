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

use cubecl::prelude::*;

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
pub(crate) fn unit_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
) -> ReduceLaunchSettings {
    unit_launch_settings_for_plane_size(client.properties().hardware.plane_size_max, problem)
}

fn unit_launch_settings_for_plane_size(
    plane_size: u32,
    problem: ReduceProblem,
) -> ReduceLaunchSettings {
    let cube_dim = CubeDim::new_1d(plane_size);
    let units_per_cube = cube_dim.num_elems() as usize;
    let cubes = problem.reduce_count.div_ceil(units_per_cube).max(1) as u32;

    ReduceLaunchSettings {
        cube_count: CubeCount::Static(cubes, 1, 1),
        cube_dim,
        blueprint: UnitReduceBlueprint {
            idle_units: problem.reduce_count % units_per_cube != 0,
        },
    }
}

#[cfg(test)]
mod tests;
