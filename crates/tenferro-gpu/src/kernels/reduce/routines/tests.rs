use cubecl::prelude::*;

use super::{unit_launch_settings_for_plane_size, ReduceProblem};

fn assert_static_cube_count(count: &CubeCount, expected_x: u32) {
    match count {
        CubeCount::Static(x, y, z) => assert_eq!((*x, *y, *z), (expected_x, 1, 1)),
        other => panic!("expected static cube count, got {other:?}"),
    }
}

#[test]
fn unit_launch_settings_uses_one_unit_per_output_element() {
    let settings = unit_launch_settings_for_plane_size(
        32,
        ReduceProblem {
            reduce_len: 5,
            reduce_count: 65,
            axis: 1,
        },
    )
    .unwrap();

    assert_static_cube_count(&settings.cube_count, 3);
    assert_eq!(settings.cube_dim, CubeDim::new_1d(32));
    assert!(settings.blueprint.idle_units);
}

#[test]
fn unit_launch_settings_marks_full_cubes_as_non_idle() {
    let settings = unit_launch_settings_for_plane_size(
        32,
        ReduceProblem {
            reduce_len: 5,
            reduce_count: 64,
            axis: 1,
        },
    )
    .unwrap();

    assert_static_cube_count(&settings.cube_count, 2);
    assert_eq!(settings.cube_dim, CubeDim::new_1d(32));
    assert!(!settings.blueprint.idle_units);
}

#[test]
fn unit_launch_settings_keeps_empty_output_launch_valid() {
    let settings = unit_launch_settings_for_plane_size(
        32,
        ReduceProblem {
            reduce_len: 5,
            reduce_count: 0,
            axis: 1,
        },
    )
    .unwrap();

    assert_static_cube_count(&settings.cube_count, 1);
    assert_eq!(settings.cube_dim, CubeDim::new_1d(32));
}

#[test]
fn unit_launch_settings_rejects_cube_count_overflow() {
    let settings = unit_launch_settings_for_plane_size(
        1,
        ReduceProblem {
            reduce_len: 5,
            reduce_count: u32::MAX as usize + 1,
            axis: 1,
        },
    );

    assert!(settings.is_err());
}
