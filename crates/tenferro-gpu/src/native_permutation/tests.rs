use super::*;

const OP: &str = "native_permutation_test";

fn plan(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> NativePermutationPlan {
    let len: usize = dims.iter().product();
    let src_len = 1 + dims
        .iter()
        .zip(src_strides)
        .map(|(&dim, &stride)| dim.saturating_sub(1) * stride.unsigned_abs())
        .sum::<usize>();
    let dst_len = 1 + dims
        .iter()
        .zip(dst_strides)
        .map(|(&dim, &stride)| dim.saturating_sub(1) * stride.unsigned_abs())
        .sum::<usize>();
    NativePermutationPlan::new(
        OP,
        dims,
        src_strides,
        dst_strides,
        0,
        src_len.max(len).max(1),
        dst_len.max(len).max(1),
        false,
    )
    .unwrap()
}

#[test]
fn identity_collapses_to_linear_copy() {
    let plan = plan(&[2, 3, 4], &[1, 2, 6], &[1, 2, 6]);
    assert_eq!(plan.kind, NativePermutationKind::LinearCopy);
    assert_eq!(plan.dims, [24]);
    assert_eq!(plan.src_strides, [1]);
    assert_eq!(plan.dst_strides, [1]);
}

#[test]
fn compact_two_dimensional_transpose_is_tiled_eligible() {
    let plan = NativePermutationPlan::for_transpose(OP, &[2, 3], &[1, 2], &[1, 0], 0, 6, 6, false)
        .unwrap();
    assert_eq!(plan.kind, NativePermutationKind::TiledTranspose);
    assert_eq!(plan.dims, [3, 2]);
    assert_eq!(plan.src_strides, [2, 1]);
    assert_eq!(plan.dst_strides, [1, 3]);
}

#[test]
fn batched_compact_transpose_is_tiled_eligible() {
    let plan = NativePermutationPlan::for_transpose(
        OP,
        &[256, 256, 240],
        &[1, 256, 65_536],
        &[1, 0, 2],
        0,
        15_728_640,
        15_728_640,
        false,
    )
    .unwrap();
    assert_eq!(plan.kind, NativePermutationKind::TiledTranspose);

    let tile = NativeTransposeTile::new(16, 8, 1, 1);
    assert_eq!(
        tile.dispatch_grid(OP, 256, 256, 240, 65_535).unwrap(),
        Some((16, 16, 240))
    );
    assert_eq!(
        tile.dispatch_grid(OP, 256, 256, 240, 16).unwrap(),
        None,
        "every grid dimension must respect the runtime limit"
    );
}

#[test]
fn batched_noncompact_transpose_remains_generic() {
    let plan = plan(&[3, 2, 4], &[2, 1, 7], &[1, 3, 6]);
    assert_eq!(plan.kind, NativePermutationKind::GenericStrided);
}

#[test]
fn transpose_and_equivalent_view_share_one_plan() {
    let transpose =
        NativePermutationPlan::for_transpose(OP, &[2, 3], &[1, 2], &[1, 0], 0, 6, 6, false)
            .unwrap();
    let view =
        NativePermutationPlan::for_contiguous_output(OP, &[3, 2], &[2, 1], 0, 6, 6, false).unwrap();
    assert_eq!(transpose, view);
}

#[test]
fn three_dimensional_swap_preserves_output_axis_order() {
    let plan = NativePermutationPlan::for_transpose(
        OP,
        &[256, 256, 240],
        &[1, 256, 65_536],
        &[1, 0, 2],
        0,
        15_728_640,
        15_728_640,
        false,
    )
    .unwrap();
    assert_eq!(plan.dims, [256, 256, 240]);
    assert_eq!(plan.src_strides, [256, 1, 65_536]);
    assert_eq!(plan.dst_strides, [1, 256, 65_536]);
}

#[test]
fn tile_selection_is_bounded_and_can_force_generic_fallback() {
    assert_eq!(
        NativeTransposeTile::parse(OP, "32x8-p1-v4").unwrap(),
        Some(NativeTransposeTile::new(32, 8, 1, 4))
    );
    assert_eq!(NativeTransposeTile::parse(OP, "generic").unwrap(), None);
    assert!(NativeTransposeTile::parse(OP, "64x1-p0-v8").is_err());
}

#[test]
fn tiled_grid_puts_the_destination_fast_tiles_on_x() {
    let tile = NativeTransposeTile::new(16, 8, 1, 1);
    assert_eq!(
        tile.dispatch_grid(OP, 512, 1_048_576, 1, 65_535).unwrap(),
        None,
        "65536 source-fast tiles exceed the per-dimension launch limit"
    );
    assert_eq!(
        tile.with_tile(32)
            .dispatch_grid(OP, 512, 1_048_576, 1, 65_535)
            .unwrap(),
        Some((16, 32_768, 1))
    );
    assert_eq!(
        tile.dispatch_grid(OP, 1024, 2048, 1, 65_535).unwrap(),
        Some((64, 128, 1)),
        "x indexes the destination-fast extent and y the source-fast extent"
    );
    assert_eq!(
        tile.dispatch_grid(OP, 512, 1_048_576, 1, 32_768).unwrap(),
        None
    );
    assert_eq!(tile.with_tile(32).shared_bytes(16), 32 * 33 * 16);
}

#[test]
fn tiled_transpose_matrix_covers_both_orientations() {
    let canonical = NativeStridedCopyPlan::new(
        OP,
        &[512, 1_048_576],
        &[1_048_576, 1],
        0,
        512 * 1_048_576,
        &[1, 512],
        0,
        512 * 1_048_576,
        false,
    )
    .unwrap();
    assert_eq!(canonical.tiled_transpose_matrix(), Some((512, 1_048_576)));

    let mirrored = NativeStridedCopyPlan::new(
        OP,
        &[1_048_576, 512],
        &[1, 1_048_576],
        0,
        512 * 1_048_576,
        &[512, 1],
        0,
        512 * 1_048_576,
        false,
    )
    .unwrap();
    assert_eq!(mirrored.tiled_transpose_matrix(), Some((512, 1_048_576)));

    // A three-axis plan that fusion cannot reduce has no tiled kernel.
    let multi_axis = NativeStridedCopyPlan::new(
        OP,
        &[1024, 1024, 512],
        &[1, 1024, 1_048_576],
        0,
        1024 * 1024 * 512,
        &[1024, 1, 1_048_576],
        0,
        1024 * 1024 * 512,
        false,
    )
    .unwrap();
    assert_eq!(multi_axis.tiled_transpose_matrix(), None);
}

#[test]
fn tile_grid_falls_back_when_a_dispatch_dimension_exceeds_the_runtime_limit() {
    let tile = NativeTransposeTile::new(16, 8, 1, 1);
    assert_eq!(
        tile.dispatch_grid(OP, 1024, 2048, 1, 65_535).unwrap(),
        Some((64, 128, 1))
    );
    assert_eq!(
        tile.dispatch_grid(OP, 4_782_976, 16, 1, 65_535).unwrap(),
        None
    );
}

#[test]
fn partial_fusion_preserves_affine_metadata() {
    let plan = plan(&[2, 3, 4], &[1, 2, 100], &[1, 2, 6]);
    assert_eq!(plan.kind, NativePermutationKind::GenericStrided);
    assert_eq!(plan.dims, [6, 4]);
    assert_eq!(plan.src_strides, [1, 100]);
    assert_eq!(plan.dst_strides, [1, 6]);
}

#[test]
fn rank_24_identity_collapses_to_linear_copy() {
    let mut dims = vec![64];
    dims.extend([2; 23]);
    let dst = compact_col_major_strides(OP, &dims).unwrap();
    let plan = plan(&dims, &dst, &dst);
    assert_eq!(plan.kind, NativePermutationKind::LinearCopy);
    assert_eq!(plan.dims.len(), 1);
}

#[test]
fn negative_stride_remains_generic_and_preserves_offset() {
    let plan = NativePermutationPlan::new(OP, &[4], &[-1], &[1], 3, 4, 4, false).unwrap();
    assert_eq!(plan.kind, NativePermutationKind::GenericStrided);
    assert_eq!(plan.src_offset, 3);
}

#[test]
fn zero_sized_plan_is_linear_without_allocations() {
    let plan = NativePermutationPlan::new(OP, &[0, 3], &[1, 0], &[1, 0], 0, 0, 0, true).unwrap();
    assert_eq!(plan.kind, NativePermutationKind::LinearCopy);
    assert_eq!(plan.len, 0);
}

#[test]
fn invalid_permutation_and_metadata_lengths_are_rejected() {
    let error = NativePermutationPlan::for_transpose(OP, &[2, 3], &[1, 2], &[0, 0], 0, 6, 6, false)
        .unwrap_err();
    assert!(matches!(error, crate::Error::Validation { .. }));

    let error = NativePermutationPlan::new(OP, &[2, 3], &[1], &[1, 2], 0, 6, 6, false).unwrap_err();
    assert!(matches!(error, crate::Error::Validation { .. }));
}

#[test]
fn product_overflow_and_source_range_violation_are_rejected() {
    let error = NativePermutationPlan::new(
        OP,
        &[usize::MAX, 2],
        &[1, 1],
        &[1, 1],
        0,
        usize::MAX,
        usize::MAX,
        false,
    )
    .unwrap_err();
    assert!(matches!(error, crate::Error::Validation { .. }));

    let error = NativePermutationPlan::new(OP, &[4], &[2], &[1], 0, 4, 4, false).unwrap_err();
    assert!(matches!(error, crate::Error::Validation { .. }));
}

#[test]
fn destination_and_allocation_overlap_are_rejected() {
    let error =
        NativePermutationPlan::new(OP, &[2, 2], &[1, 2], &[1, 1], 0, 4, 4, false).unwrap_err();
    assert!(matches!(error, crate::Error::Validation { .. }));

    let error = NativePermutationPlan::new(OP, &[4], &[1], &[1], 0, 4, 4, true).unwrap_err();
    assert!(matches!(error, crate::Error::Validation { .. }));
}

/// Issue #1891: which permutation copies reduce to one tiled 2D transpose?
///
/// The routing condition is a property of the fused affine plan, so this pins
/// the patterns it covers and the ones that stay on the generic kernel.
#[test]
fn tiled_transpose_covers_single_matrix_permutations_at_any_rank() {
    // Compact source, destination is the permuted view of a compact owner.
    fn plan(shape: &[usize], perm: &[usize]) -> NativeStridedCopyPlan {
        let src_strides = compact_col_major_strides(OP, shape).unwrap();
        let mut owner_shape = vec![0usize; shape.len()];
        for (view_axis, &owner_axis) in perm.iter().enumerate() {
            owner_shape[owner_axis] = shape[view_axis];
        }
        let owner_strides = compact_col_major_strides(OP, &owner_shape).unwrap();
        let dst_strides: Vec<isize> = perm.iter().map(|&axis| owner_strides[axis]).collect();
        let len: usize = shape.iter().product();
        NativeStridedCopyPlan::new(OP, shape, &src_strides, 0, len, &dst_strides, 0, len, false)
            .unwrap()
    }

    // One matrix: a plain 2D transpose.
    assert_eq!(
        plan(&[8192, 8192], &[1, 0]).tiled_transpose_matrix(),
        Some((8192, 8192))
    );
    // A rotation fuses the remaining axes into the matrix columns, at 3D and 4D,
    // for both orientations and for skewed extents.
    assert_eq!(
        plan(&[1024, 1024, 512], &[1, 2, 0]).tiled_transpose_matrix(),
        Some((512, 1_048_576))
    );
    assert_eq!(
        plan(&[1024, 1024, 512], &[2, 0, 1]).tiled_transpose_matrix(),
        Some((524_288, 1024))
    );
    assert_eq!(
        plan(&[64, 64, 64, 64], &[1, 2, 3, 0]).tiled_transpose_matrix(),
        Some((64, 262_144))
    );
    assert_eq!(
        plan(&[64, 64, 64, 64], &[2, 3, 0, 1]).tiled_transpose_matrix(),
        Some((4096, 4096))
    );
    assert_eq!(
        plan(&[2048, 2048, 64], &[1, 2, 0]).tiled_transpose_matrix(),
        Some((64, 4_194_304))
    );
    // A genuine three- or four-axis shuffle stays on the generic kernel, and so
    // does the identity, which is a flat copy.
    for (shape, perm) in [
        (&[1024usize, 1024, 512][..], &[0, 2, 1][..]),
        (&[1024, 1024, 512][..], &[1, 0, 2][..]),
        (&[64, 64, 64, 64][..], &[1, 0, 3, 2][..]),
        (&[64, 64, 64, 64][..], &[3, 1, 2, 0][..]),
        (&[128, 64, 32, 16][..], &[3, 1, 2, 0][..]),
    ] {
        assert_eq!(
            plan(shape, perm).tiled_transpose_matrix(),
            None,
            "{shape:?} {perm:?}"
        );
    }
}
