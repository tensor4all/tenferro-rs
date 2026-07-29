use strided_perm::plan_bilateral_fusion;
use tenferro_tensor::validate::{checked_shape_product, validate_permutation_axes};
use tenferro_tensor::{DynRank, TensorLayout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativePermutationKind {
    LinearCopy,
    GenericStrided,
    TiledTranspose,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeTransposeTile {
    pub(crate) tile: u32,
    pub(crate) block_rows: u32,
    pub(crate) padding: u32,
    pub(crate) vector_width: u32,
}

impl NativeTransposeTile {
    const DEFAULT_NAME: &'static str = "16x8-p1-v1";

    pub(crate) fn selected(op: &'static str) -> crate::Result<Option<Self>> {
        let value = std::env::var("TENFERRO_NATIVE_TRANSPOSE_TILE")
            .unwrap_or_else(|_| Self::DEFAULT_NAME.to_owned());
        Self::parse(op, &value)
    }

    fn parse(op: &'static str, value: &str) -> crate::Result<Option<Self>> {
        let config = match value {
            "generic" => return Ok(None),
            "8x8-p1-v1" => Self::new(8, 8, 1, 1),
            "16x8-p1-v1" => Self::new(16, 8, 1, 1),
            "16x8-p1-v2" => Self::new(16, 8, 1, 2),
            "32x8-p1-v1" => Self::new(32, 8, 1, 1),
            "32x8-p1-v2" => Self::new(32, 8, 1, 2),
            "32x8-p1-v4" => Self::new(32, 8, 1, 4),
            _ => {
                return Err(crate::Error::invalid_argument(
                    op,
                    "TENFERRO_NATIVE_TRANSPOSE_TILE",
                    format!(
                        "unknown tile `{value}`; expected generic, 8x8-p1-v1, \
                         16x8-p1-v1, 16x8-p1-v2, 32x8-p1-v1, 32x8-p1-v2, or 32x8-p1-v4"
                    ),
                ));
            }
        };
        Ok(Some(config))
    }

    const fn new(tile: u32, block_rows: u32, padding: u32, vector_width: u32) -> Self {
        Self {
            tile,
            block_rows,
            padding,
            vector_width,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NativePermutationPlan {
    pub(crate) kind: NativePermutationKind,
    pub(crate) dims: Vec<usize>,
    pub(crate) src_strides: Vec<isize>,
    pub(crate) dst_strides: Vec<isize>,
    pub(crate) src_offset: isize,
    pub(crate) len: usize,
}

impl NativePermutationPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        op: &'static str,
        dims: &[usize],
        src_strides: &[isize],
        dst_strides: &[isize],
        src_offset: isize,
        src_allocation_len: usize,
        dst_allocation_len: usize,
        allocations_overlap: bool,
    ) -> crate::Result<Self> {
        let len = checked_shape_product(op, "shape", dims)?;
        let expected_dst_strides = compact_col_major_strides(op, dims)?;
        if dst_strides != expected_dst_strides {
            return Err(crate::Error::invalid_argument(
                op,
                "destination strides",
                format!(
                    "native permutation destination must be compact column-major: \
                     expected {expected_dst_strides:?}, got {dst_strides:?}"
                ),
            ));
        }
        if allocations_overlap && len != 0 {
            return Err(crate::Error::invalid_argument(
                op,
                "allocations",
                "source and destination allocations overlap",
            ));
        }

        let source = TensorLayout::<DynRank>::from_parts(
            dims.to_vec().into(),
            src_strides.to_vec().into(),
            src_offset,
            src_allocation_len,
        )
        .map_err(|source| crate::Error::validation(op, source))?;
        let destination = TensorLayout::<DynRank>::from_parts(
            dims.to_vec().into(),
            dst_strides.to_vec().into(),
            0,
            dst_allocation_len,
        )
        .map_err(|source| crate::Error::validation(op, source))?;
        destination
            .validate_mutable_no_overlap()
            .map_err(|source| crate::Error::validation(op, source))?;

        let fusion = plan_bilateral_fusion(source.shape(), source.strides(), destination.strides())
            .map_err(|source| {
                crate::Error::invalid_argument(op, "fusion metadata", source.to_string())
            })?;
        let kind = classify(&fusion.dims, &fusion.src_strides, &fusion.dst_strides, len);

        Ok(Self {
            kind,
            dims: fusion.dims,
            src_strides: fusion.src_strides,
            dst_strides: fusion.dst_strides,
            src_offset,
            len,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_transpose(
        op: &'static str,
        input_shape: &[usize],
        input_strides: &[isize],
        permutation: &[usize],
        src_offset: isize,
        src_allocation_len: usize,
        dst_allocation_len: usize,
        allocations_overlap: bool,
    ) -> crate::Result<Self> {
        validate_permutation_axes(op, input_shape.len(), permutation)?;
        if input_shape.len() != input_strides.len() {
            return Err(crate::Error::rank_mismatch(
                op,
                input_shape.len(),
                input_strides.len(),
            ));
        }

        let dims: Vec<_> = permutation.iter().map(|&axis| input_shape[axis]).collect();
        let src_strides: Vec<_> = permutation
            .iter()
            .map(|&axis| input_strides[axis])
            .collect();
        let dst_strides = compact_col_major_strides(op, &dims)?;
        Self::new(
            op,
            &dims,
            &src_strides,
            &dst_strides,
            src_offset,
            src_allocation_len,
            dst_allocation_len,
            allocations_overlap,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_contiguous_output(
        op: &'static str,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        src_allocation_len: usize,
        dst_allocation_len: usize,
        allocations_overlap: bool,
    ) -> crate::Result<Self> {
        let dst_strides = compact_col_major_strides(op, dims)?;
        Self::new(
            op,
            dims,
            src_strides,
            &dst_strides,
            src_offset,
            src_allocation_len,
            dst_allocation_len,
            allocations_overlap,
        )
    }
}

fn compact_col_major_strides(op: &'static str, dims: &[usize]) -> crate::Result<Vec<isize>> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        let dim = isize::try_from(dim).map_err(|_| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("dimension {dim} cannot be represented as isize"),
            )
        })?;
        stride = stride.checked_mul(dim).ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("column-major stride overflow for shape {dims:?}"),
            )
        })?;
    }
    Ok(strides)
}

fn classify(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    len: usize,
) -> NativePermutationKind {
    if len == 0
        || (dims.len() <= 1
            && src_strides.first().is_none_or(|&stride| stride == 1)
            && dst_strides.first().is_none_or(|&stride| stride == 1))
    {
        return NativePermutationKind::LinearCopy;
    }
    if tiled_transpose_eligible(dims, src_strides, dst_strides) {
        NativePermutationKind::TiledTranspose
    } else {
        NativePermutationKind::GenericStrided
    }
}

fn tiled_transpose_eligible(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> bool {
    if dims.len() != 2 || dims.contains(&0) {
        return false;
    }
    let Some(src_axis) = src_strides.iter().position(|&stride| stride == 1) else {
        return false;
    };
    let Some(dst_axis) = dst_strides.iter().position(|&stride| stride == 1) else {
        return false;
    };
    if src_axis == dst_axis {
        return false;
    }
    let src_other = 1 - src_axis;
    let dst_other = 1 - dst_axis;
    isize::try_from(dims[src_axis]).is_ok_and(|extent| src_strides[src_other] == extent)
        && isize::try_from(dims[dst_axis]).is_ok_and(|extent| dst_strides[dst_other] == extent)
}

#[cfg(test)]
mod tests {
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
        let plan =
            NativePermutationPlan::for_transpose(OP, &[2, 3], &[1, 2], &[1, 0], 0, 6, 6, false)
                .unwrap();
        assert_eq!(plan.kind, NativePermutationKind::TiledTranspose);
        assert_eq!(plan.dims, [3, 2]);
        assert_eq!(plan.src_strides, [2, 1]);
        assert_eq!(plan.dst_strides, [1, 3]);
    }

    #[test]
    fn transpose_and_equivalent_view_share_one_plan() {
        let transpose =
            NativePermutationPlan::for_transpose(OP, &[2, 3], &[1, 2], &[1, 0], 0, 6, 6, false)
                .unwrap();
        let view =
            NativePermutationPlan::for_contiguous_output(OP, &[3, 2], &[2, 1], 0, 6, 6, false)
                .unwrap();
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
        let plan =
            NativePermutationPlan::new(OP, &[0, 3], &[1, 0], &[1, 0], 0, 0, 0, true).unwrap();
        assert_eq!(plan.kind, NativePermutationKind::LinearCopy);
        assert_eq!(plan.len, 0);
    }

    #[test]
    fn invalid_permutation_and_metadata_lengths_are_rejected() {
        let error =
            NativePermutationPlan::for_transpose(OP, &[2, 3], &[1, 2], &[0, 0], 0, 6, 6, false)
                .unwrap_err();
        assert!(matches!(error, crate::Error::Validation { .. }));

        let error =
            NativePermutationPlan::new(OP, &[2, 3], &[1], &[1, 2], 0, 6, 6, false).unwrap_err();
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
}
