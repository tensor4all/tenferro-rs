use strided_perm::plan_bilateral_fusion;
use tenferro_tensor::validate::{checked_shape_product, validate_permutation_axes};
use tenferro_tensor::{DynRank, TensorLayout};

#[cfg(test)]
mod compact_stride_tests;

pub(crate) fn compact_col_major_strides(
    op: &'static str,
    shape: &[usize],
) -> crate::Result<Vec<isize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &dim in shape {
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
                format!("column-major stride overflow for shape {shape:?}"),
            )
        })?;
    }
    Ok(strides)
}

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

    pub(crate) fn dispatch_grid(
        self,
        op: &'static str,
        dims: &[usize],
        max_dimension: u32,
    ) -> crate::Result<Option<(u32, u32, u32)>> {
        let x = u32::try_from(dims[1].div_ceil(self.tile as usize)).map_err(|_| {
            crate::Error::invalid_argument(op, "shape", "tiled transpose x grid exceeds u32::MAX")
        })?;
        let y = u32::try_from(dims[0].div_ceil(self.tile as usize)).map_err(|_| {
            crate::Error::invalid_argument(op, "shape", "tiled transpose y grid exceeds u32::MAX")
        })?;
        let z = u32::try_from(dims.get(2).copied().unwrap_or(1)).map_err(|_| {
            crate::Error::invalid_argument(op, "shape", "tiled transpose z grid exceeds u32::MAX")
        })?;
        if x > max_dimension || y > max_dimension || z > max_dimension {
            return Ok(None);
        }
        Ok(Some((x.max(1), y.max(1), z.max(1))))
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

    pub(crate) fn tiled_matrix_len(&self, op: &'static str) -> crate::Result<usize> {
        self.dims
            .first()
            .zip(self.dims.get(1))
            .and_then(|(&rows, &columns)| rows.checked_mul(columns))
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    op,
                    "shape",
                    "tiled transpose matrix extent overflows usize",
                )
            })
    }
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
    if !(dims.len() == 2 || dims.len() == 3) || dims.contains(&0) {
        return false;
    }
    let Some(src_axis) = src_strides.iter().position(|&stride| stride == 1) else {
        return false;
    };
    let Some(dst_axis) = dst_strides.iter().position(|&stride| stride == 1) else {
        return false;
    };
    if src_axis >= 2 || dst_axis >= 2 || src_axis == dst_axis {
        return false;
    }
    let src_other = 1 - src_axis;
    let dst_other = 1 - dst_axis;
    let matrix_is_transpose = isize::try_from(dims[src_axis])
        .is_ok_and(|extent| src_strides[src_other] == extent)
        && isize::try_from(dims[dst_axis]).is_ok_and(|extent| dst_strides[dst_other] == extent);
    if !matrix_is_transpose || dims.len() == 2 {
        return matrix_is_transpose;
    }

    dims[0]
        .checked_mul(dims[1])
        .and_then(|matrix_len| isize::try_from(matrix_len).ok())
        .is_some_and(|matrix_stride| {
            src_strides[2] == matrix_stride && dst_strides[2] == matrix_stride
        })
}

#[cfg(test)]
mod tests;
