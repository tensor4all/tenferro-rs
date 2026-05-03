use cubecl::prelude::*;

#[cube]
pub(crate) fn zero_value<E: CubePrimitive>() -> E {
    E::cast_from(0u32)
}

#[cube]
pub(crate) fn flat_to_multi_index(
    mut flat: usize,
    #[comptime] shape: Sequence<usize>,
) -> Array<usize> {
    let rank = shape.len();
    let mut indices = Array::<usize>::new(rank);
    #[unroll]
    for axis in 0..rank {
        let dim = comptime! { *shape.index(axis) };
        indices[axis] = flat % dim;
        flat /= dim;
    }
    indices
}

#[cube]
pub(crate) fn multi_to_flat_index(
    indices: &Array<usize>,
    #[comptime] shape: Sequence<usize>,
) -> usize {
    let rank = shape.len();
    let mut offset = 0usize;
    let mut stride = 1usize;
    #[unroll]
    for axis in 0..rank {
        let dim = comptime! { *shape.index(axis) };
        offset += indices[axis] * stride;
        stride *= dim;
    }
    offset
}

#[cube]
pub(crate) fn flat_to_tensor_index<E: CubePrimitive>(
    flat: usize,
    tensor: &Tensor<E>,
    #[comptime] rank: usize,
) -> Array<usize> {
    let mut indices = Array::<usize>::new(rank);
    #[unroll]
    for axis in 0..rank {
        indices[axis] = tensor.coordinate(flat, axis);
    }
    indices
}

#[cube]
pub(crate) fn multi_to_tensor_index<E: CubePrimitive>(
    indices: &Array<usize>,
    tensor: &Tensor<E>,
    #[comptime] rank: usize,
) -> usize {
    let mut offset = 0usize;
    #[unroll]
    for axis in 0..rank {
        offset += indices[axis] * tensor.stride(axis);
    }
    offset
}

#[cube]
pub(crate) fn axis_in_sequence(#[comptime] axes: Sequence<usize>, axis: usize) -> bool {
    let count = axes.len();
    let mut found = false;
    #[unroll]
    for pos in 0..count {
        if comptime! { *axes.index(pos) } == axis {
            found = true;
        }
    }
    found
}

#[cube]
pub(crate) fn axis_position_in_sequence(#[comptime] axes: Sequence<usize>, axis: usize) -> usize {
    let count = axes.len();
    let mut found = 0usize;
    #[unroll]
    for pos in 0..count {
        if comptime! { *axes.index(pos) } == axis {
            found = pos;
        }
    }
    found
}

#[cube]
pub(crate) fn shape_product(#[comptime] shape: Sequence<usize>) -> usize {
    let mut total = 1usize;
    let rank = shape.len();
    #[unroll]
    for axis in 0..rank {
        total *= comptime! { *shape.index(axis) };
    }
    total
}
