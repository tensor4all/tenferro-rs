use cubecl::prelude::*;

#[cube]
pub(crate) fn zero_value<E: CubePrimitive>() -> E {
    E::cast_from(0u32)
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
