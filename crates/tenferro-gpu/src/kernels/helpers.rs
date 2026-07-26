use cubecl::prelude::*;

// INVARIANT: CubeCL fixed-width Int arithmetic and integer plane collectives
// lower to modulo-2^N operations for I32/I64 on the CUDA backend. Keep every
// public integer data path routed through these named helpers so wrapping
// semantics remain explicit and auditable if CubeCL codegen changes.
#[cube]
pub(crate) fn wrapping_add<I: Int>(lhs: I, rhs: I) -> I {
    lhs + rhs
}

#[cube]
pub(crate) fn wrapping_sub<I: Int>(lhs: I, rhs: I) -> I {
    lhs - rhs
}

#[cube]
pub(crate) fn wrapping_mul<I: Int>(lhs: I, rhs: I) -> I {
    lhs * rhs
}

#[cube]
pub(crate) fn wrapping_neg<I: Int>(value: I) -> I {
    I::new(0) - value
}

#[cube]
pub(crate) fn wrapping_plane_sum<I: Int>(value: I) -> I {
    plane_sum(value)
}

#[cube]
pub(crate) fn wrapping_plane_prod<I: Int>(value: I) -> I {
    plane_prod(value)
}

// CubeCL's generic `is_nan` result uses `WithScalar<bool>`; the self-compare
// keeps these scalar kernels generic over `F: Float`.
#[allow(clippy::eq_op)]
#[cube]
pub(crate) fn nan_propagating_max<F: Float>(lhs: F, rhs: F) -> F {
    if lhs != lhs {
        lhs
    } else if rhs != rhs {
        rhs
    } else {
        lhs.max(rhs)
    }
}

#[allow(clippy::eq_op)]
#[cube]
pub(crate) fn nan_propagating_min<F: Float>(lhs: F, rhs: F) -> F {
    if lhs != lhs {
        lhs
    } else if rhs != rhs {
        rhs
    } else {
        lhs.min(rhs)
    }
}

#[allow(clippy::eq_op)]
#[cube]
pub(crate) fn plane_contains_nan<F: Float>(value: F) -> bool {
    let flag: u32 = if value != value { 1u32 } else { 0u32 };
    plane_sum(flag) > 0u32
}

// INVARIANT: The self-comparison is an intentional generic CubeCL NaN test;
// `Float::is_nan` returns `WithScalar<bool>` instead of a scalar `bool` here.
#[allow(clippy::eq_op)]
#[cube]
pub(crate) fn plane_propagate_nan<F: Float>(value: F) -> F {
    let nan_or_zero = if value != value {
        value
    } else {
        F::new(0.0_f32)
    };
    plane_sum(nan_or_zero)
}

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
