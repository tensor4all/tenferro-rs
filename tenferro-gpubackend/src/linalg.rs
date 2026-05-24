use cubecl::prelude::*;

use crate::helpers::zero_value;

#[cube]
fn one_value<E: CubePrimitive>() -> E {
    E::cast_from(1u32)
}

#[cube]
fn batch_linear_index<E: CubePrimitive>(
    tensor: &Tensor<E>,
    flat: usize,
    #[comptime] matrix_rank: usize,
    #[comptime] rank: usize,
) -> usize {
    let mut batch = 0usize;
    let mut stride = 1usize;
    #[unroll]
    for axis in matrix_rank..rank {
        let coord = tensor.coordinate(flat, axis);
        batch += coord * stride;
        stride *= tensor.shape(axis);
    }
    batch
}

#[cube]
fn matching_work_offset<Work: CubePrimitive, Out: CubePrimitive>(
    work: &Tensor<Work>,
    out: &Tensor<Out>,
    flat: usize,
    row: usize,
    col: usize,
    #[comptime] rank: usize,
) -> usize {
    let mut offset = row * work.stride(0usize) + col * work.stride(1usize);
    #[unroll]
    for axis in 2usize..rank {
        let coord = out.coordinate(flat, axis);
        offset += coord * work.stride(axis);
    }
    offset
}

#[cube(launch_unchecked)]
pub fn lu_extract_outputs<E: CubePrimitive + core::ops::Neg<Output = E>>(
    p_out: &mut Tensor<E>,
    l_out: &mut Tensor<E>,
    u_out: &mut Tensor<E>,
    parity_out: &mut Tensor<E>,
    work: &Tensor<E>,
    pivots: &Array<u32>,
    #[comptime] k: usize,
    #[comptime] rank: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < p_out.len() {
        let row = p_out.coordinate(pos, 0usize);
        let col = p_out.coordinate(pos, 1usize);
        let batch = batch_linear_index(p_out, pos, 2usize, rank);
        let mut final_row = col as u32;
        #[unroll]
        for step in 0usize..k {
            let step_u32 = step as u32;
            let pivot = pivots[step + batch * k] - 1u32;
            if final_row == step_u32 {
                final_row = pivot;
            } else if final_row == pivot {
                final_row = step_u32;
            }
        }
        p_out[pos] = if final_row == row as u32 {
            one_value::<E>()
        } else {
            zero_value::<E>()
        };
    }

    if pos < l_out.len() {
        let row = l_out.coordinate(pos, 0usize);
        let col = l_out.coordinate(pos, 1usize);
        l_out[pos] = if row < col {
            zero_value::<E>()
        } else if row == col {
            one_value::<E>()
        } else {
            let work_offset = matching_work_offset(work, l_out, pos, row, col, rank);
            work[work_offset]
        };
    }

    if pos < u_out.len() {
        let row = u_out.coordinate(pos, 0usize);
        let col = u_out.coordinate(pos, 1usize);
        u_out[pos] = if row <= col {
            let work_offset = matching_work_offset(work, u_out, pos, row, col, rank);
            work[work_offset]
        } else {
            zero_value::<E>()
        };
    }

    if pos < parity_out.len() {
        let batch = pos;
        let mut sign = one_value::<E>();
        #[unroll]
        for step in 0usize..k {
            let step_u32 = step as u32;
            let pivot = pivots[step + batch * k] - 1u32;
            if pivot != step_u32 {
                sign = -sign;
            }
        }
        parity_out[pos] = sign;
    }
}
