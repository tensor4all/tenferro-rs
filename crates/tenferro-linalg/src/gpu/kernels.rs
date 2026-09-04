#[path = "kernels/rank_revealing_qr.rs"]
mod rank_revealing_qr;
pub(super) use rank_revealing_qr::*;

use cubecl::prelude::*;
use num_complex::{Complex32, Complex64};

#[cube]
fn zero_value<E: CubePrimitive>() -> E {
    E::cast_from(0u32)
}

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

#[cube]
fn matrix_adjoint_offset<E: CubePrimitive>(
    out: &Tensor<E>,
    v: &Tensor<E>,
    flat: usize,
    #[comptime] rank: usize,
) -> usize {
    let row = out.coordinate(flat, 0usize);
    let col = out.coordinate(flat, 1usize);
    let mut offset = col * v.stride(0usize) + row * v.stride(1usize);
    #[unroll]
    for axis in 2usize..rank {
        let coord = out.coordinate(flat, axis);
        offset += coord * v.stride(axis);
    }
    offset
}

#[cube(launch_unchecked)]
pub fn fill_one_kernel<E: CubePrimitive>(out: &mut Tensor<E>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        out[pos] = one_value::<E>();
    }
}

#[cube(launch_unchecked)]
pub fn matrix_adjoint_real<E: CubePrimitive>(
    out: &mut Tensor<E>,
    v: &Tensor<E>,
    #[comptime] rank: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        out[pos] = v[matrix_adjoint_offset(out, v, pos, rank)];
    }
}

#[cube(launch_unchecked)]
pub fn matrix_adjoint_complex<C: ComplexCore>(
    out: &mut Tensor<C>,
    v: &Tensor<C>,
    #[comptime] rank: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        out[pos] = v[matrix_adjoint_offset(out, v, pos, rank)].conj();
    }
}

#[cube(launch_unchecked)]
pub fn complex32_magnitude(out: &mut Array<f32>, input: &Array<Complex32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].abs();
    }
}

#[cube(launch_unchecked)]
pub fn complex64_magnitude(out: &mut Array<f64>, input: &Array<Complex64>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = input[ABSOLUTE_POS].abs();
    }
}

#[cube(launch_unchecked)]
pub fn householder_explicit_v<E: CubePrimitive>(out: &mut Tensor<E>, packed: &Tensor<E>) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let row = out.coordinate(pos, 0usize);
        let col = out.coordinate(pos, 1usize);
        out[pos] = if row < col {
            zero_value::<E>()
        } else if row == col {
            one_value::<E>()
        } else {
            packed[row * packed.stride(0usize) + col * packed.stride(1usize)]
        };
    }
}

#[cube(launch_unchecked)]
pub fn householder_q_columns_identity<E: CubePrimitive>(out: &mut Tensor<E>, start: usize) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let row = out.coordinate(pos, 0usize);
        let col = out.coordinate(pos, 1usize);
        out[pos] = if row == start + col {
            one_value::<E>()
        } else {
            zero_value::<E>()
        };
    }
}

#[cube]
fn positive_phase_real<E: CubePrimitive + core::ops::Neg<Output = E> + PartialOrd>(
    diagonal: E,
) -> E {
    if diagonal < zero_value::<E>() {
        -one_value::<E>()
    } else {
        one_value::<E>()
    }
}

#[cube]
fn positive_phase_c32(diagonal: Complex32) -> Complex32 {
    let magnitude = diagonal.abs();
    if magnitude == 0.0f32 {
        Complex32::cast_from(1.0f32)
    } else {
        diagonal / Complex32::cast_from(magnitude)
    }
}

#[cube]
fn positive_phase_c64(diagonal: Complex64) -> Complex64 {
    let magnitude = diagonal.abs();
    if magnitude == 0.0f64 {
        Complex64::cast_from(1.0f64)
    } else {
        diagonal / Complex64::cast_from(magnitude)
    }
}

#[cube]
fn qr_diagonal_offset<E: CubePrimitive>(
    phase: &Tensor<E>,
    r: &Tensor<E>,
    pos: usize,
    index: usize,
    #[comptime] rank: usize,
) -> usize {
    let mut offset = index * r.stride(0usize) + index * r.stride(1usize);
    #[unroll]
    for axis in 2usize..rank {
        offset += phase.coordinate(pos, axis - 1usize) * r.stride(axis);
    }
    offset
}

#[cube]
fn qr_phase_offset<E: CubePrimitive>(
    out: &Tensor<E>,
    phase: &Tensor<E>,
    pos: usize,
    index: usize,
    #[comptime] rank: usize,
) -> usize {
    let mut offset = index * phase.stride(0usize);
    #[unroll]
    for axis in 2usize..rank {
        offset += out.coordinate(pos, axis) * phase.stride(axis - 1usize);
    }
    offset
}

#[cube(launch_unchecked)]
pub fn qr_phase_real<E: CubePrimitive + core::ops::Neg<Output = E> + PartialOrd>(
    phase: &mut Tensor<E>,
    r: &Tensor<E>,
    #[comptime] rank: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < phase.len() {
        let index = phase.coordinate(pos, 0usize);
        let diagonal = r[qr_diagonal_offset(phase, r, pos, index, rank)];
        phase[pos] = positive_phase_real::<E>(diagonal);
    }
}

#[cube(launch_unchecked)]
pub fn qr_phase_c32(phase: &mut Tensor<Complex32>, r: &Tensor<Complex32>, #[comptime] rank: usize) {
    let pos = ABSOLUTE_POS as usize;
    if pos < phase.len() {
        let index = phase.coordinate(pos, 0usize);
        let diagonal = r[qr_diagonal_offset(phase, r, pos, index, rank)];
        phase[pos] = positive_phase_c32(diagonal);
    }
}

#[cube(launch_unchecked)]
pub fn qr_phase_c64(phase: &mut Tensor<Complex64>, r: &Tensor<Complex64>, #[comptime] rank: usize) {
    let pos = ABSOLUTE_POS as usize;
    if pos < phase.len() {
        let index = phase.coordinate(pos, 0usize);
        let diagonal = r[qr_diagonal_offset(phase, r, pos, index, rank)];
        phase[pos] = positive_phase_c64(diagonal);
    }
}

#[cube(launch_unchecked)]
pub fn qr_apply_phase_real<E: CubePrimitive + core::ops::Mul<Output = E>>(
    q: &mut Tensor<E>,
    r: &mut Tensor<E>,
    phase: &Tensor<E>,
    q_start: usize,
    #[comptime] rank: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < q.len() {
        let index = q_start + q.coordinate(pos, 1usize);
        q[pos] = q[pos] * phase[qr_phase_offset(q, phase, pos, index, rank)];
    }
    if pos < r.len() {
        let index = r.coordinate(pos, 0usize);
        r[pos] = r[pos] * phase[qr_phase_offset(r, phase, pos, index, rank)];
    }
}

#[cube(launch_unchecked)]
pub fn qr_apply_phase_complex<C: ComplexCore>(
    q: &mut Tensor<C>,
    r: &mut Tensor<C>,
    phase: &Tensor<C>,
    q_start: usize,
    #[comptime] rank: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < q.len() {
        let index = q_start + q.coordinate(pos, 1usize);
        q[pos] = q[pos] * phase[qr_phase_offset(q, phase, pos, index, rank)];
    }
    if pos < r.len() {
        let index = r.coordinate(pos, 0usize);
        r[pos] = r[pos] * phase[qr_phase_offset(r, phase, pos, index, rank)].conj();
    }
}

#[cube(launch_unchecked)]
pub fn upper_trapezoidal_violation<E: CubePrimitive + PartialEq>(
    violation: &mut Tensor<i32>,
    input: &Tensor<E>,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < violation.len() {
        let row = input.coordinate(pos, 0usize);
        let col = input.coordinate(pos, 1usize);
        violation[pos] = if row > col && input[pos] != zero_value::<E>() {
            1i32
        } else {
            0i32
        };
    }
}

#[cube(launch_unchecked)]
pub fn householder_from_factors_assemble<E: CubePrimitive>(
    packed_out: &mut Tensor<E>,
    coeff_out: &mut Tensor<E>,
    packed_q: &Tensor<E>,
    coeff_q: &Tensor<E>,
    folded_r: &Tensor<E>,
    factor_width: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < packed_out.len() {
        let row = packed_out.coordinate(pos, 0usize);
        let col = packed_out.coordinate(pos, 1usize);
        packed_out[pos] = if row < factor_width && row <= col {
            folded_r[row * folded_r.stride(0usize) + col * folded_r.stride(1usize)]
        } else if col < factor_width && row > col {
            packed_q[row * packed_q.stride(0usize) + col * packed_q.stride(1usize)]
        } else {
            zero_value::<E>()
        };
    }
    if pos < coeff_out.len() {
        coeff_out[pos] = if pos < factor_width {
            coeff_q[pos]
        } else {
            zero_value::<E>()
        };
    }
}

#[cube(launch_unchecked)]
pub fn lu_extract_outputs<E: CubePrimitive + core::ops::Neg<Output = E>>(
    p_out: &mut Tensor<E>,
    l_out: &mut Tensor<E>,
    u_out: &mut Tensor<E>,
    parity_out: &mut Tensor<E>,
    work: &Tensor<E>,
    pivots: &Array<i32>,
    k: usize,
    #[comptime] rank: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < p_out.len() {
        let row = p_out.coordinate(pos, 0usize);
        let col = p_out.coordinate(pos, 1usize);
        let batch = batch_linear_index(p_out, pos, 2usize, rank);
        let mut final_row = col as i32;
        let mut step = 0usize;
        while step < k {
            let step_i32 = step as i32;
            let pivot = pivots[step + batch * k] - 1i32;
            if final_row == step_i32 {
                final_row = pivot;
            } else if final_row == pivot {
                final_row = step_i32;
            }
            step += 1usize;
        }
        p_out[pos] = if final_row == row as i32 {
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
        let mut step = 0usize;
        while step < k {
            let step_i32 = step as i32;
            let pivot = pivots[step + batch * k] - 1i32;
            if pivot != step_i32 {
                sign = -sign;
            }
            step += 1usize;
        }
        parity_out[pos] = sign;
    }
}

#[cube(launch_unchecked)]
pub fn lu_parity<E: CubePrimitive + core::ops::Neg<Output = E>>(
    parity_out: &mut Tensor<E>,
    pivots: &Array<i32>,
    k: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < parity_out.len() {
        let mut sign = one_value::<E>();
        let mut step = 0usize;
        while step < k {
            let step_i32 = step as i32;
            let pivot = pivots[step + pos * k] - 1i32;
            if pivot != step_i32 {
                sign = -sign;
            }
            step += 1usize;
        }
        parity_out[pos] = sign;
    }
}

#[cube(launch_unchecked)]
pub fn lu_apply_pivots<E: CubePrimitive>(
    out: &mut Tensor<E>,
    input: &Tensor<E>,
    pivots: &Array<i32>,
    k: usize,
    #[comptime] rank: usize,
    #[comptime] inverse: bool,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let row = out.coordinate(pos, 0usize);
        let col = out.coordinate(pos, 1usize);
        let batch = batch_linear_index(out, pos, 2usize, rank);
        let mut source_row = row as i32;
        if inverse {
            let mut step = 0usize;
            while step < k {
                let step_i32 = step as i32;
                let pivot = pivots[step + batch * k] - 1i32;
                if source_row == step_i32 {
                    source_row = pivot;
                } else if source_row == pivot {
                    source_row = step_i32;
                }
                step += 1usize;
            }
        } else {
            let mut step = k;
            while step > 0usize {
                step -= 1usize;
                let step_i32 = step as i32;
                let pivot = pivots[step + batch * k] - 1i32;
                if source_row == step_i32 {
                    source_row = pivot;
                } else if source_row == pivot {
                    source_row = step_i32;
                }
            }
        }

        let mut input_offset =
            (source_row as usize) * input.stride(0usize) + col * input.stride(1usize);
        #[unroll]
        for axis in 2usize..rank {
            let coord = out.coordinate(pos, axis);
            input_offset += coord * input.stride(axis);
        }
        out[pos] = input[input_offset];
    }
}
