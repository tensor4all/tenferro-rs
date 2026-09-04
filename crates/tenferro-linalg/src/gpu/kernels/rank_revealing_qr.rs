// INVARIANT: CubeCL expands kernel signatures and index expressions into
// generated Rust that triggers host-side style lints; the device IR still
// uses the validated launch domains and bounds below.
#![allow(
    clippy::too_many_arguments,
    clippy::eq_op,
    clippy::unnecessary_cast,
    clippy::neg_cmp_op_on_partial_ord
)]

use cubecl::prelude::*;
use num_complex::{Complex32, Complex64};

#[cube]
fn rrqr_zero<E: CubePrimitive>() -> E {
    E::cast_from(0u32)
}

#[cube]
fn rrqr_one<E: CubePrimitive>() -> E {
    E::cast_from(1u32)
}

#[cube(launch_unchecked)]
pub(crate) fn initialize_permutation(permutation: &mut Tensor<i64>, n: usize) {
    let pos = ABSOLUTE_POS as usize;
    if pos < permutation.len() {
        permutation[pos] = (pos % n) as i64;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn column_norms_real<
    F: Float + CubeElement + CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>,
>(
    work: &Tensor<F>,
    norms: &mut Tensor<F>,
    step: usize,
    m: usize,
    n: usize,
) {
    let task = CUBE_POS_X as usize;
    let trailing = n - step;
    let batch = task / trailing;
    let column = step + task % trailing;
    let matrix_base = batch * m * n;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut local_max = F::new(0.0f32);
    let mut invalid = 0u32.runtime();
    while row < m {
        let magnitude = work[matrix_base + row + column * m].abs();
        let finite_probe = magnitude - magnitude;
        if finite_probe != finite_probe {
            invalid = 1u32;
        } else if magnitude > local_max {
            local_max = magnitude;
        }
        row += plane_width;
    }
    let scale = plane_max(local_max);
    row = step + UNIT_POS as usize;
    let mut local_sum = F::new(0.0f32);
    if scale > F::new(0.0f32) {
        while row < m {
            let scaled = work[matrix_base + row + column * m].abs() / scale;
            local_sum += scaled * scaled;
            row += plane_width;
        }
    }
    let norm = scale * plane_sum(local_sum).sqrt();
    let norm_probe = norm - norm;
    let invalid = plane_sum(invalid) > 0u32 || norm_probe != norm_probe;
    if UNIT_POS == 0 {
        norms[column + batch * n] = if invalid { -F::new(1.0f32) } else { norm };
    }
}

#[cube(launch_unchecked)]
pub(crate) fn column_norms_c32(
    work: &Tensor<Complex32>,
    norms: &mut Tensor<f32>,
    step: usize,
    m: usize,
    n: usize,
) {
    let task = CUBE_POS_X as usize;
    let trailing = n - step;
    let batch = task / trailing;
    let column = step + task % trailing;
    let matrix_base = batch * m * n;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut local_max = 0.0f32;
    let mut invalid = 0u32.runtime();
    while row < m {
        let magnitude = work[matrix_base + row + column * m].abs();
        let finite_probe = magnitude - magnitude;
        if finite_probe != finite_probe {
            invalid = 1u32;
        } else if magnitude > local_max {
            local_max = magnitude;
        }
        row += plane_width;
    }
    let scale = plane_max(local_max);
    row = step + UNIT_POS as usize;
    let mut local_sum = 0.0f32;
    if scale > 0.0f32 {
        while row < m {
            let scaled = work[matrix_base + row + column * m].abs() / scale;
            local_sum += scaled * scaled;
            row += plane_width;
        }
    }
    let norm = scale * plane_sum(local_sum).sqrt();
    let norm_probe = norm - norm;
    let invalid = plane_sum(invalid) > 0u32 || norm_probe != norm_probe;
    if UNIT_POS == 0 {
        norms[column + batch * n] = if invalid {
            0.0f32.runtime() - 1.0f32
        } else {
            norm
        };
    }
}

#[cube(launch_unchecked)]
pub(crate) fn column_norms_c64(
    work: &Tensor<Complex64>,
    norms: &mut Tensor<f64>,
    step: usize,
    m: usize,
    n: usize,
) {
    let task = CUBE_POS_X as usize;
    let trailing = n - step;
    let batch = task / trailing;
    let column = step + task % trailing;
    let matrix_base = batch * m * n;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut local_max = 0.0f64;
    let mut invalid = 0u32.runtime();
    while row < m {
        let magnitude = work[matrix_base + row + column * m].abs();
        let finite_probe = magnitude - magnitude;
        if finite_probe != finite_probe {
            invalid = 1u32;
        } else if magnitude > local_max {
            local_max = magnitude;
        }
        row += plane_width;
    }
    let scale = plane_max(local_max);
    row = step + UNIT_POS as usize;
    let mut local_sum = 0.0f64;
    if scale > 0.0f64 {
        while row < m {
            let scaled = work[matrix_base + row + column * m].abs() / scale;
            local_sum += scaled * scaled;
            row += plane_width;
        }
    }
    let norm = scale * plane_sum(local_sum).sqrt();
    let norm_probe = norm - norm;
    let invalid = plane_sum(invalid) > 0u32 || norm_probe != norm_probe;
    if UNIT_POS == 0 {
        norms[column + batch * n] = if invalid {
            0.0f64.runtime() - 1.0f64
        } else {
            norm
        };
    }
}

#[cube(launch_unchecked)]
pub(crate) fn select_pivot<
    F: Float + CubeElement + CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>,
>(
    norms: &Tensor<F>,
    permutation: &Tensor<i64>,
    pivots: &mut Tensor<i64>,
    status: &mut Tensor<i64>,
    step: usize,
    n: usize,
) {
    let batch = CUBE_POS_X as usize;
    let mut column = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut local_max = F::new(0.0f32);
    let mut local_original = u32::MAX.runtime();
    let mut local_column = u32::MAX.runtime();
    let mut invalid = 0u32.runtime();
    while column < n {
        let offset = column + batch * n;
        let norm = norms[offset];
        let original = permutation[offset] as u32;
        if norm < F::new(0.0f32) {
            invalid = 1u32;
        } else if norm > local_max || (norm == local_max && original < local_original) {
            local_max = norm;
            local_original = original;
            local_column = column as u32;
        }
        column += plane_width;
    }
    let maximum = plane_max(local_max);
    let original = plane_min(if local_max == maximum {
        local_original
    } else {
        u32::MAX.runtime()
    });
    let selected = plane_min(if local_max == maximum && local_original == original {
        local_column
    } else {
        u32::MAX.runtime()
    });
    let any_invalid = plane_sum(invalid) > 0u32;
    if UNIT_POS == 0 {
        pivots[batch] = if selected == u32::MAX {
            step as i64
        } else {
            selected as i64
        };
        if any_invalid {
            status[batch] = 1i64;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn swap_columns<E: CubePrimitive>(
    work: &mut Tensor<E>,
    permutation: &mut Tensor<i64>,
    pivots: &Tensor<i64>,
    step: usize,
    m: usize,
    n: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    let batches = pivots.len();
    if pos < m * batches {
        let row = pos % m;
        let batch = pos / m;
        let pivot = pivots[batch] as usize;
        if pivot != step {
            let matrix_base = batch * m * n;
            let left = matrix_base + row + step * m;
            let right = matrix_base + row + pivot * m;
            let value = work[left];
            work[left] = work[right];
            work[right] = value;
            if row == 0usize {
                let left = step + batch * n;
                let right = pivot + batch * n;
                let original = permutation[left];
                permutation[left] = permutation[right];
                permutation[right] = original;
            }
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reflector_real<
    F: Float + CubeElement + CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>,
>(
    work: &mut Tensor<F>,
    coeff: &mut Tensor<F>,
    step: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let batch = CUBE_POS_X as usize;
    let matrix_base = batch * m * n;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut local_max = F::new(0.0f32);
    while row < m {
        let magnitude = work[matrix_base + row + step * m].abs();
        if magnitude > local_max {
            local_max = magnitude;
        }
        row += plane_width;
    }
    let scale = plane_max(local_max);
    row = step + UNIT_POS as usize;
    let mut local_sum = F::new(0.0f32);
    if scale > F::new(0.0f32) {
        while row < m {
            let scaled = work[matrix_base + row + step * m].abs() / scale;
            local_sum += scaled * scaled;
            row += plane_width;
        }
    }
    let norm = scale * plane_sum(local_sum).sqrt();
    let x0 = work[matrix_base + step + step * m];
    if norm == F::new(0.0f32) {
        if UNIT_POS == 0 {
            coeff[step + batch * k] = F::new(0.0f32);
        }
    } else {
        let phase = if x0 < F::new(0.0f32) {
            -F::new(1.0f32)
        } else {
            F::new(1.0f32)
        };
        let alpha = -phase * norm;
        let denominator = x0 - alpha;
        row = step + 1usize + UNIT_POS as usize;
        while row < m {
            let offset = matrix_base + row + step * m;
            work[offset] = work[offset] / denominator;
            row += plane_width;
        }
        if UNIT_POS == 0 {
            work[matrix_base + step + step * m] = alpha;
            coeff[step + batch * k] = F::new(1.0f32) + x0.abs() / norm;
        }
    }
}

#[cube]
fn c32_from_real(value: f32) -> Complex32 {
    Complex32::cast_from(value)
}

#[cube]
fn c64_from_real(value: f64) -> Complex64 {
    Complex64::cast_from(value)
}

#[cube(launch_unchecked)]
pub(crate) fn reflector_c32(
    work: &mut Tensor<Complex32>,
    coeff: &mut Tensor<Complex32>,
    step: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let batch = CUBE_POS_X as usize;
    let matrix_base = batch * m * n;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut local_max = 0.0f32;
    while row < m {
        let magnitude = work[matrix_base + row + step * m].abs();
        if magnitude > local_max {
            local_max = magnitude;
        }
        row += plane_width;
    }
    let scale = plane_max(local_max);
    row = step + UNIT_POS as usize;
    let mut local_sum = 0.0f32;
    if scale > 0.0f32 {
        while row < m {
            let scaled = work[matrix_base + row + step * m].abs() / scale;
            local_sum += scaled * scaled;
            row += plane_width;
        }
    }
    let norm = scale * plane_sum(local_sum).sqrt();
    let x0 = work[matrix_base + step + step * m];
    if norm == 0.0f32 {
        if UNIT_POS == 0 {
            coeff[step + batch * k] = Complex32::cast_from(0.0f32);
        }
    } else {
        let x0_abs = x0.abs();
        let phase = if x0_abs == 0.0f32 {
            Complex32::cast_from(1.0f32)
        } else {
            x0 / Complex32::cast_from(x0_abs)
        };
        let alpha = -phase * Complex32::cast_from(norm);
        let denominator = x0 - alpha;
        row = step + 1usize + UNIT_POS as usize;
        while row < m {
            let offset = matrix_base + row + step * m;
            work[offset] = work[offset] / denominator;
            row += plane_width;
        }
        if UNIT_POS == 0 {
            work[matrix_base + step + step * m] = alpha;
            coeff[step + batch * k] = Complex32::cast_from(1.0f32 + x0_abs / norm);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn reflector_c64(
    work: &mut Tensor<Complex64>,
    coeff: &mut Tensor<Complex64>,
    step: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    let batch = CUBE_POS_X as usize;
    let matrix_base = batch * m * n;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut local_max = 0.0f64;
    while row < m {
        let magnitude = work[matrix_base + row + step * m].abs();
        if magnitude > local_max {
            local_max = magnitude;
        }
        row += plane_width;
    }
    let scale = plane_max(local_max);
    row = step + UNIT_POS as usize;
    let mut local_sum = 0.0f64;
    if scale > 0.0f64 {
        while row < m {
            let scaled = work[matrix_base + row + step * m].abs() / scale;
            local_sum += scaled * scaled;
            row += plane_width;
        }
    }
    let norm = scale * plane_sum(local_sum).sqrt();
    let x0 = work[matrix_base + step + step * m];
    if norm == 0.0f64 {
        if UNIT_POS == 0 {
            coeff[step + batch * k] = Complex64::cast_from(0.0f64);
        }
    } else {
        let x0_abs = x0.abs();
        let phase = if x0_abs == 0.0f64 {
            Complex64::cast_from(1.0f64)
        } else {
            x0 / Complex64::cast_from(x0_abs)
        };
        let alpha = -phase * Complex64::cast_from(norm);
        let denominator = x0 - alpha;
        row = step + 1usize + UNIT_POS as usize;
        while row < m {
            let offset = matrix_base + row + step * m;
            work[offset] = work[offset] / denominator;
            row += plane_width;
        }
        if UNIT_POS == 0 {
            work[matrix_base + step + step * m] = alpha;
            coeff[step + batch * k] = Complex64::cast_from(1.0f64 + x0_abs / norm);
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn apply_reflector_real<
    F: Float + CubeElement + CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>,
>(
    packed: &Tensor<F>,
    target: &mut Tensor<F>,
    step: usize,
    first_column: usize,
    column_count: usize,
    m: usize,
    packed_columns: usize,
    target_columns: usize,
    k: usize,
    coeff: &Tensor<F>,
) {
    let task = CUBE_POS_X as usize;
    let batch = task / column_count;
    let column = first_column + task % column_count;
    let packed_base = batch * m * packed_columns;
    let target_base = batch * m * target_columns;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut dot = F::new(0.0f32);
    while row < m {
        let vector = if row == step {
            F::new(1.0f32)
        } else {
            packed[packed_base + row + step * m]
        };
        dot = dot + vector * target[target_base + row + column * m];
        row += plane_width;
    }
    dot = plane_sum(dot) * coeff[step + batch * k];
    row = step + UNIT_POS as usize;
    while row < m {
        let vector = if row == step {
            F::new(1.0f32)
        } else {
            packed[packed_base + row + step * m]
        };
        let offset = target_base + row + column * m;
        target[offset] = target[offset] - vector * dot;
        row += plane_width;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn apply_reflector_c32(
    packed: &Tensor<Complex32>,
    target: &mut Tensor<Complex32>,
    step: usize,
    first_column: usize,
    column_count: usize,
    m: usize,
    packed_columns: usize,
    target_columns: usize,
    k: usize,
    coeff: &Tensor<Complex32>,
    imaginary_unit: &Tensor<Complex32>,
) {
    let task = CUBE_POS_X as usize;
    let batch = task / column_count;
    let column = first_column + task % column_count;
    let packed_base = batch * m * packed_columns;
    let target_base = batch * m * target_columns;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut dot = Complex32::cast_from(0.0f32);
    while row < m {
        let vector = if row == step {
            Complex32::cast_from(1.0f32)
        } else {
            packed[packed_base + row + step * m]
        };
        dot = dot + vector.conj() * target[target_base + row + column * m];
        row += plane_width;
    }
    let dot_real = plane_sum(dot.real_val());
    let dot_imag = plane_sum(dot.imag_val());
    dot = (Complex32::cast_from(dot_real) + imaginary_unit[0] * Complex32::cast_from(dot_imag))
        * coeff[step + batch * k];
    row = step + UNIT_POS as usize;
    while row < m {
        let vector = if row == step {
            Complex32::cast_from(1.0f32)
        } else {
            packed[packed_base + row + step * m]
        };
        let offset = target_base + row + column * m;
        target[offset] = target[offset] - vector * dot;
        row += plane_width;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn apply_reflector_c64(
    packed: &Tensor<Complex64>,
    target: &mut Tensor<Complex64>,
    step: usize,
    first_column: usize,
    column_count: usize,
    m: usize,
    packed_columns: usize,
    target_columns: usize,
    k: usize,
    coeff: &Tensor<Complex64>,
    imaginary_unit: &Tensor<Complex64>,
) {
    let task = CUBE_POS_X as usize;
    let batch = task / column_count;
    let column = first_column + task % column_count;
    let packed_base = batch * m * packed_columns;
    let target_base = batch * m * target_columns;
    let mut row = step + UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut dot = Complex64::cast_from(0.0f64);
    while row < m {
        let vector = if row == step {
            Complex64::cast_from(1.0f64)
        } else {
            packed[packed_base + row + step * m]
        };
        dot = dot + vector.conj() * target[target_base + row + column * m];
        row += plane_width;
    }
    let dot_real = plane_sum(dot.real_val());
    let dot_imag = plane_sum(dot.imag_val());
    dot = (Complex64::cast_from(dot_real) + imaginary_unit[0] * Complex64::cast_from(dot_imag))
        * coeff[step + batch * k];
    row = step + UNIT_POS as usize;
    while row < m {
        let vector = if row == step {
            Complex64::cast_from(1.0f64)
        } else {
            packed[packed_base + row + step * m]
        };
        let offset = target_base + row + column * m;
        target[offset] = target[offset] - vector * dot;
        row += plane_width;
    }
}

#[cube(launch_unchecked)]
pub(crate) fn initialize_q<E: CubePrimitive>(q: &mut Tensor<E>, m: usize, k: usize) {
    let pos = ABSOLUTE_POS as usize;
    if pos < q.len() {
        let row = pos % m;
        let column = (pos / m) % k;
        q[pos] = if row == column {
            rrqr_one::<E>()
        } else {
            rrqr_zero::<E>()
        };
    }
}

#[cube(launch_unchecked)]
pub(crate) fn extract_r<E: CubePrimitive>(
    work: &Tensor<E>,
    r: &mut Tensor<E>,
    m: usize,
    n: usize,
    k: usize,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < r.len() {
        let row = pos % k;
        let column = (pos / k) % n;
        let batch = pos / (k * n);
        r[pos] = if row <= column {
            work[batch * m * n + row + column * m]
        } else {
            rrqr_zero::<E>()
        };
    }
}

#[cube(launch_unchecked)]
pub(crate) fn prefix_rank_real<
    F: Float + CubeElement + CubePrimitive<WithScalar<bool> = bool, WithScalar<F> = F>,
>(
    r: &Tensor<F>,
    rank: &mut Tensor<i64>,
    status: &mut Tensor<i64>,
    k: usize,
    n: usize,
    rtol: F,
    atol: F,
) {
    let batch = CUBE_POS_X as usize;
    let base = batch * k * n;
    let leading = r[base].abs();
    let relative = rtol * leading;
    let threshold = if atol > relative { atol } else { relative };
    let mut diagonal = UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut first_failure = u32::MAX.runtime();
    let mut invalid = 0u32.runtime();
    while diagonal < k {
        let value = r[base + diagonal + diagonal * k].abs();
        let finite_probe = value - value;
        if finite_probe != finite_probe {
            invalid = 1u32;
        }
        if !(value > threshold) && (diagonal as u32) < first_failure {
            first_failure = diagonal as u32;
        }
        diagonal += plane_width;
    }
    let first_failure = plane_min(first_failure);
    let any_invalid = plane_sum(invalid) > 0u32;
    if UNIT_POS == 0 {
        rank[batch] = if first_failure == u32::MAX {
            k as i64
        } else {
            first_failure as i64
        };
        if any_invalid {
            status[batch] = 1i64;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn prefix_rank_c32(
    r: &Tensor<Complex32>,
    rank: &mut Tensor<i64>,
    status: &mut Tensor<i64>,
    k: usize,
    n: usize,
    rtol: f32,
    atol: f32,
) {
    let batch = CUBE_POS_X as usize;
    let base = batch * k * n;
    let leading = r[base].abs();
    let relative = rtol * leading;
    let threshold = if atol > relative { atol } else { relative };
    let mut diagonal = UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut first_failure = u32::MAX.runtime();
    let mut invalid = 0u32.runtime();
    while diagonal < k {
        let value = r[base + diagonal + diagonal * k].abs();
        let finite_probe = value - value;
        if finite_probe != finite_probe {
            invalid = 1u32;
        }
        if !(value > threshold) && (diagonal as u32) < first_failure {
            first_failure = diagonal as u32;
        }
        diagonal += plane_width;
    }
    let first_failure = plane_min(first_failure);
    let any_invalid = plane_sum(invalid) > 0u32;
    if UNIT_POS == 0 {
        rank[batch] = if first_failure == u32::MAX {
            k as i64
        } else {
            first_failure as i64
        };
        if any_invalid {
            status[batch] = 1i64;
        }
    }
}

#[cube(launch_unchecked)]
pub(crate) fn prefix_rank_c64(
    r: &Tensor<Complex64>,
    rank: &mut Tensor<i64>,
    status: &mut Tensor<i64>,
    k: usize,
    n: usize,
    rtol: f64,
    atol: f64,
) {
    let batch = CUBE_POS_X as usize;
    let base = batch * k * n;
    let leading = r[base].abs();
    let relative = rtol * leading;
    let threshold = if atol > relative { atol } else { relative };
    let mut diagonal = UNIT_POS as usize;
    let plane_width = PLANE_DIM as usize;
    let mut first_failure = u32::MAX.runtime();
    let mut invalid = 0u32.runtime();
    while diagonal < k {
        let value = r[base + diagonal + diagonal * k].abs();
        let finite_probe = value - value;
        if finite_probe != finite_probe {
            invalid = 1u32;
        }
        if !(value > threshold) && (diagonal as u32) < first_failure {
            first_failure = diagonal as u32;
        }
        diagonal += plane_width;
    }
    let first_failure = plane_min(first_failure);
    let any_invalid = plane_sum(invalid) > 0u32;
    if UNIT_POS == 0 {
        rank[batch] = if first_failure == u32::MAX {
            k as i64
        } else {
            first_failure as i64
        };
        if any_invalid {
            status[batch] = 1i64;
        }
    }
}
