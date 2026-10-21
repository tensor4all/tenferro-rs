//! Fused CPU softmax: a PyTorch-style per-slice kernel.
//!
//! PyTorch's `_vec_softmax_lastdim` is fast because it runs one kernel over
//! contiguous slices: vectorized max, `(x - max).exp()` written directly into
//! the output, a vectorized sum of that output and a vectorized scale, inside a
//! grain-sized parallel loop with no full-size intermediate. The critical part
//! is the **vectorized floating-point reduction**: LLVM does not vectorize a
//! scalar `sum += x` reduction without fast-math, so max and sum use explicit
//! AVX2 horizontal reductions and the shifted values live in a cache-sized
//! per-task buffer.

use rayon::prelude::*;
use std::mem::MaybeUninit;

use tenferro_tensor::{DType, Tensor, TensorRead};

use crate::activation::fill_pooled;
use crate::buffer_pool::BufferPool;
use crate::materialize_tensor_read;
use crate::parallel::{group_len, task_len, MIN_BYTES_PER_TASK};
use crate::simd_math::{apply, VectorKernel};

/// Fused softmax over axis 0 for a host `F32`/`F64` tensor, if supported.
///
/// The output is a pooled, recycled tensor: a caller that holds many outputs
/// and drops them together (a benchmark, a batch loop) gets its storage back
/// from the pool instead of fresh pages from the allocator.
pub(crate) fn softmax_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axis: usize,
) -> crate::Result<Option<Tensor>> {
    let shape = input.shape().to_vec();
    if axis != 0 || shape.is_empty() || shape[0] == 0 {
        return Ok(None);
    }
    let axis_len = shape[0];
    match input.dtype() {
        DType::F32 => {
            if let Ok(data) = input.as_slice::<f32>() {
                return fill_pooled(buffers, &shape, |out| {
                    softmax_axis0_f32(data, axis_len, out)
                })
                .map(Some);
            }
            let tensor = materialize_tensor_read(buffers, "softmax", input)?;
            let data = tensor.as_slice::<f32>()?;
            fill_pooled(buffers, &shape, |out| {
                softmax_axis0_f32(data, axis_len, out)
            })
            .map(Some)
        }
        DType::F64 => {
            if let Ok(data) = input.as_slice::<f64>() {
                return fill_pooled(buffers, &shape, |out| {
                    softmax_axis0_f64(data, axis_len, out)
                })
                .map(Some);
            }
            let tensor = materialize_tensor_read(buffers, "softmax", input)?;
            let data = tensor.as_slice::<f64>()?;
            fill_pooled(buffers, &shape, |out| {
                softmax_axis0_f64(data, axis_len, out)
            })
            .map(Some)
        }
        _ => Ok(None),
    }
}

/// `exp` of `temp` into `out`, every element written; the scalar fallback
/// covers a dtype or host without the vector kernel.
fn exp_into<T: Copy + 'static + num_traits::Float>(temp: &[T], out: &mut [MaybeUninit<T>]) {
    // INVARIANT: `apply` writes every output element before returning true;
    // the fallback writes every element otherwise.
    if !apply(VectorKernel::Exp, temp, out) {
        for (slot, &value) in out.iter_mut().zip(temp) {
            slot.write(value.exp());
        }
    }
}

/// The initialized view of an output chunk that `exp_into` has filled.
///
/// # Safety
/// Every element of `chunk` must have been written.
unsafe fn initialized<T>(chunk: &mut [MaybeUninit<T>]) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut(chunk.as_mut_ptr().cast::<T>(), chunk.len()) }
}

/// Softmax over axis 0 of column-major `data`, written into `output`.
///
/// INVARIANT: every element of `output` is written (the tasks and groups
/// partition it, and each group is filled by `exp_into`).
fn softmax_axis0_f32(data: &[f32], axis_len: usize, output: &mut [MaybeUninit<f32>]) {
    let n = data.len();
    let group = group_len(n, axis_len, 4);
    let task = task_len(n, group, 4, MIN_BYTES_PER_TASK);
    output
        .par_chunks_mut(task)
        .zip(data.par_chunks(task))
        .for_each(|(out_task, in_task)| {
            // One cache-sized temp per task, reused across its groups.
            let mut temp = vec![0.0f32; group.min(in_task.len())];
            for (out_chunk, in_chunk) in out_task.chunks_mut(group).zip(in_task.chunks(group)) {
                let temp = &mut temp[..in_chunk.len()];
                if avx2_available() {
                    // SAFETY: the host reports AVX2; the group functions touch
                    // only the validated slices.
                    unsafe { shift_group_f32_avx2(in_chunk, temp, axis_len) };
                } else {
                    shift_group_f32(in_chunk, temp, axis_len);
                }
                // One vectorized exp per cache-sized group, not per slice.
                exp_into(temp, out_chunk);
                // SAFETY: `exp_into` wrote every element of the chunk.
                let out_chunk = unsafe { initialized(out_chunk) };
                if avx2_available() {
                    unsafe { scale_group_f32_avx2(out_chunk, axis_len) };
                } else {
                    scale_group_f32(out_chunk, axis_len);
                }
            }
        });
}

/// Softmax over axis 0 of column-major `data`, written into `output`.
///
/// INVARIANT: as [`softmax_axis0_f32`].
fn softmax_axis0_f64(data: &[f64], axis_len: usize, output: &mut [MaybeUninit<f64>]) {
    let n = data.len();
    let group = group_len(n, axis_len, 8);
    let task = task_len(n, group, 8, MIN_BYTES_PER_TASK);
    output
        .par_chunks_mut(task)
        .zip(data.par_chunks(task))
        .for_each(|(out_task, in_task)| {
            let mut temp = vec![0.0f64; group.min(in_task.len())];
            for (out_chunk, in_chunk) in out_task.chunks_mut(group).zip(in_task.chunks(group)) {
                let temp = &mut temp[..in_chunk.len()];
                shift_group_f64(in_chunk, temp, axis_len);
                exp_into(temp, out_chunk);
                // SAFETY: `exp_into` wrote every element of the chunk.
                let out_chunk = unsafe { initialized(out_chunk) };
                scale_group_f64(out_chunk, axis_len);
            }
        });
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

fn shift_group_f32(src: &[f32], dst: &mut [f32], axis_len: usize) {
    for (in_slice, out_slice) in src.chunks(axis_len).zip(dst.chunks_mut(axis_len)) {
        let max = in_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let safe_max = if max.is_nan() { max } else { max.max(f32::MIN) };
        for (out, &value) in out_slice.iter_mut().zip(in_slice) {
            *out = value - safe_max;
        }
    }
}

fn scale_group_f32(data: &mut [f32], axis_len: usize) {
    for slice in data.chunks_mut(axis_len) {
        let sum: f32 = slice.iter().copied().sum();
        let safe_sum = if sum.is_nan() {
            sum
        } else {
            sum.max(f32::MIN_POSITIVE)
        };
        for value in slice.iter_mut() {
            *value /= safe_sum;
        }
    }
}

fn shift_group_f64(src: &[f64], dst: &mut [f64], axis_len: usize) {
    for (in_slice, out_slice) in src.chunks(axis_len).zip(dst.chunks_mut(axis_len)) {
        let max = in_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let safe_max = if max.is_nan() { max } else { max.max(f64::MIN) };
        for (out, &value) in out_slice.iter_mut().zip(in_slice) {
            *out = value - safe_max;
        }
    }
}

fn scale_group_f64(data: &mut [f64], axis_len: usize) {
    for slice in data.chunks_mut(axis_len) {
        let sum: f64 = slice.iter().copied().sum();
        let safe_sum = if sum.is_nan() {
            sum
        } else {
            sum.max(f64::MIN_POSITIVE)
        };
        for value in slice.iter_mut() {
            *value /= safe_sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn shift_group_f32_avx2(src: &[f32], dst: &mut [f32], axis_len: usize) {
    use std::arch::x86_64::*;
    for (in_slice, out_slice) in src.chunks(axis_len).zip(dst.chunks_mut(axis_len)) {
        let n = in_slice.len();
        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 8 <= n {
            acc = _mm256_max_ps(acc, _mm256_loadu_ps(in_slice.as_ptr().add(i)));
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut max = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &value in &in_slice[i..] {
            max = max.max(value);
        }
        let safe_max = if max.is_nan() { max } else { max.max(f32::MIN) };
        let shift = _mm256_set1_ps(safe_max);
        let mut j = 0;
        while j + 8 <= n {
            let value = _mm256_loadu_ps(in_slice.as_ptr().add(j));
            _mm256_storeu_ps(out_slice.as_mut_ptr().add(j), _mm256_sub_ps(value, shift));
            j += 8;
        }
        for (out, &input) in out_slice[j..].iter_mut().zip(&in_slice[j..]) {
            *out = input - safe_max;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_group_f32_avx2(data: &mut [f32], axis_len: usize) {
    use std::arch::x86_64::*;
    for slice in data.chunks_mut(axis_len) {
        let n = slice.len();
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(slice.as_ptr().add(i)));
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum: f32 = lanes.iter().sum();
        for &value in &slice[i..] {
            sum += value;
        }
        let safe_sum = if sum.is_nan() {
            sum
        } else {
            sum.max(f32::MIN_POSITIVE)
        };
        let divisor = _mm256_set1_ps(safe_sum);
        let mut j = 0;
        while j + 8 <= n {
            let value = _mm256_loadu_ps(slice.as_ptr().add(j));
            _mm256_storeu_ps(slice.as_mut_ptr().add(j), _mm256_div_ps(value, divisor));
            j += 8;
        }
        for value in &mut slice[j..] {
            *value /= safe_sum;
        }
    }
}

#[cfg(test)]
mod tests;
