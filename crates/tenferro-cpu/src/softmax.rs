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

use crate::buffer_pool::BufferPool;
use crate::materialize_tensor_read;
use crate::parallel::{group_len, task_len, MIN_BYTES_PER_TASK};
use crate::simd_math::{apply, VectorKernel};

/// Fused softmax over axis 0 for a host `F32`/`F64` tensor, if supported.
pub(crate) fn softmax_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axis: usize,
) -> crate::Result<Option<Tensor>> {
    let shape = input.shape();
    if axis != 0 || shape.is_empty() || shape[0] == 0 {
        return Ok(None);
    }
    match input.dtype() {
        DType::F32 => {
            if let Ok(data) = input.as_slice::<f32>() {
                let output = softmax_axis0_f32(data, shape[0]);
                return Ok(Some(Tensor::from_vec_col_major(shape.to_vec(), output)?));
            }
            let tensor = materialize_tensor_read(buffers, "softmax", input)?;
            softmax_tensor(&tensor, axis)
        }
        DType::F64 => {
            if let Ok(data) = input.as_slice::<f64>() {
                let output = softmax_axis0_f64(data, shape[0]);
                return Ok(Some(Tensor::from_vec_col_major(shape.to_vec(), output)?));
            }
            let tensor = materialize_tensor_read(buffers, "softmax", input)?;
            softmax_tensor(&tensor, axis)
        }
        _ => Ok(None),
    }
}

fn softmax_tensor(tensor: &Tensor, axis: usize) -> crate::Result<Option<Tensor>> {
    let shape = tensor.shape();
    if axis != 0 || shape.is_empty() || shape[0] == 0 {
        return Ok(None);
    }
    let axis_len = shape[0];
    let output = match tensor.dtype() {
        DType::F32 => Tensor::from_vec_col_major(
            shape.to_vec(),
            softmax_axis0_f32(tensor.as_slice::<f32>()?, axis_len),
        )?,
        DType::F64 => Tensor::from_vec_col_major(
            shape.to_vec(),
            softmax_axis0_f64(tensor.as_slice::<f64>()?, axis_len),
        )?,
        _ => return Ok(None),
    };
    Ok(Some(output))
}

fn softmax_axis0_f32(data: &[f32], axis_len: usize) -> Vec<f32> {
    let n = data.len();
    let mut output = vec![0.0f32; n];
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
                // INVARIANT: `apply` writes every output element before
                // returning true; the fallback writes every element otherwise.
                let uninit = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_chunk.as_mut_ptr().cast::<MaybeUninit<f32>>(),
                        out_chunk.len(),
                    )
                };
                if !apply(VectorKernel::Exp, temp, uninit) {
                    for (out, &value) in out_chunk.iter_mut().zip(&*temp) {
                        *out = value.exp();
                    }
                }
                if avx2_available() {
                    unsafe { scale_group_f32_avx2(out_chunk, axis_len) };
                } else {
                    scale_group_f32(out_chunk, axis_len);
                }
            }
        });
    output
}

fn softmax_axis0_f64(data: &[f64], axis_len: usize) -> Vec<f64> {
    let n = data.len();
    let mut output = vec![0.0f64; n];
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
                let uninit = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_chunk.as_mut_ptr().cast::<MaybeUninit<f64>>(),
                        out_chunk.len(),
                    )
                };
                if !apply(VectorKernel::Exp, temp, uninit) {
                    for (out, &value) in out_chunk.iter_mut().zip(&*temp) {
                        *out = value.exp();
                    }
                }
                scale_group_f64(out_chunk, axis_len);
            }
        });
    output
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
mod tests {
    use super::*;

    #[test]
    fn softmax_tensor_covers_the_f32_and_f64_paths() {
        let f32_tensor =
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
        let output = softmax_tensor(&f32_tensor, 0).unwrap().unwrap();
        for column in output.as_slice::<f32>().unwrap().chunks(3) {
            let sum: f32 = column.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        let f64_tensor =
            Tensor::from_vec_col_major(vec![3, 2], vec![1.0f64, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
        let output = softmax_tensor(&f64_tensor, 0).unwrap().unwrap();
        for column in output.as_slice::<f64>().unwrap().chunks(3) {
            let sum: f64 = column.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn scalar_group_functions_match_the_reference() {
        let src = [1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0];
        let mut shifted = [0.0f32; 6];
        shift_group_f32(&src, &mut shifted, 3);
        assert_eq!(shifted, [-2.0, -1.0, 0.0, -2.0, -1.0, 0.0]);
        for value in shifted.iter_mut() {
            *value = value.exp();
        }
        scale_group_f32(&mut shifted, 3);
        for column in shifted.chunks(3) {
            let sum: f32 = column.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        let src = [1.0f64, 2.0, 3.0, -1.0, 0.0, 1.0];
        let mut shifted = [0.0f64; 6];
        shift_group_f64(&src, &mut shifted, 3);
        assert_eq!(shifted, [-2.0, -1.0, 0.0, -2.0, -1.0, 0.0]);
        for value in shifted.iter_mut() {
            *value = value.exp();
        }
        scale_group_f64(&mut shifted, 3);
        for column in shifted.chunks(3) {
            let sum: f64 = column.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_group_functions_match_the_scalar_reference() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let src = [1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0];
        let mut scalar = [0.0f32; 6];
        shift_group_f32(&src, &mut scalar, 3);
        let mut vector = [0.0f32; 6];
        // SAFETY: guarded by the runtime AVX2 check above.
        unsafe { shift_group_f32_avx2(&src, &mut vector, 3) };
        assert_eq!(scalar, vector);

        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut b = a;
        // SAFETY: guarded by the runtime AVX2 check above.
        unsafe { scale_group_f32_avx2(&mut a, 3) };
        scale_group_f32(&mut b, 3);
        assert_eq!(a, b);
    }
}
