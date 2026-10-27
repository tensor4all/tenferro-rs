//! Fused CPU softmax: a PyTorch-style per-slice kernel, over a host tensor.
//!
//! The fused pipeline itself — per-slice max and shift, `exp` written straight
//! into the output, sum of the output, in-place scale, the AVX2 `f32`
//! reductions and the per-task scratch reuse — lives in `strided-nn`
//! ([#2272](https://github.com/tensor4all/tenferro-rs/issues/2272)). This module
//! owns only the tensor boundary: axis, dtype and empty-shape decisions, the
//! materialized input, the pooled output and the caller-sized scratch.
//!
//! The kernel's fan-out goes through the operation's placement token, so a serial
//! engine stays on the calling thread and a pooled engine stays inside its own
//! pool and thread budget (#2206, #2273).

use strided_nn::{scratch_len, softmax_axis0_f32, softmax_axis0_f64};
use tenferro_tensor::{DType, Tensor, TensorRead};

use crate::activation::fill_pooled;
use crate::buffer_pool::BufferPool;
use crate::materialize_tensor_read;
use threadteam::Exec;

/// Fused softmax over axis 0 for a host `F32`/`F64` tensor, if supported.
///
/// The output is a pooled, recycled tensor: a caller that holds many outputs
/// and drops them together (a benchmark, a batch loop) gets its storage back
/// from the pool instead of fresh pages from the allocator.
pub(crate) fn softmax_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    axis: usize,
    mut exec: Exec<'_>,
) -> crate::Result<Option<Tensor>> {
    let shape = input.shape().to_vec();
    if axis != 0 || shape.is_empty() || shape[0] == 0 {
        return Ok(None);
    }
    let workers = exec.budget().get();
    match input.dtype() {
        DType::F32 => {
            if let Ok(data) = input.as_slice::<f32>() {
                return softmax_f32(workers, buffers, &shape, data, exec).map(Some);
            }
            let tensor = materialize_tensor_read(buffers, "softmax", input, exec.reborrow())?;
            softmax_f32(workers, buffers, &shape, tensor.as_slice::<f32>()?, exec).map(Some)
        }
        DType::F64 => {
            if let Ok(data) = input.as_slice::<f64>() {
                return softmax_f64(workers, buffers, &shape, data, exec).map(Some);
            }
            let tensor = materialize_tensor_read(buffers, "softmax", input, exec.reborrow())?;
            softmax_f64(workers, buffers, &shape, tensor.as_slice::<f64>()?, exec).map(Some)
        }
        _ => Ok(None),
    }
}

/// Run the `f32` kernel into a pooled output of `shape`.
///
/// INVARIANT: the kernel writes every element of the output span, and its
/// contract holds here because `shape[0]` divides `data.len()` (a tensor shape),
/// the output has the same length, and the scratch is sized for the engine's
/// worker count.
fn softmax_f32(
    workers: usize,
    buffers: &BufferPool,
    shape: &[usize],
    data: &[f32],
    exec: Exec<'_>,
) -> crate::Result<Tensor> {
    fill_pooled(buffers, shape, |out| {
        // An empty input needs no scratch; sizing it from a huge `shape[0]`
        // would overflow the element count.
        let scratch_len = if data.is_empty() {
            0
        } else {
            scratch_len::<f32>(shape[0], data.len(), workers)
        };
        let mut scratch = vec![0.0f32; scratch_len];
        softmax_axis0_f32(shape[0], data, out, &mut scratch, exec)
            .expect("validated softmax contract");
    })
}

/// Run the `f64` kernel into a pooled output of `shape`.
///
/// INVARIANT: as [`softmax_f32`].
fn softmax_f64(
    workers: usize,
    buffers: &BufferPool,
    shape: &[usize],
    data: &[f64],
    exec: Exec<'_>,
) -> crate::Result<Tensor> {
    fill_pooled(buffers, shape, |out| {
        // An empty input needs no scratch; sizing it from a huge `shape[0]`
        // would overflow the element count.
        let scratch_len = if data.is_empty() {
            0
        } else {
            scratch_len::<f64>(shape[0], data.len(), workers)
        };
        let mut scratch = vec![0.0f64; scratch_len];
        softmax_axis0_f64(shape[0], data, out, &mut scratch, exec)
            .expect("validated softmax contract");
    })
}

#[cfg(test)]
mod tests;
