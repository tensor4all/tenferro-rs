//! Fused real-float activations: one SIMD pass per element.
//!
//! `elem-simd-math` evaluates `sigmoid`, `silu`, `softplus`, `gelu` and
//! `gelu_tanh` in one vector pass each, the way PyTorch's CPU `Vectorized`
//! activations do, so no intermediate is materialized. This module owns only
//! the tensor boundary: dtype dispatch, the pooled output and the parallel
//! split from [`crate::parallel`]; the arithmetic lives in the math crate.
//!
//! The module is private: it backs the optional
//! [`tenferro_tensor::Engine::fused_activation_read`] fast path only.

use std::mem::MaybeUninit;

use tenferro_cpu_basic::PooledUninitOutput;
use tenferro_tensor::{ActivationOp, DType, Tensor, TensorRead, TensorScalar};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::materialize_tensor_read;
use crate::parallel::FanOut;

/// Elements per unit of work handed to one worker; a multiple of every SIMD
/// width, so no task ends on a partial vector except the last.
const CHUNK: usize = 2048;

/// Allocate a pooled, compact output of `shape`, let `fill` write every
/// element, and hand it back as a recycled tensor.
///
/// INVARIANT: `fill` must initialize every element of the slice it receives.
pub(crate) fn fill_pooled<T: PoolScalar + TensorScalar>(
    buffers: &BufferPool,
    shape: &[usize],
    fill: impl FnOnce(&mut [MaybeUninit<T>]),
) -> crate::Result<Tensor> {
    let mut out = PooledUninitOutput::<T>::new(buffers, shape.to_vec())?;
    fill(out.as_uninit_slice_mut());
    // SAFETY: `fill` initializes every element (the invariant above).
    unsafe { out.assume_init_recycled() }.map(Tensor::from_typed::<T>)
}

/// One slice kernel of `elem-simd-math` for a scalar type.
type Kernel<T> = fn(&[T], &mut [MaybeUninit<T>]) -> Result<(), elem_simd_math::LengthMismatch>;

/// The fused kernel for `op` over the `f32` family.
fn kernel_f32(op: ActivationOp) -> Kernel<f32> {
    use elem_simd_math::f32 as k;
    match op {
        ActivationOp::Sigmoid => k::sigmoid,
        ActivationOp::Silu => k::silu,
        ActivationOp::Softplus => k::softplus,
        ActivationOp::Gelu => k::gelu,
        ActivationOp::GeluTanh => k::gelu_tanh,
    }
}

/// The fused kernel for `op` over the `f64` family.
fn kernel_f64(op: ActivationOp) -> Kernel<f64> {
    use elem_simd_math::f64 as k;
    match op {
        ActivationOp::Sigmoid => k::sigmoid,
        ActivationOp::Silu => k::silu,
        ActivationOp::Softplus => k::softplus,
        ActivationOp::Gelu => k::gelu,
        ActivationOp::GeluTanh => k::gelu_tanh,
    }
}

/// Apply `kernel` over equal-length slices, split across the engine's pool into
/// tasks of whole chunks; small inputs stay on fewer workers and a serial engine
/// stays on the calling thread.
fn apply_parallel<T: Copy + Send + Sync>(
    fan: FanOut<'_>,
    kernel: Kernel<T>,
    src: &[T],
    dst: &mut [MaybeUninit<T>],
) {
    let n = src.len();
    let task = fan.task_len(n, CHUNK.min(n.max(1)), size_of::<T>());
    fan.run(dst, src, task, |out, input| {
        // The chunks are built from one length, so the kernel's length check
        // cannot fail.
        kernel(input, out).expect("equal-length chunks");
    });
}

/// Evaluate `op` on a contiguous slice into a fresh pooled tensor.
fn fused_slice<T: Copy + Send + Sync + PoolScalar + TensorScalar>(
    fan: FanOut<'_>,
    buffers: &BufferPool,
    kernel: Kernel<T>,
    shape: &[usize],
    data: &[T],
) -> crate::Result<Tensor> {
    // INVARIANT: `apply_parallel` covers every element of the destination
    // exactly once; each kernel writes every slot on success.
    fill_pooled(buffers, shape, |out| apply_parallel(fan, kernel, data, out))
}

/// Fused activation fast path for the CPU backend.
///
/// Returns `Ok(None)` for a dtype without a fused kernel so the caller keeps
/// the shared composite formulation. A strided or offset input is
/// materialized once, into the pool, before the single pass.
pub(crate) fn fused_activation_read_with_pool(
    fan: FanOut<'_>,
    buffers: &mut BufferPool,
    op: ActivationOp,
    input: TensorRead<'_>,
) -> crate::Result<Option<Tensor>> {
    let shape = input.shape().to_vec();
    match input.dtype() {
        DType::F32 => {
            if let Ok(data) = input.as_slice::<f32>() {
                return fused_slice(fan, buffers, kernel_f32(op), &shape, data).map(Some);
            }
            let tensor = materialize_tensor_read(buffers, op.label(), input)?;
            fused_slice(
                fan,
                buffers,
                kernel_f32(op),
                &shape,
                tensor.as_slice::<f32>()?,
            )
            .map(Some)
        }
        DType::F64 => {
            if let Ok(data) = input.as_slice::<f64>() {
                return fused_slice(fan, buffers, kernel_f64(op), &shape, data).map(Some);
            }
            let tensor = materialize_tensor_read(buffers, op.label(), input)?;
            fused_slice(
                fan,
                buffers,
                kernel_f64(op),
                &shape,
                tensor.as_slice::<f64>()?,
            )
            .map(Some)
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests;
