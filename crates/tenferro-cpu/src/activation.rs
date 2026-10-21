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

use rayon::prelude::*;
use std::mem::MaybeUninit;

use tenferro_cpu_basic::PooledUninitOutput;
use tenferro_tensor::{ActivationOp, DType, Tensor, TensorRead, TensorScalar};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::materialize_tensor_read;
use crate::parallel::{task_len, MIN_BYTES_PER_TASK};

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

/// Apply `kernel` over equal-length slices, split across the current pool
/// into tasks of whole chunks; small inputs stay on fewer workers.
fn apply_parallel<T: Copy + Send + Sync>(kernel: Kernel<T>, src: &[T], dst: &mut [MaybeUninit<T>]) {
    let n = src.len();
    let task = task_len(n, CHUNK.min(n.max(1)), size_of::<T>(), MIN_BYTES_PER_TASK);
    dst.par_chunks_mut(task)
        .zip(src.par_chunks(task))
        .for_each(|(out, input)| {
            // The chunks are built from one length, so the kernel's length
            // check cannot fail.
            kernel(input, out).expect("equal-length chunks");
        });
}

/// Evaluate `op` on a contiguous slice into a fresh pooled tensor.
fn fused_slice<T: Copy + Send + Sync + PoolScalar + TensorScalar>(
    buffers: &BufferPool,
    kernel: Kernel<T>,
    shape: &[usize],
    data: &[T],
) -> crate::Result<Tensor> {
    // INVARIANT: `apply_parallel` covers every element of the destination
    // exactly once; each kernel writes every slot on success.
    fill_pooled(buffers, shape, |out| apply_parallel(kernel, data, out))
}

/// Fused activation fast path for the CPU backend.
///
/// Returns `Ok(None)` for a dtype without a fused kernel so the caller keeps
/// the shared composite formulation. A strided or offset input is
/// materialized once, into the pool, before the single pass.
pub(crate) fn fused_activation_read_with_pool(
    buffers: &mut BufferPool,
    op: ActivationOp,
    input: TensorRead<'_>,
) -> crate::Result<Option<Tensor>> {
    let shape = input.shape().to_vec();
    match input.dtype() {
        DType::F32 => {
            if let Ok(data) = input.as_slice::<f32>() {
                return fused_slice(buffers, kernel_f32(op), &shape, data).map(Some);
            }
            let tensor = materialize_tensor_read(buffers, op.label(), input)?;
            fused_slice(buffers, kernel_f32(op), &shape, tensor.as_slice::<f32>()?).map(Some)
        }
        DType::F64 => {
            if let Ok(data) = input.as_slice::<f64>() {
                return fused_slice(buffers, kernel_f64(op), &shape, data).map(Some);
            }
            let tensor = materialize_tensor_read(buffers, op.label(), input)?;
            fused_slice(buffers, kernel_f64(op), &shape, tensor.as_slice::<f64>()?).map(Some)
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar f64 definitions, as `tenferro_runtime::composite` states them.
    fn reference(op: ActivationOp, x: f64) -> f64 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        match op {
            ActivationOp::Sigmoid => sigmoid,
            ActivationOp::Silu => x * sigmoid,
            ActivationOp::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
            ActivationOp::Gelu => 0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2)),
            ActivationOp::GeluTanh => {
                let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
        }
    }

    const OPS: [ActivationOp; 5] = [
        ActivationOp::Sigmoid,
        ActivationOp::Silu,
        ActivationOp::Softplus,
        ActivationOp::Gelu,
        ActivationOp::GeluTanh,
    ];

    #[test]
    fn every_op_matches_the_scalar_definition_in_both_dtypes() {
        // Enough elements for several parallel tasks and a ragged tail.
        let n = 3 * CHUNK + 37;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 16.0 - 8.0).collect();
        let mut pool = BufferPool::new();
        for op in OPS {
            let input = Tensor::from_vec_col_major(vec![n], xs.clone()).unwrap();
            let out =
                fused_activation_read_with_pool(&mut pool, op, TensorRead::from_tensor(&input))
                    .unwrap()
                    .expect("f64 has a fused kernel");
            for (&got, &x) in out.as_slice::<f64>().unwrap().iter().zip(&xs) {
                let want = reference(op, x);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "{op:?} f64 at {x}: got {got}, want {want}"
                );
            }
            let input32 =
                Tensor::from_vec_col_major(vec![n], xs.iter().map(|&v| v as f32).collect())
                    .unwrap();
            let out =
                fused_activation_read_with_pool(&mut pool, op, TensorRead::from_tensor(&input32))
                    .unwrap()
                    .expect("f32 has a fused kernel");
            for (&got, &x) in out.as_slice::<f32>().unwrap().iter().zip(&xs) {
                let want = reference(op, x as f32 as f64);
                assert!(
                    (got as f64 - want).abs() <= 2e-6 * want.abs().max(1.0),
                    "{op:?} f32 at {x}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn strided_input_is_materialized_once_and_integers_decline() {
        let mut pool = BufferPool::new();
        let base = tenferro_tensor::TypedTensor::<f32>::from_vec_col_major(
            vec![4, 3],
            (0..12).map(|i| i as f32 - 6.0).collect(),
        )
        .unwrap();
        let view = tenferro_tensor::TensorView::F32(base.as_view().transpose_view([1, 0]).unwrap());
        let out = fused_activation_read_with_pool(
            &mut pool,
            ActivationOp::Sigmoid,
            TensorRead::from_view(view),
        )
        .unwrap()
        .expect("f32 view");
        assert_eq!(out.shape(), &[3, 4]);
        let got = out.as_slice::<f32>().unwrap();
        for j in 0..4 {
            for i in 0..3 {
                let x = (i * 4 + j) as f32 - 6.0;
                let want = 1.0 / (1.0 + (-x).exp());
                assert!((got[i + j * 3] - want).abs() < 1e-6);
            }
        }
        let ints = Tensor::from_vec_col_major(vec![2], vec![1i32, 2]).unwrap();
        assert!(fused_activation_read_with_pool(
            &mut pool,
            ActivationOp::Silu,
            TensorRead::from_tensor(&ints)
        )
        .unwrap()
        .is_none());
    }
}
