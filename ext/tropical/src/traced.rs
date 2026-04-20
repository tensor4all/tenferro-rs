//! Traced tropical operations expressed as compositions of core `tenferro`
//! primitives.
//!
//! These wrappers accept `TracedTensor` arguments with **concrete shapes**
//! (Stage 4a scope). They lower tropical semantics into `BroadcastInDim +
//! Add + ReduceMax` (or `ReduceMin`) chains — no new core ops, no new AD
//! rules. Automatic differentiation flows through the core AD rules of
//! the underlying primitives.
//!
//! # Stage boundaries
//!
//! - Stage 4a (this module): concrete-shape compositions via the public
//!   `tenferro` facade.
//! - Stage 4b: same composition extended to symbolic-shape inputs.
//! - Stage 7: `FusedTropicalDotGeneral` as an `ExtensionOp` with argmax-
//!   based AD — not implemented here.
//!
//! # Examples
//!
//! ```
//! use tenferro::{CpuBackend, Engine, TracedTensor};
//! use tenferro_ext_tropical::traced::tropical_dot_general;
//!
//! let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
//! let b = TracedTensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
//! let mut c = tropical_dot_general(&a, &b);
//!
//! let mut engine = Engine::new(CpuBackend::new());
//! let out = c.eval(&mut engine).unwrap();
//! assert_eq!(out.shape(), &[2, 2]);
//! ```

use tenferro::TracedTensor;

/// Shape helper: panics with a clear message if `t` does not have a
/// fully-concrete rank-2 shape.
fn require_rank2_concrete(t: &TracedTensor, label: &str) -> (usize, usize) {
    assert_eq!(
        t.rank, 2,
        "{label}: tropical Stage 4a requires rank-2 inputs, got rank {}",
        t.rank
    );
    let shape = t.try_concrete_shape().unwrap_or_else(|| {
        panic!(
            "{label}: tropical Stage 4a requires concrete shapes; \
             symbolic inputs are Stage 4b"
        )
    });
    (shape[0], shape[1])
}

/// Max-plus matrix multiplication on rank-2 concrete-shape inputs.
///
/// Computes `out[i, j] = max_k (a[i, k] + b[k, j])` by lowering to:
///
/// ```text
/// BroadcastInDim(a, [M,K,N], [0,1])   // a -> [M,K,1] -> [M,K,N]
/// BroadcastInDim(b, [M,K,N], [1,2])   // b -> [1,K,N] -> [M,K,N]
/// Add                                 // elementwise sum
/// ReduceMax(axes=[1])                 // reduce K axis
/// ```
///
/// AD flows via the core AD rules of `BroadcastInDim + Add + ReduceMax` —
/// no new AD rule is required.
///
/// # Panics
///
/// Panics if `a` or `b` is not rank-2 or does not have a concrete shape,
/// or if the contraction dimensions do not match (`a.shape[1] != b.shape[0]`).
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, Engine, TracedTensor};
/// use tenferro_ext_tropical::traced::tropical_dot_general;
///
/// let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
/// let mut c = tropical_dot_general(&a, &b);
/// let mut engine = Engine::new(CpuBackend::new());
/// let _ = c.eval(&mut engine).unwrap();
/// ```
pub fn tropical_dot_general(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let (m, k_a) = require_rank2_concrete(a, "tropical_dot_general.a");
    let (k_b, n) = require_rank2_concrete(b, "tropical_dot_general.b");
    assert_eq!(
        k_a, k_b,
        "tropical_dot_general: contraction dims disagree (a.shape[1]={k_a}, b.shape[0]={k_b})"
    );
    let k = k_a;

    let a_b = a.broadcast_in_dim(&[m, k, n], &[0, 1]);
    let b_b = b.broadcast_in_dim(&[m, k, n], &[1, 2]);
    let sum = a_b.add(&b_b);
    sum.reduce_max(&[1])
}

/// Min-plus matrix multiplication on rank-2 concrete-shape inputs.
///
/// Computes `out[i, j] = min_k (a[i, k] + b[k, j])` by lowering to:
///
/// ```text
/// BroadcastInDim(a, [M,K,N], [0,1])
/// BroadcastInDim(b, [M,K,N], [1,2])
/// Add
/// ReduceMin(axes=[1])
/// ```
///
/// AD flows via the core AD rules of `BroadcastInDim + Add + ReduceMin` —
/// no new AD rule is required.
///
/// # Panics
///
/// Panics if `a` or `b` is not rank-2 or does not have a concrete shape,
/// or if the contraction dimensions do not match (`a.shape[1] != b.shape[0]`).
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, Engine, TracedTensor};
/// use tenferro_ext_tropical::traced::min_plus_dot_general;
///
/// let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
/// let mut c = min_plus_dot_general(&a, &b);
/// let mut engine = Engine::new(CpuBackend::new());
/// let _ = c.eval(&mut engine).unwrap();
/// ```
pub fn min_plus_dot_general(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let (m, k_a) = require_rank2_concrete(a, "min_plus_dot_general.a");
    let (k_b, n) = require_rank2_concrete(b, "min_plus_dot_general.b");
    assert_eq!(
        k_a, k_b,
        "min_plus_dot_general: contraction dims disagree (a.shape[1]={k_a}, b.shape[0]={k_b})"
    );
    let k = k_a;

    let a_b = a.broadcast_in_dim(&[m, k, n], &[0, 1]);
    let b_b = b.broadcast_in_dim(&[m, k, n], &[1, 2]);
    let sum = a_b.add(&b_b);
    sum.reduce_min(&[1])
}

/// Tropical (max-plus) reduction over the given axes.
///
/// In max-plus semiring, "reduction under ⊕" is just `max`. This wrapper
/// exists so user code can express intent and stay algebra-consistent with
/// the rest of the tropical surface.
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, Engine, TracedTensor};
/// use tenferro_ext_tropical::traced::tropical_reduce_sum;
///
/// let x = TracedTensor::from_vec(vec![3], vec![1.0_f64, 5.0, 2.0]);
/// let mut y = tropical_reduce_sum(&x, &[0]);
/// let mut engine = Engine::new(CpuBackend::new());
/// let out = y.eval(&mut engine).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
/// ```
pub fn tropical_reduce_sum(a: &TracedTensor, axes: &[usize]) -> TracedTensor {
    a.reduce_max(axes)
}
