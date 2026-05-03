//! Traced tropical operations.
//!
//! This module ships two flavours of tropical matmul:
//!
//! - **Composition wrappers** — [`tropical_dot_general`],
//!   [`min_plus_dot_general`], [`tropical_reduce_sum`]. Lower directly to
//!   `BroadcastInDim + Add + ReduceMax` (or `ReduceMin`). No new core ops,
//!   no new AD rules; AD flows through the existing core rules.
//! - **Fused `ExtensionOp` wrappers** — [`tropical_dot_general_fused`],
//!   [`min_plus_dot_general_fused`]. Package
//!   [`crate::fused::FusedTropicalDotGeneralOp`] into the traced graph via
//!   the `ExtensionOp` carrier. AD is emitted inside the
//!   `ExtensionOp::linearize` / `transpose_rule` methods, still targeting
//!   only core `StdTensorOp` variants so the AD closure rule holds.
//!
//! The composition and fused paths compute the **same mathematical
//! function** (and the same AD gradient modulo tie-breaking) for the
//! rank-2 cases the fused op currently covers. They exist side-by-side so the
//! fused path can be benchmarked against composition without changing the
//! composition wrappers.
//!
//! Both paths accept `TracedTensor` arguments with either **concrete
//! shapes** or **symbolic shapes**. The composition
//! wrappers derive axis sizes via [`TracedTensor::axis_sym_dim`] and feed
//! them to [`TracedTensor::broadcast_in_dim_sym`]; the fused wrappers
//! route through `tenferro::extension::apply`, whose `infer_output_meta`
//! returns [`tenferro_ops::SymDim`] shapes that remain valid under either
//! regime.
//!
//! # Boundaries
//!
//! - Composition wrappers use only the public `tenferro` facade.
//! - Fused wrappers use `FusedTropicalDotGeneral` as an `ExtensionOp` with
//!   indicator-based AD, governed by `docs/spec/extension-op.md`.
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

use std::sync::Arc;

use tenferro::extension::apply;
use tenferro::TracedTensor;
use tenferro_ops::ext_op::ExtensionOp;

use crate::fused::{register_fused_tropical, FusedTropicalDotGeneralOp, TropicalKind};

/// Shape helper: panics with a clear message if `t` is not rank-2.
fn require_rank2(t: &TracedTensor, label: &str) {
    assert_eq!(
        t.rank, 2,
        "{label}: tropical composition requires rank-2 inputs, got rank {}",
        t.rank
    );
}

/// Max-plus matrix multiplication on rank-2 inputs (concrete or symbolic
/// shape).
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
/// The `M`, `K`, `N` target-shape axes are derived as
/// [`TracedTensor::axis_sym_dim`]s so the same code path works for
/// concrete shapes (where the returned `SymDim`s are
/// `SymDim::Concrete(n)`) and for symbolic shapes (where they are
/// symbolic tensor-axis references). When broadcasting `a`, the axis
/// `N` coming from `b` is supplied as an auxiliary shape-reference
/// input; likewise when broadcasting `b`, the axis `M` coming from `a`
/// is supplied as a reference.
///
/// AD flows via the core AD rules of `BroadcastInDim + Add + ReduceMax` —
/// no new AD rule is required.
///
/// # Panics
///
/// Panics if `a` or `b` is not rank-2.
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
    require_rank2(a, "tropical_dot_general.a");
    require_rank2(b, "tropical_dot_general.b");

    let m = a.axis_sym_dim(0);
    let k = a.axis_sym_dim(1);
    let n = b.axis_sym_dim(1);
    let target_shape = [m, k, n];

    let a_b = a.broadcast_in_dim_sym(&target_shape, &[0, 1], &[b]);
    let b_b = b.broadcast_in_dim_sym(&target_shape, &[1, 2], &[a]);
    let sum = a_b.add(&b_b);
    sum.reduce_max(&[1])
}

/// Min-plus matrix multiplication on rank-2 inputs (concrete or symbolic
/// shape).
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
/// Handles concrete and symbolic inputs uniformly; see
/// [`tropical_dot_general`] for the full explanation.
///
/// AD flows via the core AD rules of `BroadcastInDim + Add + ReduceMin` —
/// no new AD rule is required.
///
/// # Panics
///
/// Panics if `a` or `b` is not rank-2.
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
    require_rank2(a, "min_plus_dot_general.a");
    require_rank2(b, "min_plus_dot_general.b");

    let m = a.axis_sym_dim(0);
    let k = a.axis_sym_dim(1);
    let n = b.axis_sym_dim(1);
    let target_shape = [m, k, n];

    let a_b = a.broadcast_in_dim_sym(&target_shape, &[0, 1], &[b]);
    let b_b = b.broadcast_in_dim_sym(&target_shape, &[1, 2], &[a]);
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

// ---------------------------------------------------------------------------
// Fused `ExtensionOp` wrappers.
// ---------------------------------------------------------------------------

/// Max-plus matrix multiplication via the fused `ExtensionOp`.
///
/// Semantically identical to [`tropical_dot_general`]: `out[i, j] = max_k
/// (a[i, k] + b[k, j])`. The difference is how the computation reaches
/// the execution backend:
///
/// - [`tropical_dot_general`] lowers to `BroadcastInDim + Add + ReduceMax`
///   at graph-build time.
/// - [`tropical_dot_general_fused`] packages a
///   [`crate::fused::FusedTropicalDotGeneralOp`] as an `ExtensionOp`, so
///   the graph carries a single `StdTensorOp::Extension` node that the
///   eager and compiled paths route to the fused primal kernel. AD still
///   emits only core ops (indicator-based VJP/JVP; see the module docs
///   on `crate::fused`).
///
/// Automatic registration: this function calls
/// [`register_fused_tropical`] on first invocation so callers can build
/// traced graphs without manually bootstrapping the registry. Tests that
/// exercise registry behaviour directly should still call
/// `register_fused_tropical` explicitly before any fused op is built.
///
/// # Panics
///
/// Panics if `a` or `b` is not rank-2.
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, Engine, TracedTensor};
/// use tenferro_ext_tropical::traced::tropical_dot_general_fused;
///
/// let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
/// let mut c = tropical_dot_general_fused(&a, &b);
/// let mut engine = Engine::new(CpuBackend::new());
/// let out = c.eval(&mut engine).unwrap();
/// assert_eq!(out.shape(), &[2, 2]);
/// ```
pub fn tropical_dot_general_fused(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    require_rank2(a, "tropical_dot_general_fused.a");
    require_rank2(b, "tropical_dot_general_fused.b");
    register_fused_tropical().expect("register_fused_tropical");
    let op: Arc<dyn ExtensionOp> = Arc::new(FusedTropicalDotGeneralOp::new(TropicalKind::MaxPlus));
    let mut outputs = apply(op, &[a, b]);
    assert_eq!(outputs.len(), 1);
    outputs.remove(0)
}

/// Min-plus matrix multiplication via the fused `ExtensionOp`.
///
/// Semantically identical to [`min_plus_dot_general`]; the packaging and
/// AD path is the fused one — see
/// [`tropical_dot_general_fused`] for the full explanation.
///
/// # Panics
///
/// Panics if `a` or `b` is not rank-2.
///
/// # Examples
///
/// ```
/// use tenferro::{CpuBackend, Engine, TracedTensor};
/// use tenferro_ext_tropical::traced::min_plus_dot_general_fused;
///
/// let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
/// let mut c = min_plus_dot_general_fused(&a, &b);
/// let mut engine = Engine::new(CpuBackend::new());
/// let _ = c.eval(&mut engine).unwrap();
/// ```
pub fn min_plus_dot_general_fused(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    require_rank2(a, "min_plus_dot_general_fused.a");
    require_rank2(b, "min_plus_dot_general_fused.b");
    register_fused_tropical().expect("register_fused_tropical");
    let op: Arc<dyn ExtensionOp> = Arc::new(FusedTropicalDotGeneralOp::new(TropicalKind::MinPlus));
    let mut outputs = apply(op, &[a, b]);
    assert_eq!(outputs.len(), 1);
    outputs.remove(0)
}
