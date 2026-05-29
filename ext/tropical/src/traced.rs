//! Traced tropical composition helpers.
//!
//! These helpers lower tropical operations to existing tenferro traced tensor
//! operations. They do not register a fused extension runtime and do not add
//! tropical-specific AD rules.
//!
//! # Examples
//!
//! ```
//! use tenferro_cpu::CpuBackend;
//! use tenferro_ext_tropical::traced::tropical_reduce_sum;
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 5.0, 2.0]);
//! let y = tropical_reduce_sum(&x, &[0]);
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let mut executor = GraphExecutor::new(CpuBackend::new());
//! let out = executor.run(&program).unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
//! ```

use tenferro_runtime::TracedTensor;

fn require_rank2(tensor: &TracedTensor, label: &str) {
    assert_eq!(
        tensor.rank, 2,
        "{label}: tropical matrix composition requires rank-2 input, got rank {}",
        tensor.rank
    );
}

fn assert_contracting_axes_compatible(a: &TracedTensor, b: &TracedTensor, label: &str) {
    if let (Some(a_shape), Some(b_shape)) = (a.try_concrete_shape(), b.try_concrete_shape()) {
        assert_eq!(
            a_shape[1], b_shape[0],
            "{label}: contracting axes must match, got a[1]={} and b[0]={}",
            a_shape[1], b_shape[0]
        );
    }
}

fn tropical_dot_general_impl(
    a: &TracedTensor,
    b: &TracedTensor,
    reduce: impl FnOnce(&TracedTensor) -> TracedTensor,
    label: &str,
) -> TracedTensor {
    require_rank2(a, &format!("{label}.a"));
    require_rank2(b, &format!("{label}.b"));
    assert_contracting_axes_compatible(a, b, label);

    let m = a.axis_sym_dim(0);
    let k = a.axis_sym_dim(1);
    let n = b.axis_sym_dim(1);
    let target_shape = [m, k, n];

    let a_b = a.broadcast_in_dim_sym(&target_shape, &[0, 1], &[b]);
    let b_b = b.broadcast_in_dim_sym(&target_shape, &[1, 2], &[a]);
    let sum = a_b.add(&b_b);
    reduce(&sum)
}

/// Max-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = max_k(a[i, k] + b[k, j])` by composing
/// `BroadcastInDim`, `Add`, and `ReduceMax` graph operations.
///
/// # Panics
///
/// Panics if either input is not rank 2. If both input shapes are concrete,
/// also panics when the contracting axes differ.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::tropical_dot_general;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
/// let c = tropical_dot_general(&a, &b);
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&c).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[23.0, 24.0, 43.0, 44.0]);
/// ```
#[must_use]
pub fn tropical_dot_general(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    tropical_dot_general_impl(a, b, |sum| sum.reduce_max(&[1]), "tropical_dot_general")
}

/// Min-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = min_k(a[i, k] + b[k, j])` by composing
/// `BroadcastInDim`, `Add`, and `ReduceMin` graph operations.
///
/// # Panics
///
/// Panics if either input is not rank 2. If both input shapes are concrete,
/// also panics when the contracting axes differ.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::min_plus_dot_general;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
/// let c = min_plus_dot_general(&a, &b);
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&c).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[6.0, 7.0, 8.0, 9.0]);
/// ```
#[must_use]
pub fn min_plus_dot_general(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    tropical_dot_general_impl(a, b, |sum| sum.reduce_min(&[1]), "min_plus_dot_general")
}

/// Tropical max-plus reduction over the given axes.
///
/// In max-plus algebra, reduction under semiring addition is ordinary maximum.
/// This wrapper exists so callers can express tropical intent while still
/// lowering to tenferro's core `ReduceMax` operation.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::tropical_reduce_sum;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 5.0, 2.0]);
/// let y = tropical_reduce_sum(&x, &[0]);
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
/// ```
#[must_use]
pub fn tropical_reduce_sum(a: &TracedTensor, axes: &[usize]) -> TracedTensor {
    a.reduce_max(axes)
}
