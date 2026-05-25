//! Traced tensor operations.
//!
//! This module is the public namespace for operations that build traced tensor
//! graphs. The operation names stay independent of execution mode; the module
//! name identifies the tensor family.

use crate::{DType, DotGeneralConfig};

pub use crate::traced::{TracedTensor, TracedTensorId};

/// Convert a traced tensor to a different dtype.
pub fn convert(input: &TracedTensor, to: DType) -> TracedTensor {
    input.convert(to)
}

/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let z = tenferro::traced_tensor::add(&x, &y);
/// ```
pub fn add(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    lhs.add(rhs)
}

/// Matrix multiplication helper for rank-2 traced tensors.
///
/// This contracts the last dimension of `a` with the first dimension of `b`.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
/// # let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
/// let c = tenferro::traced_tensor::matmul(&a, &b);
/// ```
pub fn matmul(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![a.rank - 1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    a.dot_general(b, config)
}

/// Elementwise power helper with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let base = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]);
/// # let exp = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 2.0]);
/// let y = tenferro::traced_tensor::pow(&base, &exp);
/// ```
pub fn pow(base: &TracedTensor, exp: &TracedTensor) -> TracedTensor {
    base.pow(exp)
}
