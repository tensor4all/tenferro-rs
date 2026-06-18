//! Traced tensor einsum operations.
//!
//! This module is the canonical traced tensor namespace for the einsum
//! extension crate.

use tenferro_runtime::error::Result;
use tenferro_runtime::TracedTensor;

use crate::TensorDotAxes;

pub use crate::traced::{einsum, einsum_subscripts, einsum_subscripts_with, einsum_with};

/// Build a traced NumPy-style tensor contraction from axis-pair sugar.
///
/// This helper lives in the einsum extension namespace because it is
/// contraction sugar over `dot_general`, not a linear algebra facade. It does
/// not require einsum runtime registration unless the surrounding graph also
/// contains einsum extension ops.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::TracedTensor;
/// use tenferro_einsum::{traced_tensor, TensorDotAxes};
///
/// let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let out = traced_tensor::tensordot(&lhs, &rhs, TensorDotAxes::Count(1)).unwrap();
///
/// assert_eq!(out.rank, 2);
/// ```
pub fn tensordot(
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    axes: TensorDotAxes<'_>,
) -> Result<TracedTensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.rank, rhs.rank)?;
    crate::tensordot::validate_traced_contract_dims(lhs, rhs, &config)?;
    lhs.dot_general(rhs, config)
}
