//! Concrete-tensor metadata conversion for the runtime boundary.
//!
//! Metadata inference itself lives in the semantic program builder, which infers
//! and stores a value's metadata when the operation is appended. This module only
//! converts a concrete tensor into the runtime metadata form.

use tenferro_ops::ad::context::TensorMeta;
use tenferro_ops::sym_dim::SymDim;
use tenferro_tensor::DType;
use tenferro_tensor::Tensor;

pub(crate) fn tensor_meta(dtype: DType, shape: Vec<SymDim>) -> TensorMeta {
    TensorMeta::exact(dtype, shape)
}

pub(crate) fn concrete_tensor_meta(dtype: DType, shape: &[usize]) -> TensorMeta {
    tensor_meta(dtype, shape.iter().copied().map(SymDim::from).collect())
}

/// Convert a tensor's concrete metadata into the runtime metadata form.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::ad_support::tensor_meta_from_tensor;
/// use tenferro_tensor::Tensor;
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// assert_eq!(tensor_meta_from_tensor(&tensor).rank(), 1);
/// ```
pub fn tensor_meta_from_tensor(tensor: &Tensor) -> TensorMeta {
    concrete_tensor_meta(tensor.dtype(), tensor.shape())
}
