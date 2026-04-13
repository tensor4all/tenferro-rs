use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::TensorBackend;

use crate::eager::EagerTensor;
use crate::error::Result;

/// Execute an einsum eagerly and record it for reverse-mode autodiff.
///
/// # Examples
///
/// ```
/// use tenferro::eager_einsum::eager_einsum_ad;
/// use tenferro::{EagerTensor, Tensor};
///
/// let a = EagerTensor::from_tensor(Tensor::new(
///     vec![2, 3],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ));
/// let b = EagerTensor::from_tensor(Tensor::new(
///     vec![3, 2],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ));
/// let c = eager_einsum_ad(&[&a, &b], "ij,jk->ik").unwrap();
///
/// assert_eq!(c.data().shape(), &[2, 2]);
/// assert_eq!(c.data().as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
/// ```
pub fn eager_einsum_ad<B: TensorBackend>(
    inputs: &[&EagerTensor<B>],
    subscripts: &str,
) -> Result<EagerTensor<B>> {
    EagerTensor::nary_op(
        inputs,
        StdTensorOp::NaryEinsum {
            subscripts: subscripts.to_string(),
            n_inputs: inputs.len(),
        },
    )
}
