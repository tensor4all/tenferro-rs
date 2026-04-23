//! Eager einsum helpers exposed through the `tenferro` facade.
//!
//! This module provides both immediate eager execution on concrete tensors and
//! eager reverse-mode autodiff on tracked tensors.
//!
//! # Examples
//!
//! ```rust
//! use tenferro::eager_einsum::{eager_einsum, eager_einsum_ad, eager_einsum_owned};
//! use tenferro::{CpuBackend, EagerTensor, Tensor};
//!
//! let mut backend = CpuBackend::new();
//! let a = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
//! let b = Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);
//! let dot = eager_einsum(&mut backend, &[&a, &b], "i,i->").unwrap();
//!
//! assert_eq!(dot.as_slice::<f64>().unwrap(), &[32.0]);
//!
//! let owned_dot = eager_einsum_owned(&mut backend, vec![a, b], "i,i->").unwrap();
//!
//! assert_eq!(owned_dot.as_slice::<f64>().unwrap(), &[32.0]);
//!
//! let x = EagerTensor::requires_grad(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]));
//! let y = EagerTensor::requires_grad(Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]));
//! let loss = eager_einsum_ad(&[&x, &y], "i,i->").unwrap();
//! let _ = loss.backward().unwrap();
//!
//! assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 5.0, 6.0]);
//! assert_eq!(y.grad().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
//! ```

use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::TensorBackend;

use crate::eager::EagerTensor;
use crate::error::Result;

pub use tenferro_einsum::{eager_einsum, eager_einsum_owned};

/// Execute an einsum eagerly and record it for reverse-mode autodiff.
///
/// # Examples
///
/// ```
/// use tenferro::eager_einsum::eager_einsum_ad;
/// use tenferro::{EagerTensor, Tensor};
///
/// let a = EagerTensor::from_tensor(Tensor::from_vec(
///     vec![2, 3],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ));
/// let b = EagerTensor::from_tensor(Tensor::from_vec(
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
        },
    )
}
