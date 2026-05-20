//! EagerTensor einsum implementation.
//!
//! # Examples
//!
//! ```rust
//! use tenferro::eager_tensor::einsum;
//! use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
//!
//! let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
//! let x = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]), ctx.clone());
//! let y = EagerTensor::requires_grad_in(Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]), ctx);
//! let loss = einsum(&[&x, &y], "i,i->").unwrap();
//! let _ = loss.backward().unwrap();
//!
//! assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 5.0, 6.0]);
//! assert_eq!(y.grad().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
//! ```

use tenferro_ops::std_tensor_op::{EinsumSubscripts, StdTensorOp};

use crate::eager::EagerTensor;
use crate::error::Result;
use crate::parse_einsum_subscripts;

/// Execute an einsum eagerly and record it when any input requires gradients.
///
/// # Examples
///
/// ```
/// use tenferro::eager_tensor::einsum;
/// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
///
/// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
/// let a = EagerTensor::from_tensor_in(Tensor::from_vec(
///     vec![2, 3],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ), ctx.clone());
/// let b = EagerTensor::from_tensor_in(Tensor::from_vec(
///     vec![3, 2],
///     vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
/// ), ctx.clone());
/// let c = einsum(&[&a, &b], "ij,jk->ik").unwrap();
///
/// assert_eq!(c.data().shape(), &[2, 2]);
/// assert_eq!(c.data().as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
/// ```
pub fn einsum(inputs: &[&EagerTensor], subscripts: &str) -> Result<EagerTensor> {
    let subscripts = parse_einsum_subscripts(subscripts)?;
    einsum_subscripts(inputs, &subscripts)
}

/// Execute an einsum eagerly from integer labels and record it when any input requires gradients.
pub fn einsum_subscripts(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<EagerTensor> {
    EagerTensor::nary_op(
        inputs,
        StdTensorOp::NaryEinsum {
            subscripts: subscripts.clone(),
        },
    )
}
