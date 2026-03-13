mod basics;
mod complex;
mod layout;
mod merge;
mod scalar_ops;

use num_complex::{Complex32, Complex64};

/// Runtime AD tensor wrapper.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdTensor, DynAdTensor, ScalarType};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
/// let x: DynAdTensor = AdTensor::new_primal(t).into();
/// assert_eq!(x.scalar_type(), ScalarType::F64);
/// ```
#[derive(Clone)]
pub enum DynAdTensor {
    F32(crate::AdTensor<f32>),
    F64(crate::AdTensor<f64>),
    C32(crate::AdTensor<Complex32>),
    C64(crate::AdTensor<Complex64>),
}
