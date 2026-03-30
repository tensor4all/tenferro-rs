mod accessors;
mod basics;
mod complex;
mod debug;
mod downcast;
mod eager_linalg;
mod eager_scalar;
mod eager_tensor;
mod layout;
mod merge;
mod placement;
mod promotion;
mod pullback;
mod scalar_ops;
mod shape;
mod snapshot;

pub use accessors::TypedTensorRef;
pub use downcast::TensorScalarDowncast;
pub use eager_linalg::{
    CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuResult, QrResult, SlogdetResult, SolveExResult, SvdResult,
};
use tenferro_internal_ad_core::DynAdTensor;

/// Runtime AD tensor wrapper.
///
/// `Tensor` is the canonical dynamic tensor payload for eager tensor
/// algebra in `tenferro`.
///
/// - rank-0 `Tensor` values act as scalar coefficients
/// - mixed-dtype tensor ops apply the dynamic result-type promotion rule
///   internally before execution
///   (`complex` beats `real`, and 64-bit beats 32-bit)
/// - [`Tensor::to_scalar_type`] is the explicit numeric cast boundary
/// - [`Tensor::detach`] drops tape metadata while keeping the same dynamic
///   tensor object for storage or FFI boundaries
///
/// # Examples
///
/// ```ignore
/// use num_complex::Complex64;
/// use tenferro::{ScalarType, Tensor};
/// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
///
/// let t = DenseTensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
/// let x = Tensor::from_tensor(t);
/// assert_eq!(x.scalar_type(), ScalarType::F64);
///
/// let coeff = Tensor::from_tensor(
///     DenseTensor::<Complex64>::from_slice(
///         &[Complex64::new(0.0, 2.0)],
///         &[],
///         MemoryOrder::ColumnMajor,
///     )
///         .unwrap(),
/// );
/// let y = x.scale(&coeff).unwrap();
/// assert_eq!(y.scalar_type(), ScalarType::C64);
///
/// let debug = format!("{x:?}");
/// assert!(debug.contains("preview: [1.0]"));
/// ```
#[derive(Clone)]
pub struct Tensor(pub(crate) DynAdTensor);

#[cfg(test)]
mod carrier_tests {
    use super::Tensor;
    use crate::ScalarType;
    use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

    #[test]
    fn tensor_wraps_erased_dyn_ad_carrier() {
        let dense =
            DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let tensor = Tensor::from_tensor(dense);
        assert_eq!(tensor.scalar_type(), ScalarType::F64);
        let typed = tensor.as_f64().unwrap();
        assert_eq!(typed.primal().buffer().as_slice().unwrap(), &[1.0, 2.0]);
    }
}
