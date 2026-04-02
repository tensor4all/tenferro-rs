use num_complex::{Complex32, Complex64};

use super::accessors::TypedTensorScalarTag;
use super::{Tensor, TypedTensorRef};
use crate::{Error, Result};

/// Sealed helper trait for generic scalar-type dispatch on [`Tensor`].
///
/// This trait is implemented for [`f32`], [`f64`], [`Complex32`], and
/// [`Complex64`] and is used by convenience methods such as
/// [`Tensor::try_to_vec`] and [`Tensor::try_get`] to extract typed primal
/// values from a dynamic AD tensor without requiring the caller to know which
/// enum variant is active at compile time.
///
/// The trait is **sealed** — downstream crates cannot implement it.
///
/// # Examples
///
/// ```ignore
/// use tenferro::Tensor;
///
/// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();
/// let vals = x.try_to_vec::<f64>().unwrap();
/// assert_eq!(vals, vec![1.0, 2.0, 3.0]);
/// ```
#[doc(hidden)]
pub trait TensorScalarDowncast:
    TypedTensorScalarTag + super::accessors::TypedTensorBorrowTyped
{
    /// Returns a typed tensor view if the runtime scalar type
    /// matches, or `None` otherwise.
    #[doc(hidden)]
    fn downcast_ref(tensor: &Tensor) -> Option<TypedTensorRef<'_, Self>>;

    /// Returns the [`crate::ScalarType`] tag for this scalar.
    fn scalar_type_tag() -> crate::ScalarType;
}

impl TensorScalarDowncast for f32 {
    fn downcast_ref(tensor: &Tensor) -> Option<TypedTensorRef<'_, Self>> {
        tensor.as_f32()
    }

    fn scalar_type_tag() -> crate::ScalarType {
        <Self as TypedTensorScalarTag>::scalar_type_tag()
    }
}

impl TensorScalarDowncast for f64 {
    fn downcast_ref(tensor: &Tensor) -> Option<TypedTensorRef<'_, Self>> {
        tensor.as_f64()
    }

    fn scalar_type_tag() -> crate::ScalarType {
        <Self as TypedTensorScalarTag>::scalar_type_tag()
    }
}

impl TensorScalarDowncast for Complex32 {
    fn downcast_ref(tensor: &Tensor) -> Option<TypedTensorRef<'_, Self>> {
        tensor.as_c32()
    }

    fn scalar_type_tag() -> crate::ScalarType {
        <Self as TypedTensorScalarTag>::scalar_type_tag()
    }
}

impl TensorScalarDowncast for Complex64 {
    fn downcast_ref(tensor: &Tensor) -> Option<TypedTensorRef<'_, Self>> {
        tensor.as_c64()
    }

    fn scalar_type_tag() -> crate::ScalarType {
        <Self as TypedTensorScalarTag>::scalar_type_tag()
    }
}

impl Tensor {
    /// Extracts all primal payload values as a `Vec<T>` in column-major order.
    ///
    /// `T` must match the runtime scalar type of the tensor. If the types do
    /// not match, an error is returned describing the mismatch.
    ///
    /// For dense tensors, this returns every element in column-major
    /// (Fortran) order. For structured tensors (e.g. diagonal), the
    /// compressed payload values are returned — materialize the tensor
    /// to a dense representation first if you need the full logical
    /// expansion.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();
    /// assert_eq!(x.try_to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
    ///
    /// // Type mismatch returns an error:
    /// assert!(x.try_to_vec::<f32>().is_err());
    /// ```
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro::Tensor;
    ///
    /// let z = Tensor::from_slice(
    ///     &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    ///     &[2],
    /// )
    /// .unwrap();
    /// let vals = z.try_to_vec::<Complex64>().unwrap();
    /// assert_eq!(vals.len(), 2);
    /// assert_eq!(vals[0], Complex64::new(1.0, 2.0));
    /// ```
    pub fn try_to_vec<T: TensorScalarDowncast>(&self) -> Result<Vec<T>> {
        let ad = T::downcast_ref(self).ok_or_else(|| Error::InvalidAdTensor {
            message: format!(
                "try_to_vec: scalar type mismatch — tensor is {:?}, requested {:?}",
                self.scalar_type(),
                <T as TensorScalarDowncast>::scalar_type_tag(),
            ),
        })?;
        Ok(ad.primal().to_vec())
    }

    /// Extracts a single primal element by multi-dimensional index.
    ///
    /// `T` must match the runtime scalar type of the tensor. The index
    /// addresses the primal *payload* tensor (compressed representation for
    /// structured tensors).
    ///
    /// Returns an error if the scalar types do not match or the index is out
    /// of bounds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[10.0_f64, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    /// // Column-major layout: [10, 20] is column 0, [30, 40] is column 1.
    /// assert_eq!(x.try_get::<f64>(&[0, 0]).unwrap(), 10.0);
    /// assert_eq!(x.try_get::<f64>(&[1, 0]).unwrap(), 20.0);
    /// assert_eq!(x.try_get::<f64>(&[0, 1]).unwrap(), 30.0);
    /// assert_eq!(x.try_get::<f64>(&[1, 1]).unwrap(), 40.0);
    ///
    /// // Out of bounds:
    /// assert!(x.try_get::<f64>(&[2, 0]).is_err());
    ///
    /// // Type mismatch:
    /// assert!(x.try_get::<f32>(&[0, 0]).is_err());
    /// ```
    pub fn try_get<T: TensorScalarDowncast>(&self, index: &[usize]) -> Result<T> {
        let ad = T::downcast_ref(self).ok_or_else(|| Error::InvalidAdTensor {
            message: format!(
                "try_get: scalar type mismatch — tensor is {:?}, requested {:?}",
                self.scalar_type(),
                <T as TensorScalarDowncast>::scalar_type_tag(),
            ),
        })?;
        ad.primal()
            .get(index)
            .cloned()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: format!(
                    "try_get: index {:?} is out of bounds for tensor with dims {:?}",
                    index,
                    ad.dims(),
                ),
            })
    }
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};

    use crate::Tensor;

    #[test]
    fn try_to_vec_f64() {
        let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(x.try_to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn try_to_vec_f32() {
        let x = Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
        assert_eq!(x.try_to_vec::<f32>().unwrap(), vec![1.0_f32, 2.0]);
    }

    #[test]
    fn try_to_vec_c64() {
        let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let x = Tensor::from_slice(&data, &[2]).unwrap();
        assert_eq!(x.try_to_vec::<Complex64>().unwrap(), data);
    }

    #[test]
    fn try_to_vec_c32() {
        let data = vec![Complex32::new(1.0, 2.0)];
        let x = Tensor::from_slice(&data, &[1]).unwrap();
        assert_eq!(x.try_to_vec::<Complex32>().unwrap(), data);
    }

    #[test]
    fn try_to_vec_type_mismatch() {
        let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
        assert!(x.try_to_vec::<f32>().is_err());
        assert!(x.try_to_vec::<Complex64>().is_err());
    }

    #[test]
    fn try_get_f64() {
        let x = Tensor::from_slice(&[10.0_f64, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        assert_eq!(x.try_get::<f64>(&[0, 0]).unwrap(), 10.0);
        assert_eq!(x.try_get::<f64>(&[1, 0]).unwrap(), 20.0);
        assert_eq!(x.try_get::<f64>(&[0, 1]).unwrap(), 30.0);
        assert_eq!(x.try_get::<f64>(&[1, 1]).unwrap(), 40.0);
    }

    #[test]
    fn try_get_out_of_bounds() {
        let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
        assert!(x.try_get::<f64>(&[2]).is_err());
    }

    #[test]
    fn try_get_type_mismatch() {
        let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
        assert!(x.try_get::<f32>(&[0]).is_err());
    }

    #[test]
    fn try_to_vec_scalar_tensor() {
        let x = Tensor::from_slice(&[42.0_f64], &[]).unwrap();
        assert_eq!(x.try_to_vec::<f64>().unwrap(), vec![42.0]);
    }

    #[test]
    fn try_get_scalar_tensor() {
        let x = Tensor::from_slice(&[42.0_f64], &[]).unwrap();
        assert_eq!(x.try_get::<f64>(&[]).unwrap(), 42.0);
    }
}
