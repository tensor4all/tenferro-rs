//! Result representations of the linalg operations: erased or statically typed.

use tenferro_tensor::{Erased, Repr, Tensor, TensorScalar, Typed, TypedTensor};

use crate::LinalgScalar;

/// A [`Repr`] that can also name the real, complex and index results of a
/// linalg operation.
///
/// `Erased` returns [`Tensor`] everywhere. `Typed<T>` returns
/// `TypedTensor<T::Real>` for real-valued results (singular values,
/// eigenvalues of a Hermitian matrix, ...), `TypedTensor<T::Complex>` for the
/// general eigendecomposition and `TypedTensor<i64>` for index results.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::LinalgRepr;
/// use tenferro_tensor::{Erased, Tensor, Typed};
///
/// let s = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
/// let erased = <Erased as LinalgRepr>::wrap_real(s.duplicate()?)?;
/// let typed = <Typed<f64> as LinalgRepr>::wrap_real(s.duplicate()?)?;
/// assert_eq!(erased.shape(), typed.shape());
/// // A mismatching dtype is an internal error, not a panic.
/// assert!(<Typed<f32> as LinalgRepr>::wrap_real(s).is_err());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait LinalgRepr: Repr {
    /// The real-valued result type (`Tensor` or `TypedTensor<T::Real>`).
    type Real;
    /// The complex-valued result type (`Tensor` or `TypedTensor<T::Complex>`).
    type Complex;
    /// The index result type (`Tensor` or `TypedTensor<i64>`).
    type Index;

    /// Wrap an erased real-valued result. Typed representations perform a
    /// checked downcast.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` when a typed representation receives a tensor
    /// of another dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::LinalgRepr;
    /// use tenferro_tensor::{Tensor, Typed};
    ///
    /// let s = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// assert!(<Typed<f64> as LinalgRepr>::wrap_real(s.duplicate()?).is_ok());
    /// assert!(<Typed<f32> as LinalgRepr>::wrap_real(s).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn wrap_real(t: Tensor) -> tenferro_tensor::Result<Self::Real>;

    /// Wrap an erased complex-valued result. Typed representations perform a
    /// checked downcast.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` when a typed representation receives a tensor
    /// of another dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_linalg::LinalgRepr;
    /// use tenferro_tensor::{Tensor, Typed};
    ///
    /// let c = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])?;
    /// assert!(<Typed<f64> as LinalgRepr>::wrap_complex(c.duplicate()?).is_ok());
    /// assert!(<Typed<f32> as LinalgRepr>::wrap_complex(c).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn wrap_complex(t: Tensor) -> tenferro_tensor::Result<Self::Complex>;

    /// Wrap an erased index result. Typed representations perform a checked
    /// downcast.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` when a typed representation receives a tensor
    /// of another dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::LinalgRepr;
    /// use tenferro_tensor::{Tensor, Typed};
    ///
    /// let idx = Tensor::from_vec_col_major(vec![2], vec![0_i64, 1])?;
    /// assert!(<Typed<f64> as LinalgRepr>::wrap_index(idx).is_ok());
    /// let real = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert!(<Typed<f64> as LinalgRepr>::wrap_index(real).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn wrap_index(t: Tensor) -> tenferro_tensor::Result<Self::Index>;
}

impl LinalgRepr for Erased {
    type Real = Tensor;
    type Complex = Tensor;
    type Index = Tensor;

    fn wrap_real(t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }

    fn wrap_complex(t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }

    fn wrap_index(t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }
}

impl<T: LinalgScalar> LinalgRepr for Typed<T> {
    type Real = TypedTensor<<T as TensorScalar>::Real>;
    type Complex = TypedTensor<<T as LinalgScalar>::Complex>;
    type Index = TypedTensor<i64>;

    fn wrap_real(t: Tensor) -> tenferro_tensor::Result<Self::Real> {
        <Typed<<T as TensorScalar>::Real> as Repr>::wrap_value(t)
    }

    fn wrap_complex(t: Tensor) -> tenferro_tensor::Result<Self::Complex> {
        <Typed<<T as LinalgScalar>::Complex> as Repr>::wrap_value(t)
    }

    fn wrap_index(t: Tensor) -> tenferro_tensor::Result<Self::Index> {
        <Typed<i64> as Repr>::wrap_value(t)
    }
}
