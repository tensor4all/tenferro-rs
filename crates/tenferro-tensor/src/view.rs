//! Input adapter shared by all operation families (the #2552 `AsView` grammar).
//!
//! [`AsView`] turns any tensor input into a borrowed [`TensorRead`] without
//! copying element data. [`Repr`] says whether results should come back
//! dtype-erased ([`Erased`]) or statically typed ([`Typed`]).

use std::marker::PhantomData;

use crate::{
    Error, Tensor, TensorRead, TensorScalar, TensorWrite, TypedTensor, TypedTensorView,
    TypedTensorWrite,
};

mod sealed {
    pub trait Sealed {}
}

/// A representation of tensor values: dtype-erased or statically typed.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Erased, Repr, Tensor, Typed};
///
/// let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let b = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let erased = <Erased as Repr>::wrap_value(a)?;
/// let typed = <Typed<f64> as Repr>::wrap_value(b)?;
/// assert_eq!(erased.shape(), typed.shape());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait Repr: sealed::Sealed {
    /// The value type: `Tensor` or `TypedTensor<T>`.
    type Value;
    /// The writable target type: `TensorWrite<'a>` or `TypedTensorWrite<'a, T>`.
    type Write<'a>;

    /// Wrap an erased result in this representation. Typed representations
    /// perform a checked downcast.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] when a typed representation receives a
    /// tensor of another dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Repr, Tensor, Typed};
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert!(<Typed<f64> as Repr>::wrap_value(t.duplicate()?).is_ok());
    /// assert!(<Typed<f32> as Repr>::wrap_value(t).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn wrap_value(t: Tensor) -> crate::Result<Self::Value>;

    /// Erase a writable target to [`TensorWrite`]. Metadata only.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Erased, Repr, Tensor, TensorWrite};
    ///
    /// let mut t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let write = TensorWrite::from_tensor(&mut t);
    /// let erased = <Erased as Repr>::erase_write(write);
    /// assert_eq!(erased.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn erase_write<'a>(w: Self::Write<'a>) -> TensorWrite<'a>;
}

/// The dtype-erased representation.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Erased, Repr, Tensor};
///
/// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let same = <Erased as Repr>::wrap_value(t)?;
/// assert_eq!(same.shape(), &[1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Erased;

/// The statically typed representation for scalar `T`.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Repr, Tensor, Typed};
///
/// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let typed = <Typed<f64> as Repr>::wrap_value(t)?;
/// assert_eq!(typed.shape(), &[1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub struct Typed<T>(PhantomData<fn() -> T>);

impl<T> std::fmt::Debug for Typed<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Typed")
    }
}

impl<T> Clone for Typed<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Typed<T> {}

impl<T> Default for Typed<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl sealed::Sealed for Erased {}
impl<T: TensorScalar> sealed::Sealed for Typed<T> {}

impl Repr for Erased {
    type Value = Tensor;
    type Write<'a> = TensorWrite<'a>;

    fn wrap_value(t: Tensor) -> crate::Result<Tensor> {
        Ok(t)
    }

    fn erase_write<'a>(w: Self::Write<'a>) -> TensorWrite<'a> {
        w
    }
}

impl<T: TensorScalar> Repr for Typed<T> {
    type Value = TypedTensor<T>;
    type Write<'a> = TypedTensorWrite<'a, T>;

    fn wrap_value(t: Tensor) -> crate::Result<TypedTensor<T>> {
        let actual = t.dtype();
        t.into_typed::<T>().map_err(|_| {
            Error::Internal(format!(
                "checked downcast failed: result dtype {actual:?} is not {:?}",
                T::dtype()
            ))
        })
    }

    fn erase_write<'a>(w: Self::Write<'a>) -> TensorWrite<'a> {
        w.into_tensor_write()
    }
}

/// Any tensor input that can be read as a borrowed view without copying.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{AsView, Tensor, TensorRead, TypedTensor, TypedTensorView};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let typed = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
/// let read = TensorRead::from_tensor(&tensor);
///
/// assert_eq!(tensor.as_read().shape(), typed.as_read().shape());
/// assert_eq!(read.as_read().dtype(), typed.as_read().dtype());
/// assert_eq!((&&tensor).as_read().shape(), &[2]);
///
/// let data = [1.0_f64, 2.0];
/// let view = TypedTensorView::from_col_major(&[2], &data)?;
/// assert_eq!(view.as_read().shape(), &[2]);
/// assert_eq!(view.as_read().as_slice::<f64>()?.as_ptr(), data.as_ptr());
///
/// // No element data is copied.
/// assert_eq!(
///     tensor.as_read().as_slice::<f64>()?.as_ptr(),
///     tensor.as_read().as_slice::<f64>()?.as_ptr(),
/// );
/// assert_eq!(
///     typed.as_read().as_slice::<f64>()?.as_ptr(),
///     typed.as_slice()?.as_ptr(),
/// );
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait AsView: sealed::Sealed {
    /// The representation results take for this input.
    type Repr: Repr;

    /// Borrow this input as a [`TensorRead`]. Metadata only; never copies
    /// element data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{AsView, Tensor};
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let read = t.as_read();
    /// assert_eq!(read.shape(), &[2]);
    /// assert_eq!(read.as_slice::<f64>()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn as_read(&self) -> TensorRead<'_>;
}

impl sealed::Sealed for Tensor {}
impl sealed::Sealed for TensorRead<'_> {}
impl<T: TensorScalar> sealed::Sealed for TypedTensor<T> {}
impl<T: TensorScalar> sealed::Sealed for TypedTensorView<'_, T> {}
impl<X: sealed::Sealed + ?Sized> sealed::Sealed for &X {}

impl AsView for Tensor {
    type Repr = Erased;

    fn as_read(&self) -> TensorRead<'_> {
        TensorRead::from_tensor(self)
    }
}

impl AsView for TensorRead<'_> {
    type Repr = Erased;

    fn as_read(&self) -> TensorRead<'_> {
        // `TensorRead` only holds borrows, so cloning copies metadata.
        self.clone()
    }
}

impl<T: TensorScalar> AsView for TypedTensor<T> {
    type Repr = Typed<T>;

    fn as_read(&self) -> TensorRead<'_> {
        T::tensor_read(self)
    }
}

impl<T: TensorScalar> AsView for TypedTensorView<'_, T> {
    type Repr = Typed<T>;

    fn as_read(&self) -> TensorRead<'_> {
        // The default `DynRank`/`Dynamic` view maps to `TensorView` directly,
        // so unlike `into_tensor_read` this conversion cannot fail.
        TensorRead::from_view(T::tensor_view(self.clone()))
    }
}

impl<X: AsView + ?Sized> AsView for &X {
    type Repr = X::Repr;

    fn as_read(&self) -> TensorRead<'_> {
        (**self).as_read()
    }
}

#[cfg(test)]
mod tests;
