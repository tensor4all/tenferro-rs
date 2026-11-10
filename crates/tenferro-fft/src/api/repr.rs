//! Result representations of the FFT grammar: erased inputs return
//! [`Tensor`], typed inputs return typed tensors of the transform's dtype.

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{Erased, Repr, Tensor, TensorScalar, Typed, TypedTensor};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

/// A scalar an FFT accepts: `f32`, `f64`, `Complex32` or `Complex64`.
///
/// `Complex` is the dtype of a spectrum of this scalar.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_fft::FftScalar;
///
/// fn spectrum_of<T: FftScalar>() -> &'static str {
///     std::any::type_name::<T::Complex>()
/// }
/// assert_eq!(spectrum_of::<f64>(), std::any::type_name::<Complex64>());
/// ```
pub trait FftScalar: TensorScalar + sealed::Sealed {
    /// The complex scalar of the spectrum.
    type Complex: TensorScalar;
}

impl FftScalar for f32 {
    type Complex = Complex32;
}
impl FftScalar for f64 {
    type Complex = Complex64;
}
impl FftScalar for Complex32 {
    type Complex = Complex32;
}
impl FftScalar for Complex64 {
    type Complex = Complex64;
}

/// How an FFT result is returned for an input representation.
///
/// `Erased` returns [`Tensor`]. `Typed<T>` returns `TypedTensor<T::Complex>`
/// for spectra (`fft`, `ifft`, `rfft` and their N-dimensional forms) and
/// `TypedTensor<T::Real>` for `irfft`/`irfftn`.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_fft::FftRepr;
/// use tenferro_tensor::{Erased, Tensor, Typed};
///
/// let t = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)])?;
/// let typed = <Typed<f64> as FftRepr>::wrap_complex(t)?;
/// assert_eq!(typed.shape(), &[1]);
/// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// assert!(<Erased as FftRepr>::wrap_real(t).is_ok());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait FftRepr: Repr {
    /// The complex result type.
    type Complex;
    /// The real result type.
    type Real;

    /// Wrap an erased complex result; typed representations downcast with a
    /// check.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` when a typed representation receives a
    /// tensor of another dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_fft::FftRepr;
    /// use tenferro_tensor::{Tensor, Typed};
    ///
    /// let c = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)])?;
    /// assert!(<Typed<f64> as FftRepr>::wrap_complex(c).is_ok());
    /// let r = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert!(<Typed<f64> as FftRepr>::wrap_complex(r).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn wrap_complex(t: Tensor) -> tenferro_tensor::Result<Self::Complex>;

    /// Wrap an erased real result; typed representations downcast with a
    /// check.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` when a typed representation receives a
    /// tensor of another dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_fft::FftRepr;
    /// use tenferro_tensor::{Tensor, Typed};
    ///
    /// let r = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert!(<Typed<f64> as FftRepr>::wrap_real(r.duplicate()?).is_ok());
    /// assert!(<Typed<f32> as FftRepr>::wrap_real(r).is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn wrap_real(t: Tensor) -> tenferro_tensor::Result<Self::Real>;
}

impl FftRepr for Erased {
    type Complex = Tensor;
    type Real = Tensor;

    fn wrap_complex(t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }

    fn wrap_real(t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }
}

impl<T: FftScalar> FftRepr for Typed<T> {
    type Complex = TypedTensor<T::Complex>;
    type Real = TypedTensor<<T as TensorScalar>::Real>;

    fn wrap_complex(t: Tensor) -> tenferro_tensor::Result<Self::Complex> {
        <Typed<T::Complex> as Repr>::wrap_value(t)
    }

    fn wrap_real(t: Tensor) -> tenferro_tensor::Result<Self::Real> {
        <Typed<<T as TensorScalar>::Real> as Repr>::wrap_value(t)
    }
}
