//! The one input kind of the concrete linalg surface (#2552 rule 2).
//!
//! Every concrete linalg operation is implemented once, on the borrowed
//! [`TensorRead`] view. An owned [`Tensor`], a [`TensorRead`] and a typed
//! [`TypedTensorView`] reach that implementation through [`LinalgInput`], whose
//! conversion only builds view metadata and never copies elements. The associated
//! output types make typed input give typed output: a `TypedTensorView<'_, Complex64>`
//! gets `TypedTensor<Complex64>` factors and `TypedTensor<f64>` singular values, while
//! dtype-erased input gets [`Tensor`] for every output.

use tenferro_tensor::{Tensor, TensorRead, TensorScalar, TypedTensor, TypedTensorView};

use crate::LinalgScalar;

mod private {
    pub trait Sealed {}
    impl Sealed for tenferro_tensor::Tensor {}
    impl Sealed for tenferro_tensor::TensorRead<'_> {}
    impl<T: crate::LinalgScalar> Sealed for tenferro_tensor::TypedTensor<T> {}
    impl<T: crate::LinalgScalar> Sealed for tenferro_tensor::TypedTensorView<'_, T> {}
}

/// A concrete linalg input: a metadata-only route to the borrowed view, and the
/// output types that view's dtype produces.
///
/// Implemented for [`Tensor`], [`TensorRead`], [`TypedTensor`] and [`TypedTensorView`]; the
/// trait is sealed. Operations are called through [`TensorLinalgExt`](crate::TensorLinalgExt),
/// which every `LinalgInput` implements.
///
/// | output | `Tensor`, `TensorRead` | `TypedTensor<T>`, `TypedTensorView<'_, T>` |
/// |---|---|---|
/// | [`Out`](Self::Out): the input dtype | `Tensor` | `TypedTensor<T>` |
/// | [`Real`](Self::Real): the real dtype (singular values, norms, parity) | `Tensor` | `TypedTensor<T::Real>` |
/// | [`Complex`](Self::Complex): the complex dtype (`eig`) | `Tensor` | `TypedTensor<T::Complex>` |
/// | [`Index`](Self::Index): integer metadata (pivots, rank) | `Tensor` | `TypedTensor<i64>` |
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::LinalgInput;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// fn real_output<X: LinalgInput<Real = TypedTensor<f64>>>(_: &X) {}
/// fn erased_output<X: LinalgInput<Out = Tensor, Real = Tensor>>(_: &X) {}
///
/// let typed = TypedTensor::<f64>::from_vec_col_major(vec![1, 1], vec![1.0])?;
/// real_output(&typed.as_view());
/// erased_output(&Tensor::from_typed(typed));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait LinalgInput: private::Sealed {
    /// Output with the input dtype.
    type Out;
    /// Output with the real dtype of the input.
    type Real;
    /// Output with the complex dtype of the input.
    type Complex;
    /// Output with integer (`i64`) metadata.
    type Index;

    /// The borrowed view of the input, built from metadata only.
    #[doc(hidden)]
    fn linalg_read(&self) -> TensorRead<'_>;
    /// Wrap a backend output that has the input dtype.
    #[doc(hidden)]
    fn linalg_out(tensor: Tensor) -> tenferro_tensor::Result<Self::Out>;
    /// Wrap a backend output that has the real dtype.
    #[doc(hidden)]
    fn linalg_real(tensor: Tensor) -> tenferro_tensor::Result<Self::Real>;
    /// Wrap a backend output that has the complex dtype.
    #[doc(hidden)]
    fn linalg_complex(tensor: Tensor) -> tenferro_tensor::Result<Self::Complex>;
    /// Wrap a backend output that has `i64` elements.
    #[doc(hidden)]
    fn linalg_index(tensor: Tensor) -> tenferro_tensor::Result<Self::Index>;
}

macro_rules! erased_input {
    ($ty:ty, |$this:ident| $read:expr) => {
        impl LinalgInput for $ty {
            type Out = Tensor;
            type Real = Tensor;
            type Complex = Tensor;
            type Index = Tensor;

            fn linalg_read(&self) -> TensorRead<'_> {
                let $this = self;
                $read
            }
            fn linalg_out(tensor: Tensor) -> tenferro_tensor::Result<Tensor> {
                Ok(tensor)
            }
            fn linalg_real(tensor: Tensor) -> tenferro_tensor::Result<Tensor> {
                Ok(tensor)
            }
            fn linalg_complex(tensor: Tensor) -> tenferro_tensor::Result<Tensor> {
                Ok(tensor)
            }
            fn linalg_index(tensor: Tensor) -> tenferro_tensor::Result<Tensor> {
                Ok(tensor)
            }
        }
    };
}

erased_input!(Tensor, |tensor| TensorRead::from_tensor(tensor));
erased_input!(TensorRead<'_>, |read| read.clone());

macro_rules! typed_outputs {
    () => {
        fn linalg_out(tensor: Tensor) -> tenferro_tensor::Result<Self::Out> {
            typed_output::<T>(tensor)
        }
        fn linalg_real(tensor: Tensor) -> tenferro_tensor::Result<Self::Real> {
            typed_output::<<T as TensorScalar>::Real>(tensor)
        }
        fn linalg_complex(tensor: Tensor) -> tenferro_tensor::Result<Self::Complex> {
            typed_output::<<T as LinalgScalar>::Complex>(tensor)
        }
        fn linalg_index(tensor: Tensor) -> tenferro_tensor::Result<Self::Index> {
            typed_output::<i64>(tensor)
        }
    };
}

impl<T: LinalgScalar> LinalgInput for TypedTensor<T> {
    type Out = TypedTensor<T>;
    type Real = TypedTensor<<T as TensorScalar>::Real>;
    type Complex = TypedTensor<<T as LinalgScalar>::Complex>;
    type Index = TypedTensor<i64>;

    fn linalg_read(&self) -> TensorRead<'_> {
        T::tensor_read(self)
    }
    typed_outputs!();
}

impl<T: LinalgScalar> LinalgInput for TypedTensorView<'_, T> {
    type Out = TypedTensor<T>;
    type Real = TypedTensor<<T as TensorScalar>::Real>;
    type Complex = TypedTensor<<T as LinalgScalar>::Complex>;
    type Index = TypedTensor<i64>;

    fn linalg_read(&self) -> TensorRead<'_> {
        TensorRead::from_view(T::tensor_view(self.clone()))
    }
    typed_outputs!();
}

/// Checked downcast of a backend output to the dtype the typed contract promises.
pub(crate) fn typed_output<T: TensorScalar>(
    tensor: Tensor,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if tensor.dtype() != T::dtype() {
        return Err(tenferro_tensor::Error::Internal(format!(
            "typed linalg backend contract expected {:?}, got {:?}",
            T::dtype(),
            tensor.dtype()
        )));
    }
    match T::into_typed(tensor) {
        Ok(typed) => Ok(typed),
        // INVARIANT: the dtype guard above already accepted this tensor, so the
        // refusal arm is unreachable for a matching preset scalar.
        Err(failure) => Err(tenferro_tensor::Error::Internal(format!(
            "typed linalg backend contract lost its dtype guard: {}",
            failure.error()
        ))),
    }
}
