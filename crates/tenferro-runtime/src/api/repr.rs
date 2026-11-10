//! Result types of the core grammar per input representation.

use tenferro_tensor::{
    AsView, DType, Erased, Error, Repr, Tensor, TensorScalar, Typed, TypedTensor, ValidationError,
};

/// The core result types for an input representation.
///
/// `Erased` returns [`Tensor`] everywhere. `Typed<T>` returns
/// `TypedTensor<T::Real>` for real-valued results (`abs`) and
/// `TypedTensor<bool>` for predicates (`compare`).
///
/// # Examples
///
/// ```
/// use tenferro_runtime::CoreRepr;
/// use tenferro_tensor::{Erased, Tensor, Typed};
///
/// let t = Tensor::from_vec_col_major(vec![1], vec![true])?;
/// let erased = <Erased as CoreRepr>::wrap_bool("compare", t.duplicate()?)?;
/// let typed = <Typed<f64> as CoreRepr>::wrap_bool("compare", t.duplicate()?)?;
/// assert_eq!(erased.shape(), typed.shape());
/// // A typed result of another dtype is a validation error naming the op.
/// assert!(<Typed<f64> as CoreRepr>::wrap_value_of("add", t).is_err());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait CoreRepr: Repr {
    /// The real-valued result type (`Tensor` or `TypedTensor<T::Real>`).
    type Real;
    /// The predicate result type (`Tensor` or `TypedTensor<bool>`).
    type Bool;

    /// Wrap the erased result of `op` in this representation's value type;
    /// typed representations check the dtype.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `ValidationError::DTypeMismatch` (op
    /// `op`) when a typed representation receives a tensor of another dtype.
    fn wrap_value_of(op: &'static str, t: Tensor) -> tenferro_tensor::Result<Self::Value>;

    /// Wrap the erased real-valued result of `op`; typed representations
    /// check the dtype.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `ValidationError::DTypeMismatch` when a
    /// typed representation receives a tensor of another dtype.
    fn wrap_real(op: &'static str, t: Tensor) -> tenferro_tensor::Result<Self::Real>;

    /// Wrap the erased predicate result of `op`; typed representations check
    /// the dtype.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `ValidationError::DTypeMismatch` when a
    /// typed representation receives a tensor of another dtype.
    fn wrap_bool(op: &'static str, t: Tensor) -> tenferro_tensor::Result<Self::Bool>;
}

impl CoreRepr for Erased {
    type Real = Tensor;
    type Bool = Tensor;

    fn wrap_value_of(_op: &'static str, t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }

    fn wrap_real(_op: &'static str, t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }

    fn wrap_bool(_op: &'static str, t: Tensor) -> tenferro_tensor::Result<Tensor> {
        Ok(t)
    }
}

impl<T: TensorScalar> CoreRepr for Typed<T> {
    type Real = TypedTensor<<T as TensorScalar>::Real>;
    type Bool = TypedTensor<bool>;

    fn wrap_value_of(op: &'static str, t: Tensor) -> tenferro_tensor::Result<TypedTensor<T>> {
        typed_result::<T>(op, t)
    }

    fn wrap_real(op: &'static str, t: Tensor) -> tenferro_tensor::Result<Self::Real> {
        typed_result::<<T as TensorScalar>::Real>(op, t)
    }

    fn wrap_bool(op: &'static str, t: Tensor) -> tenferro_tensor::Result<Self::Bool> {
        typed_result::<bool>(op, t)
    }
}

/// The typed result of `op`, or the dtype mismatch the typed surface has
/// always reported for a provider result of another dtype.
fn typed_result<U: TensorScalar>(
    op: &'static str,
    t: Tensor,
) -> tenferro_tensor::Result<TypedTensor<U>> {
    let actual: DType = t.dtype();
    t.into_typed::<U>().map_err(|_| {
        Error::validation(
            op,
            ValidationError::DTypeMismatch {
                expected: U::dtype(),
                actual,
            },
        )
    })
}

/// A representation that can be the condition of `where_select`: the erased
/// representation (checked at run time) or `Typed<bool>`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::SelectCondition;
/// use tenferro_tensor::{Erased, Typed};
///
/// fn is_condition<C: SelectCondition>() {}
/// is_condition::<Erased>();
/// is_condition::<Typed<bool>>();
/// ```
pub trait SelectCondition: Repr {}

impl SelectCondition for Erased {}
impl SelectCondition for Typed<bool> {}

/// The branches of `where_select` for a condition representation `C`: any
/// [`AsView`] input, available only when `C` is a [`SelectCondition`], so a
/// typed non-`bool` receiver does not offer `where_select`.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::SelectOperand;
/// use tenferro_tensor::{Tensor, Typed};
///
/// fn branch<Y: SelectOperand<Typed<bool>> + ?Sized>(_: &Y) {}
/// branch(&Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait SelectOperand<C>: AsView {}

impl<Y: AsView + ?Sized, C: SelectCondition> SelectOperand<C> for Y {}
