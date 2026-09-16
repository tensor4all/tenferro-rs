//! Scalar operations named by type.
//!
//! An operation passed to the scalar-agnostic entry points is a *type*, not a
//! closure. A closure is part of a generic function's type parameters, so every
//! call site would produce its own instantiation of the whole `strided-kernel`
//! body. A named operation keeps the instantiation keyed on the element type and
//! the operation, so two scalar sets that call the same operation share one
//! compiled kernel instead of one per set.

/// A binary operation between two scalars of the same type.
///
/// Implement this for a marker type in the crate that owns the operation. That
/// is how an external scalar's contribution supplies its own arithmetic.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::scalar_ops::{AddOp, BinaryScalarOp};
///
/// assert_eq!(<AddOp as BinaryScalarOp<f64>>::apply(1.0, 2.0), 3.0);
/// // Integers wrap, so the shared path agrees with the preset path in every build.
/// assert_eq!(<AddOp as BinaryScalarOp<i32>>::apply(i32::MAX, 1), i32::MIN);
/// ```
pub trait BinaryScalarOp<T> {
    /// Apply the operation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::scalar_ops::{BinaryScalarOp, MulOp};
    ///
    /// assert_eq!(<MulOp as BinaryScalarOp<i64>>::apply(3, 4), 12);
    /// ```
    fn apply(lhs: T, rhs: T) -> T;
}

/// Addition.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::scalar_ops::{AddOp, BinaryScalarOp};
///
/// assert_eq!(<AddOp as BinaryScalarOp<i32>>::apply(1, 2), 3);
/// ```
pub struct AddOp;

/// Subtraction.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::scalar_ops::{BinaryScalarOp, SubOp};
///
/// assert_eq!(<SubOp as BinaryScalarOp<f64>>::apply(5.0, 2.0), 3.0);
/// ```
pub struct SubOp;

/// Multiplication.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::scalar_ops::{BinaryScalarOp, MulOp};
///
/// assert_eq!(<MulOp as BinaryScalarOp<f64>>::apply(3.0, 4.0), 12.0);
/// ```
pub struct MulOp;

impl<T> BinaryScalarOp<T> for AddOp
where
    T: tenferro_tensor_core::ScalarArithmetic,
{
    /// The scalar contract's addition, which wraps for the integer members.
    ///
    /// Going through the contract rather than the operator matters for integers: the operator
    /// panics on overflow in a debug build and wraps in a release build, while the contract
    /// wraps in both, which is what the preset path and `ScalarArithmetic::scalar_add` promise.
    fn apply(lhs: T, rhs: T) -> T {
        tenferro_tensor_core::ScalarArithmetic::scalar_add(lhs, rhs)
    }
}

impl<T> BinaryScalarOp<T> for SubOp
where
    T: tenferro_tensor_core::ScalarArithmetic,
{
    /// The scalar contract's subtraction, which wraps for the integer members.
    fn apply(lhs: T, rhs: T) -> T {
        tenferro_tensor_core::ScalarArithmetic::scalar_sub(lhs, rhs)
    }
}

impl<T> BinaryScalarOp<T> for MulOp
where
    T: tenferro_tensor_core::ScalarArithmetic,
{
    /// The scalar contract's product, which wraps for the integer members.
    fn apply(lhs: T, rhs: T) -> T {
        tenferro_tensor_core::ScalarArithmetic::scalar_mul(lhs, rhs)
    }
}
