//! Shared erased-dispatch macros for concrete tensor values.
//!
//! Every crate that carries the concrete value type needs to reach a typed kernel
//! from it. Declaring the matching once here keeps each call site to the kernel it
//! calls, so a call site stops naming the variants and does not change again when
//! the value type stops being a closed enum.
//!
//! The macros expand to paths that resolve at the call site, so a caller must have
//! `Tensor` in scope.

/// Dispatch a same-variant pair of tensors to a typed kernel.
///
/// The four real and complex arms are declared once. The caller supplies the
/// operands, the typed kernel, and the expression to return for an unsupported
/// pair.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::same_variant_pair;
/// use tenferro_tensor::{Error, Tensor};
///
/// fn first_matching_pair(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor> {
///     same_variant_pair!(
///         lhs,
///         rhs,
///         |a, b| {
///             let _ = b.shape();
///             a.duplicate()
///         },
///         Err(Error::dtype_mismatch(
///             "first_matching_pair",
///             lhs.dtype(),
///             rhs.dtype()
///         ))
///     )
/// }
///
/// let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
/// let out = first_matching_pair(&lhs, &rhs)?;
/// assert_eq!(out.as_slice::<f64>()?, &[1.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[macro_export]
macro_rules! same_variant_pair {
    ($lhs:expr, $rhs:expr, |$a:ident, $b:ident| $call:expr, $fallback:expr) => {
        match ($lhs.dtype(), $rhs.dtype()) {
            ($crate::DType::F32, $crate::DType::F32) => {
                match ($lhs.as_typed::<f32>(), $rhs.as_typed::<f32>()) {
                    (Some($a), Some($b)) => $call.map(Tensor::from_typed::<f32>),
                    _ => $fallback,
                }
            }
            ($crate::DType::F64, $crate::DType::F64) => {
                match ($lhs.as_typed::<f64>(), $rhs.as_typed::<f64>()) {
                    (Some($a), Some($b)) => $call.map(Tensor::from_typed::<f64>),
                    _ => $fallback,
                }
            }
            ($crate::DType::C32, $crate::DType::C32) => {
                match (
                    $lhs.as_typed::<$crate::Complex32>(),
                    $rhs.as_typed::<$crate::Complex32>(),
                ) {
                    (Some($a), Some($b)) => $call.map(Tensor::from_typed::<$crate::Complex32>),
                    _ => $fallback,
                }
            }
            ($crate::DType::C64, $crate::DType::C64) => {
                match (
                    $lhs.as_typed::<$crate::Complex64>(),
                    $rhs.as_typed::<$crate::Complex64>(),
                ) {
                    (Some($a), Some($b)) => $call.map(Tensor::from_typed::<$crate::Complex64>),
                    _ => $fallback,
                }
            }
            _ => $fallback,
        }
    };
}

/// Dispatch a single tensor to a typed kernel and erase its result.
///
/// The typed result is wrapped by the scalar's own constructor, so a real-valued
/// result of a complex input still lands in the matching real variant without the
/// call site naming a variant.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::same_variant_unary;
/// use tenferro_tensor::{Error, Tensor, TensorScalar};
///
/// fn negate(input: &Tensor) -> tenferro_tensor::Result<Tensor> {
///     same_variant_unary!(
///         input,
///         |t| t.duplicate(),
///         |value| TensorScalar::typed_tensor_into_tensor(value),
///         Err(Error::dtype_mismatch("negate", input.dtype(), input.dtype()))
///     )
/// }
///
/// let value = Tensor::from_vec_col_major(vec![1], vec![3.0_f64])?;
/// assert_eq!(negate(&value)?.as_slice::<f64>()?, &[3.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[macro_export]
macro_rules! same_variant_unary {
    ($input:expr, |$t:ident| $call:expr, |$value:ident| $wrap:expr, $fallback:expr) => {
        match $input.dtype() {
            $crate::DType::F32 => match $input.as_typed::<f32>() {
                Some($t) => $call.map(|$value| $wrap),
                None => $fallback,
            },
            $crate::DType::F64 => match $input.as_typed::<f64>() {
                Some($t) => $call.map(|$value| $wrap),
                None => $fallback,
            },
            $crate::DType::C32 => match $input.as_typed::<$crate::Complex32>() {
                Some($t) => $call.map(|$value| $wrap),
                None => $fallback,
            },
            $crate::DType::C64 => match $input.as_typed::<$crate::Complex64>() {
                Some($t) => $call.map(|$value| $wrap),
                None => $fallback,
            },
            _ => $fallback,
        }
    };
}
