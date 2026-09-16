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
        match ($lhs, $rhs) {
            (Tensor::F32($a), Tensor::F32($b)) => $call.map(Tensor::F32),
            (Tensor::F64($a), Tensor::F64($b)) => $call.map(Tensor::F64),
            (Tensor::C32($a), Tensor::C32($b)) => $call.map(Tensor::C32),
            (Tensor::C64($a), Tensor::C64($b)) => $call.map(Tensor::C64),
            ($a, $b) => {
                let _ = (&$a, &$b);
                $fallback
            }
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
        match $input {
            Tensor::F32($t) => $call.map(|$value| $wrap),
            Tensor::F64($t) => $call.map(|$value| $wrap),
            Tensor::C32($t) => $call.map(|$value| $wrap),
            Tensor::C64($t) => $call.map(|$value| $wrap),
            _ => $fallback,
        }
    };
}
