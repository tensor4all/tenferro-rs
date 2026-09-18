//! Directed value conversions between the external scalar and `f64`.
//!
//! A conversion is a value operation with a declared rounding, range, and
//! destination allocation. It is not automatic promotion, and it creates no
//! multi-hop rule: every pair must exist on its own.

use tenferro_tensor::Tensor;
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

use crate::Df64;

/// Read the external `Df64` payload of `tensor` by its own element type.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::UnsupportedDType`] when the tensor is not an
/// externally defined `Df64` value.
fn df64_values<'a>(tensor: &'a Tensor, op: &'static str) -> tenferro_tensor::Result<&'a [Df64]> {
    match tensor.external_payload() {
        Some(payload) => payload
            .downcast_ref::<Df64>()
            .map(HostTensor::as_slice)
            .ok_or_else(|| {
                tenferro_tensor::Error::unsupported_dtype(
                    op,
                    tensor.dtype(),
                    "the payload holds a different external element type",
                )
            }),
        None => Err(tenferro_tensor::Error::unsupported_dtype(
            op,
            tensor.dtype(),
            "a directed Df64 conversion takes an externally defined Df64 tensor",
        )),
    }
}

/// Read the layout-accessible `f64` values of `tensor`.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::UnsupportedDType`] when the tensor does not
/// hold `f64` values, and [`tenferro_tensor::Error::RuntimeState`] when they are
/// not reachable as one borrowed slice.
fn f64_values<'a>(tensor: &'a Tensor, op: &'static str) -> tenferro_tensor::Result<&'a [f64]> {
    if tensor.dtype() != tenferro_tensor::DType::F64 {
        return Err(tenferro_tensor::Error::unsupported_dtype(
            op,
            tensor.dtype(),
            "a directed f64 conversion takes an f64 tensor",
        ));
    }
    tensor
        .as_slice::<f64>()
        .map_err(|source| tenferro_tensor::Error::runtime_state_source(op, source))
}

/// Convert an external `Df64` tensor to an `f64` tensor, low component included.
///
/// **Rounding:** the low component participates, so the result is the `f64`
/// nearest to `hi + lo` with the usual round-to-nearest, ties-to-even rule. This
/// is not the truncation to `hi` that reinterpreting the payload would give.
///
/// **Range:** an infinite or NaN component propagates through the sum by IEEE
/// arithmetic, and no range error is raised.
///
/// **Allocation:** a new `f64` tensor with the same shape is allocated at the
/// destination; the input payload is neither consumed nor modified.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::UnsupportedDType`] when `tensor` is not an
/// externally defined `Df64` value.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::{conversion, Df64};
/// use tenferro_tensor::Tensor;
/// use tenferro_tensor_core::{ErasedHostTensor, HostTensor};
///
/// let low = Df64 { hi: 1.0, lo: 2f64.powi(-52) };
/// let tensor = Tensor::external(ErasedHostTensor::new(
///     HostTensor::from_vec_col_major(vec![1], vec![low])?,
/// ));
///
/// // The low component is part of the value, so it is part of the result.
/// assert_eq!(conversion::to_f64(&tensor)?.as_slice::<f64>()?, &[1.0 + 2f64.powi(-52)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn to_f64(tensor: &Tensor) -> tenferro_tensor::Result<Tensor> {
    let values = df64_values(tensor, "df64_to_f64")?;
    let narrowed: Vec<f64> = values.iter().map(|value| value.hi + value.lo).collect();
    Tensor::from_vec_col_major(tensor.shape().to_vec(), narrowed)
        .map_err(|source| tenferro_tensor::Error::runtime_state_source("df64_to_f64", source))
}

/// Convert an `f64` tensor to an external `Df64` tensor.
///
/// **Rounding:** exact. Every `f64` value is a `Df64` value with a zero low
/// component, so the result's high component is the input bit for bit and its
/// low component is zero. Information a previous narrowing discarded is not
/// recovered here.
///
/// **Range:** the transformation preserves `f64` infinity and NaN unchanged.
///
/// **Allocation:** a new externally defined tensor with the same shape is
/// allocated at the destination; the input is neither consumed nor modified.
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::UnsupportedDType`] when `tensor` is not an
/// `f64` value, and [`tenferro_tensor::Error::RuntimeState`] when its values are
/// not reachable as one borrowed slice.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::{conversion, Df64};
/// use tenferro_tensor::Tensor;
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let widened = conversion::to_df64(&tensor)?;
///
/// let payload = widened.external_payload().expect("an external payload");
/// let values = payload.downcast_ref::<Df64>().expect("the payload type");
/// assert_eq!(values.as_slice(), &[Df64::from_f64(1.0), Df64::from_f64(2.0)]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn to_df64(tensor: &Tensor) -> tenferro_tensor::Result<Tensor> {
    let values = f64_values(tensor, "f64_to_df64")?;
    let widened: Vec<Df64> = values.iter().copied().map(Df64::from_f64).collect();
    let payload = HostTensor::from_vec_col_major(tensor.shape().to_vec(), widened)
        .map_err(|source| tenferro_tensor::Error::validation("f64_to_df64", source))?;
    Ok(Tensor::external(ErasedHostTensor::new(payload)))
}
