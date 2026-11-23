//! Scalar factors for `scale_real` / `scale_complex`.
//!
//! The concrete session surface and the eager surface scale by multiplying with
//! a rank-0 factor of the input dtype. This module is the single definition of
//! how a real or complex factor becomes that rank-0 tensor, so both surfaces
//! round, reject and convert factors identically.

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{ContractionScalar, DType, Error, Result, Tensor};

fn invalid_factor(op: &'static str, message: String) -> Error {
    Error::invalid_argument(op, "factor", message)
}

fn finite_real_factor(value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_factor(
            "scale_real",
            format!("real scalar must be finite, got {value}"),
        ))
    }
}

fn round_real_to_i64(value: f64) -> Result<i64> {
    let rounded = finite_real_factor(value)?.round();
    if rounded < i64::MIN as f64 || rounded >= -(i64::MIN as f64) {
        return Err(invalid_factor(
            "scale_real",
            format!("rounded real scalar {rounded} is out of i64 range"),
        ));
    }
    Ok(rounded as i64)
}

fn round_real_to_i32(value: f64) -> Result<i32> {
    let rounded = round_real_to_i64(value)?;
    i32::try_from(rounded).map_err(|_| {
        invalid_factor(
            "scale_real",
            format!("rounded real scalar {rounded} is out of i32 range"),
        )
    })
}

/// Build the rank-0 host factor that `scale_real` multiplies a `dtype` tensor by.
///
/// Integer dtypes round the factor, `Bool` maps a finite zero to `false`, and
/// complex dtypes use a zero imaginary part.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::scale::real_scale_scalar;
/// use tenferro_tensor::DType;
///
/// let factor = real_scale_scalar(DType::I32, 2.6)?;
/// assert_eq!(factor.as_slice::<i32>()?, &[3]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Validation`] with `ValidationError::InvalidArgument` for a non-finite factor,
/// a rounded factor outside the integer dtype's range, or an external dtype.
pub fn real_scale_scalar(dtype: DType, factor: f64) -> Result<Tensor> {
    match dtype {
        DType::F64 => Tensor::from_vec_col_major(vec![], vec![factor]),
        DType::F32 => Tensor::from_vec_col_major(vec![], vec![factor as f32]),
        DType::I32 => Tensor::from_vec_col_major(vec![], vec![round_real_to_i32(factor)?]),
        DType::I64 => Tensor::from_vec_col_major(vec![], vec![round_real_to_i64(factor)?]),
        DType::Bool => Tensor::from_vec_col_major(vec![], vec![finite_real_factor(factor)? != 0.0]),
        DType::C64 => Tensor::from_vec_col_major(vec![], vec![Complex64::new(factor, 0.0)]),
        DType::C32 => Tensor::from_vec_col_major(vec![], vec![Complex32::new(factor as f32, 0.0)]),
        DType::External(_) => Err(Error::invalid_argument(
            "scale_real",
            "dtype",
            "an externally defined scalar has no scale factor",
        )),
    }
}

/// Build the rank-0 host factor that `scale_complex` multiplies a `dtype` tensor by.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_runtime::scale::complex_scale_scalar;
/// use tenferro_tensor::DType;
///
/// let factor = complex_scale_scalar(DType::C64, Complex64::new(0.0, 1.0))?;
/// assert_eq!(factor.as_slice::<Complex64>()?, &[Complex64::new(0.0, 1.0)]);
/// assert!(complex_scale_scalar(DType::F64, Complex64::new(1.0, 0.0)).is_err());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Validation`] with `ValidationError::InvalidArgument` when `dtype` is not
/// complex.
pub fn complex_scale_scalar(dtype: DType, factor: Complex64) -> Result<Tensor> {
    match dtype {
        DType::C64 => Tensor::from_vec_col_major(vec![], vec![factor]),
        DType::C32 => Tensor::from_vec_col_major(
            vec![],
            vec![Complex32::new(factor.re as f32, factor.im as f32)],
        ),
        dtype => Err(Error::invalid_argument(
            "scale_complex",
            "dtype",
            format!("requires complex tensor dtype, got {dtype:?}"),
        )),
    }
}

/// The host scalar a rank-0 float or complex scale factor holds; `None` for an
/// integer or `Bool` factor, which has no [`ContractionScalar`] form.
pub(crate) fn contraction_scalar(scalar: &Tensor) -> Result<Option<ContractionScalar>> {
    Ok(match scalar.dtype() {
        DType::F32 => Some(ContractionScalar::F32(scalar.as_slice::<f32>()?[0])),
        DType::F64 => Some(ContractionScalar::F64(scalar.as_slice::<f64>()?[0])),
        DType::C32 => Some(ContractionScalar::C32(scalar.as_slice::<Complex32>()?[0])),
        DType::C64 => Some(ContractionScalar::C64(scalar.as_slice::<Complex64>()?[0])),
        _ => None,
    })
}
