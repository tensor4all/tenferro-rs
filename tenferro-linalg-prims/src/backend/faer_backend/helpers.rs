use tenferro_device::{Error, Result};

pub(crate) fn check_len(op: &str, field: &str, got: usize, need: usize) -> Result<()> {
    if got < need {
        return Err(Error::InvalidArgument(format!(
            "{op}: {field} slice length {got} < required {need}"
        )));
    }
    Ok(())
}

pub(crate) fn singular_matrix_error(op: &str) -> Error {
    Error::InvalidArgument(format!("{op}: matrix is singular"))
}

pub(crate) fn non_finite_result_error(op: &str) -> Error {
    Error::InvalidArgument(format!("{op}: solution contains non-finite values"))
}

pub(crate) fn zero_diagonal_error(op: &str, index: usize) -> Error {
    Error::InvalidArgument(format!("{op}: zero diagonal at index {index}"))
}

pub(crate) fn complex_is_finite<T: num_traits::Float>(value: num_complex::Complex<T>) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
