use tenferro_runtime::{DType, Error, Result};

pub(crate) fn validate_lstsq(
    op: &'static str,
    dtype: DType,
    a_rank: usize,
    b_rank: usize,
    shape: impl FnOnce() -> Result<(usize, usize)>,
    wide_error: impl FnOnce(String) -> Error,
) -> Result<()> {
    ensure_float_or_complex(op, dtype)?;
    ensure_min_rank(op, a_rank, 2)?;
    ensure_min_rank(op, b_rank, 2)?;
    let (m, n) = shape()?;
    if m < n {
        return Err(wide_error(format!(
            "lstsq requires a tall or square matrix (rows {m} >= cols {n}); \
             underdetermined (wide) systems are not supported"
        )));
    }
    Ok(())
}

pub(crate) fn ensure_float_or_complex(op: &'static str, dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        DType::I32 | DType::I64 | DType::Bool => Err(Error::TensorRuntime(
            crate::error::unsupported_dtype(op, dtype),
        )),
    }
}

fn ensure_min_rank(op: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        return Err(Error::TensorRuntime(tenferro_tensor::Error::rank_mismatch(
            op, expected, actual,
        )));
    }
    Ok(())
}
