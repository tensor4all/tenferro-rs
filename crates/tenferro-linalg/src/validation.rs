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

// Validate the complete solve before a composite performs its first LU kernel.
#[cfg(feature = "autodiff")]
pub(crate) fn validate_solve_inputs(
    a_dtype: DType,
    a: &[usize],
    b_dtype: DType,
    b: &[usize],
) -> Result<()> {
    const OP: &str = "solve";
    ensure_float_or_complex(OP, a_dtype)?;
    if a_dtype != b_dtype {
        return Err(tenferro_tensor::Error::dtype_mismatch(OP, a_dtype, b_dtype).into());
    }
    ensure_min_rank(OP, a.len(), 2)?;
    ensure_min_rank(OP, b.len(), 2)?;
    // INVARIANT: ranks were checked above; compute dimensions precede batches.
    let rhs_batch = if b.len() == a.len() - 1 {
        &b[1..]
    } else {
        &b[2..]
    };
    if a[0] != a[1] || b[0] != a[0] || rhs_batch != &a[2..] {
        return Err(tenferro_tensor::Error::shape_mismatch(OP, a.to_vec(), b.to_vec()).into());
    }
    Ok(())
}

pub(crate) fn ensure_float_or_complex(op: &'static str, dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        DType::I32 | DType::I64 | DType::Bool | DType::External(_) => Err(Error::TensorRuntime(
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
