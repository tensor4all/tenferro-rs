use tenferro_tensor::{DType, SliceConfig, Tensor, TensorElementwise, TensorIndexing};

const OP: &str = "cuda_fft";

/// Complete a one-sided real FFT spectrum using existing same-placement CUDA
/// structural and elementwise operations.
pub(crate) fn complete<S>(
    session: &mut S,
    one_sided: Tensor,
    n: usize,
) -> tenferro_tensor::Result<Tensor>
where
    S: TensorElementwise + TensorIndexing,
{
    if n == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            OP,
            "transform length",
            "must be positive",
        ));
    }
    if !matches!(one_sided.dtype(), DType::C32 | DType::C64) {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            OP,
            DType::C32,
            one_sided.dtype(),
        ));
    }
    let rank = one_sided.shape().len();
    let last = rank.checked_sub(1).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(OP, "rank", "FFT requires rank >= 1")
    })?;
    let half = n
        .checked_div(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(OP, "transform length", "overflows usize")
        })?;
    let actual_half = one_sided.shape().get(last).copied().ok_or_else(|| {
        tenferro_tensor::Error::Internal("one-sided spectrum rank metadata is inconsistent".into())
    })?;
    if actual_half != half {
        return Err(tenferro_tensor::Error::invalid_argument(
            OP,
            "spectrum",
            format!("one-sided spectrum axis length mismatch: expected {half}, got {actual_half}"),
        ));
    }

    let mirror_end = if (n & 1) == 0 {
        half.checked_sub(1).ok_or_else(|| {
            tenferro_tensor::Error::Internal("even Hermitian half-spectrum underflow".into())
        })?
    } else {
        half
    };
    let mut starts = vec![0; rank];
    let mut limits = one_sided.shape().to_vec();
    let strides = vec![1; rank];
    limits[last] = mirror_end;
    starts[last] = 1;
    let mirrored = <S as TensorIndexing>::slice(
        session,
        &one_sided,
        &SliceConfig {
            starts,
            limits,
            strides,
        },
    )?;
    let mirrored = <S as TensorIndexing>::reverse(session, &mirrored, &[last])?;
    let mirrored = <S as TensorElementwise>::conj(session, &mirrored)?;
    let output = <S as TensorIndexing>::concatenate(session, &[&one_sided, &mirrored], last)?;

    let expected_len = if mirror_end == 0 {
        0
    } else {
        half.checked_add(mirror_end - 1).ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(
                OP,
                "output shape",
                "Hermitian spectrum length overflows usize",
            )
        })?
    };
    if output.shape().get(last).copied() != Some(expected_len) {
        return Err(tenferro_tensor::Error::Internal(
            "Hermitian completion returned an unexpected final-axis length".into(),
        ));
    }
    Ok(output)
}
