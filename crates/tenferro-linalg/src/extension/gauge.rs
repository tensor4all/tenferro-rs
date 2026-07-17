use num_complex::{Complex32, Complex64};
use tenferro_tensor::{Error, Tensor};

use super::{EighGauge, QrGauge};

pub(crate) fn apply_eigh_gauge(
    gauge: EighGauge,
    outputs: &mut [Tensor],
) -> tenferro_tensor::Result<()> {
    match gauge {
        EighGauge::Raw => Ok(()),
        EighGauge::CanonicalPivot => apply_canonical_pivot_eigh_gauge(outputs),
    }
}

pub(crate) fn apply_qr_gauge(
    gauge: QrGauge,
    outputs: &mut [Tensor],
) -> tenferro_tensor::Result<()> {
    match gauge {
        QrGauge::Raw => Ok(()),
        QrGauge::PositiveDiagonal => apply_positive_diagonal_qr_gauge(outputs),
    }
}

fn apply_canonical_pivot_eigh_gauge(outputs: &mut [Tensor]) -> tenferro_tensor::Result<()> {
    if outputs.len() != 2 {
        return Err(Error::invalid_argument(
            "tenferro-linalg.eigh",
            "outputs",
            format!(
                "canonical eigh gauge expected two outputs, got {}",
                outputs.len()
            ),
        ));
    }

    let (values_slice, vectors_slice) = outputs.split_at_mut(1);
    let values_shape = values_slice[0].shape().to_vec();
    let vectors = &mut vectors_slice[0];
    let vectors_shape = vectors.shape().to_vec();
    if values_shape.is_empty() || vectors_shape.len() < 2 {
        return Err(Error::invalid_argument(
            "tenferro-linalg.eigh",
            "outputs",
            format!(
                "canonical eigh gauge expected values rank >= 1 and vectors rank >= 2; got values={values_shape:?}, vectors={vectors_shape:?}"
            ),
        ));
    }

    let n = vectors_shape[0];
    if vectors_shape[1] != n || values_shape[0] != n || values_shape[1..] != vectors_shape[2..] {
        return Err(Error::invalid_argument(
            "tenferro-linalg.eigh",
            "outputs",
            format!(
                "canonical eigh gauge expected compatible eigendecomposition shapes, got values={values_shape:?}, vectors={vectors_shape:?}"
            ),
        ));
    }
    let batch_count = checked_batch_count("tenferro-linalg.eigh", &vectors_shape[2..])?;

    match vectors {
        Tensor::F64(vectors) => canonicalize_eigh_gauge_f64(
            vectors.host_data_mut()?,
            n,
            batch_count,
            "tenferro-linalg.eigh",
        ),
        Tensor::F32(vectors) => canonicalize_eigh_gauge_f32(
            vectors.host_data_mut()?,
            n,
            batch_count,
            "tenferro-linalg.eigh",
        ),
        Tensor::C64(vectors) => canonicalize_eigh_gauge_c64(
            vectors.host_data_mut()?,
            n,
            batch_count,
            "tenferro-linalg.eigh",
        ),
        Tensor::C32(vectors) => canonicalize_eigh_gauge_c32(
            vectors.host_data_mut()?,
            n,
            batch_count,
            "tenferro-linalg.eigh",
        ),
        vectors => Err(Error::unsupported(
            "tenferro-linalg.eigh",
            format!("unsupported eigenvector dtype {:?}", vectors.dtype()),
        )),
    }
}

fn apply_positive_diagonal_qr_gauge(outputs: &mut [Tensor]) -> tenferro_tensor::Result<()> {
    if outputs.len() != 2 {
        return Err(Error::invalid_argument(
            "tenferro-linalg.qr",
            "outputs",
            format!(
                "positive-diagonal QR gauge expected two outputs, got {}",
                outputs.len()
            ),
        ));
    }

    let (q_slice, r_slice) = outputs.split_at_mut(1);
    let q = &mut q_slice[0];
    let r = &mut r_slice[0];
    let q_shape = q.shape().to_vec();
    let r_shape = r.shape().to_vec();
    if q_shape.len() < 2 || r_shape.len() < 2 {
        return Err(Error::invalid_argument(
            "tenferro-linalg.qr",
            "outputs",
            format!(
                "positive-diagonal QR gauge expected Q and R rank >= 2; got Q={q_shape:?}, R={r_shape:?}"
            ),
        ));
    }

    let m = q_shape[0];
    let k = q_shape[1];
    let n = r_shape[1];
    if r_shape[0] != k || q_shape[2..] != r_shape[2..] {
        return Err(Error::invalid_argument(
            "tenferro-linalg.qr",
            "outputs",
            format!(
                "positive-diagonal QR gauge expected compatible QR shapes, got Q={q_shape:?}, R={r_shape:?}"
            ),
        ));
    }
    let batch_count = checked_batch_count("tenferro-linalg.qr", &q_shape[2..])?;

    match (q, r) {
        (Tensor::F64(q), Tensor::F64(r)) => canonicalize_qr_gauge_f64(
            q.host_data_mut()?,
            r.host_data_mut()?,
            m,
            k,
            n,
            batch_count,
            "tenferro-linalg.qr",
        ),
        (Tensor::F32(q), Tensor::F32(r)) => canonicalize_qr_gauge_f32(
            q.host_data_mut()?,
            r.host_data_mut()?,
            m,
            k,
            n,
            batch_count,
            "tenferro-linalg.qr",
        ),
        (Tensor::C64(q), Tensor::C64(r)) => canonicalize_qr_gauge_c64(
            q.host_data_mut()?,
            r.host_data_mut()?,
            m,
            k,
            n,
            batch_count,
            "tenferro-linalg.qr",
        ),
        (Tensor::C32(q), Tensor::C32(r)) => canonicalize_qr_gauge_c32(
            q.host_data_mut()?,
            r.host_data_mut()?,
            m,
            k,
            n,
            batch_count,
            "tenferro-linalg.qr",
        ),
        (q, r) => Err(Error::dtype_mismatch(
            "tenferro-linalg.qr",
            q.dtype(),
            r.dtype(),
        )),
    }
}

fn canonicalize_eigh_gauge_f64(
    vectors: &mut [f64],
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let batch_stride = checked_square_len(op, n)?;
    for batch in 0..batch_count {
        let base = checked_batch_offset(op, batch, batch_stride)?;
        for col in 0..n {
            let pivot = max_abs_pivot_f64(vectors, base, n, col);
            if vectors[base + pivot + n * col] < 0.0 {
                for row in 0..n {
                    let offset = base + row + n * col;
                    vectors[offset] = -vectors[offset];
                }
            }
        }
    }
    Ok(())
}

fn canonicalize_eigh_gauge_f32(
    vectors: &mut [f32],
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let batch_stride = checked_square_len(op, n)?;
    for batch in 0..batch_count {
        let base = checked_batch_offset(op, batch, batch_stride)?;
        for col in 0..n {
            let pivot = max_abs_pivot_f32(vectors, base, n, col);
            if vectors[base + pivot + n * col] < 0.0 {
                for row in 0..n {
                    let offset = base + row + n * col;
                    vectors[offset] = -vectors[offset];
                }
            }
        }
    }
    Ok(())
}

fn canonicalize_eigh_gauge_c64(
    vectors: &mut [Complex64],
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let batch_stride = checked_square_len(op, n)?;
    for batch in 0..batch_count {
        let base = checked_batch_offset(op, batch, batch_stride)?;
        for col in 0..n {
            let pivot = max_abs_pivot_c64(vectors, base, n, col);
            let pivot_value = vectors[base + pivot + n * col];
            let pivot_norm = pivot_value.norm();
            if pivot_norm == 0.0 {
                continue;
            }
            let phase = pivot_value.conj() / pivot_norm;
            for row in 0..n {
                let offset = base + row + n * col;
                vectors[offset] *= phase;
            }
        }
    }
    Ok(())
}

fn canonicalize_eigh_gauge_c32(
    vectors: &mut [Complex32],
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let batch_stride = checked_square_len(op, n)?;
    for batch in 0..batch_count {
        let base = checked_batch_offset(op, batch, batch_stride)?;
        for col in 0..n {
            let pivot = max_abs_pivot_c32(vectors, base, n, col);
            let pivot_value = vectors[base + pivot + n * col];
            let pivot_norm = pivot_value.norm();
            if pivot_norm == 0.0 {
                continue;
            }
            let phase = pivot_value.conj() / pivot_norm;
            for row in 0..n {
                let offset = base + row + n * col;
                vectors[offset] *= phase;
            }
        }
    }
    Ok(())
}

fn canonicalize_qr_gauge_f64(
    q: &mut [f64],
    r: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let q_stride = checked_matrix_len(op, m, k)?;
    let r_stride = checked_matrix_len(op, k, n)?;
    for batch in 0..batch_count {
        let q_base = checked_batch_offset(op, batch, q_stride)?;
        let r_base = checked_batch_offset(op, batch, r_stride)?;
        for diag in 0..k {
            if r[r_base + diag + k * diag] < 0.0 {
                for row in 0..m {
                    q[q_base + row + m * diag] = -q[q_base + row + m * diag];
                }
                for col in 0..n {
                    r[r_base + diag + k * col] = -r[r_base + diag + k * col];
                }
            }
        }
    }
    Ok(())
}

fn canonicalize_qr_gauge_f32(
    q: &mut [f32],
    r: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let q_stride = checked_matrix_len(op, m, k)?;
    let r_stride = checked_matrix_len(op, k, n)?;
    for batch in 0..batch_count {
        let q_base = checked_batch_offset(op, batch, q_stride)?;
        let r_base = checked_batch_offset(op, batch, r_stride)?;
        for diag in 0..k {
            if r[r_base + diag + k * diag] < 0.0 {
                for row in 0..m {
                    q[q_base + row + m * diag] = -q[q_base + row + m * diag];
                }
                for col in 0..n {
                    r[r_base + diag + k * col] = -r[r_base + diag + k * col];
                }
            }
        }
    }
    Ok(())
}

fn canonicalize_qr_gauge_c64(
    q: &mut [Complex64],
    r: &mut [Complex64],
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let q_stride = checked_matrix_len(op, m, k)?;
    let r_stride = checked_matrix_len(op, k, n)?;
    for batch in 0..batch_count {
        let q_base = checked_batch_offset(op, batch, q_stride)?;
        let r_base = checked_batch_offset(op, batch, r_stride)?;
        for diag in 0..k {
            let diagonal = r[r_base + diag + k * diag];
            let norm = diagonal.norm();
            if norm == 0.0 {
                continue;
            }
            let q_phase = diagonal / norm;
            let r_phase = diagonal.conj() / norm;
            for row in 0..m {
                q[q_base + row + m * diag] *= q_phase;
            }
            for col in 0..n {
                r[r_base + diag + k * col] *= r_phase;
            }
        }
    }
    Ok(())
}

fn canonicalize_qr_gauge_c32(
    q: &mut [Complex32],
    r: &mut [Complex32],
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let q_stride = checked_matrix_len(op, m, k)?;
    let r_stride = checked_matrix_len(op, k, n)?;
    for batch in 0..batch_count {
        let q_base = checked_batch_offset(op, batch, q_stride)?;
        let r_base = checked_batch_offset(op, batch, r_stride)?;
        for diag in 0..k {
            let diagonal = r[r_base + diag + k * diag];
            let norm = diagonal.norm();
            if norm == 0.0 {
                continue;
            }
            let q_phase = diagonal / norm;
            let r_phase = diagonal.conj() / norm;
            for row in 0..m {
                q[q_base + row + m * diag] *= q_phase;
            }
            for col in 0..n {
                r[r_base + diag + k * col] *= r_phase;
            }
        }
    }
    Ok(())
}

fn max_abs_pivot_f64(values: &[f64], base: usize, rows: usize, col: usize) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = values[base + rows * col].abs();
    for row in 1..rows {
        let candidate_abs = values[base + row + rows * col].abs();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn max_abs_pivot_f32(values: &[f32], base: usize, rows: usize, col: usize) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = values[base + rows * col].abs();
    for row in 1..rows {
        let candidate_abs = values[base + row + rows * col].abs();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn max_abs_pivot_c64(values: &[Complex64], base: usize, rows: usize, col: usize) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = values[base + rows * col].norm_sqr();
    for row in 1..rows {
        let candidate_abs = values[base + row + rows * col].norm_sqr();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn max_abs_pivot_c32(values: &[Complex32], base: usize, rows: usize, col: usize) -> usize {
    let mut pivot = 0;
    let mut pivot_abs = values[base + rows * col].norm_sqr();
    for row in 1..rows {
        let candidate_abs = values[base + row + rows * col].norm_sqr();
        if candidate_abs > pivot_abs {
            pivot = row;
            pivot_abs = candidate_abs;
        }
    }
    pivot
}

fn checked_batch_count(op: &'static str, batch_shape: &[usize]) -> tenferro_tensor::Result<usize> {
    batch_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| invalid_config(op, "batch element count overflow"))
    })
}

fn checked_square_len(op: &'static str, n: usize) -> tenferro_tensor::Result<usize> {
    checked_matrix_len(op, n, n)
}

fn checked_matrix_len(
    op: &'static str,
    rows: usize,
    cols: usize,
) -> tenferro_tensor::Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_config(op, "matrix element count overflow"))
}

fn checked_batch_offset(
    op: &'static str,
    batch: usize,
    stride: usize,
) -> tenferro_tensor::Result<usize> {
    batch
        .checked_mul(stride)
        .ok_or_else(|| invalid_config(op, "batch offset overflow"))
}

fn invalid_config(op: &'static str, message: &'static str) -> Error {
    Error::invalid_argument(op, "configuration", message)
}
