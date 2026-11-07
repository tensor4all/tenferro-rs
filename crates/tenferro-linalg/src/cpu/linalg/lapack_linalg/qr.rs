use crate::cpu::linalg::{compact_host_data, HostView};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;
use tlinalg_blas::Op;

use super::helpers::{
    checked_product, has_zero_dim, matrix_dims, matrix_with_batch_shape, pooled_output, provider,
    split_core_and_batch_result, tensor_from_vec_with_template, validate_buffer_len,
    vector_with_batch_shape, LapackLinalg,
};

type CompactQrResult<T> = (TypedTensor<T>, TypedTensor<T>);

/// The leading `rows x cols` upper triangle of a column-major factor with `source_rows` rows.
fn leading_upper_triangle_from_lapack<T: Copy + Default>(
    data: &[T],
    source_rows: usize,
    rows: usize,
    cols: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let len = checked_product("lapack_linalg", "upper triangle", &[rows, cols])?;
    let mut out = vec![T::default(); len];
    for col in 0..cols {
        for row in 0..rows.min(col + 1) {
            out[row + col * rows] = data[row + col * source_rows];
        }
    }
    Ok(out)
}

/// Factor one compact `rows x cols` matrix in place; returns its `tau` coefficients.
fn geqrf_2d<T: LapackLinalg>(
    buffers: &mut BufferPool,
    data: &mut [T],
    rows: usize,
    cols: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    validate_buffer_len(
        "compact_factor_2d",
        "matrix",
        data.len(),
        checked_product("compact_factor_2d", "matrix", &[rows, cols])?,
    )?;
    let mut tau = Vec::new();
    T::householder_factor(buffers, rows, cols, data, &mut tau)
        .map_err(provider(Op::HouseholderFactor))?;
    Ok(tau)
}

/// Apply the compact reflectors to `c` in place; the provider takes its scratch from the pool.
// INVARIANT: these buffers and dimensions mirror the provider reflector ABI.
#[allow(clippy::too_many_arguments)]
fn apply_reflectors_2d<T: LapackLinalg>(
    buffers: &mut BufferPool,
    a: &[T],
    a_cols: usize,
    tau: &[T],
    c: &mut [T],
    m: usize,
    p: usize,
    k: usize,
    transpose: bool,
) -> tenferro_tensor::Result<()> {
    T::apply_reflectors(buffers, m, a_cols, p, k, transpose, a, tau, c)
        .map_err(provider(Op::HouseholderApply))
}

fn validate_state<T: 'static>(
    packed: &HostView<'_, T>,
    tau: &HostView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize, usize)> {
    let (m, n) = matrix_dims(packed, op)?;
    if tau.shape().len() != 1 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            1,
            tau.shape().len(),
        ));
    }
    let k = m.min(n);
    if tau.shape()[0] != k {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![k],
            vec![tau.shape()[0]],
        ));
    }
    Ok((m, n, k))
}

pub(crate) fn compact_factor_2d<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &HostView<'_, T>,
) -> tenferro_tensor::Result<CompactQrResult<T>> {
    let (m, n) = matrix_dims(input, "compact_factor_2d")?;
    let k = m.min(n);
    let input_data = compact_host_data("compact_factor_2d", input)?;
    let mut packed = buffers.acquire_with_capacity::<T>(input_data.len());
    packed.extend_from_slice(input_data);
    let tau = if has_zero_dim(input.shape()) {
        Vec::new()
    } else {
        geqrf_2d(buffers, &mut packed, m, n)?
    };
    Ok((
        tensor_from_vec_with_template(vec![m, n], packed, input)?,
        tensor_from_vec_with_template(vec![k], tau, input)?,
    ))
}

pub(crate) fn append_2d<T: LapackLinalg>(
    buffers: &mut BufferPool,
    packed: &HostView<'_, T>,
    tau: &HostView<'_, T>,
    input: &HostView<'_, T>,
) -> tenferro_tensor::Result<CompactQrResult<T>> {
    let (m, old_n, old_k) = validate_state(packed, tau, "append_2d")?;
    let (input_m, p) = matrix_dims(input, "append_2d")?;
    if input_m != m {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "append_2d",
            vec![m, p],
            vec![input_m, p],
        ));
    }
    if p == 0 {
        return Ok((
            tensor_from_vec_with_template(
                packed.shape().to_vec(),
                compact_host_data("append_2d", packed)?.to_vec(),
                packed,
            )?,
            tensor_from_vec_with_template(
                tau.shape().to_vec(),
                compact_host_data("append_2d", tau)?.to_vec(),
                tau,
            )?,
        ));
    }
    let new_n = old_n.checked_add(p).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument("append_2d", "shape", "column count overflow")
    })?;
    let new_k = m.min(new_n);
    let input_data = compact_host_data("append_2d", input)?;
    let mut transformed = buffers.acquire_with_capacity::<T>(input_data.len());
    transformed.extend_from_slice(input_data);
    apply_reflectors_2d::<T>(
        buffers,
        compact_host_data("append_2d", packed)?,
        old_n,
        compact_host_data("append_2d", tau)?,
        &mut transformed,
        m,
        p,
        old_k,
        true,
    )?;

    let trailing_rows = m - old_k;
    let mut trailing = buffers.acquire_with_capacity::<T>(checked_product(
        "append_2d",
        "trailing block",
        &[trailing_rows, p],
    )?);
    for col in 0..p {
        trailing.extend_from_slice(&transformed[col * m + old_k..(col + 1) * m]);
    }
    let new_tau = if trailing_rows == 0 {
        Vec::new()
    } else {
        geqrf_2d(buffers, &mut trailing, trailing_rows, p)?
    };

    let packed_len = checked_product("append_2d", "packed state", &[m, new_n])?;
    let mut output = buffers.acquire_with_capacity::<T>(packed_len);
    output.extend_from_slice(compact_host_data("append_2d", packed)?);
    for col in 0..p {
        output.extend_from_slice(&transformed[col * m..col * m + old_k]);
        output.extend_from_slice(&trailing[col * trailing_rows..(col + 1) * trailing_rows]);
    }
    let tau_data = compact_host_data("append_2d", tau)?;
    let mut output_tau = buffers.acquire_with_capacity::<T>(new_k);
    output_tau.extend_from_slice(tau_data);
    output_tau.extend_from_slice(&new_tau);
    validate_buffer_len("append_2d", "coefficients", output_tau.len(), new_k)?;
    Ok((
        tensor_from_vec_with_template(vec![m, new_n], output, packed)?,
        tensor_from_vec_with_template(vec![new_k], output_tau, packed)?,
    ))
}

pub(crate) fn from_factors_2d<T: LapackLinalg>(
    buffers: &mut BufferPool,
    q: &HostView<'_, T>,
    r: &HostView<'_, T>,
) -> tenferro_tensor::Result<CompactQrResult<T>> {
    let (m, s) = matrix_dims(q, "from_factors_2d")?;
    let (r_s, n) = matrix_dims(r, "from_factors_2d")?;
    if r_s != s {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "from_factors_2d",
            vec![s, n],
            vec![r_s, n],
        ));
    }
    if s > m.min(n) {
        return Err(tenferro_tensor::Error::invalid_argument(
            "from_factors_2d",
            "shape",
            "Q column count must not exceed min(Q rows, R columns)",
        ));
    }
    let k = m.min(n);
    let r_data = compact_host_data("from_factors_2d", r)?;
    for col in 0..n.min(s) {
        for row in col + 1..s {
            if r_data[row + col * s] != T::default() {
                return Err(tenferro_tensor::Error::invalid_argument(
                    "from_factors_2d",
                    "R",
                    "must be upper trapezoidal",
                ));
            }
        }
    }
    if has_zero_dim(q.shape()) || has_zero_dim(r.shape()) {
        return Ok((
            tensor_from_vec_with_template(
                vec![m, n],
                buffers.acquire_zeroed::<T>(checked_product(
                    "from_factors_2d",
                    "packed state",
                    &[m, n],
                )?),
                q,
            )?,
            tensor_from_vec_with_template(vec![k], buffers.acquire_zeroed::<T>(k), q)?,
        ));
    }

    let q_data = compact_host_data("from_factors_2d", q)?;
    let mut q_packed = buffers.acquire_with_capacity::<T>(q_data.len());
    q_packed.extend_from_slice(q_data);
    let q_tau = geqrf_2d(buffers, &mut q_packed, m, s)?;
    let t = leading_upper_triangle_from_lapack(&q_packed, m, s, s)?;
    let mut folded =
        buffers.acquire_zeroed::<T>(checked_product("from_factors_2d", "folded R", &[s, n])?);
    T::gemm_2d(&t, s, s, r_data, n, &mut folded)?;

    let mut packed =
        buffers.acquire_zeroed::<T>(checked_product("from_factors_2d", "packed state", &[m, n])?);
    for col in 0..n {
        for row in 0..m {
            packed[row + col * m] = if col < s && row > col {
                q_packed[row + col * m]
            } else if row < s {
                folded[row + col * s]
            } else {
                T::default()
            };
        }
    }
    let mut tau = buffers.acquire_zeroed::<T>(k);
    tau[..s].copy_from_slice(&q_tau);
    Ok((
        tensor_from_vec_with_template(vec![m, n], packed, q)?,
        tensor_from_vec_with_template(vec![k], tau, q)?,
    ))
}

pub(crate) fn raw_r_2d<T: LapackLinalg>(
    packed: &HostView<'_, T>,
    tau: &HostView<'_, T>,
    positive_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (m, n, k) = validate_state(packed, tau, "raw_r_2d")?;
    let mut r = vec![T::default(); checked_product("raw_r_2d", "R", &[k, n])?];
    let packed_data = compact_host_data("raw_r_2d", packed)?;
    for col in 0..n {
        for row in 0..k.min(col + 1) {
            r[row + col * k] = packed_data[row + col * m];
        }
    }
    if positive_diagonal {
        for diag in 0..k {
            let phase = T::r_phase(r[diag + diag * k]);
            for col in 0..n {
                r[diag + col * k] = r[diag + col * k] * phase;
            }
        }
    }
    tensor_from_vec_with_template(vec![k, n], r, packed)
}

pub(crate) fn q_columns_2d<T: LapackLinalg>(
    buffers: &mut BufferPool,
    packed: &HostView<'_, T>,
    tau: &HostView<'_, T>,
    start: usize,
    end: usize,
    positive_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (m, n, k) = validate_state(packed, tau, "q_columns_2d")?;
    // Full-Q width: columns `k..m` span the orthogonal complement of the
    // input's column space. Applying the compact reflectors to those identity
    // columns produces them directly, so no separate `?orgqr` path is needed.
    if start > end || end > m {
        return Err(tenferro_tensor::Error::invalid_argument(
            "q_columns_2d",
            "range",
            format!("range {start}..{end} is outside 0..{m}"),
        ));
    }
    let columns = end - start;
    let mut q = vec![T::default(); checked_product("q_columns_2d", "Q", &[m, columns])?];
    for col in 0..columns {
        q[start + col + col * m] = T::unit();
    }
    if columns != 0 {
        apply_reflectors_2d::<T>(
            buffers,
            compact_host_data("q_columns_2d", packed)?,
            n,
            compact_host_data("q_columns_2d", tau)?,
            &mut q,
            m,
            columns,
            k,
            false,
        )?;
        if positive_diagonal {
            let packed_data = compact_host_data("q_columns_2d", packed)?;
            // The gauge is defined by R's diagonal, so it fixes only the first
            // `k` columns; a complement column has no diagonal to fix.
            for col in 0..columns {
                let diag = start + col;
                if diag >= k {
                    break;
                }
                let phase = T::q_phase(packed_data[diag + diag * m]);
                for row in 0..m {
                    q[row + col * m] = q[row + col * m] * phase;
                }
            }
        }
    }
    tensor_from_vec_with_template(vec![m, columns], q, packed)
}

pub(crate) fn qr<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &HostView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "qr";
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, OP)?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    let q_shape = matrix_with_batch_shape(m, k, batch_shape);
    let r_shape = matrix_with_batch_shape(k, n, batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(q_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(r_shape, Vec::new(), input)?,
        ]);
    }
    let mut q = pooled_output::<T>(buffers, OP, "Q", &q_shape)?;
    let mut r = pooled_output::<T>(buffers, OP, "R", &r_shape)?;
    let view = input.clone().into_dynamic();
    T::qr(buffers, super::super::raw_view(OP, &view)?, &mut q, &mut r).map_err(provider(Op::Qr))?;
    Ok(vec![
        tensor_from_vec_with_template(q_shape, q, input)?,
        tensor_from_vec_with_template(r_shape, r, input)?,
    ])
}

/// Column-pivoted QR with the host's rank decision.
///
/// The host screens non-finite input first (as the faer route does, so both report it first); the
/// provider gives an all-zero item the canonical zero-rank factors and factors the rest in one
/// batched call; the host decides each rank from the `R` diagonal.
pub(crate) fn rank_revealing_qr<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &HostView<'_, T>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<crate::cpu::linalg::rank_revealing_qr::TypedRrqr<T>> {
    const OP: &str = "rank_revealing_qr";
    crate::rank_revealing_qr::validate_rank_revealing_qr_options(OP, options)?;
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, OP)?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    let q_shape = matrix_with_batch_shape(m, k, batch_shape);
    let r_shape = matrix_with_batch_shape(k, n, batch_shape);
    let p_shape = vector_with_batch_shape(n, batch_shape);
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let view = input.clone().into_dynamic();
    crate::cpu::linalg::rank_revealing_qr::screen_non_finite(OP, &view, batch, |value: T| {
        value.is_finite_value()
    })?;
    let mut q = pooled_output::<T>(buffers, OP, "Q", &q_shape)?;
    let mut r = pooled_output::<T>(buffers, OP, "R", &r_shape)?;
    let mut permutation = Vec::with_capacity(checked_product(OP, "permutation", &p_shape)?);
    if batch > 0 {
        T::rank_revealing_qr(
            buffers,
            super::super::raw_view(OP, &view)?,
            tlinalg_blas::qr::RankRevealingQrOutputs {
                q: &mut q,
                r: &mut r,
                permutation: &mut permutation,
            },
        )
        .map_err(provider(Op::RankRevealingQr))?;
    }
    let ranks = crate::cpu::linalg::rank_revealing_qr::batch_ranks(
        &r,
        k,
        n,
        batch,
        options,
        |value: T| value.rank_magnitude(),
    )?;
    Ok(crate::RankRevealingQrResult {
        q: tensor_from_vec_with_template(q_shape, q, input)?,
        r: tensor_from_vec_with_template(r_shape, r, input)?,
        column_permutation: tensor_from_vec_with_template(p_shape, permutation, input)?,
        rank: tensor_from_vec_with_template(batch_shape.to_vec(), ranks, input)?,
    })
}

#[cfg(test)]
#[path = "qr_tests.rs"]
mod tests;
