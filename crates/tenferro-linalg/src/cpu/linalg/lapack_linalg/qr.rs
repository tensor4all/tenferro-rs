use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    check_lapack_info, checked_product, dim_i32, has_zero_dim, leading_upper_triangle_from_lapack,
    matrix_dims, matrix_with_batch_shape, pooled_zeroed, release_scratch,
    split_core_and_batch_result, tensor_from_vec_with_template, work_len,
};

type CompactQrResult<T> = (TypedTensor<T>, TypedTensor<T>);

pub(crate) trait LapackQr:
    Clone + Copy + Default + PartialEq + PoolScalar + std::ops::Mul<Output = Self>
{
    fn geqrf_2d(
        buffers: &mut BufferPool,
        data: &mut [Self],
        rows: usize,
        cols: usize,
    ) -> tenferro_tensor::Result<Vec<Self>>;

    /// Apply the compact reflectors to `c` in place.
    ///
    /// The workspace query and `work` buffer come from `buffers`, so a repeated
    /// QR does not pay the allocator for vendor scratch on every call.
    #[allow(clippy::too_many_arguments)]
    fn apply_reflectors_2d(
        buffers: &mut BufferPool,
        a: &[Self],
        a_cols: usize,
        tau: &[Self],
        c: &mut [Self],
        m: usize,
        p: usize,
        k: usize,
        transpose: bool,
    ) -> tenferro_tensor::Result<()>;

    fn gemm_2d(
        a: &[Self],
        a_rows: usize,
        a_cols: usize,
        b: &[Self],
        b_cols: usize,
        c: &mut [Self],
    ) -> tenferro_tensor::Result<()>;

    fn one() -> Self;
    fn r_phase(diagonal: Self) -> Self;
    fn q_phase(diagonal: Self) -> Self;

    fn factor_work(
        m: i32,
        n: i32,
        data: &mut [Self],
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
    ) -> tenferro_tensor::Result<()>;

    fn generate_work(
        m: i32,
        k: i32,
        data: &mut [Self],
        tau: &[Self],
        work: &mut [Self],
        lwork: i32,
    ) -> tenferro_tensor::Result<()>;

    fn qr_work_len(value: Self) -> tenferro_tensor::Result<i32>;
}

fn validate_state<T>(
    packed: &TypedTensor<T>,
    tau: &TypedTensor<T>,
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

fn validate_buffer_len(
    op: &'static str,
    role: &'static str,
    actual: usize,
    expected: usize,
) -> tenferro_tensor::Result<()> {
    if actual != expected {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            role,
            format!("expected {expected} elements, got {actual}"),
        ));
    }
    Ok(())
}

pub(crate) fn compact_factor_2d<T: LapackQr>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<CompactQrResult<T>> {
    let (m, n) = matrix_dims(input, "compact_factor_2d")?;
    let k = m.min(n);
    let input_data = input.host_data()?;
    let mut packed = buffers.acquire_with_capacity::<T>(input_data.len());
    packed.extend_from_slice(input_data);
    let tau = if has_zero_dim(input.shape()) {
        Vec::new()
    } else {
        T::geqrf_2d(buffers, &mut packed, m, n)?
    };
    Ok((
        tensor_from_vec_with_template(vec![m, n], packed, input)?,
        tensor_from_vec_with_template(vec![k], tau, input)?,
    ))
}

pub(crate) fn append_2d<T: LapackQr>(
    buffers: &mut BufferPool,
    packed: &TypedTensor<T>,
    tau: &TypedTensor<T>,
    input: &TypedTensor<T>,
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
                packed.host_data()?.to_vec(),
                packed,
            )?,
            tensor_from_vec_with_template(tau.shape().to_vec(), tau.host_data()?.to_vec(), tau)?,
        ));
    }
    let new_n = old_n.checked_add(p).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument("append_2d", "shape", "column count overflow")
    })?;
    let new_k = m.min(new_n);
    let input_data = input.host_data()?;
    let mut transformed = buffers.acquire_with_capacity::<T>(input_data.len());
    transformed.extend_from_slice(input_data);
    T::apply_reflectors_2d(
        buffers,
        packed.host_data()?,
        old_n,
        tau.host_data()?,
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
        T::geqrf_2d(buffers, &mut trailing, trailing_rows, p)?
    };

    let packed_len = checked_product("append_2d", "packed state", &[m, new_n])?;
    let mut output = buffers.acquire_with_capacity::<T>(packed_len);
    output.extend_from_slice(packed.host_data()?);
    for col in 0..p {
        output.extend_from_slice(&transformed[col * m..col * m + old_k]);
        output.extend_from_slice(&trailing[col * trailing_rows..(col + 1) * trailing_rows]);
    }
    let tau_data = tau.host_data()?;
    let mut output_tau = buffers.acquire_with_capacity::<T>(new_k);
    output_tau.extend_from_slice(tau_data);
    output_tau.extend_from_slice(&new_tau);
    validate_buffer_len("append_2d", "coefficients", output_tau.len(), new_k)?;
    Ok((
        tensor_from_vec_with_template(vec![m, new_n], output, packed)?,
        tensor_from_vec_with_template(vec![new_k], output_tau, packed)?,
    ))
}

pub(crate) fn from_factors_2d<T: LapackQr>(
    buffers: &mut BufferPool,
    q: &TypedTensor<T>,
    r: &TypedTensor<T>,
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
    let r_data = r.host_data()?;
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

    let q_data = q.host_data()?;
    let mut q_packed = buffers.acquire_with_capacity::<T>(q_data.len());
    q_packed.extend_from_slice(q_data);
    let q_tau = T::geqrf_2d(buffers, &mut q_packed, m, s)?;
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

pub(crate) fn raw_r_2d<T: LapackQr>(
    packed: &TypedTensor<T>,
    tau: &TypedTensor<T>,
    positive_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (m, n, k) = validate_state(packed, tau, "raw_r_2d")?;
    let mut r = vec![T::default(); checked_product("raw_r_2d", "R", &[k, n])?];
    let packed_data = packed.host_data()?;
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

pub(crate) fn q_columns_2d<T: LapackQr>(
    buffers: &mut BufferPool,
    packed: &TypedTensor<T>,
    tau: &TypedTensor<T>,
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
        q[start + col + col * m] = T::one();
    }
    if columns != 0 {
        T::apply_reflectors_2d(
            buffers,
            packed.host_data()?,
            n,
            tau.host_data()?,
            &mut q,
            m,
            columns,
            k,
            false,
        )?;
        if positive_diagonal {
            let packed_data = packed.host_data()?;
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

macro_rules! impl_real_qr {
    ($scalar:ty, $geqrf:path, $orgqr:path, $ormqr:path, $gemm:path, $geqrf_name:literal, $orgqr_name:literal, $ormqr_name:literal) => {
        impl LapackQr for $scalar {
            fn geqrf_2d(
                buffers: &mut BufferPool,
                data: &mut [Self],
                rows: usize,
                cols: usize,
            ) -> tenferro_tensor::Result<Vec<Self>> {
                let k = rows.min(cols);
                validate_buffer_len(
                    "compact_factor_2d",
                    "matrix",
                    data.len(),
                    checked_product("compact_factor_2d", "matrix", &[rows, cols])?,
                )?;
                if k == 0 {
                    return Ok(Vec::new());
                }
                let rows_i32 = dim_i32(rows, "compact_factor_2d")?;
                let cols_i32 = dim_i32(cols, "compact_factor_2d")?;
                let mut tau = pooled_zeroed::<$scalar>(buffers, k);
                let mut query = pooled_zeroed::<$scalar>(buffers, 1);
                let mut info = 0;
                // SAFETY: validated column-major storage covers rows*cols, tau has
                // min(rows, cols) entries, and lwork=-1 writes only the query slot.
                unsafe {
                    $geqrf(
                        rows_i32, cols_i32, data, rows_i32, &mut tau, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info(
                    "compact_factor_2d",
                    concat!($geqrf_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0] as f64, "compact_factor_2d", $geqrf_name)?;
                let mut work = pooled_zeroed::<$scalar>(buffers, lwork as usize);
                // SAFETY: dimensions and workspace come from checked input and the
                // successful LAPACK query; all mutable slices remain live and disjoint.
                unsafe {
                    $geqrf(
                        rows_i32, cols_i32, data, rows_i32, &mut tau, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("compact_factor_2d", $geqrf_name, info)?;
                release_scratch(buffers, query);
                release_scratch(buffers, work);
                Ok(tau)
            }

            #[allow(clippy::too_many_arguments)]
            fn apply_reflectors_2d(
                buffers: &mut BufferPool,
                a: &[Self],
                a_cols: usize,
                tau: &[Self],
                c: &mut [Self],
                m: usize,
                p: usize,
                k: usize,
                transpose: bool,
            ) -> tenferro_tensor::Result<()> {
                if k > a_cols || k > m {
                    return Err(tenferro_tensor::Error::invalid_argument(
                        "apply_reflectors_2d",
                        "dimensions",
                        "reflector count exceeds matrix dimensions",
                    ));
                }
                validate_buffer_len(
                    "apply_reflectors_2d",
                    "A",
                    a.len(),
                    checked_product("apply_reflectors_2d", "A", &[m, a_cols])?,
                )?;
                validate_buffer_len("apply_reflectors_2d", "tau", tau.len(), k)?;
                validate_buffer_len(
                    "apply_reflectors_2d",
                    "C",
                    c.len(),
                    checked_product("apply_reflectors_2d", "C", &[m, p])?,
                )?;
                if m == 0 || p == 0 || k == 0 {
                    return Ok(());
                }
                let m_i32 = dim_i32(m, "apply_reflectors_2d")?;
                let p_i32 = dim_i32(p, "apply_reflectors_2d")?;
                let k_i32 = dim_i32(k, "apply_reflectors_2d")?;
                let mut query = pooled_zeroed::<$scalar>(buffers, 1);
                let mut info = 0;
                let trans = if transpose { b'T' } else { b'N' };
                // SAFETY: validated A, tau, and C slices satisfy LAPACK dimensions;
                // lwork=-1 writes only the single query element.
                unsafe {
                    $ormqr(
                        b'L', trans, m_i32, p_i32, k_i32, a, m_i32, tau, c, m_i32, &mut query, -1,
                        &mut info,
                    );
                }
                check_lapack_info(
                    "apply_reflectors_2d",
                    concat!($ormqr_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0] as f64, "apply_reflectors_2d", $ormqr_name)?;
                let mut work = pooled_zeroed::<$scalar>(buffers, lwork as usize);
                // SAFETY: checked dimensions and queried workspace cover the full
                // reflector application; A/tau are read-only and C is uniquely mutable.
                unsafe {
                    $ormqr(
                        b'L', trans, m_i32, p_i32, k_i32, a, m_i32, tau, c, m_i32, &mut work,
                        lwork, &mut info,
                    );
                }
                let result = check_lapack_info("apply_reflectors_2d", $ormqr_name, info);
                release_scratch(buffers, query);
                release_scratch(buffers, work);
                result
            }

            fn gemm_2d(
                a: &[Self],
                a_rows: usize,
                a_cols: usize,
                b: &[Self],
                b_cols: usize,
                c: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                validate_buffer_len(
                    "from_factors_2d",
                    "T",
                    a.len(),
                    checked_product("from_factors_2d", "T", &[a_rows, a_cols])?,
                )?;
                validate_buffer_len(
                    "from_factors_2d",
                    "R",
                    b.len(),
                    checked_product("from_factors_2d", "R", &[a_cols, b_cols])?,
                )?;
                validate_buffer_len(
                    "from_factors_2d",
                    "folded R",
                    c.len(),
                    checked_product("from_factors_2d", "folded R", &[a_rows, b_cols])?,
                )?;
                let m = dim_i32(a_rows, "from_factors_2d")?;
                let n = dim_i32(b_cols, "from_factors_2d")?;
                let k = dim_i32(a_cols, "from_factors_2d")?;
                // SAFETY: the checked slice lengths cover m*k, k*n, and m*n
                // column-major elements; dimensions and leading dimensions fit LP64 i32.
                unsafe {
                    $gemm(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        m,
                        n,
                        k,
                        1.0,
                        a.as_ptr(),
                        m,
                        b.as_ptr(),
                        k,
                        0.0,
                        c.as_mut_ptr(),
                        m,
                    );
                }
                Ok(())
            }

            fn one() -> Self {
                1.0
            }
            fn r_phase(diagonal: Self) -> Self {
                if diagonal < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }
            fn q_phase(diagonal: Self) -> Self {
                Self::r_phase(diagonal)
            }

            fn factor_work(
                m: i32,
                n: i32,
                data: &mut [Self],
                tau: &mut [Self],
                work: &mut [Self],
                lwork: i32,
            ) -> tenferro_tensor::Result<()> {
                let mut info = 0;
                // SAFETY: qr validates m*n storage, min(m,n) tau entries and
                // provides either the query slot or the queried workspace length.
                unsafe {
                    $geqrf(m, n, data, m, tau, work, lwork, &mut info);
                }
                check_lapack_info("qr", $geqrf_name, info)
            }

            fn generate_work(
                m: i32,
                k: i32,
                data: &mut [Self],
                tau: &[Self],
                work: &mut [Self],
                lwork: i32,
            ) -> tenferro_tensor::Result<()> {
                let mut info = 0;
                // SAFETY: qr supplies m*k reflector entries, k tau entries,
                // k<=m, and either one query slot or sufficient queried workspace.
                unsafe {
                    $orgqr(m, k, k, data, m, tau, work, lwork, &mut info);
                }
                check_lapack_info("qr", $orgqr_name, info)
            }

            fn qr_work_len(value: Self) -> tenferro_tensor::Result<i32> {
                work_len(value as f64, "qr", "QR workspace")
            }
        }
    };
}

macro_rules! impl_complex_qr {
    ($complex:ty, $geqrf:path, $ungqr:path, $unmqr:path, $gemm:path, $geqrf_name:literal, $ungqr_name:literal, $unmqr_name:literal) => {
        impl LapackQr for $complex {
            fn geqrf_2d(
                buffers: &mut BufferPool,
                data: &mut [Self],
                rows: usize,
                cols: usize,
            ) -> tenferro_tensor::Result<Vec<Self>> {
                let k = rows.min(cols);
                validate_buffer_len(
                    "compact_factor_2d",
                    "matrix",
                    data.len(),
                    checked_product("compact_factor_2d", "matrix", &[rows, cols])?,
                )?;
                if k == 0 {
                    return Ok(Vec::new());
                }
                let rows_i32 = dim_i32(rows, "compact_factor_2d")?;
                let cols_i32 = dim_i32(cols, "compact_factor_2d")?;
                let mut tau = pooled_zeroed::<$complex>(buffers, k);
                let mut query = pooled_zeroed::<$complex>(buffers, 1);
                let mut info = 0;
                // SAFETY: validated column-major storage covers rows*cols, tau has
                // min(rows, cols) entries, and lwork=-1 writes only the query slot.
                unsafe {
                    $geqrf(
                        rows_i32, cols_i32, data, rows_i32, &mut tau, &mut query, -1, &mut info,
                    );
                }
                check_lapack_info(
                    "compact_factor_2d",
                    concat!($geqrf_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0].re as f64, "compact_factor_2d", $geqrf_name)?;
                let mut work = pooled_zeroed::<$complex>(buffers, lwork as usize);
                // SAFETY: dimensions and workspace come from checked input and the
                // successful LAPACK query; all mutable slices remain live and disjoint.
                unsafe {
                    $geqrf(
                        rows_i32, cols_i32, data, rows_i32, &mut tau, &mut work, lwork, &mut info,
                    );
                }
                check_lapack_info("compact_factor_2d", $geqrf_name, info)?;
                release_scratch(buffers, query);
                release_scratch(buffers, work);
                Ok(tau)
            }

            #[allow(clippy::too_many_arguments)]
            fn apply_reflectors_2d(
                buffers: &mut BufferPool,
                a: &[Self],
                a_cols: usize,
                tau: &[Self],
                c: &mut [Self],
                m: usize,
                p: usize,
                k: usize,
                transpose: bool,
            ) -> tenferro_tensor::Result<()> {
                if k > a_cols || k > m {
                    return Err(tenferro_tensor::Error::invalid_argument(
                        "apply_reflectors_2d",
                        "dimensions",
                        "reflector count exceeds matrix dimensions",
                    ));
                }
                validate_buffer_len(
                    "apply_reflectors_2d",
                    "A",
                    a.len(),
                    checked_product("apply_reflectors_2d", "A", &[m, a_cols])?,
                )?;
                validate_buffer_len("apply_reflectors_2d", "tau", tau.len(), k)?;
                validate_buffer_len(
                    "apply_reflectors_2d",
                    "C",
                    c.len(),
                    checked_product("apply_reflectors_2d", "C", &[m, p])?,
                )?;
                if m == 0 || p == 0 || k == 0 {
                    return Ok(());
                }
                let m_i32 = dim_i32(m, "apply_reflectors_2d")?;
                let p_i32 = dim_i32(p, "apply_reflectors_2d")?;
                let k_i32 = dim_i32(k, "apply_reflectors_2d")?;
                let mut query = pooled_zeroed::<$complex>(buffers, 1);
                let mut info = 0;
                let trans = if transpose { b'C' } else { b'N' };
                // SAFETY: validated A, tau, and C slices satisfy LAPACK dimensions;
                // lwork=-1 writes only the single query element.
                unsafe {
                    $unmqr(
                        b'L', trans, m_i32, p_i32, k_i32, a, m_i32, tau, c, m_i32, &mut query, -1,
                        &mut info,
                    );
                }
                check_lapack_info(
                    "apply_reflectors_2d",
                    concat!($unmqr_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0].re as f64, "apply_reflectors_2d", $unmqr_name)?;
                let mut work = pooled_zeroed::<$complex>(buffers, lwork as usize);
                // SAFETY: checked dimensions and queried workspace cover the full
                // reflector application; A/tau are read-only and C is uniquely mutable.
                unsafe {
                    $unmqr(
                        b'L', trans, m_i32, p_i32, k_i32, a, m_i32, tau, c, m_i32, &mut work,
                        lwork, &mut info,
                    );
                }
                let result = check_lapack_info("apply_reflectors_2d", $unmqr_name, info);
                release_scratch(buffers, query);
                release_scratch(buffers, work);
                result
            }

            fn gemm_2d(
                a: &[Self],
                a_rows: usize,
                a_cols: usize,
                b: &[Self],
                b_cols: usize,
                c: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                validate_buffer_len(
                    "from_factors_2d",
                    "T",
                    a.len(),
                    checked_product("from_factors_2d", "T", &[a_rows, a_cols])?,
                )?;
                validate_buffer_len(
                    "from_factors_2d",
                    "R",
                    b.len(),
                    checked_product("from_factors_2d", "R", &[a_cols, b_cols])?,
                )?;
                validate_buffer_len(
                    "from_factors_2d",
                    "folded R",
                    c.len(),
                    checked_product("from_factors_2d", "folded R", &[a_rows, b_cols])?,
                )?;
                let m = dim_i32(a_rows, "from_factors_2d")?;
                let n = dim_i32(b_cols, "from_factors_2d")?;
                let k = dim_i32(a_cols, "from_factors_2d")?;
                let alpha = <$complex>::new(1.0, 0.0);
                let beta = <$complex>::new(0.0, 0.0);
                // SAFETY: the checked slice lengths cover m*k, k*n, and m*n
                // column-major elements; dimensions and leading dimensions fit LP64 i32.
                unsafe {
                    $gemm(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        m,
                        n,
                        k,
                        (&alpha as *const $complex).cast(),
                        a.as_ptr().cast(),
                        m,
                        b.as_ptr().cast(),
                        k,
                        (&beta as *const $complex).cast(),
                        c.as_mut_ptr().cast(),
                        m,
                    );
                }
                Ok(())
            }

            fn one() -> Self {
                <$complex>::new(1.0, 0.0)
            }
            fn r_phase(diagonal: Self) -> Self {
                let norm = diagonal.norm();
                if norm == 0.0 {
                    Self::one()
                } else {
                    diagonal.conj() / norm
                }
            }
            fn q_phase(diagonal: Self) -> Self {
                let norm = diagonal.norm();
                if norm == 0.0 {
                    Self::one()
                } else {
                    diagonal / norm
                }
            }

            fn factor_work(
                m: i32,
                n: i32,
                data: &mut [Self],
                tau: &mut [Self],
                work: &mut [Self],
                lwork: i32,
            ) -> tenferro_tensor::Result<()> {
                let mut info = 0;
                // SAFETY: qr validates m*n storage, min(m,n) tau entries and
                // provides either the query slot or the queried workspace length.
                unsafe {
                    $geqrf(m, n, data, m, tau, work, lwork, &mut info);
                }
                check_lapack_info("qr", $geqrf_name, info)
            }

            fn generate_work(
                m: i32,
                k: i32,
                data: &mut [Self],
                tau: &[Self],
                work: &mut [Self],
                lwork: i32,
            ) -> tenferro_tensor::Result<()> {
                let mut info = 0;
                // SAFETY: qr supplies m*k reflector entries, k tau entries,
                // k<=m, and either one query slot or sufficient queried workspace.
                unsafe {
                    $ungqr(m, k, k, data, m, tau, work, lwork, &mut info);
                }
                check_lapack_info("qr", $ungqr_name, info)
            }

            fn qr_work_len(value: Self) -> tenferro_tensor::Result<i32> {
                work_len(value.re as f64, "qr", "QR workspace")
            }
        }
    };
}

impl_real_qr!(
    f32,
    lapack::sgeqrf,
    lapack::sorgqr,
    lapack::sormqr,
    cblas_sys::cblas_sgemm,
    "sgeqrf",
    "sorgqr",
    "sormqr"
);
impl_real_qr!(
    f64,
    lapack::dgeqrf,
    lapack::dorgqr,
    lapack::dormqr,
    cblas_sys::cblas_dgemm,
    "dgeqrf",
    "dorgqr",
    "dormqr"
);
impl_complex_qr!(
    Complex32,
    lapack::cgeqrf,
    lapack::cungqr,
    lapack::cunmqr,
    cblas_sys::cblas_cgemm,
    "cgeqrf",
    "cungqr",
    "cunmqr"
);
impl_complex_qr!(
    Complex64,
    lapack::zgeqrf,
    lapack::zungqr,
    lapack::zunmqr,
    cblas_sys::cblas_zgemm,
    "zgeqrf",
    "zungqr",
    "zunmqr"
);

pub(crate) trait LapackRankRevealingQr: LapackQr {
    fn is_finite(self) -> bool;
    fn magnitude(self) -> f64;
    fn geqp3(
        buffers: &mut BufferPool,
        data: &mut [Self],
        rows: usize,
        cols: usize,
        permutation: &mut [i32],
        tau: &mut [Self],
    ) -> tenferro_tensor::Result<()>;
    fn generate_q(
        buffers: &mut BufferPool,
        q: &mut [Self],
        rows: usize,
        cols: usize,
        tau: &[Self],
    ) -> tenferro_tensor::Result<()>;
}

macro_rules! impl_real_rank_revealing_qr {
    ($scalar:ty, $geqp3:path, $orgqr:path, $geqp3_name:literal, $orgqr_name:literal) => {
        impl LapackRankRevealingQr for $scalar {
            fn is_finite(self) -> bool {
                self.is_finite()
            }

            fn magnitude(self) -> f64 {
                self.abs() as f64
            }

            fn geqp3(
                buffers: &mut BufferPool,
                data: &mut [Self],
                rows: usize,
                cols: usize,
                permutation: &mut [i32],
                tau: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let m = dim_i32(rows, "rank_revealing_qr")?;
                let n = dim_i32(cols, "rank_revealing_qr")?;
                let mut query = [0.0 as $scalar];
                let mut info = 0;
                // SAFETY: all matrix, pivot, tau, and query lengths follow the
                // checked LAPACK dimensions; lwork=-1 writes only query[0].
                unsafe {
                    $geqp3(m, n, data, m, permutation, tau, &mut query, -1, &mut info);
                }
                check_lapack_info(
                    "rank_revealing_qr",
                    concat!($geqp3_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0] as f64, "rank_revealing_qr", $geqp3_name)?;
                let mut work = buffers.acquire_zeroed::<$scalar>(lwork as usize);
                // SAFETY: dimensions and workspace were validated by the
                // successful query; all mutable buffers are live and disjoint.
                unsafe {
                    $geqp3(m, n, data, m, permutation, tau, &mut work, lwork, &mut info);
                }
                <$scalar as PoolScalar>::pool_release(buffers, work);
                check_lapack_info("rank_revealing_qr", $geqp3_name, info)
            }

            fn generate_q(
                buffers: &mut BufferPool,
                q: &mut [Self],
                rows: usize,
                cols: usize,
                tau: &[Self],
            ) -> tenferro_tensor::Result<()> {
                let m = dim_i32(rows, "rank_revealing_qr")?;
                let k = dim_i32(cols, "rank_revealing_qr")?;
                let mut query = [0.0 as $scalar];
                let mut info = 0;
                // SAFETY: q is checked m-by-k column-major storage, tau has k
                // entries, and lwork=-1 writes only query[0].
                unsafe { $orgqr(m, k, k, q, m, tau, &mut query, -1, &mut info) };
                check_lapack_info(
                    "rank_revealing_qr",
                    concat!($orgqr_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0] as f64, "rank_revealing_qr", $orgqr_name)?;
                let mut work = buffers.acquire_zeroed::<$scalar>(lwork as usize);
                // SAFETY: the successful query validated the dimensions and
                // workspace; q, tau, work, and info remain live and disjoint.
                unsafe { $orgqr(m, k, k, q, m, tau, &mut work, lwork, &mut info) };
                <$scalar as PoolScalar>::pool_release(buffers, work);
                check_lapack_info("rank_revealing_qr", $orgqr_name, info)
            }
        }
    };
}

macro_rules! impl_complex_rank_revealing_qr {
    ($complex:ty, $real:ty, $geqp3:path, $ungqr:path, $geqp3_name:literal, $ungqr_name:literal) => {
        impl LapackRankRevealingQr for $complex {
            fn is_finite(self) -> bool {
                self.re.is_finite() && self.im.is_finite()
            }

            fn magnitude(self) -> f64 {
                self.norm() as f64
            }

            fn geqp3(
                buffers: &mut BufferPool,
                data: &mut [Self],
                rows: usize,
                cols: usize,
                permutation: &mut [i32],
                tau: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let m = dim_i32(rows, "rank_revealing_qr")?;
                let n = dim_i32(cols, "rank_revealing_qr")?;
                let rwork_len =
                    checked_product("rank_revealing_qr", "GEQP3 real workspace", &[2, cols])?;
                let mut rwork = buffers.acquire_zeroed::<$real>(rwork_len);
                let mut query = [<$complex>::new(0.0, 0.0)];
                let mut info = 0;
                // SAFETY: all matrix, pivot, tau, real-work, and query lengths
                // follow checked LAPACK dimensions; lwork=-1 writes query[0].
                unsafe {
                    $geqp3(
                        m,
                        n,
                        data,
                        m,
                        permutation,
                        tau,
                        &mut query,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_lapack_info(
                    "rank_revealing_qr",
                    concat!($geqp3_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0].re as f64, "rank_revealing_qr", $geqp3_name)?;
                let mut work = buffers.acquire_zeroed::<$complex>(lwork as usize);
                // SAFETY: dimensions and workspace were validated by the
                // successful query; all mutable buffers are live and disjoint.
                unsafe {
                    $geqp3(
                        m,
                        n,
                        data,
                        m,
                        permutation,
                        tau,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }
                <$complex as PoolScalar>::pool_release(buffers, work);
                <$real as PoolScalar>::pool_release(buffers, rwork);
                check_lapack_info("rank_revealing_qr", $geqp3_name, info)
            }

            fn generate_q(
                buffers: &mut BufferPool,
                q: &mut [Self],
                rows: usize,
                cols: usize,
                tau: &[Self],
            ) -> tenferro_tensor::Result<()> {
                let m = dim_i32(rows, "rank_revealing_qr")?;
                let k = dim_i32(cols, "rank_revealing_qr")?;
                let mut query = [<$complex>::new(0.0, 0.0)];
                let mut info = 0;
                // SAFETY: q is checked m-by-k column-major storage, tau has k
                // entries, and lwork=-1 writes only query[0].
                unsafe { $ungqr(m, k, k, q, m, tau, &mut query, -1, &mut info) };
                check_lapack_info(
                    "rank_revealing_qr",
                    concat!($ungqr_name, "(work query)"),
                    info,
                )?;
                let lwork = work_len(query[0].re as f64, "rank_revealing_qr", $ungqr_name)?;
                let mut work = buffers.acquire_zeroed::<$complex>(lwork as usize);
                // SAFETY: the successful query validated the dimensions and
                // workspace; q, tau, work, and info remain live and disjoint.
                unsafe { $ungqr(m, k, k, q, m, tau, &mut work, lwork, &mut info) };
                <$complex as PoolScalar>::pool_release(buffers, work);
                check_lapack_info("rank_revealing_qr", $ungqr_name, info)
            }
        }
    };
}

impl_real_rank_revealing_qr!(f32, lapack::sgeqp3, lapack::sorgqr, "sgeqp3", "sorgqr");
impl_real_rank_revealing_qr!(f64, lapack::dgeqp3, lapack::dorgqr, "dgeqp3", "dorgqr");
impl_complex_rank_revealing_qr!(
    Complex32,
    f32,
    lapack::cgeqp3,
    lapack::cungqr,
    "cgeqp3",
    "cungqr"
);
impl_complex_rank_revealing_qr!(
    Complex64,
    f64,
    lapack::zgeqp3,
    lapack::zungqr,
    "zgeqp3",
    "zungqr"
);

fn normalize_lapack_permutation(permutation: &[i32]) -> tenferro_tensor::Result<Vec<i64>> {
    let n = permutation.len();
    let mut seen = vec![false; n];
    let mut normalized = Vec::with_capacity(n);
    for &column in permutation {
        let zero_based = column
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok());
        let Some(zero_based) = zero_based.filter(|&value| value < n && !seen[value]) else {
            return Err(tenferro_tensor::Error::invalid_argument(
                "rank_revealing_qr",
                "column_permutation",
                "LAPACK GEQP3 returned an invalid pivot permutation",
            ));
        };
        seen[zero_based] = true;
        normalized.push(i64::try_from(zero_based).map_err(|_| {
            tenferro_tensor::Error::invalid_argument(
                "rank_revealing_qr",
                "column_permutation",
                "column index exceeds i64 range",
            )
        })?);
    }
    Ok(normalized)
}

fn rank_revealing_qr_2d<T: LapackRankRevealingQr>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<crate::cpu::linalg::rank_revealing_qr::TypedRrqr<T>> {
    let (m, n) = matrix_dims(input, "rank_revealing_qr")?;
    let k = m.min(n);
    let input_data = input.host_data()?;
    if input_data.iter().any(|&value| !value.is_finite()) {
        return Err(crate::error::into_tensor_error(
            "rank_revealing_qr",
            crate::Error::NonFinite {
                op: "rank_revealing_qr",
                role: "input",
            },
        ));
    }
    if input_data.iter().all(|&value| value.magnitude() == 0.0) {
        return crate::cpu::linalg::rank_revealing_qr::zero_matrix_result(input, T::one());
    }

    let mut qr = input_data.to_vec();
    let mut permutation = vec![0_i32; n];
    let mut tau = vec![T::default(); k];
    T::geqp3(buffers, &mut qr, m, n, &mut permutation, &mut tau)?;
    let r = leading_upper_triangle_from_lapack(&qr, m, k, n)?;
    let rank = crate::cpu::linalg::rank_revealing_qr::prefix_rank(
        (0..k).map(|index| r[index + index * k].magnitude()),
        options,
    )?;
    let mut q = Vec::with_capacity(checked_product("rank_revealing_qr", "Q matrix", &[m, k])?);
    for column in 0..k {
        let start = column.checked_mul(m).ok_or_else(|| {
            tenferro_tensor::Error::validation(
                "rank_revealing_qr",
                tenferro_tensor::ValidationError::IntegerOverflow,
            )
        })?;
        q.extend_from_slice(&qr[start..start + m]);
    }
    T::generate_q(buffers, &mut q, m, k, &tau)?;

    Ok(crate::RankRevealingQrResult {
        q: tensor_from_vec_with_template(vec![m, k], q, input)?,
        r: tensor_from_vec_with_template(vec![k, n], r, input)?,
        column_permutation: tensor_from_vec_with_template(
            vec![n],
            normalize_lapack_permutation(&permutation)?,
            input,
        )?,
        rank: tensor_from_vec_with_template(vec![], vec![rank], input)?,
    })
}

pub(crate) fn qr<T: LapackQr>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "qr")?;
        let m = matrix_shape[0];
        let n = matrix_shape[1];
        let k = m.min(n);
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            )?,
        ]);
    }
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "qr")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    let (mi, ni, ki) = (dim_i32(m, "qr")?, dim_i32(n, "qr")?, dim_i32(k, "qr")?);
    let matrix_len = checked_product("qr", "matrix", &[m, n])?;
    let q_len = checked_product("qr", "Q matrix", &[m, k])?;
    let q_shape = matrix_with_batch_shape(m, k, batch_shape);
    let r_shape = matrix_with_batch_shape(k, n, batch_shape);
    let mut q = buffers.acquire_with_capacity::<T>(checked_product("qr", "Q", &q_shape)?);
    let mut r = buffers.acquire_with_capacity::<T>(checked_product("qr", "R", &r_shape)?);
    let data = input.host_data()?;
    let mut packed = buffers.acquire_with_capacity::<T>(matrix_len);
    packed.extend_from_slice(&data[..matrix_len]);
    let mut tau = pooled_zeroed::<T>(buffers, k);
    let mut query = [T::default()];
    T::factor_work(mi, ni, &mut packed, &mut tau, &mut query, -1)?;
    let factor_len = T::qr_work_len(query[0])?;
    T::generate_work(mi, ki, &mut packed[..q_len], &tau, &mut query, -1)?;
    let lwork = factor_len.max(T::qr_work_len(query[0])?);
    let mut work = pooled_zeroed::<T>(buffers, lwork as usize);
    // INVARIANT: validated nonempty compact input has complete m*n chunks;
    // reduced Q occupies the first m*k entries, k<=n. All batches share the
    // queried dimensions and scratch, but never alias the original input.
    // LAPACK owns threading; the batch loop only prepares provider calls.
    for matrix in data.chunks_exact(matrix_len) {
        packed.copy_from_slice(matrix);
        T::factor_work(mi, ni, &mut packed, &mut tau, &mut work, lwork)?;
        r.extend_from_slice(&leading_upper_triangle_from_lapack(&packed, m, k, n)?);
        T::generate_work(mi, ki, &mut packed[..q_len], &tau, &mut work, lwork)?;
        q.extend_from_slice(&packed[..q_len]);
    }
    release_scratch(buffers, packed);
    release_scratch(buffers, tau);
    release_scratch(buffers, work);
    Ok(vec![
        tensor_from_vec_with_template(q_shape, q, input)?,
        tensor_from_vec_with_template(r_shape, r, input)?,
    ])
}

pub(crate) fn rank_revealing_qr<T: LapackRankRevealingQr>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<crate::cpu::linalg::rank_revealing_qr::TypedRrqr<T>> {
    crate::rank_revealing_qr::validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
    crate::cpu::linalg::rank_revealing_qr::batched(buffers, input, |buffers, batch| {
        rank_revealing_qr_2d(buffers, batch, options)
    })
}

#[cfg(test)]
#[path = "qr_tests.rs"]
mod tests;
