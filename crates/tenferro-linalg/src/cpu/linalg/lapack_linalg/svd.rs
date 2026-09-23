use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batch_element_count, check_lapack_info, checked_product, dim_i32, has_zero_dim,
    matrix_with_batch_shape, pooled_copy, pooled_zeroed, release_scratch,
    split_core_and_batch_result, tensor_from_vec_with_template, vector_with_batch_shape, work_len,
};

/// Which singular factors a batched SVD computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SvdMode {
    /// Thin factors: `U` is `m x k` and `Vt` is `k x n`.
    Thin,
    /// Full factors: `U` is `m x m` and `Vt` is `n x n`, so the trailing
    /// columns and rows span the left and right nullspaces.
    Full,
    /// Singular values only.
    Values,
}

impl SvdMode {
    /// LAPACK job letter shared by `gesdd` (`jobz`) and `gesvd` (`jobu`, `jobvt`).
    fn job(self) -> u8 {
        match self {
            Self::Thin => b'S',
            Self::Full => b'A',
            Self::Values => b'N',
        }
    }

    /// `(U columns, Vt rows)` of one factor pair; zero in values-only mode.
    fn factor_dims(self, m: usize, n: usize) -> (usize, usize) {
        let k = m.min(n);
        match self {
            Self::Thin => (k, k),
            Self::Full => (m, n),
            Self::Values => (0, 0),
        }
    }
}

/// Per-matrix buffer lengths of one batched SVD call.
struct SvdLayout {
    batch: usize,
    k: usize,
    a_len: usize,
    u_len: usize,
    vt_len: usize,
    ldu: i32,
    ldvt: i32,
}

impl SvdLayout {
    /// Check that all four buffers describe the same number of nonempty
    /// `m x n` problems in `mode`.
    fn new(
        op: &'static str,
        mode: SvdMode,
        m: usize,
        n: usize,
        lens: [usize; 4],
    ) -> tenferro_tensor::Result<Self> {
        let [a, s, u, vt] = lens;
        let k = m.min(n);
        let (u_cols, vt_rows) = mode.factor_dims(m, n);
        let a_len = checked_product(op, "matrix", &[m, n])?;
        let u_len = checked_product(op, "left singular vectors", &[m, u_cols])?;
        let vt_len = checked_product(op, "right singular vectors", &[vt_rows, n])?;
        let batch = a.checked_div(a_len).unwrap_or(0);
        let consistent = a_len != 0
            && a == checked_product(op, "matrix batch", &[a_len, batch])?
            && s == checked_product(op, "singular value batch", &[k, batch])?
            && u == checked_product(op, "left factor batch", &[u_len, batch])?
            && vt == checked_product(op, "right factor batch", &[vt_len, batch])?;
        if !consistent {
            return Err(tenferro_tensor::Error::Internal(format!(
                "{op}: SVD buffers describe different batches"
            )));
        }
        let (ldu, ldvt) = if mode == SvdMode::Values {
            (1, 1)
        } else {
            (dim_i32(m, op)?, dim_i32(vt_rows, op)?)
        };
        Ok(Self {
            batch,
            k,
            a_len,
            u_len,
            vt_len,
            ldu,
            ldvt,
        })
    }

    /// Mutable views of matrix `index` in each of the four buffers.
    fn chunk<'a, T, R>(
        &self,
        index: usize,
        a: &'a mut [T],
        s: &'a mut [R],
        u: &'a mut [T],
        vt: &'a mut [T],
    ) -> (&'a mut [T], &'a mut [R], &'a mut [T], &'a mut [T]) {
        (
            &mut a[index * self.a_len..(index + 1) * self.a_len],
            &mut s[index * self.k..(index + 1) * self.k],
            &mut u[index * self.u_len..(index + 1) * self.u_len],
            &mut vt[index * self.vt_len..(index + 1) * self.vt_len],
        )
    }
}

pub(crate) trait LapackSvd: Clone + Copy + Default + PoolScalar {
    type Real: Clone + Copy + Default + PoolScalar + tenferro_tensor::TensorScalar;

    /// The multiplicative unit, used to build the identity factor a full
    /// decomposition still owes for an empty core dimension.
    fn unit() -> Self;

    /// Decompose every compact `m x n` matrix of `a` (destroyed), writing
    /// `min(m, n)` descending singular values per matrix into `s` and, unless
    /// `mode` is [`SvdMode::Values`], the factors into `u` and `vt`.
    ///
    /// The workspace is queried once and reused across the batch.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` for inconsistent buffer lengths, an invalid
    /// workspace error for a bad LAPACK query, and `NonConvergence` or an
    /// illegal-argument error from the LAPACK driver.
    #[allow(clippy::too_many_arguments)]
    fn svd_batched(
        buffers: &mut BufferPool,
        op: &'static str,
        mode: SvdMode,
        m: usize,
        n: usize,
        a: &mut [Self],
        s: &mut [<Self as LapackSvd>::Real],
        u: &mut [Self],
        vt: &mut [Self],
    ) -> tenferro_tensor::Result<()>;

    /// Real singular values in this scalar type, reusing the buffer when the
    /// scalar is real.
    fn values_as_scalar(
        buffers: &mut BufferPool,
        values: Vec<<Self as LapackSvd>::Real>,
    ) -> Vec<Self>;
}

#[cfg(not(feature = "provider-inject"))]
fn gesdd_iwork_len(k: usize) -> tenferro_tensor::Result<usize> {
    checked_product("svd", "integer workspace", &[8, k.max(1)])
}

#[cfg(not(feature = "provider-inject"))]
fn complex_gesdd_rwork_len(jobz: u8, m: usize, n: usize) -> tenferro_tensor::Result<usize> {
    let mn = m.min(n);
    let mx = m.max(n);
    if jobz == b'N' {
        return checked_product("svd", "real workspace", &[5, mn.max(1)]);
    }
    let threshold = checked_product("svd", "workspace crossover", &[10, mn])?;
    let square_term = checked_product("svd", "real workspace square term", &[5, mn, mn])?;
    let linear_term = checked_product("svd", "real workspace linear term", &[5, mn])?;
    let small_shape_len = square_term.checked_add(linear_term).ok_or_else(|| {
        tenferro_tensor::Error::validation("svd", tenferro_tensor::ValidationError::IntegerOverflow)
    })?;
    if mx > threshold {
        return Ok(small_shape_len);
    }
    let rectangular_term = checked_product("svd", "real workspace rectangular term", &[2, mx, mn])?;
    let second_square_term =
        checked_product("svd", "real workspace secondary square term", &[2, mn, mn])?;
    let large_shape_len = rectangular_term
        .checked_add(second_square_term)
        .and_then(|len| len.checked_add(mn))
        .ok_or_else(|| {
            tenferro_tensor::Error::validation(
                "svd",
                tenferro_tensor::ValidationError::IntegerOverflow,
            )
        })?;
    Ok(small_shape_len.max(large_shape_len))
}

#[cfg(feature = "provider-inject")]
macro_rules! impl_real_svd {
    ($scalar:ty, $gesvd:path, $routine:literal) => {
        impl LapackSvd for $scalar {
            type Real = $scalar;

            fn unit() -> Self {
                1.0
            }

            #[allow(clippy::too_many_arguments)]
            fn svd_batched(
                buffers: &mut BufferPool,
                op: &'static str,
                mode: SvdMode,
                m: usize,
                n: usize,
                a: &mut [Self],
                s: &mut [$scalar],
                u: &mut [Self],
                vt: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let layout = SvdLayout::new(op, mode, m, n, [a.len(), s.len(), u.len(), vt.len()])?;
                if layout.batch == 0 {
                    return Ok(());
                }
                let job = mode.job();
                let m_i32 = dim_i32(m, op)?;
                let n_i32 = dim_i32(n, op)?;
                let (ldu, ldvt) = (layout.ldu, layout.ldvt);
                let mut query = [<$scalar>::default(); 1];
                let mut info = 0;
                {
                    let (a0, s0, u0, vt0) = layout.chunk(0, a, s, u, vt);
                    // SAFETY: the first chunk of every buffer matches the
                    // validated `m x n` problem in `mode` (lengths checked by
                    // `SvdLayout::new`); `lwork = -1` writes only `query`.
                    unsafe {
                        $gesvd(
                            job, job, m_i32, n_i32, a0, m_i32, s0, u0, ldu, vt0, ldvt, &mut query,
                            -1, &mut info,
                        );
                    }
                }
                check_lapack_info(op, concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, op, $routine)?;
                let mut work = pooled_zeroed::<$scalar>(buffers, lwork as usize);
                // INVARIANT: `SvdLayout::new` checked that every buffer holds
                // `layout.batch` blocks, and the workspace depends only on
                // `(mode, m, n)`, so it is reused across the batch. The serial
                // loop is intentional: the LAPACK provider owns threading.
                for index in 0..layout.batch {
                    let (a_i, s_i, u_i, vt_i) = layout.chunk(index, a, s, u, vt);
                    // SAFETY: each chunk matches the validated problem and
                    // `work` has the queried length.
                    unsafe {
                        $gesvd(
                            job, job, m_i32, n_i32, a_i, m_i32, s_i, u_i, ldu, vt_i, ldvt,
                            &mut work, lwork, &mut info,
                        );
                    }
                    check_lapack_info(op, $routine, info)?;
                }
                release_scratch(buffers, work);
                Ok(())
            }

            fn values_as_scalar(_buffers: &mut BufferPool, values: Vec<Self>) -> Vec<Self> {
                values
            }
        }
    };
}

#[cfg(not(feature = "provider-inject"))]
macro_rules! impl_real_svd {
    ($scalar:ty, $gesdd:path, $routine:literal) => {
        impl LapackSvd for $scalar {
            type Real = $scalar;

            fn unit() -> Self {
                1.0
            }

            #[allow(clippy::too_many_arguments)]
            fn svd_batched(
                buffers: &mut BufferPool,
                op: &'static str,
                mode: SvdMode,
                m: usize,
                n: usize,
                a: &mut [Self],
                s: &mut [$scalar],
                u: &mut [Self],
                vt: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let layout = SvdLayout::new(op, mode, m, n, [a.len(), s.len(), u.len(), vt.len()])?;
                if layout.batch == 0 {
                    return Ok(());
                }
                let job = mode.job();
                let m_i32 = dim_i32(m, op)?;
                let n_i32 = dim_i32(n, op)?;
                let (ldu, ldvt) = (layout.ldu, layout.ldvt);
                let mut iwork = pooled_zeroed::<i32>(buffers, gesdd_iwork_len(layout.k)?);
                let mut query = [<$scalar>::default(); 1];
                let mut info = 0;
                {
                    let (a0, s0, u0, vt0) = layout.chunk(0, a, s, u, vt);
                    // SAFETY: the first chunk of every buffer matches the
                    // validated `m x n` problem in `mode` (lengths checked by
                    // `SvdLayout::new`); `lwork = -1` writes only `query`.
                    unsafe {
                        $gesdd(
                            job, m_i32, n_i32, a0, m_i32, s0, u0, ldu, vt0, ldvt, &mut query, -1,
                            &mut iwork, &mut info,
                        );
                    }
                }
                check_lapack_info(op, concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, op, $routine)?;
                let mut work = pooled_zeroed::<$scalar>(buffers, lwork as usize);
                // INVARIANT: `SvdLayout::new` checked that every buffer holds
                // `layout.batch` blocks, and the workspace depends only on
                // `(mode, m, n)`, so it is reused across the batch. The serial
                // loop is intentional: the LAPACK provider owns threading.
                for index in 0..layout.batch {
                    let (a_i, s_i, u_i, vt_i) = layout.chunk(index, a, s, u, vt);
                    // SAFETY: each chunk matches the validated problem and
                    // `work` has the queried length.
                    unsafe {
                        $gesdd(
                            job, m_i32, n_i32, a_i, m_i32, s_i, u_i, ldu, vt_i, ldvt, &mut work,
                            lwork, &mut iwork, &mut info,
                        );
                    }
                    check_lapack_info(op, $routine, info)?;
                }
                release_scratch(buffers, iwork);
                release_scratch(buffers, work);
                Ok(())
            }

            fn values_as_scalar(_buffers: &mut BufferPool, values: Vec<Self>) -> Vec<Self> {
                values
            }
        }
    };
}

#[cfg(feature = "provider-inject")]
macro_rules! impl_complex_svd {
    ($complex:ty, $real:ty, $gesvd:path, $routine:literal) => {
        impl LapackSvd for $complex {
            type Real = $real;

            fn unit() -> Self {
                <$complex>::new(1.0, 0.0)
            }

            #[allow(clippy::too_many_arguments)]
            fn svd_batched(
                buffers: &mut BufferPool,
                op: &'static str,
                mode: SvdMode,
                m: usize,
                n: usize,
                a: &mut [Self],
                s: &mut [$real],
                u: &mut [Self],
                vt: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let layout = SvdLayout::new(op, mode, m, n, [a.len(), s.len(), u.len(), vt.len()])?;
                if layout.batch == 0 {
                    return Ok(());
                }
                let job = mode.job();
                let m_i32 = dim_i32(m, op)?;
                let n_i32 = dim_i32(n, op)?;
                let (ldu, ldvt) = (layout.ldu, layout.ldvt);
                let rwork_len = checked_product(op, "real workspace", &[5, layout.k.max(1)])?;
                let mut rwork = pooled_zeroed::<$real>(buffers, rwork_len);
                let mut query = [<$complex>::default(); 1];
                let mut info = 0;
                {
                    let (a0, s0, u0, vt0) = layout.chunk(0, a, s, u, vt);
                    // SAFETY: the first chunk of every buffer matches the
                    // validated `m x n` problem in `mode` (lengths checked by
                    // `SvdLayout::new`); `lwork = -1` writes only `query`.
                    unsafe {
                        $gesvd(
                            job, job, m_i32, n_i32, a0, m_i32, s0, u0, ldu, vt0, ldvt, &mut query,
                            -1, &mut rwork, &mut info,
                        );
                    }
                }
                check_lapack_info(op, concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, op, $routine)?;
                let mut work = pooled_zeroed::<$complex>(buffers, lwork as usize);
                // INVARIANT: `SvdLayout::new` checked that every buffer holds
                // `layout.batch` blocks, and the workspace depends only on
                // `(mode, m, n)`, so it is reused across the batch. The serial
                // loop is intentional: the LAPACK provider owns threading.
                for index in 0..layout.batch {
                    let (a_i, s_i, u_i, vt_i) = layout.chunk(index, a, s, u, vt);
                    // SAFETY: each chunk matches the validated problem and
                    // `work` has the queried length.
                    unsafe {
                        $gesvd(
                            job, job, m_i32, n_i32, a_i, m_i32, s_i, u_i, ldu, vt_i, ldvt,
                            &mut work, lwork, &mut rwork, &mut info,
                        );
                    }
                    check_lapack_info(op, $routine, info)?;
                }
                release_scratch(buffers, rwork);
                release_scratch(buffers, work);
                Ok(())
            }

            fn values_as_scalar(buffers: &mut BufferPool, values: Vec<$real>) -> Vec<Self> {
                let mut converted = buffers.acquire_with_capacity::<$complex>(values.len());
                converted.extend(values.iter().map(|&value| <$complex>::new(value, 0.0)));
                release_scratch(buffers, values);
                converted
            }
        }
    };
}

#[cfg(not(feature = "provider-inject"))]
macro_rules! impl_complex_svd {
    ($complex:ty, $real:ty, $gesdd:path, $routine:literal) => {
        impl LapackSvd for $complex {
            type Real = $real;

            fn unit() -> Self {
                <$complex>::new(1.0, 0.0)
            }

            #[allow(clippy::too_many_arguments)]
            fn svd_batched(
                buffers: &mut BufferPool,
                op: &'static str,
                mode: SvdMode,
                m: usize,
                n: usize,
                a: &mut [Self],
                s: &mut [$real],
                u: &mut [Self],
                vt: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let layout = SvdLayout::new(op, mode, m, n, [a.len(), s.len(), u.len(), vt.len()])?;
                if layout.batch == 0 {
                    return Ok(());
                }
                let job = mode.job();
                let m_i32 = dim_i32(m, op)?;
                let n_i32 = dim_i32(n, op)?;
                let (ldu, ldvt) = (layout.ldu, layout.ldvt);
                let mut rwork =
                    pooled_zeroed::<$real>(buffers, complex_gesdd_rwork_len(job, m, n)?);
                let mut iwork = pooled_zeroed::<i32>(buffers, gesdd_iwork_len(layout.k)?);
                let mut query = [<$complex>::default(); 1];
                let mut info = 0;
                {
                    let (a0, s0, u0, vt0) = layout.chunk(0, a, s, u, vt);
                    // SAFETY: the first chunk of every buffer matches the
                    // validated `m x n` problem in `mode` (lengths checked by
                    // `SvdLayout::new`); `lwork = -1` writes only `query`.
                    unsafe {
                        $gesdd(
                            job, m_i32, n_i32, a0, m_i32, s0, u0, ldu, vt0, ldvt, &mut query, -1,
                            &mut rwork, &mut iwork, &mut info,
                        );
                    }
                }
                check_lapack_info(op, concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, op, $routine)?;
                let mut work = pooled_zeroed::<$complex>(buffers, lwork as usize);
                // INVARIANT: `SvdLayout::new` checked that every buffer holds
                // `layout.batch` blocks, and the workspace depends only on
                // `(mode, m, n)`, so it is reused across the batch. The serial
                // loop is intentional: the LAPACK provider owns threading.
                for index in 0..layout.batch {
                    let (a_i, s_i, u_i, vt_i) = layout.chunk(index, a, s, u, vt);
                    // SAFETY: each chunk matches the validated problem and
                    // `work` has the queried length.
                    unsafe {
                        $gesdd(
                            job, m_i32, n_i32, a_i, m_i32, s_i, u_i, ldu, vt_i, ldvt, &mut work,
                            lwork, &mut rwork, &mut iwork, &mut info,
                        );
                    }
                    check_lapack_info(op, $routine, info)?;
                }
                release_scratch(buffers, rwork);
                release_scratch(buffers, iwork);
                release_scratch(buffers, work);
                Ok(())
            }

            fn values_as_scalar(buffers: &mut BufferPool, values: Vec<$real>) -> Vec<Self> {
                let mut converted = buffers.acquire_with_capacity::<$complex>(values.len());
                converted.extend(values.iter().map(|&value| <$complex>::new(value, 0.0)));
                release_scratch(buffers, values);
                converted
            }
        }
    };
}

#[cfg(not(feature = "provider-inject"))]
impl_real_svd!(f32, lapack::sgesdd, "sgesdd");
#[cfg(not(feature = "provider-inject"))]
impl_real_svd!(f64, lapack::dgesdd, "dgesdd");
#[cfg(not(feature = "provider-inject"))]
impl_complex_svd!(Complex32, f32, lapack::cgesdd, "cgesdd");
#[cfg(not(feature = "provider-inject"))]
impl_complex_svd!(Complex64, f64, lapack::zgesdd, "zgesdd");

#[cfg(feature = "provider-inject")]
impl_real_svd!(f32, lapack::sgesvd, "sgesvd");
#[cfg(feature = "provider-inject")]
impl_real_svd!(f64, lapack::dgesvd, "dgesvd");
#[cfg(feature = "provider-inject")]
impl_complex_svd!(Complex32, f32, lapack::cgesvd, "cgesvd");
#[cfg(feature = "provider-inject")]
impl_complex_svd!(Complex64, f64, lapack::zgesvd, "zgesvd");

/// Pooled factor buffers for a batched SVD in `mode`, returned as
/// `(a copy, values, U, Vt)`.
#[allow(clippy::type_complexity)]
fn svd_buffers<T: LapackSvd>(
    buffers: &mut BufferPool,
    op: &'static str,
    mode: SvdMode,
    m: usize,
    n: usize,
    batch_shape: &[usize],
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(Vec<T>, Vec<<T as LapackSvd>::Real>, Vec<T>, Vec<T>)> {
    let batch = batch_element_count(op, batch_shape)?;
    let (u_cols, vt_rows) = mode.factor_dims(m, n);
    let s_len = checked_product(op, "singular values", &[m.min(n), batch])?;
    let u_len = checked_product(op, "left singular vectors", &[m, u_cols, batch])?;
    let vt_len = checked_product(op, "right singular vectors", &[vt_rows, n, batch])?;
    let mut a = pooled_copy(buffers, input.host_data()?);
    let mut s = pooled_zeroed::<<T as LapackSvd>::Real>(buffers, s_len);
    let mut u = pooled_zeroed::<T>(buffers, u_len);
    let mut vt = pooled_zeroed::<T>(buffers, vt_len);
    T::svd_batched(buffers, op, mode, m, n, &mut a, &mut s, &mut u, &mut vt)?;
    Ok((a, s, u, vt))
}

pub(crate) fn svd<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
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
    let (a, s, u, vt) = svd_buffers(buffers, "svd", SvdMode::Thin, m, n, batch_shape, input)?;
    release_scratch(buffers, a);
    let s = T::values_as_scalar(buffers, s);
    Ok(vec![
        tensor_from_vec_with_template(matrix_with_batch_shape(m, k, batch_shape), u, input)?,
        tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), s, input)?,
        tensor_from_vec_with_template(matrix_with_batch_shape(k, n, batch_shape), vt, input)?,
    ])
}

pub(crate) fn svd_full<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_full")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    if has_zero_dim(input.shape()) {
        return empty_full_svd_outputs("svd_full", m, n, batch_shape, input);
    }
    let (a, s, u, vt) = svd_buffers(buffers, "svd_full", SvdMode::Full, m, n, batch_shape, input)?;
    release_scratch(buffers, a);
    let s = T::values_as_scalar(buffers, s);
    Ok(vec![
        tensor_from_vec_with_template(matrix_with_batch_shape(m, m, batch_shape), u, input)?,
        tensor_from_vec_with_template(vector_with_batch_shape(m.min(n), batch_shape), s, input)?,
        tensor_from_vec_with_template(matrix_with_batch_shape(n, n, batch_shape), vt, input)?,
    ])
}

/// Full-SVD factors for an input with an empty core dimension.
///
/// The full variant keeps its `m x m` and `n x n` output shapes even when the
/// other core dimension is zero, so the factor for the non-empty dimension is
/// the identity rather than an empty tensor. This mirrors the faer provider so
/// one public call has one shape and unitarity contract.
fn empty_full_svd_outputs<T: LapackSvd, U>(
    op: &'static str,
    m: usize,
    n: usize,
    batch_shape: &[usize],
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let blocks = checked_product(op, "batch shape", batch_shape)?;
    Ok(vec![
        tensor_from_vec_with_template(
            matrix_with_batch_shape(m, m, batch_shape),
            identity_blocks::<T>(op, m, blocks)?,
            template,
        )?,
        tensor_from_vec_with_template(
            vector_with_batch_shape(m.min(n), batch_shape),
            Vec::new(),
            template,
        )?,
        tensor_from_vec_with_template(
            matrix_with_batch_shape(n, n, batch_shape),
            identity_blocks::<T>(op, n, blocks)?,
            template,
        )?,
    ])
}

/// `blocks` column-major `dim x dim` identity matrices laid out back to back.
fn identity_blocks<T: LapackSvd>(
    op: &'static str,
    dim: usize,
    blocks: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let per_block = checked_product(op, "identity block", &[dim, dim])?;
    let len = checked_product(op, "identity stack", &[per_block, blocks])?;
    let mut data = vec![T::default(); len];
    for block in 0..blocks {
        let base = block * per_block;
        for index in 0..dim {
            data[base + index + index * dim] = T::unit();
        }
    }
    Ok(data)
}

pub(crate) fn svd_values<T: LapackSvd>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<<T as LapackSvd>::Real>> {
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_values")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return tensor_from_vec_with_template(
            vector_with_batch_shape(k, batch_shape),
            Vec::new(),
            input,
        );
    }
    let (a, s, u, vt) = svd_buffers(
        buffers,
        "svd_values",
        SvdMode::Values,
        m,
        n,
        batch_shape,
        input,
    )?;
    release_scratch(buffers, a);
    release_scratch(buffers, u);
    release_scratch(buffers, vt);
    tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), s, input)
}
