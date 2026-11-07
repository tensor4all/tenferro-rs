//! Borrowed-`Exec` factorization entries with typed outputs (#2462); the contract is the
//! "Borrowed-`Exec` factorization entries" section of the crate documentation.
//!
//! # Three input rules behind three spellings of the same factorization
//!
//! The crate now offers the same six factorizations through three shapes, and they accept
//! **different inputs**. That is deliberate, and it is the thing to read before choosing one:
//!
//! | entry | input | rule |
//! |---|---|---|
//! | `svd(&Tensor)` and siblings | an owned tensor | copied in through its own view: no layout restriction, because the callee already owns the bytes |
//! | `svd_read(TensorRead)` and siblings | any read | the `*_read` rule: rank 2 in place, everything else materialized, and a **negative stride is gathered** |
//! | `svd_with_exec(&mut Exec, &mut BufferPool, &TypedTensorView)` and siblings | a borrowed view | the `*_impl` rule: `matrix_core_and_batch` + `raw_view`, read in place, and a **negative stride is `Unsupported`** |
//!
//! The rule follows from the promise, not from taste. A borrowed entry promises "no copy", so it
//! cannot gather a negative stride and has to refuse it. A read entry promises "any read", so it
//! must pay for the gather. Neither behaviour may be relaxed to match the other: the parity test
//! `a_negative_stride_is_unsupported_for_every_entry` asserts both halves on the same view, so
//! making them agree fails a test that says why they do not.

use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::{TypedTensor, TypedTensorView};

use crate::LinalgScalar;

/// Singular value decomposition of a borrowed matrix or batch, on the caller's bundle and pool.
///
/// Returns `(U, S, Vᴴ)` with `S` in the real scalar. With `full == false` the shapes are
/// `[m, k, b..]`, `[k, b..]` and `[k, n, b..]` with `k = min(m, n)`; with `full == true` they are
/// `[m, m, b..]`, `[k, b..]` and `[n, n, b..]`. See the [contract](crate#borrowed-exec-factorization-entries).
///
/// # Errors
///
/// - `Error::Validation` with `ValidationError::RankMismatch` when the view has fewer than two
///   dimensions.
/// - [`Error::Unsupported`](tenferro_tensor::Error::Unsupported) when a stride is negative.
/// - [`Error::RuntimeState`](tenferro_tensor::Error::RuntimeState) when the view's storage is not
///   host memory.
/// - `Error::Validation` with `ValidationError::InvalidArgument` when an output element count
///   overflows `usize`, or when the view's offsets reach outside its storage (role `layout`).
/// - A failure the provider reports, translated by [`map_error`](crate::error::map_error), whose
///   table gives the variant per `tlinalg` error kind; a numerical failure is `Error::Extension`
///   with kind `NumericalFailure` and a downcastable [`Error`](crate::Error) payload.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
/// use tenferro_tensor::TypedTensor;
///
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 2.0])?;
/// let mut engine = CpuEngine::serial();
/// let mut buffers = BufferPool::new();
/// let (u, s, vt) = engine.with_exec(|exec| {
///     tenferro_linalg::svd_with_exec(exec, &mut buffers, &a.as_view(), false)
/// })??;
/// assert_eq!(s.host_data()?, &[3.0, 2.0]);
/// assert_eq!((u.shape(), vt.shape()), (&[2, 2][..], &[2, 2][..]));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[allow(clippy::type_complexity)]
pub fn svd_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    full: bool,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T::RealScalar>, TypedTensor<T>)> {
    T::svd_with_exec(exec, buffers, view, full)
}

/// Singular values of a borrowed matrix or batch, in the real scalar, shape `[min(m, n), b..]`,
/// descending. See the [contract](crate#borrowed-exec-factorization-entries).
///
/// # Errors
///
/// - `Error::Validation` with `ValidationError::RankMismatch` when the view has fewer than two
///   dimensions.
/// - [`Error::Unsupported`](tenferro_tensor::Error::Unsupported) when a stride is negative.
/// - [`Error::RuntimeState`](tenferro_tensor::Error::RuntimeState) when the view's storage is not
///   host memory.
/// - `Error::Validation` with `ValidationError::InvalidArgument` when an output element count
///   overflows `usize`, or when the view's offsets reach outside its storage (role `layout`).
/// - A failure the provider reports, translated by [`map_error`](crate::error::map_error), whose
///   table gives the variant per `tlinalg` error kind; a numerical failure is `Error::Extension`
///   with kind `NumericalFailure` and a downcastable [`Error`](crate::Error) payload.
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex64;
/// use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
/// use tenferro_tensor::TypedTensor;
///
/// let a = TypedTensor::<Complex64>::from_vec_col_major(
///     vec![2, 2],
///     vec![
///         Complex64::new(0.0, 4.0),
///         Complex64::new(0.0, 0.0),
///         Complex64::new(0.0, 0.0),
///         Complex64::new(1.0, 0.0),
///     ],
/// )?;
/// let mut engine = CpuEngine::serial();
/// let mut buffers = BufferPool::new();
/// let s: TypedTensor<f64> = engine.with_exec(|exec| {
///     tenferro_linalg::svd_values_with_exec(exec, &mut buffers, &a.as_view())
/// })??;
/// assert_eq!(s.host_data()?, &[4.0, 1.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn svd_values_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    T::svd_values_with_exec(exec, buffers, view)
}

/// Thin QR of a borrowed matrix or batch: `(Q, R)` of shapes `[m, k, b..]` and `[k, n, b..]`,
/// `k = min(m, n)`. See the [contract](crate#borrowed-exec-factorization-entries).
///
/// # Errors
///
/// - `Error::Validation` with `ValidationError::RankMismatch` when the view has fewer than two
///   dimensions.
/// - [`Error::Unsupported`](tenferro_tensor::Error::Unsupported) when a stride is negative.
/// - [`Error::RuntimeState`](tenferro_tensor::Error::RuntimeState) when the view's storage is not
///   host memory.
/// - `Error::Validation` with `ValidationError::InvalidArgument` when an output element count
///   overflows `usize`, or when the view's offsets reach outside its storage (role `layout`).
/// - A failure the provider reports, translated by [`map_error`](crate::error::map_error), whose
///   table gives the variant per `tlinalg` error kind; a numerical failure is `Error::Extension`
///   with kind `NumericalFailure` and a downcastable [`Error`](crate::Error) payload.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
/// use tenferro_tensor::TypedTensor;
///
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;
/// let mut engine = CpuEngine::serial();
/// let mut buffers = BufferPool::new();
/// let (q, r) = engine.with_exec(|exec| {
///     tenferro_linalg::qr_with_exec(exec, &mut buffers, &a.as_view())
/// })??;
/// assert_eq!((q.shape(), r.shape()), (&[3, 2][..], &[2, 2][..]));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn qr_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
    T::qr_with_exec(exec, buffers, view)
}

/// Hermitian eigendecomposition of a borrowed square matrix or batch: `(values, vectors)` of
/// shapes `[n, b..]` (real scalar, ascending) and `[n, n, b..]`. See the [contract](crate#borrowed-exec-factorization-entries).
///
/// # Errors
///
/// - `Error::Validation` with `ValidationError::RankMismatch` when the view has fewer than two
///   dimensions.
/// - `Error::Validation` with `ValidationError::ShapeMismatch` when the matrix is not square
///   (`m != n`).
/// - [`Error::Unsupported`](tenferro_tensor::Error::Unsupported) when a stride is negative.
/// - [`Error::RuntimeState`](tenferro_tensor::Error::RuntimeState) when the view's storage is not
///   host memory.
/// - `Error::Validation` with `ValidationError::InvalidArgument` when an output element count
///   overflows `usize`, or when the view's offsets reach outside its storage (role `layout`).
/// - A failure the provider reports, translated by [`map_error`](crate::error::map_error), whose
///   table gives the variant per `tlinalg` error kind; a numerical failure is `Error::Extension`
///   with kind `NumericalFailure` and a downcastable [`Error`](crate::Error) payload.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
/// use tenferro_tensor::TypedTensor;
///
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 0.0, 0.0, 1.0])?;
/// let mut engine = CpuEngine::serial();
/// let mut buffers = BufferPool::new();
/// let (values, vectors) = engine.with_exec(|exec| {
///     tenferro_linalg::eigh_with_exec(exec, &mut buffers, &a.as_view())
/// })??;
/// assert_eq!(values.host_data()?, &[1.0, 2.0]);
/// assert_eq!(vectors.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn eigh_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<(TypedTensor<T::RealScalar>, TypedTensor<T>)> {
    T::eigh_with_exec(exec, buffers, view)
}

/// Hermitian eigenvalues of a borrowed square matrix or batch, in the real scalar, shape
/// `[n, b..]`, ascending. See the [contract](crate#borrowed-exec-factorization-entries).
///
/// # Errors
///
/// - `Error::Validation` with `ValidationError::RankMismatch` when the view has fewer than two
///   dimensions.
/// - `Error::Validation` with `ValidationError::ShapeMismatch` when the matrix is not square
///   (`m != n`).
/// - [`Error::Unsupported`](tenferro_tensor::Error::Unsupported) when a stride is negative.
/// - [`Error::RuntimeState`](tenferro_tensor::Error::RuntimeState) when the view's storage is not
///   host memory.
/// - `Error::Validation` with `ValidationError::InvalidArgument` when an output element count
///   overflows `usize`, or when the view's offsets reach outside its storage (role `layout`).
/// - A failure the provider reports, translated by [`map_error`](crate::error::map_error), whose
///   table gives the variant per `tlinalg` error kind; a numerical failure is `Error::Extension`
///   with kind `NumericalFailure` and a downcastable [`Error`](crate::Error) payload.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
/// use tenferro_tensor::TypedTensor;
///
/// let a = TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![5.0, 0.0, 0.0, -1.0])?;
/// let mut engine = CpuEngine::serial();
/// let mut buffers = BufferPool::new();
/// let values = engine.with_exec(|exec| {
///     tenferro_linalg::eigh_values_with_exec(exec, &mut buffers, &a.as_view())
/// })??;
/// assert_eq!(values.host_data()?, &[-1.0, 5.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn eigh_values_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    T::eigh_values_with_exec(exec, buffers, view)
}

/// General eigenvalues of a borrowed square matrix or batch, in the complex scalar
/// [`LinalgScalar::Complex`], shape `[n, b..]`.
///
/// The order of the eigenvalues within one matrix is the provider's and is not part of the
/// contract. See the [contract](crate#borrowed-exec-factorization-entries).
///
/// # Errors
///
/// - `Error::Validation` with `ValidationError::RankMismatch` when the view has fewer than two
///   dimensions.
/// - `Error::Validation` with `ValidationError::ShapeMismatch` when the matrix is not square
///   (`m != n`).
/// - [`Error::Unsupported`](tenferro_tensor::Error::Unsupported) when a stride is negative.
/// - [`Error::RuntimeState`](tenferro_tensor::Error::RuntimeState) when the view's storage is not
///   host memory.
/// - `Error::Validation` with `ValidationError::InvalidArgument` when an output element count
///   overflows `usize`, or when the view's offsets reach outside its storage (role `layout`).
/// - A failure the provider reports, translated by [`map_error`](crate::error::map_error), whose
///   table gives the variant per `tlinalg` error kind; a numerical failure is `Error::Extension`
///   with kind `NumericalFailure` and a downcastable [`Error`](crate::Error) payload.
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex64;
/// use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
/// use tenferro_tensor::TypedTensor;
///
/// // A rotation by 90 degrees has eigenvalues ±i.
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![0.0, 1.0, -1.0, 0.0])?;
/// let mut engine = CpuEngine::serial();
/// let mut buffers = BufferPool::new();
/// let values: TypedTensor<Complex64> = engine.with_exec(|exec| {
///     tenferro_linalg::eig_values_with_exec(exec, &mut buffers, &a.as_view())
/// })??;
/// let mut imag: Vec<f64> = values.host_data()?.iter().map(|value| value.im).collect();
/// imag.sort_by(f64::total_cmp);
/// assert!((imag[0] + 1.0).abs() < 1e-12 && (imag[1] - 1.0).abs() < 1e-12);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn eig_values_with_exec<T: LinalgScalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::Complex>> {
    T::eig_values_with_exec(exec, buffers, view)
}
